"""Entrena sentimiento NEGATIVO / POSITIVO y guarda su evaluación."""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from resenas.entrenamiento import CLASES, cargar_datos, evaluar, manifiesto, separar

BASE = Path(__file__).resolve().parent
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def entrenar_lineal(registros, particiones, args):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.metrics import f1_score
    from resenas.texto import marcar_negacion

    # Vocabulario e IDF se ajustan exclusivamente con entrenamiento.
    texto = FeatureUnion([
        ("palabras", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), max_features=20000, sublinear_tf=True)),
        ("caracteres", TfidfVectorizer(strip_accents="unicode", analyzer="char_wb", ngram_range=(3, 5), max_features=30000, sublinear_tf=True)),
        ("negacion", TfidfVectorizer(preprocessor=marcar_negacion, ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
    ])
    ids = particiones["train"]
    x_train = texto.fit_transform([registros[i]["texto"] for i in ids])
    x_val = texto.transform([registros[i]["texto"] for i in particiones["validation"]])
    y_train = [registros[i]["etiqueta"] for i in ids]
    y_val = [registros[i]["etiqueta"] for i in particiones["validation"]]
    candidatos, mejor, mejor_f1 = [], None, -1.0
    tiene_ampliacion = any(registros[i].get("ampliacion", False) for i in ids)
    for regularizacion in (0.5, 2.0, 8.0):
        for peso in ((1.0, 3.0) if tiene_ampliacion else (1.0,)):
            pesos = np.array([peso if registros[i].get("ampliacion", False) else 1.0 for i in ids])
            clasificador = LogisticRegression(C=regularizacion, class_weight="balanced", max_iter=1500, random_state=args.semilla)
            clasificador.fit(x_train, y_train, sample_weight=pesos)
            f1 = float(f1_score(y_val, clasificador.predict(x_val), labels=CLASES, average="macro", zero_division=0))
            candidatos.append({"C": regularizacion, "peso_ampliacion": peso, "validation_f1_macro": f1})
            print(json.dumps(candidatos[-1]), flush=True)
            if f1 > mejor_f1:
                mejor, mejor_f1 = clasificador, f1
                elegida = candidatos[-1]
    modelo = Pipeline([("texto", texto), ("clasificador", mejor)])

    def predecir(indices):
        return modelo.predict([registros[i]["texto"] for i in indices])

    return modelo, predecir, {"seleccion_validacion": {"criterio": "F1 macro de validación, sin consultar prueba", "elegida": elegida, "candidatos": candidatos}}


def entrenar_beto(registros, particiones, args):
    import numpy as np
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

    tf.keras.utils.set_random_seed(args.semilla)
    tokenizador = AutoTokenizer.from_pretrained(args.base)
    etiquetas = np.array([CLASES.index(r["etiqueta"]) for r in registros])

    def dataset(ids, entrenando=False):
        entradas = tokenizador([registros[i]["texto"] for i in ids], truncation=True, padding="max_length", max_length=128, return_tensors="tf")
        resultado = tf.data.Dataset.from_tensor_slices((dict(entradas), etiquetas[ids]))
        if entrenando:
            resultado = resultado.shuffle(len(ids), seed=args.semilla)
        return resultado.batch(args.batch).prefetch(tf.data.AUTOTUNE)

    modelo = TFAutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(CLASES), id2label=dict(enumerate(CLASES)),
        label2id={clase: i for i, clase in enumerate(CLASES)}, ignore_mismatched_sizes=True,
    )
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    pesos = compute_class_weight("balanced", classes=np.arange(len(CLASES)), y=etiquetas[particiones["train"]])
    historia = modelo.fit(
        dataset(particiones["train"], True), validation_data=dataset(particiones["validation"]),
        epochs=args.epocas, class_weight={i: float(p) for i, p in enumerate(pesos)},
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)],
    )

    def predecir(ids):
        logits = modelo.predict(dataset(ids)).logits
        return np.array(CLASES)[np.argmax(logits, axis=-1)]

    return (modelo, tokenizador), predecir, {
        "modelo_base": args.base, "revision_modelo_base": getattr(modelo.config, "_commit_hash", None),
        "historial": {k: [float(v) for v in valores] for k, valores in historia.history.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    disponibles = [p for p in [BASE / "reviews_limpio.csv", BASE / "dataset_resenas_completo_300.csv"] if p.is_file()]
    parser.add_argument("--datos", type=Path, nargs="+", default=disponibles, help="Uno o más CSV. Se excluyen filas requiere_revision=true.")
    parser.add_argument("--salida", type=Path, default=BASE / "modelos" / "resenas-chile")
    parser.add_argument("--motor", choices=["lineal", "beto"], default="lineal")
    parser.add_argument("--base", default="dccuchile/bert-base-spanish-wwm-uncased", help="Modelo base o carpeta local para --motor beto.")
    parser.add_argument("--epocas", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--ampliacion", type=Path, nargs="+", default=[BASE / "datos" / "ampliacion_entrenamiento.csv"] if (BASE / "datos" / "ampliacion_entrenamiento.csv").is_file() else [])
    parser.add_argument("--sin-ampliacion", action="store_true", help="No añade el CSV complementario de entrenamiento.")
    args = parser.parse_args()
    if not args.datos:
        parser.error("Indica al menos un CSV mediante --datos.")
    if args.epocas < 1 or args.batch < 1:
        parser.error("Las épocas y el lote deben ser positivos.")
    if args.salida.exists() and (not args.salida.is_dir() or any(args.salida.iterdir())):
        parser.error("La salida debe ser una carpeta nueva o vacía; no se sobrescriben modelos.")
    try:
        registros, limpieza, fuentes = cargar_datos(args.datos)
        particiones = separar(registros, args.semilla)
        ampliacion = {"n": 0, "fuentes": [], "limpieza": {}}
        if args.ampliacion and not args.sin_ampliacion:
            import numpy as np
            adicionales, limpieza_extra, fuentes_extra = cargar_datos(args.ampliacion)
            conocidas = {r["huella"] for r in registros}
            adicionales = [r for r in adicionales if r["huella"] not in conocidas]
            ampliacion = {"n": len(adicionales), "fuentes": fuentes_extra, "limpieza": limpieza_extra, "uso": "solo_entrenamiento"}
            inicio = len(registros)
            for registro in adicionales:
                registro["ampliacion"] = True
            registros.extend(adicionales)
            particiones["train"] = np.concatenate([particiones["train"], np.arange(inicio, len(registros), dtype=int)])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"motor": args.motor, "limpieza": limpieza, "clases": dict(Counter(r["etiqueta"] for r in registros))}, ensure_ascii=True), flush=True)
    entrenar = entrenar_lineal if args.motor == "lineal" else entrenar_beto
    modelo, predecir, extras = entrenar(registros, particiones, args)
    mayoritaria = Counter(registros[i]["etiqueta"] for i in particiones["train"]).most_common(1)[0][0]
    prueba = predecir(particiones["test"])
    reporte = {
        "formato": 2, "motor": args.motor, "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "semilla": args.semilla, "clases": CLASES, "fuentes": fuentes, "limpieza": limpieza,
        "distribucion": dict(Counter(r["etiqueta"] for r in registros)),
        "ampliacion_entrenamiento": ampliacion,
        "particiones": {nombre: {"n": len(ids), "clases": dict(Counter(registros[i]["etiqueta"] for i in ids))} for nombre, ids in particiones.items()},
        "validation": evaluar(registros, particiones["validation"], predecir(particiones["validation"])),
        "test": evaluar(registros, particiones["test"], prueba),
        "baseline_mayoria": evaluar(registros, particiones["test"], [mayoritaria] * len(prueba)),
        "limitaciones": ["Etiquetas originales derivadas de estrellas; etiquetas sintéticas generadas por IA sin auditoría humana completa.",
                         "Partición estratificada por texto único; no separa paráfrasis, productos, usuarios ni fechas.",
                         "Los subconjuntos de modismos son pequeños y sintéticos; no demuestran dominio del español chileno.",
                         "La clasificación binaria no comprende todas las ironías ni valida la coherencia de los comentarios.",
                         "Los puntajes de inferencia no están calibrados."],
        "versiones": {"python": sys.version, **{p: version(p) for p in (["numpy", "scikit-learn", "joblib"] + (["tensorflow", "tf-keras", "transformers"] if args.motor == "beto" else []))}},
        **extras,
    }
    args.salida.mkdir(parents=True, exist_ok=True)
    if args.motor == "lineal":
        import joblib
        joblib.dump(modelo, args.salida / "clasificador.joblib", compress=3)
        pesos = args.salida / "clasificador.joblib"
        metadata = {"motor": "lineal", "clases": CLASES, "sha256_pesos": hashlib.sha256(pesos.read_bytes()).hexdigest()}
        reporte["sha256_pesos"] = metadata["sha256_pesos"]
        (args.salida / "modelo.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        modelo[0].save_pretrained(args.salida)
        modelo[1].save_pretrained(args.salida)
    (args.salida / "evaluacion.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.salida / "particiones.json").write_text(json.dumps(manifiesto(registros, particiones), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"salida": str(args.salida), "test_accuracy": reporte["test"]["accuracy"], "test_f1_macro": reporte["test"]["f1_macro"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
