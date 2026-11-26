"""Datos, particiones y métricas compartidos por los dos motores."""
import hashlib
from collections import Counter

from .datos import extraer_estrellas, leer_csv, normalizar_texto

CLASES = ["NEGATIVO", "POSITIVO"]


def cargar_datos(rutas):
    grupos, resumen, fuentes = {}, Counter(), []
    for ruta in rutas:
        fuentes.append({"archivo": ruta.name, "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest()})
        for fila in leer_csv(ruta, requeridas=("Comentario",)):
            resumen["originales"] += 1
            if normalizar_texto(fila.get("requiere_revision")).casefold() in {"true", "si", "sí", "1"}:
                resumen["pendientes_revision"] += 1
                continue
            if normalizar_texto(fila.get("sarcasmo")).casefold() in {"si", "sí", "ambiguo", "true", "1"}:
                resumen["marcadas_sarcasmo_excluidas"] += 1
                continue
            texto = normalizar_texto(fila.get("Comentario"))
            estrellas = extraer_estrellas(fila.get("Estrellas"))
            explicito = normalizar_texto(fila.get("sentimiento")).upper()
            if explicito == "NEUTRAL" or estrellas == 3:
                resumen["neutrales_excluidas"] += 1
                continue
            derivado = None if estrellas is None else CLASES[0 if estrellas < 3 else 1]
            if not texto or (explicito and explicito not in CLASES) or (not explicito and derivado is None):
                resumen["invalidas"] += 1
                continue
            if explicito and derivado and explicito != derivado:
                resumen["sentimiento_estrellas_incompatibles"] += 1
                continue
            etiqueta = explicito or derivado
            clave = texto.casefold()
            registro = {
                "texto": texto, "etiqueta": etiqueta,
                "origen": normalizar_texto(fila.get("origen")) or "resenas_originales",
                "modismos": bool(normalizar_texto(fila.get("modismos"))),
                "huella": hashlib.sha256(clave.encode("utf-8")).hexdigest(),
            }
            if clave in grupos:
                resumen["duplicadas"] += 1
                grupos[clave]["etiquetas"].add(etiqueta)
                anterior = grupos[clave]["registro"]
                if anterior["origen"] != registro["origen"]:
                    anterior["origen"] = "origen_compartido"
            else:
                grupos[clave] = {"registro": registro, "etiquetas": {etiqueta}}
    registros = []
    for grupo in grupos.values():
        if len(grupo["etiquetas"]) != 1:
            resumen["textos_conflictivos"] += 1
        else:
            registros.append(grupo["registro"])
    resumen["conservadas"] = len(registros)
    return registros, dict(resumen), fuentes


def separar(registros, semilla):
    import numpy as np
    from sklearn.model_selection import train_test_split

    conteos = Counter(r["etiqueta"] for r in registros)
    if any(conteos[clase] < 10 for clase in CLASES):
        raise ValueError("Se necesitan al menos 10 reseñas únicas positivas y 10 negativas.")
    indices = np.arange(len(registros))
    estratos = [f'{r["etiqueta"]}|{r["origen"]}' for r in registros]
    if min(Counter(estratos).values()) < 10:
        estratos = [r["etiqueta"] for r in registros]
    entrenamiento, resto = train_test_split(indices, test_size=0.30, stratify=estratos, random_state=semilla)
    validacion, prueba = train_test_split(resto, test_size=0.5, stratify=np.array(estratos)[resto], random_state=semilla)
    return {"train": entrenamiento, "validation": validacion, "test": prueba}


def metricas(reales, predichas):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    return {
        "n": len(reales), "accuracy": float(accuracy_score(reales, predichas)),
        "f1_macro": float(f1_score(reales, predichas, labels=CLASES, average="macro", zero_division=0)),
        "por_clase": classification_report(reales, predichas, labels=CLASES, output_dict=True, zero_division=0),
        "matriz_confusion": confusion_matrix(reales, predichas, labels=CLASES).tolist(),
    }


def evaluar(registros, indices, predicciones):
    reales = [registros[i]["etiqueta"] for i in indices]
    resultado = metricas(reales, list(predicciones))
    subconjuntos = {f"origen:{origen}": [j for j, i in enumerate(indices) if registros[i]["origen"] == origen]
                   for origen in sorted({registros[i]["origen"] for i in indices})}
    subconjuntos["modismos_sinteticos"] = [j for j, i in enumerate(indices) if registros[i]["origen"] == "sintetico_ia" and registros[i]["modismos"]]
    resultado["subconjuntos"] = {
        nombre: metricas([reales[j] for j in posiciones], [predicciones[j] for j in posiciones])
        for nombre, posiciones in subconjuntos.items() if posiciones
    }
    return resultado


def manifiesto(registros, particiones):
    return {
        nombre: [{"sha256_texto": registros[i]["huella"], "etiqueta": registros[i]["etiqueta"],
                  "origen": registros[i]["origen"],
                  "modismos": registros[i]["modismos"]} for i in ids]
        for nombre, ids in particiones.items()
    }
