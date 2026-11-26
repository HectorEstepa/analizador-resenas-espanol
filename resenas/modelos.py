"""Carga diferida: importar la API no descarga ni carga modelos."""
import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

TEMAS = [
    "calidad del producto", "envío y logística", "atención al cliente",
    "precio y oferta", "problema con el producto", "recomendación positiva",
    "diseño y apariencia", "funcionalidad y rendimiento", "durabilidad del producto",
]


class Modelos:
    def __init__(self, ruta, contexto=False):
        self.ruta = Path(ruta)
        self.tiene_contexto = contexto
        self._modelo = self._tokenizador = self._contexto = None
        self._metadata = None

    def informacion(self):
        if self._metadata is None:
            archivo = self.ruta / "modelo.json"
            if archivo.is_file():
                datos = json.loads(archivo.read_text(encoding="utf-8"))
                if datos.get("motor") != "lineal":
                    raise RuntimeError("Motor del modelo no reconocido.")
                self._metadata = datos
            else:
                archivo = self.ruta / "config.json"
                config = json.loads(archivo.read_text(encoding="utf-8")) if archivo.is_file() else {}
                etiquetas = config.get("id2label", {"0": "NEGATIVO", "1": "POSITIVO"})
                clases = [etiquetas[str(i)].upper() for i in range(len(etiquetas))]
                if clases == ["LABEL_0", "LABEL_1"]:
                    clases = ["NEGATIVO", "POSITIVO"]
                self._metadata = {"motor": "beto", "clases": clases}
            clases = self._metadata.get("clases", [])
            if len(clases) != 2 or set(clases) != {"NEGATIVO", "POSITIVO"}:
                self._metadata = None
                raise RuntimeError("Selecciona un modelo binario con las clases NEGATIVO y POSITIVO.")
        clases = self._metadata["clases"]
        return {"motor": self._metadata["motor"], "clases": clases}

    def metricas(self):
        ruta = self.ruta / "evaluacion.json"
        if not ruta.is_file():
            return {"disponible": False, "motivo": "Este modelo no tiene un informe de prueba independiente en el formato actual."}
        reporte = json.loads(ruta.read_text(encoding="utf-8"))
        if reporte.get("formato") != 2:
            return {"disponible": False, "motivo": "El informe pertenece a una versión anterior. Vuelve a entrenar para generar las métricas actuales."}
        info = self.informacion()
        if info["motor"] == "lineal" and reporte.get("sha256_pesos") != self._metadata.get("sha256_pesos"):
            return {"disponible": False, "motivo": "El informe no corresponde a los pesos de este modelo."}
        return {"disponible": True, "reporte": reporte}

    def _cargar_local(self):
        if self._modelo is not None:
            return
        informacion = self.informacion()
        if informacion["motor"] == "lineal":
            import joblib

            archivo = self.ruta / "clasificador.joblib"
            huella = hashlib.sha256(archivo.read_bytes()).hexdigest()
            if huella != self._metadata.get("sha256_pesos"):
                raise RuntimeError("La huella de los pesos no coincide con modelo.json.")
            modelo = joblib.load(archivo)
            if set(modelo.classes_) != set(informacion["clases"]):
                raise RuntimeError("Los pesos y los metadatos tienen clases distintas.")
            self._modelo = modelo
            return
        if not (self.ruta / "config.json").is_file():
            raise RuntimeError("No se encontró el modelo local. Ejecuta entrenar.py o indica --modelo.")
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

        tokenizador = AutoTokenizer.from_pretrained(self.ruta, local_files_only=True)
        modelo = TFAutoModelForSequenceClassification.from_pretrained(self.ruta, local_files_only=True)
        self._tokenizador, self._modelo = tokenizador, modelo

    def sentimiento(self, texto):
        self._cargar_local()
        informacion = self.informacion()
        if informacion["motor"] == "lineal":
            probabilidades = self._modelo.predict_proba([texto])[0]
            clases = list(self._modelo.classes_)
            origen = "TF-IDF + regresión logística · sentimiento positivo o negativo"
        else:
            import tensorflow as tf

            entradas = self._tokenizador(texto, truncation=True, max_length=128, return_tensors="tf")
            logits = self._modelo(**entradas, training=False).logits
            probabilidades = tf.nn.softmax(logits, axis=-1).numpy()[0]
            clases = informacion["clases"]
            origen = "BETO entrenado localmente · clasificación binaria"
        indice = int(probabilidades.argmax())
        return {
            "etiqueta": clases[indice],
            "puntaje": float(probabilidades[indice]),
            "origen": origen,
            "probabilidades": {clase: float(p) for clase, p in zip(clases, probabilidades)},
        }

    @staticmethod
    def _pipeline_externo(tarea, nombre):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        # Reutiliza primero los pesos completos de la caché, sin comprobar la red.
        # En una instalación nueva los descarga si todavía no están disponibles.
        for solo_local in (True, False):
            try:
                tokenizador = AutoTokenizer.from_pretrained(nombre, use_fast=False, local_files_only=solo_local)
                modelo = AutoModelForSequenceClassification.from_pretrained(nombre, local_files_only=solo_local)
                return pipeline(tarea, framework="pt", model=modelo, tokenizer=tokenizador)
            except OSError:
                if not solo_local:
                    raise

    def intencion(self, texto):
        if self._contexto is None:
            self._contexto = self._pipeline_externo(
                "zero-shot-classification", "Recognai/bert-base-spanish-wwm-cased-xnli",
            )
        resultado = self._contexto(
            texto, candidate_labels=TEMAS,
            hypothesis_template="Esta reseña trata sobre {}.", multi_label=False,
        )
        return {
            "etiqueta": resultado["labels"][0],
            "puntaje": float(resultado["scores"][0]),
            "origen": "Modelo externo Recognai · tema aproximado por zero-shot",
        }
