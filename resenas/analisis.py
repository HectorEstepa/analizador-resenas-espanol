"""Validación y contrato de resultados, independientes de TensorFlow."""
import logging
import threading

from .datos import normalizar_texto

MAX_CARACTERES = 2000


def validar_texto(valor):
    if not isinstance(valor, str):
        raise ValueError("La reseña debe ser texto.")
    if len(valor) > MAX_CARACTERES:
        raise ValueError(f"Escribe hasta {MAX_CARACTERES} caracteres.")
    texto = normalizar_texto(valor)
    if len(texto.split()) < 2:
        raise ValueError("Escribe una reseña de al menos dos palabras.")
    if len(texto.split()) > 200:
        raise ValueError("Escribe una reseña de hasta 200 palabras.")
    if sum(letra.isalpha() for letra in texto) < 3:
        raise ValueError("La reseña debe contener palabras.")
    return texto


class Analizador:
    def __init__(self, modelos):
        self.modelos = modelos
        self.bloqueo = threading.Lock()

    def analizar(self, valor):
        texto = validar_texto(valor)
        with self.bloqueo:
            sentimiento = self.modelos.sentimiento(texto)
            intencion = None
            avisos = []
            if self.modelos.tiene_contexto:
                try:
                    intencion = self.modelos.intencion(texto)
                except Exception:
                    logging.getLogger(__name__).exception("Falló el modelo opcional de intención")
                    avisos.append("No se pudo cargar o ejecutar el modelo opcional de intención.")
        return {
            "sentimiento": sentimiento,
            "intencion": intencion,
            "avisos": avisos,
            "advertencia": (
                "Los puntajes no son probabilidades calibradas de acierto. "
                "La clasificación de intención no valida si un comentario es verdadero o spam."
            ),
        }
