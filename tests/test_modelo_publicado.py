"""Casos de desarrollo: protegen fallos conocidos, no miden generalización."""
import hashlib
import json
import unittest
from pathlib import Path

from app import crear_app
from resenas.datos import normalizar_texto
from resenas.modelos import Modelos
from resenas.texto import marcar_negacion

BASE = Path(__file__).resolve().parents[1]


class ModeloPublicadoTests(unittest.TestCase):
    def test_regresion_quejas_y_contrastes(self):
        modelos = Modelos(BASE / "modelo_publicado")
        cliente = crear_app(modelos).test_client()
        casos = [
            ("NEGATIVO", "El producto llegó roto y nadie respondió a mi reclamo."),
            ("NEGATIVO", "El paquete venía quebrado y nadie me dio solución."),
            ("NEGATIVO", "No contestan y mi compra sigue sin llegar."),
            ("NEGATIVO", "Me cobraron, no entregaron nada y siguen ignorándome."),
            ("NEGATIVO", "La calidad es pésima. Se rompió el primer día."),
            ("POSITIVO", "El producto llegó intacto y respondieron todas mis preguntas."),
            ("POSITIVO", "No llegó roto; venía bien protegido y funciona perfecto."),
            ("POSITIVO", "La atención fue excelente y recibí todo a tiempo."),
            ("POSITIVO", "Me ayudaron con el cambio y quedé muy conforme."),
        ]
        particiones = json.loads((BASE / "modelo_publicado" / "particiones.json").read_text(encoding="utf-8"))
        conocidas = {fila["sha256_texto"] for filas in particiones.values() for fila in filas}
        for esperado, texto in casos:
            with self.subTest(texto=texto):
                # No se añadió literalmente el texto a los datos para memorizarlo.
                huella = hashlib.sha256(normalizar_texto(texto).casefold().encode("utf-8")).hexdigest()
                self.assertNotIn(huella, conocidas)
                respuesta = cliente.post("/api/analizar", json={"texto": texto})
                self.assertEqual(respuesta.status_code, 200)
                self.assertEqual(respuesta.json["sentimiento"]["etiqueta"], esperado)

    def test_negacion_no_atraviesa_puntuacion(self):
        self.assertEqual(marcar_negacion("No funciona. Buen envío."), "no neg_funciona buen envio")

    def test_ampliacion_esta_solo_en_entrenamiento(self):
        particiones = json.loads((BASE / "modelo_publicado" / "particiones.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(f["origen"] == "ampliacion_sintetica" for f in particiones["train"]), 120)
        for nombre in ("validation", "test"):
            self.assertFalse(any(f["origen"] == "ampliacion_sintetica" for f in particiones[nombre]))


if __name__ == "__main__":
    unittest.main()
