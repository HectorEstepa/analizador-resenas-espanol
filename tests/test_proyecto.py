import unittest

from app import crear_app
from resenas.analisis import Analizador, validar_texto
from resenas.datos import extraer_estrellas, limpiar_filas


class ModelosFalsos:
    tiene_contexto = False

    def sentimiento(self, texto):
        return {"etiqueta": "POSITIVO", "puntaje": 0.8, "origen": "Doble de prueba"}


class DatosTests(unittest.TestCase):
    def test_formatos_y_rango(self):
        for texto in ["Clasificación 5 de 5 estrellas", "5", "5.0", "5 estrellas", "5/5"]:
            self.assertEqual(extraer_estrellas(texto), 5)
        for texto in ["0", "6", "15 estrellas", "3.5", "", None, "5 de 10"]:
            self.assertIsNone(extraer_estrellas(texto))

    def test_deduplicacion_y_conflictos(self):
        filas = [
            {"Comentario": " Muy   bueno ", "Estrellas": "5"},
            {"Comentario": "muy bueno", "Estrellas": "5"},
            {"Comentario": "Algo malo", "Estrellas": "1"},
            {"Comentario": "algo malo", "Estrellas": "4"},
            {"Comentario": "", "Estrellas": "5"},
        ]
        limpias, resumen = limpiar_filas(filas)
        self.assertEqual(limpias, [{"Comentario": "Muy bueno", "Estrellas": 5}])
        self.assertEqual(resumen["textos_con_etiquetas_conflictivas"], 1)

    def test_unicode_equivalente(self):
        filas = [{"Comentario": "café bueno", "Estrellas": "5"}, {"Comentario": "cafe\u0301 bueno", "Estrellas": "5"}]
        self.assertEqual(len(limpiar_filas(filas)[0]), 1)


class AnalisisTests(unittest.TestCase):
    def test_validacion(self):
        for valor in [None, 42, "hola", "1234 5678", "a " * 201, "x" * 2001]:
            with self.assertRaises(ValueError):
                validar_texto(valor)
        self.assertEqual(validar_texto("  MUY BUENO  "), "MUY BUENO")

    def test_binario_no_inventa_temas(self):
        resultado = Analizador(ModelosFalsos()).analizar("Excelente producto")
        self.assertNotIn("estrellas", resultado)
        self.assertIsNone(resultado["intencion"])
        self.assertEqual(resultado["sentimiento"]["etiqueta"], "POSITIVO")

    def test_no_calcula_ni_extrae_estrellas(self):
        class SinEstrellas(ModelosFalsos):
            def estrellas(self, texto):
                raise AssertionError("No debe ejecutarse")

        resultado = Analizador(SinEstrellas()).analizar("Le doy 4 de 5 estrellas")
        self.assertNotIn("estrellas", resultado)

    def test_modelos_opcionales(self):
        class Opcionales(ModelosFalsos):
            tiene_contexto = True

            def intencion(self, texto):
                return {"etiqueta": "calidad", "puntaje": 0.6, "origen": "Externo"}

        resultado = Analizador(Opcionales()).analizar("Buena calidad")
        self.assertEqual(resultado["intencion"]["etiqueta"], "calidad")

    def test_fallo_opcional_conserva_sentimiento(self):
        class OpcionalRoto(ModelosFalsos):
            tiene_contexto = True

            def intencion(self, texto):
                raise RuntimeError("No disponible")

        with self.assertLogs("resenas.analisis", level="ERROR"):
            resultado = Analizador(OpcionalRoto()).analizar("Buen producto")
        self.assertEqual(resultado["sentimiento"]["etiqueta"], "POSITIVO")
        self.assertIsNone(resultado["intencion"])
        self.assertEqual(len(resultado["avisos"]), 1)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.cliente = crear_app(ModelosFalsos()).test_client()

    def test_pagina_y_recursos(self):
        for ruta in ["/", "/static/app.js", "/static/styles.css", "/api/estado"]:
            with self.cliente.get(ruta) as respuesta:
                self.assertEqual(respuesta.status_code, 200)

    def test_analizar(self):
        respuesta = self.cliente.post("/api/analizar", json={"texto": "Excelente producto"})
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(respuesta.json["sentimiento"]["etiqueta"], "POSITIVO")

    def test_errores_de_entrada(self):
        for datos in [{}, {"texto": None}, {"texto": "hola"}, [], {"texto": 123}]:
            self.assertEqual(self.cliente.post("/api/analizar", json=datos).status_code, 400)
        self.assertEqual(self.cliente.post("/api/analizar", data="x").status_code, 415)
        self.assertEqual(self.cliente.post("/api/analizar", data="{", content_type="application/json").status_code, 400)

    def test_limite_tamano(self):
        self.assertEqual(self.cliente.post("/api/analizar", json={"texto": "x" * 20000}).status_code, 413)

    def test_origen_ajeno(self):
        respuesta = self.cliente.post("/api/analizar", json={"texto": "Buen producto"}, headers={"Origin": "https://otro.example"})
        self.assertEqual(respuesta.status_code, 403)

    def test_error_inferencia_sin_detalles_internos(self):
        class Rotos(ModelosFalsos):
            def sentimiento(self, texto):
                raise RuntimeError("ruta-privada-de-prueba")

        with self.assertLogs("app", level="ERROR"):
            respuesta = crear_app(Rotos()).test_client().post("/api/analizar", json={"texto": "Buen producto"})
        self.assertEqual(respuesta.status_code, 503)
        self.assertNotIn("ruta-privada-de-prueba", respuesta.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
