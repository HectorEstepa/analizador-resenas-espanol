import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app import crear_app
from resenas.entrenamiento import CLASES, cargar_datos, separar
from resenas.modelos import Modelos

BASE = Path(__file__).resolve().parents[1]


class PreparacionTests(unittest.TestCase):
    def cargar(self, filas):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "resenas.csv"
            with ruta.open("w", encoding="utf-8", newline="") as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=["Comentario", "Estrellas", "sentimiento", "requiere_revision", "origen", "sarcasmo", "modismos"])
                escritor.writeheader()
                escritor.writerows(filas)
            return cargar_datos([ruta])

    def test_exclusion_neutral_revision_y_sarcasmo(self):
        registros, resumen, _ = self.cargar([
            {"Comentario": "Un producto normal", "Estrellas": "3"},
            {"Comentario": "Comentario sin estrellas", "sentimiento": "neutral"},
            {"Comentario": "Es maravilloso", "Estrellas": "5", "sentimiento": "negativo", "requiere_revision": "true"},
            {"Comentario": "Es malo", "Estrellas": "5", "sentimiento": "negativo"},
            {"Comentario": "Excelente compra", "Estrellas": "5"},
            {"Comentario": "Pesima compra", "Estrellas": "1"},
            {"Comentario": "Maravilloso, llego roto", "Estrellas": "1", "sarcasmo": "si"},
        ])
        self.assertEqual([r["etiqueta"] for r in registros], ["POSITIVO", "NEGATIVO"])
        self.assertEqual(resumen["neutrales_excluidas"], 2)
        self.assertEqual(resumen["marcadas_sarcasmo_excluidas"], 1)
        self.assertEqual(resumen["pendientes_revision"], 1)
        self.assertEqual(resumen["sentimiento_estrellas_incompatibles"], 1)

    def test_conflictos_entre_archivos_se_descartan(self):
        registros, resumen, _ = self.cargar([
            {"Comentario": "Producto bueno", "Estrellas": "5"},
            {"Comentario": " producto   BUENO ", "Estrellas": "1"},
        ])
        self.assertEqual(registros, [])
        self.assertEqual(resumen["textos_conflictivos"], 1)

    def test_sin_negativos_no_entrena(self):
        with self.assertRaisesRegex(ValueError, "negativas"):
            separar([{"etiqueta": "POSITIVO", "origen": "prueba"}] * 30, 42)

    def test_configuracion_binaria_rechaza_ternaria(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal)
            config = ruta / "config.json"
            config.write_text('{}', encoding="utf-8")
            self.assertEqual(Modelos(ruta).informacion()["clases"], CLASES)
            config.write_text(json.dumps({"id2label": dict(enumerate(CLASES))}), encoding="utf-8")
            self.assertEqual(Modelos(ruta).informacion()["clases"], CLASES)
            config.write_text(json.dumps({"id2label": {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                Modelos(ruta).informacion()


class FlujoCompletoTests(unittest.TestCase):
    def test_entrenamiento_guardado_inferencia_y_metricas(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            datos, salida = raiz / "datos.csv", raiz / "modelo"
            with datos.open("w", encoding="utf-8", newline="") as archivo:
                escritor = csv.writer(archivo)
                escritor.writerow(["Comentario", "Estrellas"])
                for i in range(30):
                    escritor.writerow([f"Excelente maravilloso delicioso compra recomendada ejemplo {i}", 5])
                    escritor.writerow([f"Horrible roto defectuoso pesimo reclamo ejemplo {i}", 1])
                    escritor.writerow([f"Normal promedio intermedio indiferente sin opinion ejemplo {i}", 3])
            resultado = subprocess.run([sys.executable, "-B", str(BASE / "entrenar.py"), "--datos", str(datos), "--salida", str(salida), "--sin-ampliacion"], capture_output=True, text=True, timeout=90)
            self.assertEqual(resultado.returncode, 0, resultado.stderr)
            modelo = Modelos(salida)
            self.assertEqual(modelo.informacion()["clases"], ["NEGATIVO", "POSITIVO"])
            positivo = modelo.sentimiento("Excelente maravilloso delicioso compra recomendada")
            self.assertEqual(positivo["etiqueta"], "POSITIVO")
            self.assertEqual(set(positivo["probabilidades"]), set(CLASES))
            self.assertAlmostEqual(sum(positivo["probabilidades"].values()), 1.0)
            particiones = json.loads((salida / "particiones.json").read_text(encoding="utf-8"))
            huellas = {nombre: {r["sha256_texto"] for r in filas} for nombre, filas in particiones.items()}
            self.assertFalse(huellas["train"] & huellas["test"])
            self.assertFalse(huellas["validation"] & huellas["test"])
            self.assertFalse(huellas["train"] & huellas["validation"])
            cliente = crear_app(modelo).test_client()
            self.assertEqual(cliente.get("/api/estado").json["modelo"]["clases"], CLASES)
            reporte = cliente.get("/api/metricas").json
            self.assertTrue(reporte["disponible"])
            self.assertEqual(reporte["reporte"]["test"]["n"], len(huellas["test"]))
            respuesta = cliente.post("/api/analizar", json={"texto": "Horrible roto defectuoso pesimo reclamo"})
            self.assertEqual(respuesta.status_code, 200)
            self.assertEqual(respuesta.json["sentimiento"]["etiqueta"], "NEGATIVO")
            self.assertNotIn("NEUTRAL", respuesta.json["sentimiento"]["probabilidades"])
            # Una evaluación externa no debe presentar el dataset de entrenamiento como nuevo.
            evaluacion = subprocess.run([sys.executable, "-B", str(BASE / "evaluar.py"), "--datos", str(datos), "--modelo", str(salida), "--salida", str(raiz / "externa.json")], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(evaluacion.returncode, 0)
            self.assertIn("No quedan rese", evaluacion.stderr)
            datos_extra = raiz / "nuevos.csv"
            with datos_extra.open("w", encoding="utf-8", newline="") as archivo:
                escritor = csv.writer(archivo)
                escritor.writerow(["Comentario", "sentimiento", "origen"])
                escritor.writerow(["Entrega maravillosa en una compra distinta", "positivo", "prueba_sintetica"])
                escritor.writerow(["Servicio horrible en otra ciudad", "negativo", "prueba_sintetica"])
                escritor.writerow(["Sin opinion sobre esta compra diferente", "neutral", "prueba_sintetica"])
            externa = raiz / "externa.json"
            evaluacion = subprocess.run([sys.executable, "-B", str(BASE / "evaluar.py"), "--datos", str(datos_extra), "--modelo", str(salida), "--salida", str(externa)], capture_output=True, text=True, timeout=30)
            self.assertEqual(evaluacion.returncode, 0, evaluacion.stderr)
            informe = json.loads(externa.read_text(encoding="utf-8"))
            self.assertEqual(informe["evaluacion"]["n"], 2)
            self.assertTrue(informe["solapamiento_comprobado"])
            # Los pesos publicados se verifican antes de deserializar.
            metadata = json.loads((salida / "modelo.json").read_text(encoding="utf-8"))
            metadata["sha256_pesos"] = "incorrecto"
            (salida / "modelo.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertFalse(Modelos(salida).metricas()["disponible"])
            with self.assertRaisesRegex(RuntimeError, "huella"):
                Modelos(salida).sentimiento("Comentario de prueba")


if __name__ == "__main__":
    unittest.main()
