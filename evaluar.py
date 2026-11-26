"""Evalúa un CSV adicional sin entrenar ni incluir textos ya usados por el modelo."""
import argparse
import json
from pathlib import Path

from resenas.entrenamiento import CLASES, cargar_datos, evaluar
from resenas.modelos import Modelos

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, nargs="+", required=True)
    parser.add_argument("--modelo", type=Path, default=BASE / "modelo_publicado")
    parser.add_argument("--salida", type=Path, default=BASE / "evaluacion_externa.json")
    args = parser.parse_args()
    if args.salida.exists():
        parser.error("La salida ya existe; indica otro archivo para conservarla.")
    try:
        registros, limpieza, fuentes = cargar_datos(args.datos)
        ruta_particiones = args.modelo / "particiones.json"
        conocidos = set()
        if ruta_particiones.is_file():
            particiones = json.loads(ruta_particiones.read_text(encoding="utf-8"))
            conocidos = {fila["sha256_texto"] for filas in particiones.values() for fila in filas}
        nuevos = [r for r in registros if r["huella"] not in conocidos]
        if not nuevos:
            parser.error("No quedan reseñas nuevas después de excluir los textos de las particiones del modelo.")
        modelos = Modelos(args.modelo)
        predicciones = [modelos.sentimiento(r["texto"])["etiqueta"] for r in nuevos]
        reporte = {
            "clases": CLASES, "modelo": args.modelo.name, "fuentes": fuentes, "limpieza": limpieza,
            "textos_ya_conocidos_excluidos": len(registros) - len(nuevos),
            "solapamiento_comprobado": ruta_particiones.is_file(),
            "evaluacion": evaluar(nuevos, list(range(len(nuevos))), predicciones),
            "nota": "Se excluyen coincidencias exactas con todas las particiones conocidas. Esto no detecta paráfrasis ni verifica las etiquetas humanas; si falta particiones.json no se puede comprobar independencia.",
        }
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"archivo": str(args.salida), "n": len(nuevos), "f1_macro": reporte["evaluacion"]["f1_macro"]}, ensure_ascii=True))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
