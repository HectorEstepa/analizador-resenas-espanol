"""Uso: python limpio.py --entrada datos.csv --salida reviews_limpio.csv"""
import argparse
import json
from pathlib import Path

from resenas.datos import guardar_csv, leer_csv, limpiar_filas

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Limpia y deduplica reseñas.")
    parser.add_argument("--entrada", type=Path, default=BASE / "datos.csv")
    parser.add_argument("--salida", type=Path, default=BASE / "reviews_limpio.csv")
    args = parser.parse_args()
    filas, resumen = limpiar_filas(leer_csv(args.entrada))
    guardar_csv(args.salida, filas)
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
