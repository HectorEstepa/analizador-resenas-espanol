"""Limpieza sin dependencias de aprendizaje automático."""
import csv
import re
import unicodedata
from collections import Counter


def normalizar_texto(texto):
    return " ".join(unicodedata.normalize("NFC", str(texto or "")).split())


def extraer_estrellas(valor):
    texto = normalizar_texto(valor).lower()
    patrones = (
        r"([1-5])(?:\.0)?",
        r"(?:clasificación|clasificacion)\s+([1-5])\s+de\s+5\s+estrellas?",
        r"([1-5])\s*(?:de\s*5|/\s*5|estrellas?)",
    )
    for patron in patrones:
        coincidencia = re.fullmatch(patron, texto)
        if coincidencia:
            return int(coincidencia.group(1))
    return None


def limpiar_filas(filas):
    grupos = {}
    resumen = Counter()
    for fila in filas:
        resumen["originales"] += 1
        texto = normalizar_texto(fila.get("Comentario"))
        estrellas = extraer_estrellas(fila.get("Estrellas"))
        if not texto or estrellas is None:
            resumen["invalidas"] += 1
            continue
        clave = texto.casefold()
        if clave in grupos:
            resumen["duplicadas"] += 1
            grupos[clave][2].add(estrellas)
        else:
            grupos[clave] = [texto, estrellas, {estrellas}]
    limpias = []
    for texto, estrellas, etiquetas in grupos.values():
        if len(etiquetas) > 1:
            resumen["textos_con_etiquetas_conflictivas"] += 1
            continue
        limpias.append({"Comentario": texto, "Estrellas": estrellas})
    resumen["conservadas"] = len(limpias)
    return limpias, dict(resumen)


def leer_csv(ruta, requeridas=("Comentario", "Estrellas")):
    with open(ruta, encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if not set(requeridas).issubset(lector.fieldnames or []):
            raise ValueError("El CSV debe incluir las columnas: " + ", ".join(requeridas))
        return list(lector)


def guardar_csv(ruta, filas):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["Comentario", "Estrellas"])
        escritor.writeheader()
        escritor.writerows(filas)
