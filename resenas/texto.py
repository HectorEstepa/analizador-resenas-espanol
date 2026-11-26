"""Características de negación para el clasificador estadístico."""
import re
import unicodedata


def marcar_negacion(texto):
    """Marca hasta tres palabras tras una negación, sin asignar sentimiento.

    La puntuación cierra el alcance. Es una aproximación sintáctica, no un
    intérprete del lenguaje ni una regla que fuerce una clase de salida.
    """
    texto = "".join(c for c in unicodedata.normalize("NFKD", texto.casefold()) if not unicodedata.combining(c))
    tokens = re.findall(r"\w+|[^\w\s]", texto)
    salida, restantes = [], 0
    for token in tokens:
        if token in {"no", "sin", "nunca", "jamas", "ni", "nadie"}:
            restantes = 3
            salida.append(token)
        elif not token.isalnum():
            restantes = 0
        else:
            salida.append("neg_" + token if restantes else token)
            restantes = max(0, restantes - 1)
    return " ".join(salida)
