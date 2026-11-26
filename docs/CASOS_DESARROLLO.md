# Corrección del ejemplo negativo

El modelo anterior clasificaba «El producto llegó roto y nadie respondió a mi reclamo.» como positivo, con puntaje 0,680. El modelo corregido lo clasifica como negativo, con puntaje 0,815. Los puntajes no son probabilidades calibradas de acertar.

Se amplió el entrenamiento con 120 ejemplos sintéticos balanceados de daños, entregas, atención y devoluciones. Se normalizaron variantes con y sin tilde y se añadió una representación del alcance de negación: marca hasta tres palabras después de una negación y se detiene ante puntuación. Es una característica aprendida por el clasificador, no una regla que asigne sentimiento. La API y los botones recorren la misma inferencia y no contienen excepciones para el ejemplo.

Se seleccionó regularización y peso del complemento mediante validación, manteniendo prueba aparte. Las huellas de los textos de la partición de prueba coinciden con la versión binaria anterior. El complemento no aparece en validación ni prueba.

Se comprobaron once frases de desarrollo: cinco quejas y seis comentarios favorables, incluidos contrastes con negaciones y reclamos resueltos. Nueve reciben la clase esperada y están protegidos por una prueba de regresión. Ninguna frase de ese grupo se copió literalmente al entrenamiento; hay ejemplos relacionados. Estos casos se usaron durante el desarrollo y **no son una estimación independiente de calidad**.

## Errores que siguen abiertos

| Texto | Clase esperada | Resultado actual |
| --- | --- | --- |
| Solucionaron mi reclamo y me devolvieron el dinero. | POSITIVO | NEGATIVO (0,608) |
| No tuve problemas con el producto ni con el envío. | POSITIVO | NEGATIVO (0,656) |

Estos fallos muestran que el modelo estadístico todavía no interpreta todo el contexto. No se fuerzan sus respuestas ni se ocultan esos ejemplos. Una ampliación con etiquetas revisadas y una evaluación externa más extensa son necesarias antes de afirmar comprensión general del lenguaje.
