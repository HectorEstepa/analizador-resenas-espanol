# Versión binaria

Se retiraron la clase neutral y los ejemplos de sarcasmo de la interfaz. El modelo ligero se entrenó de nuevo con las clases NEGATIVO y POSITIVO; no se redistribuyeron artificialmente los puntajes del modelo anterior.

El entrenamiento excluye neutrales, tres estrellas, filas pendientes de revisión y las marcadas como sarcasmo o ambiguas. Los CSV originales y los pesos de BETO se conservan. Las ironías no etiquetadas no se detectan automáticamente.

Se publican los nuevos pesos ligeros, su evaluación y el manifiesto de particiones. Las métricas anteriores de tres clases se respaldan y no se muestran como si correspondieran al modelo nuevo. El README explica cómo usar los pesos incluidos, entrenar un modelo propio o BETO binario y evaluar reseñas nuevas.

Se verificaron 22 pruebas automatizadas, entrenamiento real del modelo ligero, inferencia, API y flujo de navegador. El panel muestra dos clases y no incluye opciones de neutral o sarcasmo. La captura se actualizó en `vista.png`.

Las métricas de esta tarea binaria y su partición de prueba no se comparan directamente con las del modelo anterior de tres clases. Los límites del dataset y la procedencia pendiente se mantienen documentados en README.

## Corrección de quejas directas

Se corrigió la regresión del ejemplo negativo con normalización de tildes, representación de negación y 120 ejemplos sintéticos complementarios exclusivamente en entrenamiento. Se eligen hiperparámetros con validación y se conserva la misma partición de prueba. El informe actual reemplaza las cifras de la versión anterior. Las limitaciones y dos fallos de desarrollo aún abiertos están en CASOS_DESARROLLO.md.
