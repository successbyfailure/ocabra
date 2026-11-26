"""Diagnóstico de fallos de benchmark: ¿es el modelo o es cómo lo servimos?

Toma los casos que fallaron en un intento de benchmark y, antes de proponerlos
para entrenar, descarta que el fallo venga del servido: plantilla de chat,
parámetros de muestreo efectivos, límites de contexto o salida, o el propio
arnés del benchmark. Solo lo que sobrevive a esas comprobaciones queda como
«modelo» y se exporta como candidato a entrenamiento.

Piezas:

- ``cases``: extrae los casos fallidos del espacio de trabajo de un intento.
- ``reference``: la ficha oficial del modelo en Hugging Face (plantilla y
  muestreo recomendado).
- ``serving``: lo que de verdad ve el backend (prompt renderizado, parámetros
  efectivos, contexto por slot), consultado al worker vivo.
- ``replay``: repite casos por la cola normal de oCabra y los puntúa con el
  checker oficial de cada app, dentro de su entorno.
- ``classify`` y ``report``: categoría por caso, informe y exportación JSONL.
- ``jobs``: los diagnósticos como trabajos en disco que ejecuta la cola de
  benchmarks (nunca a la vez que un benchmark).
"""
