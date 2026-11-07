# Modelos

Esta página recogerá, por modelo, cómo rinde de verdad en este despliegue:
latencia al primer token, tokens por segundo, contexto útil, comportamiento con
herramientas y razonamiento, y en qué casos conviene cada uno.

Está pendiente de un benchmark intensivo que se está ejecutando. Mientras
llegue, los datos que sí puedes consultar en vivo:

- `GET /v1/models` — modelos disponibles, sus capacidades (`chat`, `tools`,
  `vision`, `reasoning`, `embeddings`) y el contexto que anuncia cada uno.
- `GET /ocabra/stats/performance` — latencias reales medidas sobre el tráfico de
  este servidor, con la espera en cola, el arranque en frío y el tiempo del
  motor separados (ver la entrada 0.39.1 del [changelog](/docs/changelog)).

Para elegir un tope de salida o un nivel de razonamiento con criterio, lee
antes las entradas 0.41.0 y 0.42.0 del [changelog](/docs/changelog): no todos
los modelos distinguen los niveles de `reasoning_effort`.
