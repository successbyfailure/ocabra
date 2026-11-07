# Changelog

Cambios relevantes para quien construye contra la API de oCabra. No es el
historial completo del proyecto: aquí solo está lo que puede afectar a tu
cliente — cambios de comportamiento, parámetros nuevos y errores que estabas
sufriendo sin saberlo.

Cada versión marca con **Acción** lo que puede requerir que toques tu código.
La versión que estás usando la devuelve `GET /health`.

---

## 0.45.0 — 2026-11-07

### Nuevo

- **Esta página.** `GET /docs/changelog` recoge los cambios que afectan a tu
  cliente, y `GET /docs/models` recogerá cómo rinde cada modelo en este
  despliegue. Ambas se sirven también en Markdown crudo añadiendo `.md`, que es
  la misma fuente sin maquetar — pensado para agentes y para `curl`.

---

## 0.44.0 — 2026-11-07

### Cambios de comportamiento

**Si no mandas `max_tokens`, oCabra pone un tope por ti.** Antes, una petición
sin `max_tokens` generaba sin límite alguno: el modelo podía seguir hasta
agotar el contexto o hasta el timeout de inferencia, y entonces recibías un
`504` sin nada aprovechable. Ahora se inyecta el tope declarado del modelo (o
65536 si no declara ninguno), acotado por el contexto efectivo. Una generación
desbocada termina con `finish_reason: "length"` y el texto generado hasta ese
punto, que sí puedes usar.

Un `max_tokens` o `max_completion_tokens` explícito tuyo **siempre manda**;
esto solo rellena cuando callas.

**Acción**: si dependías de generaciones sin tope, pon `max_tokens`
explícitamente con el valor que necesites.

### Arreglos

- **`reasoning_effort: "off"` fallaba en modelos Qwen 3.5 servidos por vLLM.**
  La plantilla de chat de esa familia solo acepta `xhigh`, `medium` y `low`, y
  lanza excepción con cualquier otro valor; oCabra enviaba `"off"` tal cual.
  También fallaba `"high"`. Ahora `off` se traduce a `enable_thinking: false` y
  `high` a `xhigh`.
- **El tope de salida podía superar la ventana de contexto.** Varios modelos
  declaran 65536 tokens de salida con presets de 32k o 40960, así que el tope
  inyectado pedía más salida que contexto disponible. Ahora se acota.

---

## 0.43.0 — 2026-11-07

### Arreglos

- **`content` como lista de partes rompía en `qwen3:32b` y `qwen3:8b`.** Si
  enviabas `content: [{"type": "text", "text": "..."}]` (la forma con *content
  parts*, que usa el SDK de OpenAI) junto con `tools`, la petición fallaba con
  un error de plantilla. La causa: la plantilla de chat que trae Ollama para
  qwen3 es una revisión anterior a la oficial de Qwen y asume que `content` es
  una cadena. oCabra ahora aplana las partes de solo texto antes de enviar. El
  prompt resultante es idéntico; las listas con imagen, audio o vídeo se dejan
  intactas.
- **Al cerrar la conexión, la generación sigue parada.** Antes, en peticiones
  **no** streaming, si tu cliente se rendía por timeout, oCabra seguía
  generando hasta su propio límite de 1500 s, con la GPU ocupada. Ahora se
  detecta la desconexión y se cancela; la petición queda registrada como `499`.
- **Los modelos de embeddings ya no se anuncian como modelos de razonamiento.**
  `GET /v1/models` marcaba `reasoning: true` en `qwen3-embedding:*` porque sus
  pesos traen la plantilla de chat de qwen3. Si filtrabas modelos por esa
  capacidad, te aparecían de más.

---

## 0.42.0 — 2026-11-07

### Cambios de comportamiento

**`reasoning_effort` ahora hace algo en los modelos que lo soportan.** El nivel
viajaba en un sitio que la plantilla de chat nunca lee, así que las plantillas
que sí distinguen niveles caían a su valor por defecto — `xhigh` en la familia
Qwen 3.5, el esfuerzo máximo. En esos modelos `low`, `medium` y `high` daban
exactamente el mismo resultado; ahora no.

De las plantillas del parque, **solo las de la familia Qwen 3.5 distinguen
niveles** (`qwen3.8:27b` y sus variantes). El resto, incluida la de Gemma 4,
solo tienen interruptor de encendido/apagado: en ellas el nivel sigue siendo
indistinguible, pero sí cambia el tope duro de pensamiento (ver 0.41.0).

**Acción**: si usabas `high` contra un modelo Qwen 3.5 esperando el máximo, lo
sigues teniendo. Si usabas `low` esperando respuestas más rápidas y no lo
notabas, ahora lo notarás.

---

## 0.41.0 — 2026-11-07

### Cambios de comportamiento

**El pensamiento tiene tope real.** `reasoning_effort` era solo una sugerencia
que la plantilla podía ignorar, y nada impedía que un modelo pensara hasta
consumir toda la salida: llegó a ocurrir con 62.072 tokens de razonamiento y
132.263 de salida total antes de morir por timeout, devolviendo `content`
vacío. Ahora cada nivel lleva un tope de tokens de pensamiento (`low` 2048,
`medium` 16384, `high` sin tope). Al agotarse, el motor corta el razonamiento y
fuerza la respuesta final.

Si una respuesta llega vacía con `finish_reason: "length"`, es que el
pensamiento se comió el presupuesto: sube `max_tokens` o baja
`reasoning_effort`.

---

## 0.40.0 — 2026-11-07

### Arreglos

- **`POST /ocabra/models/refresh-tokenizer-fingerprints` devolvía 500 siempre.**
- **Realtime: los ids de hablante podían corromperse a mitad de sesión.** Al
  reanudar con `conversation_id` un registro guardado por otro modelo de
  diarización, el centroide se truncaba en silencio y dos hablantes distintos
  podían fusionarse. Ahora se detecta y se descarta el registro incompatible.
- **Batches: menos latencia ajena.** La lectura del JSONL de entrada, del audio
  y las imágenes de cada línea, y la escritura de los artefactos generados,
  bloqueaban el servidor entero mientras duraban. Afectaba al tiempo de
  respuesta de **todas** las peticiones concurrentes, no solo a los batches.
- **TRT-LLM: la estimación de VRAM se quedaba hasta 4 veces corta** al compilar
  motores (un modelo de 14B se estimaba como de 4B, uno de 27B como de 7B), lo
  que provocaba fallos de memoria.

---

## 0.39.0 y 0.39.1 — 2026-11-06

### Arreglos

- **Las estadísticas se perdían de forma intermitente.** Si consumes
  `/ocabra/stats/*`, había huecos: las tareas que las registraban podían ser
  recolectadas a medias bajo carga. Se observó un hueco de ~55 minutos.
- **`/ocabra/stats/performance` separa las tres latencias.** Antes el TTFT
  mezclaba espera en cola, arranque en frío del modelo y tiempo real del motor,
  lo que hacía imposible distinguir un modelo lento de uno que simplemente
  estaba frío. Nuevos campos: `ttftEngineP50Ms`, `ttftEngineP95Ms`,
  `queueP95Ms`, `maxQueueMs`, `coldStartRequests`, `loadP50Ms`, `maxLoadMs`.
  Los `ttftP50Ms` / `ttftP95Ms` anteriores siguen ahí con el valor bruto.

---

## 0.36.0 y 0.31.1 — 2026-11-06

### Arreglos de concurrencia

Si recibías `429` en ráfagas, probablemente era esto:

- **Los endpoints de encoder estaban limitados a 4 peticiones concurrentes.**
  Afectaba a `/v1/embeddings`, `/v1/rerank`, `/v1/score`, `/v1/pooling` y
  `/v1/classify`. El límite venía de un cálculo pensado para modelos
  generativos que se les aplicaba por error.
- **Los modelos llama.cpp sin preset de runtime también estaban capados a 4**,
  en lugar de a sus slots reales (hasta 16).

---

## 0.30.0 y 0.30.1 — 2026-11-05

- **`reasoning_effort` tiene un valor por defecto configurable** (global y por
  modelo), `medium` de serie. Antes, una petición que no lo mandaba dejaba al
  modelo razonar sin cota.
- **`diarize: true` falla explícitamente** en los backends que no la soportan,
  en lugar de devolver una transcripción silenciosamente sin hablantes.

---

## 0.29.0 — 2026-11-04

### Arreglos

- **Chat con `tools` devolvía 502 en modelos llama.cpp** si algún parámetro de
  la herramienta usaba un JSON Schema con `type` como objeto o unión (por
  ejemplo `{"type": {"enum": [...]}}`). Cualquier petición con esa forma
  tumbaba la conversación entera.
- **Al cancelar un stream se cierra la conexión con el motor**, que aborta la
  generación y libera la GPU en lugar de seguir generando para nadie.

---

## Cómo leer esta página desde un agente

Hay una versión en Markdown sin maquetar en `GET /docs/changelog.md`, que es la
misma fuente que genera esta página. La referencia completa de la API está en
`GET /docs` (OpenAPI/Swagger) y `GET /openapi.json`.
