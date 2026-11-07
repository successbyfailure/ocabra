# Changelog

Cambios relevantes para quien construye contra la API de oCabra. No es el
historial completo del proyecto: aquí solo está lo que puede afectar a tu
cliente — cambios de comportamiento, parámetros nuevos y errores que estabas
sufriendo sin saberlo.

Cada versión marca con **Acción** lo que puede requerir que toques tu código.
Si algo dejó de funcionar o empezó a comportarse raro, busca aquí la versión
que estás usando y lee hacia arriba: la devuelve `GET /health`.

Para leerlo desde un agente hay una copia en Markdown sin maquetar en
`GET /docs/changelog.md`, que es la misma fuente que genera esta página. La
referencia completa de los endpoints está en `GET /docs` y `GET /openapi.json`,
y cómo rinde cada modelo, en [`/docs/models`](/docs/models).

---

## 0.50.1 — 2026-11-07

### Arreglos

- **`reasoning_effort: "high"` devolvía `500` en los modelos Qwen 3.5**
  (`qwen3.8:27b`, `qwen3.8:27b-abliterated`, `ravenx-chaos-agent`). Su
  plantilla solo acepta `xhigh`, `medium` y `low`, y aunque oCabra ya traducía
  `high` → `xhigh`, mandaba además el valor crudo en un segundo campo que el
  motor lee con prioridad: la plantilla recibía `high` y lanzaba
  "Unexpected reasoning effort high". Afectaba a toda petición con `high`,
  desde 0.42.0. `low`, `medium` y `off` nunca estuvieron afectados.

### Documentación

- **[`/docs/models`](/docs/models) prometía que `gemma4:26b` está "siempre
  cargado"**. No es cierto y conviene que lo sepas: su pin es evictable y
  reserva 23.600 MB de los 24.576 de la tarjeta, así que cuando otro modelo
  grande se la queda no puede volver hasta que quede libre, y entre tanto
  responde `409`. La página lo explica ahora y apunta a
  `recomendado-ha-chatbots`, que enruta al que esté cargado.

---

## 0.50.0 — 2026-11-07

### Sin cambios para tu cliente

- Esta versión solo cambia el panel de administración (editor de modelos).
  La API, los nombres de modelo y las respuestas son los mismos que en 0.49.0.

---

## 0.49.0 — 2026-11-07

### Cambios de comportamiento

- **Los nombres que llevan el motor como sufijo dejan de existir.**
  `gemma4:26b-ollama`, `qwen3-coder:30b-ollama`, `qwen3-embedding:8b-ollama`,
  `mistral:7b-llamacpp-mks` y el resto de esa forma devuelven ahora `404`, en
  los endpoints de OpenAI y en los de Ollama. El error te dice los dos nombres
  que sí valen, así que no hace falta volver aquí para arreglarlo.
  **Acción:** cámbialos por el nombre del modelo (`gemma4:26b`), que es el
  normal; o por `<motor>/<modelo>` (`ollama/gemma4:26b`) si de verdad
  necesitas ese motor en concreto. Esa segunda forma no ha cambiado y sigue
  dándote el mismo despliegue, con sus mismos presets de contexto.

  Por qué: el motor que ejecuta un modelo es una decisión de este servidor y
  puede cambiar. Un nombre que lo lleva dentro te deja fijado al lado perdedor
  de ese cambio sin que te enteres.

---

## 0.48.0 — 2026-11-07

### Sin cambios para tu cliente

- Esta versión solo cambia el panel de administración (gestión de modelos y
  del explorador de descargas). La API, los nombres de modelo y las respuestas
  son los mismos que en 0.47.1.

---

## 0.47.1 — 2026-11-07

### Documentación

- **[`/docs/models`](/docs/models) ya tiene datos.** Por modelo: tokens por
  segundo y velocidad de prefill medidos sobre el tráfico real de este
  servidor, contexto por defecto y presets, cuáles arrancan en frío y cuánto
  tardan, qué modelos distinguen de verdad los niveles de `reasoning_effort`
  (solo tres) y cuáles solo tienen interruptor, dimensiones de los embeddings,
  y qué significan los `409`, `429` y `504` que puedes recibir. La copia sin
  maquetar para agentes está en `GET /docs/models.md`.

---

## 0.47.0 — 2026-11-07

### Cambios de comportamiento

- **Llama al modelo por su nombre y se cargará con el backend que tenga
  configurado.** El motor que sirve cada modelo (llama.cpp, Ollama, vLLM,
  Whisper…) es una decisión de este servidor, no tuya: se cambia desde la
  administración y puede cambiar sin avisarte. Tú pides `gemma4:26b` y recibes
  el despliegue que esté configurado en ese momento.
- **Cada modelo aparece una sola vez en `GET /v1/models` y `GET /api/tags`**,
  con el nombre que tienes que usar.
- **Si de verdad necesitas un motor concreto, pídelo como
  `<motor>/<modelo>`**: `ollama/gemma4:26b`, `ollama/qwen3-embedding:8b`. Esa
  es la forma de hacerlo (y desde 0.49.0, la única).
- **El acceso se concede por modelo**, no por motor: si tu clave puede usar un
  modelo, lo puede usar sea cual sea el backend que lo sirva.

### Arreglos

- **En los endpoints de Ollama (`/api/chat`, `/api/generate`,
  `/api/embeddings`, `/api/show`), un nombre servido por dos motores a la vez
  se resolvía al primero que apareciese**, que podía no ser el configurado.
  Ahora gana siempre el que está configurado para ese modelo.

---

## 0.46.1 — 2026-11-07

### Cambios de comportamiento

- **`nemotron-3-nano:30b` pasa a 512k tokens de contexto** (antes 64k). Es el
  modelo con la ventana más grande del servidor; su techo de arquitectura es
  1.048.576 y cabe entero en las GPUs, así que puede subirse más si lo
  necesitas. Ten en cuenta que leer un prompt de ese tamaño lleva varios
  minutos antes de que empiece a responder: para uso interactivo, 256k es un
  punto más cómodo.

### Arreglos

- **Las páginas de `/docs/changelog` y `/docs/models` devolvían 404.** Se
  anunciaron en 0.45.0 pero el proxy no enrutaba las subrutas de `/docs`, así
  que nunca llegaron a verse desde fuera. Afectaba igual a
  `/docs/oauth2-redirect`, el retorno de OAuth del Swagger.
- **`nemotron-3-nano:30b` se rechazaba con `insufficient_vram`** al pedirle más
  de unos 300k tokens de contexto, aunque cupiese de sobra: el cálculo de
  memoria no reconocía que la mayor parte de sus capas no guardan caché de
  atención y le reservaba casi nueve veces lo que necesita.

---

## 0.46.0 — 2026-11-07

### Cambios de comportamiento

**Por dentro, oCabra separa el modelo del motor que lo ejecuta. Los nombres
que usas no cambian.** Lo único que necesitas saber es el nombre del modelo
—`gemma4:26b`, `whisper-large-v3-turbo`—; el servidor decide con qué motor lo
sirve.

- Los nombres de modelo, los routers `recomendado-*` y los alias siguen
  funcionando igual.
- `recomendado-ha-chatbots` usa ahora la configuración por defecto de cada uno
  de sus modelos.
- En las estadísticas (`/ocabra/stats/*`), `model_id` es siempre el nombre del
  modelo y el motor va en un campo aparte; lo que pidió el cliente se guarda en
  `requested_model`.

**Acción**: ninguna. Si en algún momento guardaste un id interno con el motor
delante, deja de depender de esa forma y usa el nombre del modelo: es lo único
que el servidor garantiza.

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

De los modelos de este servidor, **solo tres distinguen niveles**:
`qwen3.8:27b`, `qwen3.8:27b-abliterated` y `ravenx-chaos-agent`, todos de la
familia Qwen 3.5. El resto, incluidos los Gemma 4 y `ravenx`, solo tienen
interruptor de encendido/apagado: en ellos el nivel sigue siendo
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

- **Realtime: los ids de hablante podían corromperse a mitad de sesión.** Al
  reanudar con `conversation_id` un registro guardado por otro modelo de
  diarización, el centroide se truncaba en silencio y dos hablantes distintos
  podían fusionarse. Ahora se detecta y se descarta el registro incompatible.
- **Batches: menos latencia ajena.** La lectura del JSONL de entrada, del audio
  y las imágenes de cada línea, y la escritura de los artefactos generados,
  bloqueaban el servidor entero mientras duraban. Afectaba al tiempo de
  respuesta de **todas** las peticiones concurrentes, no solo a los batches.

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
- **Las peticiones que se iban a un servidor federado heredaban el límite de
  concurrencia del local**, en vez del que tuviera el modelo en el servidor que
  de verdad las atendía.

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
