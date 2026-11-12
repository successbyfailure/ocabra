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

## 0.75.0 — 2026-11-12

### Cambios

- **Acción: los modelos de Ollama ya no tienen presets de contexto.** Ollama
  pasa a ser un único servidor para todos sus modelos, con el paralelismo y el
  contexto máximo globales (por defecto 2 peticiones simultáneas y 65.536
  tokens). Se han borrado los presets 32K/64K/128K/MAX de los perfiles
  `-ollama`, y `POST /ocabra/profiles/{id}/runtime-presets` responde `400`
  para modelos de Ollama. Si mandabas `ocabra.context_preset` a un perfil
  `-ollama`, ya no hace nada: el modelo carga con el contexto global. Para
  contextos grandes usa el despliegue de llama.cpp del mismo modelo.
- **El `num_ctx` que pide un cliente a un modelo de Ollama se recorta al
  máximo global** (o al contexto propio del modelo, si es menor). Antes, un
  `options.num_ctx` mayor obligaba a Ollama a recargar el modelo con una KV
  para la que nadie había reservado memoria.
- **El paralelismo y el contexto máximo de Ollama** se leen en
  `GET /ocabra/config` (`ollamaNumParallel` y `ollamaMaxContext`) y se cambian
  con `PATCH` (solo administradores). Aplicarlos reinicia Ollama
  (`POST /ocabra/ollama/server/apply-settings`), que descarga todos sus
  modelos. `GET /ocabra/ollama/server/settings` dice qué tiene el servidor en
  marcha y si quedan cambios por aplicar.

---

## 0.74.2 — 2026-11-12

### Novedades

- **Los embeddings guardan su velocidad.** Las peticiones a `/v1/embeddings`,
  `/api/embed` y `/api/embeddings` no registraban ninguna cifra de
  rendimiento: el motor no la devuelve, y `promptTokensPerSecond` salía vacío
  en `/ocabra/stats` y en las últimas peticiones del dashboard. Ahora se
  calcula con los tokens de entrada y el tiempo que tarda el worker, sin
  contar la espera en cola de oCabra ni la carga del modelo. Así se puede
  comparar un mismo modelo en CPU y en GPU. Si el worker atiende varias
  peticiones en paralelo, la cifra incluye su cola interna. La respuesta de
  los endpoints no cambia.
- **Las transcripciones guardan el perfil que pidió el cliente.** Una
  transcripción a `whisper-large-v3-turbo-diarized` se registraba con el
  nombre del worker, que comparte con `whisper-large-v3-turbo`, así que no
  quedaba rastro de que fuera diarizada. Ahora se guarda el nombre pedido.
  Las filas de `GET /ocabra/stats/recent` lo traen en `requestedModel`;
  `modelId` sigue siendo el modelo.
- **`GET /ocabra/models/{id}/insights`** (nuevo): ritmos medidos de prefill y
  generación, salida típica y tiempo de carga en frío del modelo; si su
  plantilla de chat lee `reasoning_effort` y `enable_thinking`; y si está en
  mantenimiento.

---

## 0.74.0 — 2026-11-12

### Cambios

- **Acción: cambiar la configuración de un modelo exige el rol
  `model_manager`.** `PATCH /ocabra/models/{id}` aceptaba a cualquier usuario
  autenticado, aunque esa configuración acaba en la línea de comandos del
  worker. Ahora responde `403` a los usuarios normales, igual que perfiles y
  presets. Cargar y descargar modelos sigue abierto a todos.
- **`PATCH /ocabra/models/{id}` valida `extraConfig`.** Una opción con un tipo
  o un valor imposible (`ctx_size: "32k"`, `parallel_slots: 0`, un
  `cache_type_k` que no existe) devuelve `400` con `detail.errors`
  (`key`, `label`, `message`) en vez de guardarse y fallar en la siguiente
  carga. Las claves desconocidas se siguen aceptando.
- **`vramEstimateMb` en el `PATCH` ya tiene efecto.** Antes se descartaba en
  silencio. Ahora fija la reserva de VRAM en los modelos llama.cpp y vLLM, y
  da `400` en los backends que no la admiten. Cambiar `backendType` da `400`:
  un modelo cambia de backend añadiendo un despliegue.
- **Los `request_defaults` de un router se aplican.** Antes el router se
  sustituía por el perfil elegido y sus valores por defecto se perdían. Ahora
  van por encima de los del perfil destino y por debajo de lo que mande el
  cliente.
- **Los `load_overrides` de un perfil se fusionan con la configuración del
  modelo.** Un perfil que cambiaba una sola opción de vLLM o llama.cpp
  sustituía la sección entera, y el resto de opciones del modelo se perdían en
  ese worker. Ahora solo cambia lo que el perfil indica; un `null` quita la
  opción.
- **Los horarios de un modelo funcionan con cualquier política de carga.**
  Solo actuaban sobre los modelos `warm`; en `pin` y `on_demand` no hacían
  nada. Un modelo fijado ya no se recarga dentro de su franja.
- **`GET /ocabra/profiles/{id}/runtime-presets/resolved` devuelve la VRAM
  estimada de cada preset**: `estimated_vram_mb` y `vram_by_gpu`.
- **Un `null` en `mmproj_model_id` o en `speculative.draft_model_id` quita el
  proyector o el draft.** Antes se ignoraba y el modelo seguía cargándolos.

### Correcciones

- `GET /ocabra/models/{id}/profiles` y
  `GET /ocabra/models/{id}/speculative-candidates` respondían siempre `404`
  ("model '<id>/profiles' not found"): otra ruta se los comía.
- vLLM con un preset de KV `q8` o `q4` arrancaba sin `--kv-cache-dtype`, así
  que usaba fp16 mientras la comprobación de si cabe calculaba la mitad o un
  cuarto. Ahora arranca en `fp8`, que es lo que vLLM admite, y el cálculo lo
  tiene en cuenta.
- El `keep_alive_seconds` de un modelo llama.cpp del fork de oCabra
  (`llamacpp_mks`) guardado en su sección `llama_cpp` se ignoraba y el modelo
  se descargaba con el tiempo global.
- `split_mode: "tensor"` se marcaba como no válido en la estimación de
  memoria aunque llama.cpp lo acepta.
- TensorRT-LLM: el contexto se pasaba como presupuesto de tokens por paso
  (`--max_num_tokens`) en vez de como longitud máxima (`--max_seq_len`).
- Un horario mal formado en el `PATCH` daba `500`; ahora da `400`.

---

## 0.73.4 — 2026-11-12

### Correcciones

- **Las transcripciones diarizadas ya no fallan con `500 Internal Server Error`
  cuando la tarjeta de whisper está llena.** Desde la 0.73.3 el diarizador se
  carga bajo demanda, y si su tarjeta no tenía sitio (por ejemplo, con un
  modelo de embeddings grande al lado) la transcripción se quedaba sin
  memoria de GPU. Ahora, antes de diarizar, oCabra desaloja modelos ociosos de
  esa tarjeta para hacerle sitio, como hacía cuando el diarizado era un worker
  aparte. Si aun así falta memoria, la respuesta es `503` con
  `code: "transcription_out_of_memory"` (o `diarization_out_of_memory`) y
  `Retry-After`, en vez de un 500.
- **Batches: los errores transitorios vuelven a reencolarse.** El procesador
  de lotes no leía el código de error de las respuestas de oCabra, así que
  `insufficient_vram` y las faltas de memoria de whisper fallaban la línea a
  la primera en vez de reintentarla. Ahora se reintentan hasta el plazo de
  reintento del lote. Además, el `error` de las líneas fallidas en el fichero
  de errores pasa a ser el objeto de error (`code`, `message`, `type`,
  `param`), no el envoltorio `{"detail": {"error": …}}`.

---

## 0.73.3 — 2026-11-12

### Correcciones

- **Un perfil whisper normal ya no diariza por su cuenta.** El perfil normal y
  su `-diarized` podían acabar en el mismo proceso. Si ese proceso lo había
  arrancado el diarizado, las transcripciones del perfil normal salían
  diarizadas: más lentas y con hablantes que nadie había pedido. Ahora
  oCabra dice en cada petición si se diariza: el perfil `-diarized` diariza y
  el normal no, salvo que el cliente mande `diarize` explícito.
- **El diarizador se libera cuando no se usa.** El de los perfiles
  `-diarized` se quedaba cargado para siempre (unos 2 GB de VRAM). Ahora se
  descarga tras 5 minutos sin diarizar, como en el resto. La primera
  diarización después de esa pausa tarda unos segundos más, lo que cuesta
  volver a cargarlo.
- **El perfil normal y el `-diarized` del mismo modelo comparten un solo
  worker.** Antes podían cargar dos copias de los mismos pesos, que no
  cabían juntas en su tarjeta: alternar los dos tipos de petición hacía que
  cada una desalojase a la otra. Esto sustituye al desvío de la 0.72.3.

---

## 0.73.2 — 2026-11-12

### Correcciones

- **El ritmo en vivo de las peticiones que generan tool calls ya es el real.**
  El contador de tokens en curso solo sumaba el texto y el razonamiento, no los
  argumentos de las llamadas a herramientas. Un agente que escribe un fichero
  dentro de una tool call aparecía en `/ocabra/models/activity` (y en el
  dashboard) parado, a ~1 t/s, mientras el modelo generaba a su ritmo normal.
  El registro final ya era correcto porque sale de la cifra del backend; si el
  backend no la da, ahora también cuenta esos argumentos.

---

## 0.73.1 — 2026-11-11

### Arreglos

- **Las transcripciones guardan siempre la duración del audio.** Antes solo
  se conocía si la respuesta la traía, y con `response_format=json` (el
  formato por defecto) no venía: `audioSeconds` quedaba vacío y no había
  factor de tiempo real. Ahora el worker la manda en la cabecera
  `X-Audio-Duration`, en todos los formatos. `verbose_json` incluye además
  `duration`, como la API de OpenAI. Solo aplica a las transcripciones
  nuevas.

---

## 0.73.0 — 2026-11-11

### Cambios

- **Avance y velocidad de las transcripciones en curso con whisper.**
  `/ocabra/models/activity` añade a cada transcripción en curso un objeto
  `audio`. Trae la fase (`transcribing` o `diarizing`), la duración total del
  audio, los segundos ya transcritos, el porcentaje y `realtimeFactor`: los
  segundos de audio por segundo de reloj (14 = catorce veces más rápido que
  el audio). El tiempo restante se calcula con esa velocidad. Solo funciona
  con el backend faster-whisper; whisper.cpp no publica su avance. Durante la
  diarización no hay porcentaje.
- **Duración del audio en las transcripciones terminadas.**
  `/ocabra/stats/recent` y el feed `finished` de la actividad añaden
  `audioSeconds`. Junto con la duración de la petición da su factor de
  tiempo real.

---

## 0.72.3 — 2026-11-11

### Correcciones

- **Alternar transcripciones normales y diarizadas del mismo modelo ya no
  recarga el modelo cada vez.** `faster-whisper-large-v3` y
  `faster-whisper-large-v3-diarized` corrían en dos procesos con los mismos
  pesos, que no cabían juntos en su tarjeta: cada cambio de tipo de petición
  desalojaba al otro y la petición esperaba de 13 a 20 s a que cargase. Ahora,
  si el diarizado está cargado y el normal no, la petición sin diarizar se
  transcribe en el diarizado (sin diarizar). La respuesta es la misma.

---

## 0.72.2 — 2026-11-11

### Correcciones

- **Las peticiones en curso de audio, imágenes, vídeo, voz en tiempo real y
  agentes ya dicen de quién son.** En `/ocabra/models/activity` (y en la
  sección Peticiones del dashboard) salían sin `username`, como "anónimo",
  porque esos endpoints marcaban la petición en curso sin el usuario. Las de
  chat, completions y embeddings ya lo traían.

---

## 0.72.1 — 2026-11-11

### Cambios

- **Al arrancar, los modelos fijados (PIN) ya no desalojan a nadie.** Tras el
  minuto de espera de la 0.71.0, cada modelo fijado solo carga si cabe en la
  VRAM libre. Si la tarjeta está ocupada por otro modelo usado hace poco, ese
  modelo se queda, y el fijado vuelve a comprobarlo cada dos minutos hasta que
  haya sitio. Pedir un modelo fijado sigue cargándolo en el acto, como
  cualquier petición.

---

## 0.72.0 — 2026-11-11

### Cambios

- **`/ocabra/models/activity` añade `finished`: las peticiones terminadas en
  los últimos 20 s**, incluidos los rechazos 429. De cada una da usuario,
  llave, tipo, estado, tokens (entrada, cache y salida), ritmos de prefill y
  generación, TTFT, espera en cola y duración. Las peticiones muy cortas
  (embeddings, transcripciones pequeñas) empiezan y terminan entre dos
  refrescos del snapshot y antes no llegaban a verse. Nunca incluye IP ni
  user agent.

---

## 0.71.0 — 2026-11-11

### Cambios

- **Los modelos fijados (PIN) se cargan un minuto después de arrancar.** Antes
  empezaban a cargar en el mismo instante del arranque, y si en ese momento
  llegaba una petición, su modelo y los fijados se peleaban por la VRAM: la
  petición podía esperar mucho más de lo normal, o fallar, mientras
  un modelo desalojaba a otro. Ahora la petición carga primero lo que
  necesita, y los fijados entran después, cuando acaba esa carga y solo si
  siguen descargados. Durante ese primer minuto, pedir un modelo fijado lo
  carga en el acto, como cualquier otro, así que la primera petición a uno de
  ellos tras un reinicio puede tardar lo que tarde en cargarse.

---

## 0.70.10 — 2026-11-11

### Cambios

- **`/ocabra/models/activity` añade `ramByModel`.** Da los MB de RAM del host
  (RSS en vivo) de los modelos cuyo backend mantiene pesos en RAM aunque
  corran en GPU. De momento solo Strata. Antes la RAM de un modelo solo se
  medía si corría entero en CPU, y una vez al cargarlo: un modelo de Strata en
  las dos GPU figuraba con 0 MB de RAM y ocupaba unos 26 GB.

---

## 0.70.9 — 2026-11-11

### Arreglos

- **Cargar la visión en un modelo de dos GPU ya no recorta de más.** El hueco
  para el proyector se calculaba en la tarjeta donde estaba antes (la de
  whisper), no en la que lo carga ahora: `qwen3.8:27b` `extended` bajaba a
  225.280 tokens al recibir la primera imagen, unos 20.000 más de lo necesario.

**Acción:** ninguna.

---

## 0.70.8 — 2026-11-11

### Arreglos

- **0.70.7 rompió la carga de los presets de dos GPU con visión** (por ejemplo
  `qwen3.8:27b` con contexto `extended`): respondían 503 `Failed to load model
  ... exited with rc=2`. Durante unos minutos de esta noche. Arreglado.

**Acción:** ninguna. Si recibiste ese 503, repite la petición.

---

## 0.70.7 — 2026-11-11

### Arreglos

- **El proyector de visión de 0.70.6 seguía cargándose en la GPU de whisper.**
  El backend de llama.cpp no recibía el estado de las GPU, así que no sabía cuál
  era la tarjeta grande. Por la misma causa, `split_strategy: "evenly"` no
  calculaba el reparto entre tarjetas y lo dejaba en manos de llama.cpp.

**Acción:** ninguna.

---

## 0.70.6 — 2026-11-11

### Arreglos

- **Diarizar junto a un modelo que llena las dos tarjetas ya no da 500.** Con
  `qwen3.8:27b` en su preset de 262k y la visión cargada, una transcripción con
  `diarize=true` se quedaba sin memoria en la GPU de whisper y devolvía un 500
  opaco. Ahora se intenta, por orden:
  1. El proyector de visión de los modelos repartidos entre dos GPU se carga en
     la tarjeta grande, no en la que comparte con whisper.
  2. Si pyannote se queda sin memoria, reintenta con lotes más pequeños
     (32 → 16 → 8 → 4). Diariza más despacio, pero no echa a nadie.
  3. Si ni así cabe, se expulsa el modelo de esa tarjeta, como en cualquier
     carga sin sitio, y se repite la transcripción. Ese modelo tarda en volver
     a cargarse en su siguiente petición.
  4. Si tampoco hay nada que expulsar, responde **503** con código
     `diarization_out_of_memory` y `Retry-After: 30`.

**Acción:** ninguna. Si tratas los 503, ya puedes reintentar este según
`Retry-After` o repetir la petición sin `diarize`.

---

## 0.70.5 — 2026-11-11

### Arreglos

- **Al cargar la visión, el modelo cede solo el contexto que hace falta.**
  Antes lo reducía a la mitad: `qwen3.8:27b` pasaba de 114.688 a 57.344 tokens
  cuando con la visión cargada le cabían unos 90.000. Si el preset tiene una
  medición, ahora usa el coste real por tarjeta y recorta lo justo.

**Acción:** ninguna.

---

## 0.70.4 — 2026-11-11

### Arreglos

- **La primera imagen después de cargar un modelo ya no lo tumba.** La 0.70.0
  hacía sitio al proyector de visión, pero si la imagen llegaba nada más
  terminar la carga, oCabra todavía veía la tarjeta vacía, no hacía sitio y el
  modelo se caía. Ahora mide la memoria de cada proceso en ese mismo momento.

**Acción:** ninguna.

---

## 0.70.3 — 2026-11-11

### Arreglos

- **`/ocabra/models` muestra por fin las capacidades de la plantilla en los
  modelos de llama.cpp no cargados.** La 0.70.2 lo arreglaba solo a medias: en
  los modelos que no se habían cargado desde el último reinicio seguía saliendo
  `flint:40b` sin herramientas ni razonamiento.

**Acción:** ninguna.

---

## 0.70.2 — 2026-11-11

### Arreglos

- **Las capacidades de los modelos no cargados ahora sí salen de su
  plantilla.** En la 0.70.1 los listados seguían mostrando las capacidades
  antiguas (`flint:40b` sin herramientas ni razonamiento). Ahora se leen de la
  plantilla del modelo la primera vez que hacen falta.

**Acción:** ninguna.

---

## 0.70.1 — 2026-11-11

### Arreglos

- **`/ocabra/models` también muestra bien las herramientas y el razonamiento
  de los modelos de llama.cpp que no están cargados.** La 0.69.0 lo arregló
  solo en `/v1/models`.
- **Medir el contexto de un preset ya no deja fuera a los modelos que tuvo que
  descargar.** Esos modelos se recargaban antes de que la tarjeta hubiera
  liberado la memoria y se quedaban en error. Ahora oCabra espera a que la
  memoria quede libre y reintenta la carga.

**Acción:** ninguna.

---

## 0.70.0 — 2026-11-11

### Arreglos

- **La primera imagen ya no tumba un modelo que tiene la tarjeta llena.** Antes,
  si un modelo de llama.cpp ocupaba toda la tarjeta y llegaba una imagen, el
  proyector de visión no cabía y el worker se caía. El cliente recibía la
  conexión cortada (`incomplete chunked read`), como le pasó a `gemma4:26b` el
  10 de noviembre. Ahora oCabra reduce primero el contexto lo justo para que
  quepa el proyector y luego lo carga. Si ni así cabe, la petición falla con un
  error que lo explica y el modelo sigue vivo para las peticiones sin imagen.
- **Transcribir con diarización ya no se queda sin memoria si la RTX 3060 está
  llena.** Diarizar cuesta unos 3,4 GB más que transcribir. Antes de una
  transcripción que lo necesita, oCabra reduce el contexto del modelo grande
  que comparte la tarjeta. El diarizador se descarga tras 5 minutos sin
  diarizar y devuelve esa memoria.

### Nuevo

- **Medición del contexto máximo de un preset**, desde la página de modelos
  (perfil → presets → «Medir contexto máximo») o con
  `POST /ocabra/profiles/{perfil}/runtime-presets/{preset}/context-probe`.
  Mide en la tarjeta real hasta dónde cabe el contexto, dejando el margen de
  seguridad (`vram_buffer_mb`). Mientras mide, descarga lo que haya en esas
  GPU y el modelo responde `503` con `Retry-After`. «Aplicar» lo guarda en el
  preset.
- **Los presets de dos tarjetas medidos dejan sitio a Whisper en la RTX 3060.**
  Arrancan con el contexto que deja libre la reserva (2.700 MB por defecto) y
  crecen hasta el máximo medido solo cuando una petición lo pide. A los 120 s
  sin peticiones vuelven al contexto de arranque.

**Acción:** ninguna. Si una petición pide más contexto del que el modelo tiene
en ese momento, la primera vez tarda unos segundos más mientras crece.

---

## 0.69.0 — 2026-11-11

### Cambios de comportamiento

- **`reasoning_effort: "low"` y `"medium"` ahora piensan menos en los modelos de
  llama.cpp.** El tope de tokens de pensamiento pasa a 1.000 con `low` y a
  4.096 con `medium`; antes eran 2.048 y 16.384. Muchas plantillas ignoran el
  nivel (`gemma4`, `ornith`, `nemotron`, `ravenx`) y `flint:40b` ni siquiera
  deja apagar el pensamiento, así que el tope es lo único que distingue un
  nivel de otro. Hasta ahora `flint:40b` pensaba lo mismo (unos 750 tokens) con
  cualquier nivel, también en tareas de clasificación rápida. La tabla por
  modelo está en [`/docs/models`](/docs/models#razonamiento).
- **Si no mandas `reasoning_effort` nada cambia.** Se aplica un tope de 16.384
  tokens, el mismo que tenías antes con el nivel por defecto. Si un modelo
  tiene configurado su propio nivel por defecto, se usa el tope de ese nivel.

### Arreglos

- **`/v1/models` y `/ocabra/models` muestran bien las herramientas y el
  razonamiento de los modelos de llama.cpp que no están cargados.** Se leen
  de la plantilla del propio fichero del modelo. Antes salía la capacidad
  guardada de una carga antigua o deducida del nombre: `flint:40b` aparecía
  sin herramientas ni razonamiento aunque los tiene. Al cargarse siempre se
  usaban los correctos.

**Acción:** si pides `low` o `medium` y necesitas más pensamiento, pide `high`,
o fija tú el tope con `reasoning_budget_tokens`.

---

## 0.68.5 — 2026-11-11

### Cambios de comportamiento

- **`/docs/changelog.md` y `/docs/models.md` se leen formateadas en el
  navegador.** Antes el navegador mostraba el Markdown como texto plano. Ahora,
  si la petición prefiere `text/html` (lo que hace un navegador), recibe la
  misma página que `/docs/changelog` o `/docs/models`. `curl`, los scripts y
  los agentes que mandan `*/*` o piden `text/markdown` siguen recibiendo el
  Markdown sin maquetar. La respuesta lleva `Vary: Accept`.

**Acción:** solo si tu agente pide la página con cabeceras de navegador
(`Accept: text/html,…`) y espera Markdown. En ese caso añade `?raw=1` o pide
`Accept: text/markdown`.

---

## 0.68.4 — 2026-11-11

### Arreglos

- **Un router (`recomendado-*`) devolvía `409` mientras había una sesión de
  voz (`/v1/realtime`) abierta con su modelo.** El router aparta a propósito
  el modelo que está usando una sesión de voz, para que la voz no se frene, y
  probaba a cargar otro; si el otro no cabía en la GPU, la petición fallaba con
  `409 Need … MB VRAM`. Ahora, si no hay ningún otro modelo que se pueda
  servir, la petición comparte el modelo de la sesión de voz: tarda algo más,
  pero responde.

---

## 0.68.3 — 2026-11-11

### Arreglos

- **«Cargar» volvía a fallar en los modelos con presets.** La 0.68.2 devolvía
  un 500 en vez de cargar el preset por defecto. No descargaba nada.

**Acción:** ninguna.

---

## 0.68.2 — 2026-11-11

### Arreglos

- **«Cargar» en la página de modelos usa el preset por defecto.** Si el
  modelo tiene presets, carga el mismo worker que usaría un chat: por
  ejemplo, `ornith-1.5:35b` carga su preset «normal», con 131k de contexto en
  la RTX 3090. Antes cargaba la configuración base, que en `ornith-1.5:35b`
  pedía 262k en una sola tarjeta y no cabía en ninguna. Fallaba con 409.
- **Una carga imposible ya no descarga a los demás antes de fallar.** Si un
  modelo cabe en una sola tarjeta, oCabra comprueba si cabe en la mejor de
  ellas, en vez de sumar la memoria libre de todas. Antes, una carga que no
  podía caber descargaba modelos fijados (Whisper y `gemma4:26b`) y luego
  fallaba de todos modos. Esos modelos tardaban hasta 2 minutos en volver.

**Acción:** ninguna.

---

## 0.68.1 — 2026-11-11

### Arreglos

- **El criterio de 0.68.0 también se aplica cuando un perfil fija la
  tarjeta.** Si un preset obligaba a cargar en una tarjeta concreta, oCabra
  seguía descargando el modelo grande antes de probar con los ociosos más
  baratos de recargar y con reducir el contexto.

**Acción:** ninguna.

---

## 0.68.0 — 2026-11-11

### Cambios de comportamiento

- **Para hacer sitio, oCabra descarga primero lo que menos cuesta recargar.**
  Cuando un modelo no cabe en una tarjeta y reducir el contexto de otro no
  basta, oCabra prueba antes la combinación más barata: descargar modelos
  ociosos que tardan poco en volver y reducir el contexto del grande para
  cubrir el resto. Antes descargaba el modelo grande entero, aunque fuera el
  que estaba en uso. Por ejemplo, cargar `qwen3-embedding:4b` en la RTX 3060
  descargaba `qwen3.8:27b`, que tarda entre 45 y 100 s en recargar. Ahora
  descarga Whisper, que vuelve en unos 10 s, y deja `qwen3.8:27b` cargado con
  menos contexto. El coste se mide con los tiempos de carga reales de cada
  modelo. Solo se consideran modelos sin peticiones en curso. Los fijados de
  forma permanente y los que tienen una sesión en tiempo real nunca se
  descargan.

**Acción:** ninguna. Un modelo fijado pero ocioso (como Whisper) puede tardar
unos segundos en su siguiente petición si se descargó para hacer sitio.

---

## 0.67.6 — 2026-11-11

### Arreglos

- **`/ocabra/models/activity` da el primer token y un restante útil en las
  peticiones a llama.cpp sin streaming.** `firstTokenAfterMs` llegaba a
  `null`, porque `/slots` no lo da; ahora se reconstruye con el ritmo
  medido. Además, el tiempo restante daba 0 cuando la petición ya había
  superado la salida típica del modelo. Ahora se calcula hasta su
  `max_tokens`, y el campo nuevo `remainingIsUpperBound` indica que es un
  máximo.

---

## 0.67.4 — 2026-11-10

### Arreglos

- **Las transcripciones que tardaban más de 5 minutos fallaban siempre**, con
  `503 worker_unavailable` y tras unos 10 minutos de espera. Afectaba sobre
  todo a grabaciones largas con diarización (`whisper-large-v3-diarized`),
  también dentro de `/v1/batches`. Ahora pueden tardar tanto como el límite
  general de inferencia del servidor (25 minutos). Si aun así no terminan,
  la respuesta es `504` con `code: "transcription_timeout"` y llega en cuanto
  se cumple el plazo, sin reintentos que dupliquen la espera. **Acción:** si
  tu cliente reintentaba los `503` de transcripción, trata el `504
  transcription_timeout` como definitivo para ese audio (prueba sin
  diarización, con un modelo más rápido o troceando la grabación).

---

## 0.67.3 — 2026-11-10

### Arreglos

- **Recuperar el contexto ya no descarga modelos sin necesidad.** Cuando un
  modelo con el contexto reducido lo recuperaba, oCabra contaba la memoria de
  los demás modelos por su reserva, no por lo que usan de verdad. Por eso podía
  descargar, por ejemplo, el modelo de transcripción, que reserva 3,5 GB y usa
  unos 0,4 GB, para un hueco que en realidad ya había. Ahora mira la memoria
  libre real de la tarjeta.

**Acción:** ninguna.

---

## 0.67.2 — 2026-11-10

### Arreglos

- **La reducción de contexto de 0.67.1 ya evita la descarga.** El contexto se
  reducía bien en la otra tarjeta, pero oCabra decidía con la cifra de memoria
  libre anterior a la reducción, porque esa cifra se refresca cada ~2 s. Así
  que descargaba el modelo de todos modos. Ahora espera a que la cifra refleje
  lo liberado antes de decidir.

**Acción:** ninguna.

---

## 0.67.1 — 2026-11-10

### Arreglos

- **El ajuste de contexto de 0.67.0 ahora prueba todas las tarjetas.** Antes
  solo reducía el contexto en la tarjeta preferida del modelo que se iba a
  cargar. Si esa tarjeta no daba bastante, descargaba otro modelo aunque
  reducir en la otra tarjeta habría bastado. Pasaba, por ejemplo, con
  `qwen3-embedding:4b` y `qwen3.8:27b`: se echaban mutuamente y cada
  petición al otro pagaba una recarga. Ahora oCabra reduce el contexto
  donde haga falta antes de descargar nada. Además, si una reducción no
  va a llegar, lo ve al primer paso y no pierde segundos en intentarlo.

**Acción:** ninguna.

---

## 0.67.0 — 2026-11-10

### Cambios de comportamiento

- **Menos recargas largas en los modelos de llama.cpp.** Si hace falta VRAM
  para otro modelo y un modelo de llama.cpp está ocioso, oCabra primero le
  reduce el contexto en caliente, a la mitad cada vez y con un mínimo de
  32.768 tokens. Solo lo descarga si con eso no basta. Antes lo descargaba
  siempre, y la siguiente petición pagaba una carga de 30 s a varios minutos.
  Cambiar el contexto tarda menos de medio segundo.
- **Una petición que no cabe en el contexto reducido lo recupera.** El modelo
  vuelve a su contexto completo antes de atenderla. Si la VRAM que soltó ya la
  ocupa otro modelo, se descarga ese. Antes respondía 503
  `context_temporarily_reduced` en cuanto la VRAM no estaba libre. Ese 503
  sigue existiendo, pero ya solo sale si ni descargando cabe; trae
  `Retry-After`.
- Cuando el modelo reducido se queda ocioso y hay sitio, recupera solo su
  contexto completo.

**Acción:** ninguna. Si lees `max_context` o el contexto del modelo en
`/ocabra/models`, ahora puede bajar un rato y volver a subir. El valor actual
está en `live_context`, y el de carga en `live_context.nominal`.

---

## 0.66.0 — 2026-11-10

### Nuevo

- **Backend Strata y perfil `qwen3.8-coder:iq1-m`.** Sirve chat, streaming,
  razonamiento y herramientas con 65.536 tokens de contexto y dos peticiones
  paralelas. Se carga bajo demanda en las dos tarjetas y puede tardar varios
  minutos en arrancar.
- **Reserva multi-GPU asimétrica.** Los backends pueden declarar el consumo
  validado por tarjeta. El perfil Strata reserva aproximadamente 11,3 GiB en
  la RTX 3060 y 23,7 GiB en la RTX 3090, en vez de repartir el total al 50 %.

**Acción:** ninguna para clientes OpenAI u Ollama; usa el nombre público del
modelo como cualquier otro.

---

## 0.65.1 — 2026-11-10

### Arreglos

- **`/ocabra/models/activity` refleja el contexto cambiado en vivo.** Antes,
  si se cambiaba el contexto de un worker de llama.cpp sin recargarlo, la
  clave `contexts` seguía mostrando el tamaño con el que se cargó. Ahora
  `activeMaxTokens` es el tamaño real; `liveResized` indica que se cambió y
  `nominalMaxTokens` da el tamaño de carga.

---

## 0.65.0 — 2026-11-10

### Nuevo

- **Los administradores pueden cambiar en vivo el contexto de un modelo de
  llama.cpp ya cargado**, sin recargarlo, con
  `POST /ocabra/models/{model_id}/context`. El cuerpo admite `context`
  (tamaño de la caché KV en tokens), `parallel` (slots), `ubatch` y `batch`, o
  `restore` para volver al tamaño con el que se cargó. El cambio tarda unos
  0,15 s en lugar de las decenas de segundos de una recarga.

  Está **desactivado por defecto** (`llama_cpp_live_context_enabled`) y solo
  funciona con el backend `llamacpp_mks` cuando ejecuta una versión del fork
  que lo admite. Mientras se aplica, las peticiones a ese modelo esperan a que
  termine. Si un modelo está funcionando con un contexto reducido y llega una
  petición que no cabe, vuelve a crecer solo antes de atenderla. Si no puede,
  la petición recibe `503` con `Retry-After`.

---

## 0.64.0 — 2026-11-10

### Cambios

- **Vuelve el cálculo de memoria con la medición real de la 0.63.0, ahora
  acotado a lo que cabe en cada tarjeta.** El servidor apunta la memoria real
  que ocupa cada modelo de llama.cpp al cargarse y, la siguiente vez que se
  carga con la misma configuración, reserva esa cantidad en lugar de su
  estimación si esta se quedaba corta. Así evita los fallos `503` al cargar un
  modelo junto a otros: antes reservaba de menos, empezaba a cargar y la GPU
  se quedaba sin memoria.

  La diferencia con la 0.63.0 es que la reserva ya no puede pasar de lo que
  la tarjeta ofrece vacía. En la 0.63.0 un modelo que llena casi una tarjeta
  entera recibía `409 insufficient_vram` en su segunda carga. Ahora carga
  igual que antes, aunque para hacerle sitio se descarguen todos los demás.

---

## 0.63.1 — 2026-11-10

### Arreglos

- **Se retira el cambio de la 0.63.0, que impedía recargar algunos modelos.**
  Un modelo que ocupa casi toda una tarjeta (`qwen3-coder:30b` en la RTX 3090)
  cargaba bien la primera vez, pero en la siguiente carga el servidor le
  reservaba más memoria de la que la tarjeta ofrece y respondía
  `409 insufficient_vram`. Duró unos minutos. El servidor vuelve a calcular la
  memoria como antes de la 0.63.0, y la medición real se sigue guardando sin
  aplicarla todavía.

---

## 0.63.0 — 2026-11-10

### Cambios

- **Los modelos de llama.cpp ya no fallan al cargar por estimar mal su
  memoria.** Antes de cargar un modelo, el servidor calcula cuánta VRAM
  necesitará para decidir dónde cabe y a quién descargar. Para llama.cpp ese
  cálculo dejaba fuera una parte que depende de cada modelo, y en los
  repartidos entre las dos GPUs se quedaba corto en más de 3 GB. El servidor
  creía que cabía, empezaba a cargar y la GPU se quedaba sin memoria: la
  petición fallaba con `503` aunque hubiera bastado con descargar otro modelo
  primero.

  Ahora apunta la memoria real que ocupa cada modelo al cargarse y la usa la
  siguiente vez que se cargue con la misma configuración. La cifra solo sirve
  para reservar más, nunca menos, y se descarta si cambia la configuración del
  modelo. La primera carga de cada configuración sigue usando la estimación.

---

## 0.62.5 — 2026-11-10

### Arreglos

- **Los modelos vLLM con `vram_estimate_mb` vuelven a cargar.** Tras la
  unificación de modelos (0.46.0) el servidor dejaba de ver ese ajuste
  y estimaba la VRAM solo por el tamaño de los pesos. Con ese cálculo
  podía rechazar una carga que sí cabía, con el error `Need N MB VRAM`.
  Le pasaba, por ejemplo, a `bge-reranker-v2-m3` en `/v1/rerank`.

---

## 0.62.4 — 2026-11-10

### Arreglos

- **`estimatedRemainingMs` ya no inventa un valor cuando no se puede
  saber.** En una petición sin streaming cuyo prompt resulta más largo de lo
  estimado, el servidor no conoce el total de tokens. Ahora `estimatedRemainingMs`
  y `expectedDurationMs` llegan a `null` en lugar de quedarse fijos en un
  valor falso.
- **El tiempo restante ya no salta al empezar a generar.** En el primer
  instante de la generación, antes de tener ritmo en vivo, se usa el ritmo
  de generación habitual del modelo. Y `expectedDurationMs` cuadra con el
  restante: es lo transcurrido más lo que falta.

---

## 0.62.2 — 2026-11-10

### Arreglos

- **El tiempo restante usa el ritmo medido en la propia petición.** Si hay
  ritmo en vivo de prefill o de generación, se calcula con él y no con la
  mediana histórica del modelo, que puede ser varias veces más lenta. En
  gemma4:26b, el histórico era de 726 t/s y la petición iba a 2.300 t/s:
  marcaba 41 s restantes en una petición que terminó a los 20.

---

## 0.62.1 — 2026-11-10

### Arreglos

- **El prefill de las peticiones sin streaming ya no marca un porcentaje
  falso.** El servidor de llama.cpp no publica el total de tokens del prompt
  mientras lo procesa: el dato que se tomaba por total era solo lo cargado
  hasta ese momento, y el progreso salía siempre cerca del 98 %. Ahora
  `prompt.processedTokens` es el dato real, `prompt.totalKnown` vale `false`
  y `prefillPct` es `null`. En streaming el total sí es real y el porcentaje
  no cambia.

---

## 0.62.0 — 2026-11-10

### Cambios

- **`/ocabra/models/activity` distingue las peticiones sin streaming.** En
  la API de OpenAI, una petición sin `stream` en el cuerpo responde de una
  vez, y ahora se trata así: antes se tomaba por streaming y se marcaba como
  parada a los 15 s porque no emite nada hasta el final. En la API nativa de
  Ollama (`/api/*`), la ausencia sigue significando streaming.
- **Progreso real también para las peticiones sin streaming a llama.cpp.**
  Se lee del endpoint `/slots` del llama-server: avance del prompt, tokens
  generados, ritmos y si espera turno dentro del backend.
- **Campo nuevo `stalled` en cada petición.** Solo vale `true` cuando hay una
  señal de vida que vigilar (tokens en streaming o progreso real del prefill)
  y lleva 15 s parada. Las peticiones sin ninguna señal ya no salen como
  paradas.
- **Duraciones estimadas con los ritmos medidos de cada modelo.** Se calculan
  como prompt ÷ prefill medido + salida esperada ÷ generación medida (medianas
  de las últimas 6 h), en lugar de con la recta global del estimador, que se
  quedaba muy corta con prompts largos.
- **Clave nueva `modelStates`.** Indica si un modelo está cargando (cuánto
  lleva y cuánto suele tardar), esperando espacio en GPU (y a qué modelo
  desaloja) o esperando una GPU retenida por un servicio.
- **`GET /ocabra/host/stats` añade `cpu_model` y `cpu_power_max_w`**, con su
  fuente en `cpu_power_max_source` (`config`, `rapl` o `tdp`). El máximo se
  puede fijar con `CPU_POWER_MAX_W`.

---

## 0.61.2 — 2026-11-09

### Arreglos

- **`estimatedRemainingMs` ya no cuenta hacia atrás mientras el modelo se
  carga o espera GPU.** Una petición en `loading` o `waiting_for_gpu` aún no
  ha empezado, pero su tiempo restante bajaba con el reloj y podía llegar a 0
  antes de procesar nada. Ahora conserva su duración estimada completa hasta
  que empieza, igual que `backend_wait`.

---

## 0.61.1 — 2026-11-09

### Arreglos

- **`/ocabra/models/activity` ya no da por "en prefill" una petición que el
  backend aún no ha empezado.** llama.cpp procesa los prompts de uno en uno:
  si llegaban varias peticiones largas a la vez, las que esperaban turno
  dentro del llama-server salían en prefill durante minutos. Ahora tienen su
  propio `stage`: `backend_wait`.
- **El tiempo restante estimado ya no llega a cero en mitad de un prefill
  largo.** Partía del recuento aproximado de tokens del prompt, que puede
  quedarse en una cuarta parte del real. Con llama.cpp se recalcula con el
  total real en cuanto se conoce, y nunca es menor que el prefill que falta
  a su ritmo medido.

---

## 0.61.0 — 2026-11-09

### Cambios

- **Streaming con llama.cpp: el servidor pide al backend el progreso del
  prefill.** Para enseñar en el dashboard cuánto lleva procesado cada prompt,
  oCabra añade `return_progress` a las peticiones en streaming que van a un
  modelo `llama_cpp` o `llamacpp_mks`. Tu cliente no recibe nada nuevo: esos
  chunks de progreso se consumen dentro del servidor y el stream que te llega
  es el de siempre. Si tú mandabas `return_progress: true`, los sigues
  recibiendo igual que antes.
- **`GET /ocabra/models/activity` trae el estado real de cada petición en
  curso.** Cada petición incluye ahora `stage` (`queued`, `loading`,
  `waiting_for_gpu`, `prefill`, `generating` o `processing` si no hay
  streaming), su puesto en la cola de admisión, el progreso del prefill
  (`prompt`, `prefillPct`), los t/s de prefill y de generación medidos en su
  propia fase, y la duración y espera estimadas. Hay una clave nueva,
  `contexts`, con los niveles de contexto de cada modelo cargado por
  colocación de GPU (en modelos vLLM, `usagePct` con la ocupación real del
  pool KV), y otra, `vramByGpu`, con los MB que cada modelo ocupa en cada GPU
  según NVML. En la cola, `predictedStartOrder` y `queueWaitEstimateMs`
  tienen en cuenta el KV libre, no solo los slots: una petición pequeña puede
  entrar antes que una grande que todavía no cabe. En backends que no
  informan del prefill (Ollama, vLLM), `prefillPct` se estima con el ritmo de
  prefill medido del modelo y `prefillSource` lo indica. Los campos
  anteriores no cambian. `tokensPerSecond`
  sigue dividiendo la salida entre todo el tiempo transcurrido; para el
  ritmo real de generación usa `generationTokensPerSecond`.
- **El mismo snapshot llega por WebSocket.** Si en `/ocabra/ws` mandas
  `{"subscribe": "models_activity"}`, recibes un mensaje `models_activity`
  cada segundo hasta que mandas `{"unsubscribe": "models_activity"}`.
- **Más endpoints de lectura abiertos a cualquier usuario autenticado.**
  `GET /ocabra/stats/recent`, `GET /ocabra/stats/requests`,
  `GET /ocabra/stats/tokens`, `GET /ocabra/services` y
  `GET /ocabra/downloads` ya no exigen rol `model_manager`; antes devolvían
  403 a un usuario normal. Crear, modificar o cancelar sigue igual.
- **`GET /ocabra/stats/recent` solo da la IP y el user agent a los admin.**
  Para el resto de roles, `clientAddr` y `userAgent` llegan siempre a
  `null`.

---

## 0.60.3 — 2026-11-09

### Arreglos

- **Otro caso del mismo 409: el servidor descargaba un modelo para hacer
  sitio y comprobaba la GPU antes de que terminara de liberarse.** Daba por
  liberada la memoria en cuanto subía 1 GB, aunque el modelo descargado
  ocupase más de 20. Medio segundo después veía la tarjeta a medio vaciar,
  concluía que no cabía y devolvía `409 insufficient_vram`. Ese hueco lo
  aprovechaba además el modelo recién descargado para volver a cargarse.
  Ahora espera a que la GPU tenga la memoria necesaria, o a que deje de
  liberar más, antes de decidir.

  Los 409 de `gemma4:26b` con la RTX 3090 a unos 23.300 MB libres eran este
  caso: la tarjeta estaba prácticamente vacía, solo que todavía no se había
  terminado de leer así.

---

## 0.60.2 — 2026-11-09

### Arreglos

- **Un modelo grande podía rechazarse con 409 aunque hubiese sitio si se
  descargaba otro.** Antes de descargar modelos para hacer hueco, el servidor
  calcula si la operación puede funcionar. Ese cálculo usaba lo que se
  estimaba que ocupaba cada modelo cargado, no lo que ocupaba de verdad.
  Cuando la estimación se quedaba corta, daba la descarga por inútil y
  rechazaba la carga con `409 insufficient_vram`, aunque descargar ese modelo
  habría dejado sitio de sobra.

  Lo sufría sobre todo `gemma4:26b`: unas 10 de cada 180 peticiones en los
  tres días previos, siempre que otro modelo grande ocupaba la RTX 3090. Ahora
  el cálculo usa la memoria real de cada proceso y la carga sigue adelante.

  Si reintentabas esos 409 con espera, ya no te harán falta en este caso. Un
  409 sigue significando que no cabe ni descargando todo lo que se puede
  descargar.

---

## 0.60.1 — 2026-11-09

### Arreglos

- **`/api/chat` y `/api/generate` en streaming devolvían
  `prompt_eval_count: 0` y `eval_count: 0`** en el último trozo, en todos los
  modelos servidos por llama.cpp (casi todos). La respuesta era correcta; solo
  faltaban los conteos de tokens. Si tu cliente calcula consumo o velocidad
  con esos campos, ahora recibe los valores reales.
- Ojo al calcular la velocidad de lectura del prompt con esos campos:
  `prompt_eval_count` incluye los tokens que se sirvieron de la caché, pero
  `prompt_eval_duration` solo mide los que se procesaron de verdad (igual que
  en Ollama). Con una caché caliente, `prompt_eval_count / prompt_eval_duration`
  sale muy por encima de la velocidad real. Para la tasa real usa
  `promptTokensPerSecond` de `GET /ocabra/stats/recent`, que la da el propio
  motor.

---

## 0.60.0 — 2026-11-09

### Nuevo

- **Cada petición dice ahora cuánto esperó y a qué velocidad fue.**
  `GET /ocabra/stats/recent` (y el equivalente de cada usuario) añade por
  petición `queueMs`, `promptTokensPerSecond` (lectura del prompt),
  `predictedTokensPerSecond` (generación), `generationMs`, `cachedTokens`,
  `effectiveContext`, `runtimePreset` y `gpuIndex`.

  Son tres cosas que conviene mirar por separado: una espera larga es
  contención, un prefill lento es un prompt enorme y una generación lenta es
  el modelo o la tarjeta. Estaban guardadas, pero solo se publicaban
  promediadas por modelo en `/ocabra/stats/performance`, que es justo donde
  esas diferencias se pierden.

---

## 0.59.1 — 2026-11-09

### Arreglos

- **`/api/chat` y `/api/generate` devolvían tiempos falsos.** `eval_duration`
  traía el reloj de pared completo —cola, carga del modelo y lectura del
  prompt incluidas— en vez del tiempo de generación, y `prompt_eval_duration`
  no se enviaba. Si calculabas tokens/s dividiendo por esos campos, te salía
  bastante menos de lo que el servidor hace en realidad. Ahora los dos llevan
  lo que mide el motor, y `total_duration` sigue siendo el reloj de pared.

- **El tamaño de una petición se calcula ahora exactamente.** Para decidir
  ventana y reservar caché, el servidor estimaba el tamaño del prompt juntando
  su texto. Ahora le pide al propio modelo que aplique su plantilla de chat y
  cuenta eso, que es literalmente lo que va a leer: medido contra un turno con
  herramientas, el cálculo da 7.833 tokens y el motor cuenta 7.833 (antes,
  8.490).

### Nuevo

- **`GET /ocabra/models` dice ahora cuántas peticiones atiende un modelo a la
  vez y cómo reparte su contexto.** Tres campos nuevos en `capabilities`:
  `parallel_slots`, `kv_cache_shared` y `context_per_request`. El último es el
  que querías: cuánto contexto puede gastar **una** petición. No es siempre el
  mismo que `context_length` — con caché compartida una sola petición puede
  usar la ventana entera, y sin ella cada plaza recibe su parte. Antes había
  que deducirlo de la configuración del backend, y quien lo dedujo pensando en
  llama.cpp de serie mostraba un contexto que no es el que se sirve.

  Y mientras un modelo está descargado, esos campos salían con los valores por
  defecto del servidor en vez de los suyos: `flint:40b`, que corre con 262.144
  en un pool de cuatro plazas, aparecía como 4.096 con una sola plaza. Ahora,
  descargado, manda lo que tiene configurado.

- **Estadísticas de rendimiento también para los endpoints de Ollama.** Las
  velocidades de lectura de prompt y de generación ya se guardan para
  `/api/chat` y `/api/generate`; antes solo se registraban por la vía de
  OpenAI, y un 29 % del tráfico quedaba sin ningún dato con el que comparar
  modelos.

---

## 0.58.0 — 2026-11-09

### Arreglos

- **Una petición con herramientas ya no se cuenta a la mitad.** Lo que el
  asistente pide a una herramienta viaja fuera del campo de texto del mensaje,
  y ninguno de los dos contadores lo miraba. En un cliente de programación eso
  es la mitad del prompt: una petición que el motor contó en 196.549 tokens
  aparecía como 35.383 en el panel y se medía internamente como 102.200. Ahora
  se cuentan también los argumentos de las llamadas a herramientas.

- **Una petición grande ya no reserva toda la caché compartida.** Cuando el
  tamaño real del prompt no se puede medir —el modelo aún está cargando, o su
  tokenizador no contesta a tiempo— el servidor asume el máximo para no
  rechazar la petición por una estimación. Pero ese máximo se estaba apuntando
  también como caché ocupada, así que **una sola petición bloqueaba las cuatro
  plazas del modelo** hasta terminar, y el panel mostraba la caché al 100 %.
  Ahora, cuando no hay medida, se reserva lo que le corresponde a una plaza;
  elegir la ventana grande y reservar la caché entera son decisiones distintas.

---

## 0.57.0 — 2026-11-09

### Retirado

- **`bonsai-27b-ternario` ya no está disponible,** y con él el motor ternario
  que solo servía a ese modelo. Lo sustituye **`bonsai-2:27b`**, el mismo
  modelo en una cuantización más nueva que además trae visión y la escalera de
  contexto completa (262.144 tokens).

  **Acción** si lo nombrabas: cambia a `bonsai-2:27b`. Medido en la misma
  tarjeta, `bonsai-2:27b` da 65,6 tokens/s en la 3090 y 32,9 en la 3060.

---

## 0.56.0 — 2026-11-09

### Mejoras

- **El modelo ternario (`bonsai-27b-ternario`) va un 62 % más rápido:** de 35 a
  **57 tokens/s** en la misma tarjeta. Su motor estaba compilado en julio y se
  ha actualizado a la versión del 6 de noviembre, que son tres meses de mejoras
  de llama.cpp además de las propias de ese formato.

### Retirado

- **El motor `bitnet-server` de Microsoft ya no está instalado.** Llevaba
  dentro un llama.cpp de enero y no lo usaba ningún modelo del servidor. Si
  instalas un modelo en formato `i2_s` (el de Microsoft BitNet), ahora el error
  lo dice y explica cómo recuperarlo; antes se intentaba arrancar un binario
  inexistente. Los modelos en formato PrismML (`Q1_0`, `Q2_0`) no se ven
  afectados.

### Arreglos

- **Un modelo ternario instalado desde Ollama ya elige bien su motor.** Se
  decidía por el nombre del fichero, y el de un modelo instalado como tag es su
  sha256, así que arrancaba el motor equivocado y el servidor se abortaba. Ahora
  se decide por la cuantización que declara el propio fichero.

---

## 0.55.4 — 2026-11-09

### Arreglos

- **Un modelo recién instalado ya se puede usar sin configurarlo a mano.** Al
  darlo de alta sin parámetros —lo que hace la pantalla de instalación— el
  despliegue de llama.cpp se quedaba sin ninguno, y llama.cpp solo hace lo que
  se le dice: cargaba **en CPU** y con **4.096 tokens** de contexto. Un modelo
  de 20 GB así no llega a cargar. Ahora el alta deja capas en GPU, el contexto
  nativo que declare el fichero del modelo (hasta 262.144), atención rápida,
  caché de contexto cuantizada y cuatro peticiones en paralelo. Si el tag trae
  visión, el proyector se carga con la primera imagen en vez de ocupar memoria
  en todas las peticiones de texto.

  Lo que configures tú sigue mandando: estos valores solo rellenan lo que no
  venga puesto.

- **Un modelo ternario recién instalado tampoco se queda en 256 tokens.** El
  backend ternario reparte la ventana entre las peticiones en paralelo, y con
  sus valores por defecto (4.096 de contexto entre 16 peticiones) cada petición
  se quedaba con 256 tokens. Es lo que le pasaba a `bonsai-27b-ternario` hasta
  0.51.3; ahora el alta pone el contexto del propio fichero y cuatro peticiones
  en paralelo. Y el tiempo de arranque de estos modelos ya se puede configurar
  por modelo: los 30 segundos de antes valían para uno de 3 GB y no para uno de
  8, que fallaba con un error que parecía del modelo y era del cronómetro.
- **Instalar un modelo desde Ollama ya no puede acabar en nada.** El alta
  elegía llama.cpp para cualquier tag y el registro se rechazaba cuando el
  modelo necesitaba otro backend: la descarga terminaba al 100 % y el modelo no
  aparecía en ninguna parte, con el motivo solo en los registros del servidor.
  Ahora el alta pregunta a la misma tabla de compatibilidad que valida el
  registro.
- **Un nombre de familia ya no manda un modelo al backend equivocado.** Para
  decidir si un modelo es de 1 bit se miraba el nombre completo, así que
  cualquier cosa llamada «bonsai» iba al backend ternario aunque no lo fuera.
  `bonsai-2:27b` es un modelo normal con visión y acababa en un backend que no
  sabe leerlo. Ahora solo deciden los marcadores de cuantización
  (`i2_s`, `q1_0`, `1.58bit`, `tq1_0`, `tq2_0`, `bitnet`).

### Retirado

- **`nemotron3:33b` y `vibevoice-asr-bitnet` ya no están.** El primero era el
  último modelo sobre Ollama y su rendimiento variaba diez veces entre
  peticiones según cómo repartiera Ollama entre CPU y GPU; el segundo no tenía
  su backend instalado, así que nunca pudo responder.

### Nuevo

- **Tres modelos más:** `ornith-1.5:35b` (35B con visión, 158 tok/s),
  `flint:40b` (40B, 107 tok/s) y `nemotron-3.5-lightning:30b` (30B, 148 tok/s,
  reparte entre las dos tarjetas). Los tres con contexto largo: 262.144 tokens
  los dos primeros, 131.072 el tercero.

  `qwen3.8:27b-abliterated` sigue igual; solo cambió de dónde se leen sus
  pesos, que ahora es su repositorio original de HuggingFace.

---

## 0.54.1 — 2026-11-09

### Arreglos

- **`POST /ocabra/models/update-all` usa ahora la procedencia del artefacto.**
  Conserva el namespace real de HuggingFace, actualiza por Ollama los modelos
  que llama.cpp sirve desde un tag de Ollama y soporta GGUF con repositorio y
  fichero explícitos. Ya no intenta descargar nombres locales inventados.
- **Una fuente compartida se descarga una sola vez.** Si los despliegues de
  Ollama y llama.cpp sirven los mismos pesos, el segundo figura en `skipped`
  indicando qué despliegue cubre su actualización.
- **Actualizar ya no crea despliegues nuevos.** El auto-registro queda limitado
  a instalaciones iniciales; un repull solo refresca los pesos y metadatos del
  modelo existente.

---

## 0.54.0 — 2026-11-08

Esta versión no cambia el contrato HTTP. Añade una herramienta operativa para
ejecutar manualmente datasets mantenidos por las aplicaciones, empezando por
Escriba, y conservar resultados con la revisión del dataset y la versión del
servidor medidas.

---

## 0.53.1 — 2026-11-08

### Retirado

- **`nemotron3:33b` ya no está disponible** (devuelve `404` y no aparece en
  `GET /v1/models`). Era el último modelo que quedaba sobre Ollama, que decide
  por su cuenta el reparto entre CPU y GPU al cargar, y según cómo cayera daba
  entre 9 y 100 tokens por segundo para la misma petición. `nemotron-3-nano:30b`
  no tiene nada que ver y sigue donde estaba.

  **Acción** si lo nombrabas: usa `qwen3.8:27b` o `gemma4:26b` para tareas
  generales, o `recomendado-sota` si prefieres no elegir.

---

## 0.53.0 — 2026-11-08

### Cambios de comportamiento

- **`POST /v1/audio/transcriptions` con varios audios ahora responde `400`.**
  Antes se quedaba con el último y devolvía `200` sin decir nada de los otros:
  mandabas cuatro WAV, recibías una transcripción y parecía correcta. El
  endpoint transcribe un audio por petición, igual que el de OpenAI.

  **Acción** si mandabas varios: mándalos de uno en uno, o usa `/v1/batches`
  con una línea por audio (cada uno subido con `POST /v1/files` y
  `purpose="batch_input_audio"`). Eso ya funcionaba y procesa todos; lo que no
  existe es meter varios en un solo `multipart`.

---

## 0.52.1 — 2026-11-08

### Arreglos

- **La autodetección de idioma de 0.52.0 solo llegó a las transcripciones con
  diarización.** Las normales no pasaban por el sitio donde se aplicó, así que
  seguían volviendo en inglés. Ahora cubre las dos. Si probaste con 0.52.0 y te
  salió igual que antes, era esto.

---

## 0.52.0 — 2026-11-08

### Arreglos

- **La transcripción en CPU devolvía el texto en inglés.** Si no mandabas
  `language`, el modelo de respaldo de `recomendado-stt` (`whisper.cpp`, el que
  entra cuando las tarjetas están ocupadas) asumía inglés y **traducía** en vez
  de transcribir: un clip en español volvía como "The report on European budget
  was approved yesterday". El modelo de GPU detecta el idioma solo, así que la
  misma petición daba un idioma u otro según qué worker la sirviera, sin ningún
  error de por medio. Ahora el worker de CPU arranca autodetectando el idioma,
  así que sin `language` transcribe en el idioma que oye. Si lo mandas, manda
  lo tuyo.

  Se notaba más en audios cortos, donde la traducción además resume: en un clip
  de 3,8 segundos con 12 palabras volvían 9, en inglés.

### Nuevo

- **`GET /version` y `GET /api/version`.** Devuelven `{"version": "..."}` sin
  necesidad de clave. `/version` daba un 404 que no venía del servidor sino del
  panel web, y `/api/version` —que forma parte de la API de Ollama y la usan
  sus clientes para decidir qué soporta el servidor— no existía. `GET /health`
  sigue trayendo la versión igual que antes.

---

## 0.51.9 — 2026-11-08

### Arreglos

- **Guardar la configuración de un modelo desde el panel ya no le quita la
  decodificación especulativa.** Era la causa de fondo del `503` de
  `gemma4:12b` descrito en 0.51.8: una edición desde el panel borró el tipo
  de borrador. En los modelos con MTP sin borrador externo (`qwen3.8:27b`), la
  misma edición la habría desactivado entera sin dar ningún error, solo
  generando más despacio.

---

## 0.51.8 — 2026-11-08

### Arreglos

- **`gemma4:12b` respondía `503` a todo.** Desde las 02:13 UTC del 08/11, cada
  petición a `gemma4:12b` fallaba con `Failed to load model`: el proceso del
  modelo se abortaba al arrancar y ninguna carga llegaba a completarse. La
  causa estaba en el modelo borrador que acompaña al principal para acelerar
  la generación: su configuración se quedó sin indicar de qué tipo era, y el
  servidor se lo pasaba igual al motor. Eso no es "ir sin acelerar": el motor
  carga el borrador y se aborta al arrancar, así que el modelo no llegaba a
  estar disponible nunca. Ahora el tipo se deduce del propio fichero del
  borrador, y cuando no se puede deducir el borrador se ignora y el modelo
  carga sin aceleración en vez de no cargar. `gemma4:12b` vuelve a responder,
  y con el borrador activo: ~116 tok/s.

  Le puede haber pasado a cualquier modelo al que su paquete le añada un
  borrador. Si viste `503` con `Failed to load model` en un modelo que antes
  iba, es esto.

---

## 0.51.7 — 2026-11-08

### Sin cambios para tu cliente

- Ajuste interno de cómo el servidor calcula qué configuraciones de contexto
  caben en cada GPU, para que coincida con lo que de verdad puede cargar. Los
  nombres de modelo, sus contextos y las respuestas son los mismos que en 0.51.6.

---

## 0.51.6 — 2026-11-08

### Arreglos

- **Si pedías un preset de contexto concreto, el `400` por tamaño volvía a
  decidirse con una estimación.** El arreglo de 0.51.0 comparaba el tamaño del
  prompt con la ventana *más grande* del modelo, pero cuando pides un preset
  con `ocabra.context_preset` el rechazo se compara con **ese** preset. Un
  prompt de ~19.000 tokens reales contra `qwen3:32b` con el preset `normal`
  (30.720) se estimaba en 34.311 y se rechazaba, aunque cabía de sobra y el
  modelo estaba cargado. Peor: cuando no había conteo exacto disponible, el
  recorte "no rechaces por estimación" usaba la ventana mayor y **fabricaba**
  un rechazo que con el número real no existía. Ahora se mide y se recorta
  contra el preset que de verdad decide.

- **El error de activación de visión culpaba al motor en vez de a la VRAM.**
  Cuando una petición con imagen no cabía, recibías
  `On-demand vision requires the oCabra custom llama.cpp runtime with hot
  mmproj support`, que manda a mirar el sitio equivocado: el proyector visual
  se asigna en VRAM al activarlo, así que la causa habitual es que no quepa
  junto a la caché del modelo. Ese mensaje queda solo para cuando el motor de
  verdad no implementa la activación en caliente (`404`/`405`/`501`); el resto
  de fallos dicen qué pasó y sugieren un preset de contexto más pequeño.

## 0.51.5 — 2026-11-08

### Sin cambios para tu cliente

- Corrección interna del panel de administración. La API, los nombres de
  modelo y las respuestas son los mismos que en 0.51.4.

---

## 0.51.4 — 2026-11-08

### Arreglos

- **`gemma4:26b` rechazó todas las peticiones durante unos dos minutos**
  (2026-11-08, 03:19–03:21 UTC) con `400 context_length_exceeded`, incluso las
  más cortas. Lo causó un cambio de configuración del servidor que ya se ha
  deshecho, y el servidor impide ahora que un cambio así vuelva a dejar un
  modelo sin ninguna configuración de contexto utilizable.

---

## 0.51.3 — 2026-11-08

### Arreglos

- **`bonsai-27b-ternario` solo admitía 256 tokens por petición** y rechazaba
  casi cualquier prompt con `request (N tokens) exceeds the available context
  size (256 tokens)`. Ahora acepta hasta **98.304 tokens**, con 4 peticiones
  simultáneas que comparten ese contexto. Su ventana nativa es de 262.144; se
  queda en 96k para poder convivir en la misma GPU con la transcripción.

---

## 0.51.2 — 2026-11-08

### Documentación

- **[`/docs/models`](/docs/models) reescrita con un benchmark propio.** Por
  modelo: velocidad en tres tamaños de tarea y con concurrencia, calidad de
  extracción (cobertura y precisión sobre 100 afirmaciones de referencia),
  rendimiento con documentos de 15.000 y 25.000 palabras, latencia al primer
  token, WER y DER de los cinco perfiles de transcripción, y NDCG/MRR de los
  embeddings y el reranker. Con avisos concretos: `mistral:7b` no produce JSON
  válido, `turbo-latam int8` tiene un WER del 20 %, `nemotron3:33b` puede caer
  a CPU y `qwen3:32b` tarda 22 s en empezar.

### Arreglos

- **Con `reasoning_effort` y un `max_tokens` ajustado, la respuesta llegaba
  vacía.** El nivel concedía un presupuesto de pensamiento fijo sin mirar
  cuánta salida tenías: `medium` daba 16.384 tokens de pensamiento aunque tu
  `max_tokens` fuese 4.096, así que el pensamiento se comía la respuesta
  entera y recibías `content: ""` con `finish_reason: "length"`. Pasaba igual
  con `low` por debajo de 2.048 de salida, y `high` no llevaba tope ninguno.
  Ahora el presupuesto se recorta para dejar siempre sitio a la respuesta —un
  cuarto de la salida, mínimo 256 tokens—, y con salida holgada el nivel manda
  sin cambios (`medium` sobre 65.536 sigue siendo 16.384). **Acción:** si
  habías bajado el nivel de razonamiento para esquivar esto, puedes volver a
  subirlo.

---

## 0.51.1 — 2026-11-08

### Arreglos

- **El arreglo del `400` de 0.51.0 se dejaba fuera a cuatro modelos**, entre
  ellos `gemma4:26b`. Son los que no tienen presets de contexto
  (`gemma4:26b`, `gemma4:12b`, `mistral:7b`, `ravenx`): su petición sale del
  planificador por una vía más corta y el cálculo aproximado llegaba sin
  corregir a la puerta de admisión, que lo compara con el pool de caché del
  motor. Un prompt de 230.011 tokens reales contra `gemma4:26b` se estimaba
  en 705.372 —3,07 veces el tamaño verdadero— y recibías
  `request needs 705372 KV tokens but the pool holds 262144`, aunque cabía de
  sobra en la ventana de 262.144. Ahora esos cuatro modelos pasan por la misma
  corrección: conteo exacto si hay un runtime cargado, y si no, se intenta en
  vez de rechazar.

---

## 0.51.0 — 2026-11-07

### Arreglos

- **Un prompt que cabía podía recibir `400` `context_length_exceeded`.** El
  tamaño del prompt se decidía con un cálculo aproximado sobre caracteres y
  palabras que, medido contra el tokenizador de verdad, se queda entre el 0,94x
  y el 2,42x del valor real según el contenido: con texto de palabras largas se
  pasaba más del doble. Un prompt de ~32.000 tokens reales se estimaba en
  ~55.000 y `qwen3:32b` (ventana de 40.960) lo rechazaba aunque entraba de
  sobra. Afectaba igual a `qwen3:8b` y a cualquier modelo cuya ventana
  estuvieras aprovechando a fondo.

  Ahora, antes de rechazar: si hay un runtime del modelo cargado se le pide el
  conteo **exacto** a su tokenizador y se decide con ese número; y si no hay
  ninguno cargado no se rechaza por estimación — se intenta con la ventana más
  grande disponible y, si de verdad no cabe, el error lo da el motor, que es el
  único que cuenta bien. El cálculo aproximado se sigue usando para elegir qué
  ventana cargar, donde equivocarse no cuesta nada.

- **`gemma4:26b` podía quedarse inalcanzable con `409` durante horas.** Su pin
  es evictable: cuando otro modelo grande (`qwen3.8:27b`, `ravenx`) se queda la
  tarjeta, `gemma4:26b` sale. Para eso hay un vigilante que lo vuelve a cargar
  en cuanto hay sitio, pero **se rendía a los 12 minutos**: a quien se quedase
  la tarjeta le bastaba con aguantarla ese rato y `gemma4:26b` no volvía hasta
  el siguiente reinicio del servidor. Ahora el vigilante no tiene plazo:
  reintenta mientras el modelo siga fijado y descargado, y entra solo cuando la
  tarjeta queda libre de verdad (nunca desaloja para colarse).

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
