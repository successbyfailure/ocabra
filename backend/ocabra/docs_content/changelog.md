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

## 0.145.0 — 2026-11-27

### Cambios

- **Disco del pod al lanzarlo.** `POST /ocabra/cloud/nodes`,
  `POST /ocabra/cloud/requests` y el `cloud` de las ejecuciones y planes de
  benchmark aceptan `container_disk_gb` (20–2000). Sin él se usa el de los
  ajustes de RunPod, como antes.
- **Las baterías con pod borran cada modelo al terminar sus pruebas.** En
  cuanto ninguna prueba de la batería necesita ya un modelo, se borra del pod
  con sus ficheros. Así una batería de muchos modelos cabe en un disco
  pequeño. Un modelo que comparte fichero con otro aún pendiente
  (`llamacpp_mks/x` y `ollama/x`) se conserva hasta que acaben los dos.
- **Una descarga de la batería que aún no cabe espera.** Si el pod no tiene
  sitio para empezar una descarga, el modelo queda «esperando disco» y se
  vuelve a intentar al liberarse espacio. Antes fallaba la batería entera.
- **Los pods exigen 5 GB libres para descargar, no 30.** El mínimo del hub
  (`download_min_free_gb`) se comía casi todo el disco de un pod de 80 GB.

**Acción:** ninguna.

---

## 0.144.1 — 2026-11-27

### Arreglos

- **`/_node/<nodo>/` ya no se sirve en otra máquina si ese nodo no está.**
  Antes, con el nodo desconectado (un túnel reconectando tras un reinicio),
  la petición caía en el reparto normal y se servía donde hubiese el modelo,
  aunque la URL pedía ese nodo. Para un benchmark eso es medir otra máquina.
  Ahora responde 503 con `Retry-After: 15` y código `pinned_node_unavailable`.
  La cabecera `X-Ocabra-Node` sigue como antes: si el nodo no está, se usa el
  reparto normal y se anota.

**Acción:** si llamas por `/_node/<nodo>/`, reintenta los 503 respetando
`Retry-After`.

---

## 0.144.0 — 2026-11-27

### Cambios

- **Benchmarks con un backend concreto.** `POST /ocabra/benchmarks/runs` (y los
  planes) aceptan en `models` el id de despliegue `<backend>/<modelo>`
  (`ollama/gemma4:12b`) aunque ese despliegue no exista aún. Basta con que el
  backend pueda leer el fichero del modelo: en este servidor se crea al medir,
  y en un pod se monta con ese backend. Los nombres derivados
  (`gemma4:12b-ollama`), que la API de inferencia rechaza, se traducen solos.
  Un backend que no sirve ese fichero da 400 al crear la ejecución.
- **Ejecuciones en paralelo por máquina.** La cola de benchmarks mide a la vez
  en máquinas distintas (este servidor, cada nodo, cada pod). Antes una medida
  en un pod dejaba esperando a otra en un nodo de la federación hasta que
  terminase. En una misma máquina sigue habiendo una sola a la vez.
- **Modelos de Ollama en pods.** La imagen `ocabra-node` baja y arranca su
  propio Ollama la primera vez que un modelo lo necesita, con la versión, el
  paralelismo y el contexto del hub. Necesita un nodo 0.144.0 o posterior.
- **Una batería con pod ya no falla entera por un modelo.** Si un modelo no se
  puede copiar al pod, se lanza el pod con los demás y ese modelo queda
  fallido con su motivo en `error`. Antes no se alquilaba nada.

### Arreglos

- **Peticiones de usuarios servidas en el pod de un benchmark.** La federación
  trataba el pod alquilado para un benchmark como un nodo más. Si tenía
  cargado el modelo que pedías (o el destino de un router como
  `recomendado-sota`), tu petición podía acabar allí: la medida salía
  contaminada y tus datos pasaban por esa máquina. Ahora, mientras dure el
  benchmark, el pod solo recibe las peticiones del propio benchmark.

**Acción:** ninguna.
## 0.143.1 — VRAM mínima inclusiva

El selector de GPU de benchmarks incluye las tarjetas que igualan el mínimo.
Por ejemplo, con 24 GB se pueden elegir tarjetas de 24 GB o más.

---

## 0.143.0 — Selección de GPU para benchmarks en pod

Las ejecuciones y planes de benchmark aceptan `cloud.cloud_type: "BOTH"`.
Para cada GPU elegida, por orden de preferencia, se prueba Community y luego
Secure si hay oferta. En modo `zero_data` o con volumen de red solo se usa Secure.
El selector también permite limitar las tarjetas a las que superan un umbral
de VRAM por GPU.

---

## 0.142.0 — Benchmarks multimodelo en pod

Los pods propios preparan hasta dos modelos en paralelo y empiezan a medir el
primero disponible mientras siguen las descargas. Los resultados conservan el
identificador del perfil medido. Para dos transferencias simultáneas, el hub y
la imagen del nodo deben incluir esta versión.

## 0.141.1 — 2026-11-26

### Arreglos

- **El ranking puntuaba BFCL simple_python con la media non-live.** Esa media
  incluye categorías que la suite no mide y salía ~0,08 aunque el modelo
  acertase el 93 %. Ahora se usa la métrica principal que declara el runner
  (`primary_metric`, la precisión simple), también para ejecuciones antiguas, y
  el runner deja de guardar esa media en `simple_python`.

**Acción:** ninguna.

---

## 0.141.0 — 2026-11-26

### Nuevo (administración de benchmarks)

- **El diagnóstico de fallos repite casos de Corrala, Escriba y Fulcrum.**
  Usa el `ocabra_replay.py` de cada app: una pasada con el muestreo del
  benchmark y otra con el del fabricante (pisado en el servidor, solo para las
  peticiones del runner). Escriba entra en el diagnóstico (suite `llm`) y en
  Fulcrum cada repetición del escenario es un caso. Hace falta medir con las
  versiones de las apps que guardan los casos; las ejecuciones anteriores
  siguen como «sin verificar».

**Acción:** ninguna.

---

## 0.140.2 — 2026-11-26

### Arreglos

- **Un corte momentáneo del túnel con un pod daba su modelo por fallido.**
  Cuando el pod reconectaba su túnel con el hub, la petición de precarga en
  curso caía con «federation tunnel closed» y el modelo quedaba en error (y un
  benchmark con pod propio se cancelaba y apagaba la máquina). Ahora un corte
  de transporte se reintenta en la siguiente pasada; solo se da por fallido
  tras cinco cortes seguidos. En el ranking, una misma GPU en Community y en
  Secure Cloud cuenta como hardware distinto.

**Acción:** ninguna.

---

## 0.140.0 — 2026-11-26

### Nuevo (administración)

- **Baterías de benchmarks.** `POST /ocabra/benchmarks/runs` (y los planes)
  aceptan `battery`: más pruebas `{app, suite, suite_args}` con los mismos
  modelos y el mismo nodo o pod. Se crea una ejecución por prueba (comparten
  `batchId`) y la respuesta lleva `batchRuns`. Con pod propio se pide una sola
  máquina para todas, que se apaga al terminar la última.
- **Cuánto tarda un pod en estar listo.** Cada nodo de la nube guarda su línea
  de tiempo y `GET /ocabra/cloud/nodes` devuelve `durations`: espera de
  máquina, arranque, conexión con oCabra, preparación (backends y modelos) y
  total, más lo que tardó cada elemento de la precarga. Los intentos de
  benchmark en un pod lo guardan en `environment.node.cloud.readiness`.

### Arreglos

- **Los pods de benchmark no se pedían mientras otra ejecución medía.** El
  aprovisionamiento corre ahora en su propio bucle.

**Acción:** ninguna en la API de inferencia.

---

## 0.139.1 — 2026-11-26

### Arreglos

- **Benchmarks en nodos remotos: el backend y la GPU del nodo no llegaban.**
  Las estadísticas del nodo se pedían con un límite que el endpoint no admite
  (500 > 200) y se quedaban sin leer, así que el intento volvía a usar las del
  hub. Ahora se leen; en nodos antiguos que no publican el índice de GPU y
  tienen una sola, se toma esa y se marca como deducida.

**Acción:** ninguna.

---

## 0.139.0 — 2026-11-26

### Nuevo (administración)

- **Peticiones de máquina en la nube que esperan capacidad.**
  `POST /ocabra/cloud/requests` acepta lo mismo que lanzar un nodo más
  `alternatives` (otras GPUs aceptables, por orden), `wait_minutes` (360) y
  `retry_interval_minutes` (2). Si ninguna GPU tiene máquinas libres, la
  petición queda en `waiting` y se reintenta sola hasta lanzar el pod o
  caducar; otros fallos (presupuesto, API key) la cierran como `failed`.
  `GET /ocabra/cloud/requests` y `POST /ocabra/cloud/requests/{id}/cancel`.
- **Benchmarks con pod propio.** `POST /ocabra/benchmarks/runs` y los planes
  aceptan `cloud` (GPUs aceptables por orden, número, nube, backends, espera de
  capacidad, tope para estar listo, vida máxima y `terminate_after`). La
  ejecución pide la máquina, espera a que se una con sus modelos cargados,
  mide en ella y, si se pidió, la apaga al terminar, al fallar o al cancelarla.

### Arreglos

- **Benchmarks en nodos remotos: backend, GPU y versiones del nodo.** Hasta
  ahora un intento medido en otro nodo tomaba el backend y las GPU del estado
  local del hub, y las versiones de backend quedaban vacías. Ahora salen de las
  estadísticas y de `/ocabra/backends` del propio nodo, junto con su driver y
  CUDA y, si es un pod, su tipo de GPU, imagen y coste.

**Acción:** ninguna en la API de inferencia.

---

## 0.138.0 — 2026-11-26

### Nuevo (administración de benchmarks)

- **Ejecuciones con el muestreo del fabricante.** `POST /ocabra/benchmarks/runs`
  (y los planes) aceptan `sampling: "recommended"`: oCabra pisa en el servidor
  temperatura, `top_p`, `top_k`, `min_p` y penalizaciones de las peticiones del
  runner con los de la ficha oficial del modelo en Hugging Face. Solo afecta a
  las peticiones que llevan la clave de servicio de los benchmarks, nunca a las
  de usuarios. Por defecto, `"benchmark"` (lo de siempre). No se combina con
  `node`. Estas ejecuciones no cuentan en el ranking.
- **Notas invalidadas.** Las ejecuciones afectadas por un fallo conocido del
  arnés llevan `scoringIssue` (`reason`, `fixedIn`) y no cuentan en el ranking.
  De momento, las de BFCL medidas antes de 0.137.0.
- **Diagnósticos: cancelar y borrar.**
  `POST /ocabra/benchmarks/diagnoses/{id}/cancel` y
  `DELETE /ocabra/benchmarks/diagnoses/{id}`; los terminados se borran solos a
  los `BENCHMARK_DIAGNOSIS_RETENTION_DAYS` días (30).

**Acción:** ninguna en la API de inferencia.

---

## 0.137.0 — 2026-11-26

### Nuevo

- **Diagnóstico de fallos de benchmark** (administración, rol `model_manager`):
  `POST /ocabra/benchmarks/attempts/{attempt_id}/diagnoses` toma los casos que
  fallaron en un intento y clasifica cada uno en plantilla, sampling, límites,
  cuantización/servido o modelo. Compara el prompt servido con el de la
  plantilla oficial de Hugging Face, los parámetros efectivos con los del
  fabricante, mira si la salida se cortó o el prompt no cabía, y repite una
  muestra de casos con el muestreo recomendado por la cola normal. Lo que queda
  como «modelo» se exporta en JSONL, separado en entrenamiento y evaluación.
  Estado e informe en `GET /ocabra/benchmarks/diagnoses/{id}` y ficheros en
  `GET /ocabra/benchmarks/diagnoses/{id}/files/{name}`.

### Arreglos

- **BFCL puntuaba como «nombre de función incorrecto» respuestas correctas.**
  Las APIs tipo OpenAI no admiten puntos en los nombres de función, así que
  BFCL manda `math.factorial` como `math_factorial` y solo lo deshace al
  puntuar si el modelo se registra con `underscore_to_dot`; nuestro runner no lo
  hacía. Re-puntuado con el checker oficial, eran 516 de 771 fallos de
  gemma4:12b y 527 de 745 de gemma4:26b. Las notas de BFCL anteriores a esta
  versión están infravaloradas: repite la medida para comparar.

**Acción:** ninguna en la API de inferencia.

---

## 0.136.0 — 2026-11-26

Sin cambios en la API de inferencia. La sección de administración Benchmarks
gana la suite `fulcrum/orquestacion`: mide si un modelo sabe coordinar
varias sesiones de agentes durante muchos turnos (despacho, textos fijos,
máximo simultáneo, orden de fases, cola, honestidad), y entra en el ranking
global de modelos.

**Acción:** ninguna.

---

## 0.135.2 — 2026-11-24

### Arreglos

- **`qwen3.8-flash-next-coder:iq1_m` respondía 503 a todas las peticiones.**
  No cabía en la 3090: con 128k de contexto y el borrador MTP en la tarjeta
  principal, cada carga moría al reservar memoria. Ahora arranca con 80k de
  contexto (el prompt más largo que ha recibido fueron 64k), se reparte
  siempre entre las dos GPU y oCabra reserva lo que de verdad ocupa
  (32,5 GB), así que hace sitio antes de cargarlo.

**Acción:** ninguna. Si mandabas prompts de más de 80k a este modelo, ahora
recibirás un error de contexto en lugar de un 503.

---

## 0.135.0 — 2026-11-24

### Cambiado

- **Embeddings y pooling cancelan al desconectarse el cliente.** Antes, si
  cerrabas la conexión de una petición a `/v1/embeddings` o `/v1/pooling` antes
  de recibir la respuesta, el servidor seguía calculándola hasta su propio
  timeout, ocupando GPU y slot de admisión. Ahora la desconexión cancela el
  trabajo en curso (como ya hacía chat) y la petición termina con 499. La
  concurrencia de estos endpoints no cambia.

## 0.134.0 — 2026-11-24

### Nuevo

- **Prefijo `/_node/<nodo>/` para forzar el nodo de la federación** sin
  cabeceras: `/_node/<nodo>/v1/chat/completions` es `/v1/chat/completions` con
  `X-Ocabra-Node: <nodo>`. Pensado para clientes que solo dejan configurar la
  URL base (`base_url=https://…/_node/Mapache/v1`). Vale delante de `/v1/`,
  `/api/`, `/health`, `/version` y `/ready`. Si el nodo no tiene el modelo,
  la petición se sirve donde se serviría sin forzar, igual que con la
  cabecera.

---

## 0.133.0 — 2026-11-24

### Cambiado

- **Entrada malformada detectada antes.** Una petición a
  `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/pooling`,
  `/v1/images/generations`, `/v1/audio/generate` o a `/api/chat`,
  `/api/generate`, `/api/embed` a la que le falta un campo obligatorio
  (`model`, `messages`, `input`, `prompt`…) o que manda un campo conocido con
  el tipo equivocado responde ahora **422** con un mensaje claro que nombra el
  campo, en vez del 500 opaco que daba al fallar más adelante. Los campos
  desconocidos y los valores que ya enviabas se siguen aceptando igual.

## 0.132.0 — 2026-11-24

### Nuevo

- **Retención de estadísticas (opcional).** Las tablas de estadísticas
  (`gpu_stats`, `server_stats`, `request_stats`) ya pueden podarse
  automáticamente por antigüedad. Viene **desactivada**: no se borra nada hasta
  que la actives en Ajustes. Al activarla, por defecto conserva `request_stats`
  365 días y las muestras de GPU/servidor 90 días (configurable por tabla;
  0 = conservar siempre).

## 0.131.0 — 2026-11-23

### Nuevo

- **Progreso interno y tiempo restante en benchmarks.** El seguimiento separa
  los modelos aceptados de los casos completados en la prueba actual. IFEval y
  BFCL informan cada caso y la fase de scoring; los demás runners muestran
  actividad indeterminada hasta que adopten el mismo contrato.
- **Estimación viva de finalización.** Cada ejecución muestra cuánto queda para
  el modelo actual y para la cola completa, combinando el ritmo observado con
  el histórico del mismo perfil. Las pausas y los intentos descartados por
  contaminación no inflan la estimación.

### API

- `GET /ocabra/benchmarks/runs` y su detalle añaden `testProgress` con
  `completed`, `total`, `unit`, `stage` y `updatedAt`, o `null` cuando el runner
  no ofrece avance estructurado.

## 0.130.1 — 2026-11-23

### Arreglos

- **Las estimaciones incluyen el trabajo previo a una reanudación.** Si un
  modelo se pausa o falla y continúa desde checkpoint, se suman los tramos de
  la cadena aceptada; un intento descartado por contaminación tardía reinicia
  la cadena. Los modelos ya aceptados de una ejecución todavía activa también
  alimentan inmediatamente el cálculo.
- **Los gráficos de benchmarks recuperan contraste y contexto.** Los tooltips
  tienen fondo opaco, borde y sombra, y el diagrama calidad-rendimiento muestra
  los nombres de ambos ejes.

## 0.130.0 — 2026-11-23

### Nuevo

- **Estimación de duración antes de lanzar un benchmark.** El formulario usa
  los tiempos limpios del mismo modelo y suite, escala perfiles equivalentes
  cuando es posible y muestra rango, confianza y desglose por modelo. También
  avisa de preparaciones iniciales y de que las pausas por tráfico real no
  forman parte del cálculo.

## 0.129.0 — 2026-11-23

### Nuevo

- **Ranking explicable de modelos y backends.** Resultados combina Corrala,
  Escriba, BFCL e IFEval con fiabilidad, rendimiento y eficiencia GPU. Incluye
  perfiles de pesos, configuración personalizada, cobertura mínima, filtros por
  versión/hardware, desglose de la puntuación y frontera de Pareto; nunca
  modifica las métricas oficiales guardadas por cada benchmark.

## 0.128.0 — 2026-11-23

### Nuevo

- **Resultados de benchmarks como vista principal.** La página separa la
  comparativa, el seguimiento de ejecuciones y el lanzamiento de nuevas
  pruebas. Los resultados ofrecen rankings gráficos por métrica y prueba,
  líderes, detalle tabular e histórico; Corrala añade una calidad global
  ponderada por el número real de casos.

---

## 0.127.5 — 2026-11-22

### Arreglos

- **El benchmark de Corrala vuelve a conectarse a su base desechable.** Al
  derivar `ocabra_bench` desde la conexión principal se conserva ahora la
  contraseña real; antes SQLAlchemy la sustituía por `***` y todos los intentos
  terminaban con un error de autenticación de PostgreSQL.

---

## 0.127.4 — 2026-11-21

### Cambios

- **llama.cpp upstream y el fork Makespace se actualizan juntos.** El backend
  `llama_cpp` incorpora los últimos cambios de `ggml-org/llama.cpp`, y
  `llamacpp_mks` conserva la visión bajo demanda y el cambio de contexto en
  vivo sobre esa misma base actualizada. Las imágenes nativas y la imagen de
  nodo usan exactamente los commits validados.

### Arreglos

- **Las actualizaciones de backends nativos instalados desde código vuelven a
  notificarse.** Como no tienen paquetes Python, ahora se compara el commit
  realmente compilado con el commit publicado en la imagen OCI. También se
  reconocen instalaciones antiguas que aún no guardaban ese dato en sus
  metadatos.

---

## 0.127.3 — 2026-11-21

### Arreglos

- **Modelos GGUF con número de cabezas por capa ya no salen como «no cabe».**
  Algunas arquitecturas nuevas (p. ej. `inkling2`) guardan
  `attention.head_count` como una lista por capa; el servidor no sabía leerla,
  daba «no se pudo leer el header del GGUF» y marcaba todos los peldaños como
  que no cabían, aunque sí cabían.

---

## 0.127.2 — 2026-11-21

### Arreglos

- **Los modelos de Ollama en el formato nuevo ya no se dan de alta en
  llama.cpp.** Algunos tags recientes (p. ej. `inkling-2-mini`) no traen un
  GGUF sino un fichero por tensor (`format: safetensors`, motor MLX de
  Ollama). Se registraban en llamacpp_mks, el editor decía «No cabe: sin
  model_path del GGUF» y el modelo no podía cargar nunca. Ahora se registran
  en el backend `ollama`, el único que los sirve.
- **Borrar un modelo GGUF descargado de Hugging Face libera de verdad el
  disco.** La carpeta se buscaba por el nombre público del modelo y no la
  encontraba: la API respondía 200 y el fichero se quedaba (16 GB en un
  caso). Ahora se borra la carpeta de la descarga, salvo que otro modelo la
  use.
- **Registrar un modelo que el servidor ya veía en Ollama lo guarda.** Antes
  `POST /ocabra/models` devolvía 200 sin persistir nada, y el modelo se
  quedaba sin nombre público en `/v1/models`.

---

## 0.127.1 — 2026-11-20

### Corregido

- **El modelo de vídeo se llama `ltx-2.5` en todas partes.** Su id canónico
  era `comfyui_video/video` (el nombre de la carpeta de pesos) y en
  estadísticas y en `/v1/models` salía como «video». Ahora es
  `comfyui_video/ltx-2.5`. **Acción:** solo si usabas el id canónico; el
  nombre `ltx-2.5` sigue funcionando igual.

---

## 0.127.0 — 2026-11-20

### Nuevo

- **Planes programables de benchmarks.** La API interna permite guardar una
  configuración completa, ejecutarla al momento o programarla por días de la
  semana, hora y zona horaria. No crea otra ejecución si el mismo plan conserva
  una activa.
- **Referencias para comparar resultados.** Una ejecución terminada puede ser
  la referencia única de su aplicación y suite. El listado conserva esa
  referencia aunque quede fuera del límite de ejecuciones recientes; la UI
  presenta los deltas de calidad y rendimiento junto a versiones, backend y
  GPU.

---

## 0.126.1 — 2026-11-20

### Corregido

- **Versión exacta de backends nativos en benchmarks.** Los módulos compilados
  desde Git, como `llamacpp_mks`, no tienen una versión de paquete Python y se
  mostraban como «versión no informada». Cada intento conserva ahora el commit
  completo de las fuentes que produjeron el binario y la UI presenta su hash
  corto. Para instalaciones Python u OCI se mantienen la versión de paquete y
  el digest de imagen respectivamente.

---

## 0.126.0 — 2026-11-20

### Nuevo

- **`/v1/videos` parte de imágenes.** `input_reference` (el campo de OpenAI)
  usa una imagen como primer fotograma y el vídeo la anima siguiendo el
  prompt. Se acepta como fichero en multipart (lo que envía el SDK de OpenAI),
  o en JSON como `data:` URL, URL `http(s)` pública o `{"image_url": ...}`.
  Si no mandas `size`, el vídeo toma la proporción de la imagen.
- **`last_frame`** (extensión, necesita `input_reference`): la imagen del
  último fotograma; el vídeo hace la transición de una a otra.
- Hasta 20 MB por imagen (`VIDEO_MAX_IMAGE_MB`); PNG, JPEG, WebP o GIF.

**Acción:** si mandabas `input_reference` y tratabas el 400 como «no
soportado», ahora se usa de verdad.

---

## 0.125.2 — 2026-11-20

### Corregido

- **`POST /v1/videos` con el modelo de vídeo sin cargar.** Respondía 400
  `model_not_capable` porque un modelo que nunca se ha cargado no publica aún
  sus capacidades, y ese es el estado normal del modelo de vídeo (ocupa una
  GPU entera y solo se carga para generar). Ahora se encola y el modelo se
  carga cuando le llega el turno.

---

## 0.125.0 — 2026-11-20

### Nuevo

- **Generación de vídeo con audio: `/v1/videos`**, compatible con la API de
  vídeo de OpenAI. Es asíncrona: `POST /v1/videos` encola y devuelve el objeto
  `video` en `queued`; consulta `GET /v1/videos/{id}` (`status`, `progress`,
  `queue_position`) y descarga `GET /v1/videos/{id}/content` cuando esté en
  `completed` (MP4 H.264 + AAC; `?variant=thumbnail` da un JPEG). También
  `GET /v1/videos` y `DELETE /v1/videos/{id}`. El primer modelo es
  **LTX-2.5** (backend `comfyui_video`): un vídeo de 5 s a 1280x704 tarda
  ~65 s con el modelo cargado (~85 s si hay que cargarlo). Diferencias con OpenAI: `seconds` es
  cualquier entero de 1 a 10, `size` se redondea a múltiplos de 64
  (`1280x720` sale `1280x704`), y `seed`/`negative_prompt` son extensiones.
  Se genera un vídeo a la vez por nodo y el fichero se borra a las 24 h. No
  está disponible en nodos `zero_data`.

### Corregido

- **`/ocabra/video/upscale` con modelos recién descargados.** Tenía el mismo
  fallo que la 0.124.1 arregló en `/v1/*`: con un preset de runtime buscaba el
  worker con otra clave y respondía 503 aunque el modelo estuviera cargado.
- **El modelo de diarización configurado se ignoraba.** Un perfil de Whisper
  con `diarization_model_id` (por ejemplo pyannote community-1) diarizaba
  siempre con pyannote 3.1, porque el modelo solo llegaba al worker si la
  diarización estaba activada al cargarlo, y ahora se decide en cada
  petición. Ahora llega siempre.

---

## 0.124.1 — 2026-11-20

### Corregido

- **Embeddings, rerank, pooling, audio y escalado con modelos recién
  descargados.** Un modelo dado de alta desde Explorar (o por
  `POST /ocabra/downloads`) recibe un preset de runtime `normal`, y su worker
  se registraba con una clave distinta de la que buscaban `/v1/embeddings`,
  `/api/embed`, `/v1/rerank`, `/v1/score`, `/v1/pooling`, `/v1/classify`,
  `/v1/audio/*` y `/v1/images/upscale`. El resultado era un **503
  `worker_unloaded`** («it was unloaded while the request waited») en todas las
  peticiones, aunque el modelo estuviera cargado. Los modelos antiguos sin
  preset no estaban afectados. Ahora esos endpoints usan el worker que se acaba
  de resolver.

---

## 0.124.0 — 2026-11-20

### Nuevo

- **Benchmarks reproducibles y comparables por backend.** El selector agrupa
  perfiles por modelo y muestra cada backend como una opción independiente.
  Cada intento conserva commit de dataset, versiones de oCabra, Python,
  scorer y módulo de backend, preset/contexto efectivos, nodo y todas las GPU
  utilizadas. La descarga JSON incluye esa evidencia además de las métricas.
- **Reintento desde checkpoint.** Una ejecución fallida se puede reanudar en
  el modelo que falló sin repetir los ya aceptados. La interfaz muestra además
  la fase exacta de preparación o ejecución y el historial de hardware.

### Arreglos

- **Ningún intento contaminado se acepta como resultado.** Si se desactiva la
  pausa automática, el mismo modelo se reencola inmediatamente; si está
  activa, espera el periodo sin tráfico. Los fallos inesperados cierran el
  intento en curso para que también pueda reanudarse con seguridad.
- **IFEval puede ejecutar el scorer oficial hasta el final.** El adaptador
  prepara en su caché persistente el recurso `punkt_tab` que NLTK requiere
  pero el proyecto oficial no descarga. Un intento ya generado reutiliza sus
  respuestas y solo repite la puntuación.

---

## 0.123.1 — 2026-11-20

### Arreglos

- **BFCL ya arranca su scorer oficial.** La revisión fijada de BFCL importa
  `soundfile` a través de `qwen-agent`, pero no lo declara en sus dependencias;
  el runner terminaba con código 1 antes de la primera inferencia. oCabra añade
  ahora esa dependencia al entorno aislado de BFCL.

---

## 0.123.0 — 2026-11-20

### Nuevo

- **BFCL v4 e IFEval en la cola de benchmarks.** El catálogo interno ofrece
  ahora las suites públicas `bfcl/simple_python`, `bfcl/core`, `ifeval/quick`
  e `ifeval/full`. Usan revisiones oficiales fijadas, conservan checkpoints
  por caso y pasan por la misma pausa y repetición automática cuando hay
  tráfico interactivo. La primera ejecución prepara sus dependencias aisladas
  en el cache persistente de benchmarks.

---

## 0.122.3 — 2026-11-20

### Arreglos

- **Los errores de `/v1/images/upscale` traen `error.code`.** Antes llegaban
  con `code: null` y solo se distinguían por el status. Ahora: 413
  `image_too_large` (más píxeles de entrada de los admitidos) o
  `file_too_large` (fichero por encima del límite en MB); 400
  `invalid_image` (ilegible, o salida demasiado grande para el formato),
  `empty_image`, `invalid_value`, `invalid_output_format`,
  `invalid_response_format` y `missing_required_parameter`. Lee `error.code`
  y `error.message` de la raíz del cuerpo: `detail` se mantiene por
  compatibilidad (ver 0.121.0).
- Las estadísticas de un escalado guardan también los bytes entregados y el
  formato: la misma foto pesaba 10,3 MB en PNG y 0,7 MB en WebP.

---

## 0.122.2 — 2026-11-20

### Arreglos

- **Las estadísticas de `/v1/images/upscale` guardan el tamaño de cada foto.**
  Antes solo constaba que había una imagen: una de 1080p (~10 s) y una 4K
  (~45 s) contaban igual, y la estimación de duración no podía tenerlo en
  cuenta. Ahora se guardan la resolución de entrada, la de la red y la
  entregada (y las teselas), la velocidad se da también en megapíxeles por
  segundo y la duración esperada crece con los megapíxeles. Además, la
  velocidad de las peticiones que no se miden en tokens (transcripción, voz,
  imágenes, escalado…) ya no cuenta la espera a que cargase el modelo.

---

## 0.122.1 — 2026-11-19

### Arreglos

- **Los benchmarks ya pueden clonar repositorios privados de GitHub.**
  `BENCHMARK_GIT_TOKEN` se envía con el esquema HTTP que entiende el transporte
  Git HTTPS. Antes se enviaba como Bearer y GitHub rechazaba el clon aunque el
  token tuviera acceso al repositorio.

---

## 0.122.0 — 2026-11-19

### Nuevo

- **Benchmarks de aplicaciones desde el panel.** Un gestor de modelos puede
  encolar suites de Escriba o Corrala para varios modelos, consultar sus
  resultados e intentos y pausar, reanudar o cancelar la ejecución. oCabra
  pausa automáticamente cuando detecta tráfico interactivo, conserva el
  checkpoint y repite los intentos que se hayan solapado con peticiones
  externas. Las rutas internas viven bajo `/ocabra/benchmarks/*`.

---

## 0.121.1 — 2026-11-19

### Arreglos

- **Transcripciones largas: el primer segmento llega mucho antes.** Con audios
  de más de una hora, `/v1/audio/transcriptions` pasaba 3-4 minutos sin
  devolver nada antes de empezar a transcribir (1 h 35 min: ~225 s). Ahora son
  ~30 s, la transcripción es idéntica y la transcripción completa tarda menos
  de la mitad. Además, el avance que da el servidor (`audio.totalSeconds` en
  `/ocabra/models/activity`) muestra la duración del audio desde el primer
  momento, también mientras prepara.

---

## 0.121.0 — 2026-11-19

### Cambiado

- **Errores con el formato de cada API (y `detail` obsoleto).** Hasta ahora
  los errores de `/v1/*` llegaban como `{"detail": {"error": {...}}}` (y a
  veces `{"detail": "texto"}`), un formato que los SDK de OpenAI no leen:
  veías un error genérico sin el mensaje ni el código reales. Ahora todas las
  respuestas de error de `/v1/*` llevan además, en el nivel superior, el objeto
  `error` de OpenAI `{"message", "type", "param", "code"}`, también en los
  errores de validación (422, con el campo afectado en `param`). Las de `/api/*`
  (compatibilidad Ollama) llevan `{"error": "texto"}`, que es lo que esperan
  los clientes de Ollama. Los códigos de estado y las cabeceras (`Retry-After`
  incluida) no cambian. **Acción (obsoleto):** `detail` se mantiene de momento
  con el mismo contenido para no romper a nadie, pero se retirará en una
  versión futura. Si tu cliente lee `detail`, pásate a `error`
  (`error.message` / `error.code` en `/v1`, `error` en `/api`). `/ocabra/*` no
  cambia.
- **Estimaciones con el ACL de modelos.** `POST /ocabra/estimate` y
  `POST /ocabra/models/{id}/memory-estimate` responden 404 para modelos que no
  tienes concedidos, como ya hacía `GET /models`. `run_probe=true` en
  `memory-estimate` (arranca un proceso vLLM) y `GET /ocabra/trtllm/estimate`
  piden ahora el rol `model_manager`.
- **Reescalado de vídeo.** `POST /ocabra/video/upscale` rechaza con 413 los
  segmentos de más de 512 MB (configurable con `VIDEO_UPSCALE_MAX_UPLOAD_MB`).
  La respuesta solo lleva las cabeceras de estadísticas `x-ocabra-*`
  documentadas.
- **Modelos locales en carpeta.** `POST /ocabra/registry/local/register`
  acepta carpetas de modelo (safetensors de HF, diffusers…) para vllm, sglang,
  tensorrt_llm, diffusers, mage, upscaler y flashvsr. **Acción:** `model_ref`
  tiene que ser la ruta de la carpeta bajo MODELS_DIR, o `org/name` para las
  carpetas en `huggingface/org--name`.

### Arreglos

- **Matar una sesión Realtime la cierra de verdad.**
  `POST /ocabra/sessions/{id}/kill` solo la quitaba de la lista: el cliente
  seguía conectado y la respuesta en curso seguía generando. Ahora el cliente
  recibe un evento `error` con `code: "session_closed"`, el WebSocket se cierra
  con el código 4001 y la respuesta añade `connection_closed`.
- **Descargas.** `GET /ocabra/downloads/{id}/stream` termina solo cuando la
  descarga acaba (completada, fallida o cancelada) y su primer evento es el
  estado actual. `GET /ocabra/downloads/{id}` y su stream funcionan con el rol
  `user`, como la lista: antes daban 403 y el panel de descargas de los
  usuarios sin rol de gestor reintentaba en bucle.
- **Ollama.** Dos POST seguidos a `/ocabra/ollama/server/update` (o
  `apply-settings`) ya no lanzan la operación dos veces: el segundo recibe 409.
- **Agentes.** Borrar un agente lo quita de los subagentes de los demás. Antes
  un agente creado después con el mismo nombre se les colgaba solo. La
  respuesta añade `detached_from`.
- **Servicios.** `GET /ocabra/services/{id}/generations` da 404 para un
  servicio que no existe, en vez de `[]`.

## 0.120.0 — 2026-11-19

### Nuevo

- **`POST /v1/images/upscale`: mejorar la resolución de una foto.** Hasta
  ahora solo se podía escalar vídeo (`/ocabra/video/upscale`), y
  `/v1/images/edits` rechaza estos modelos porque es img2img con prompt. Ahora
  mandas la foto en multipart (`image`, `model`) con cualquiera de los modelos
  de escalado (`realesrgan-x4plus`, `ultrasharp-x4`, `remacri-x4`,
  `real-esrgan-compact-x4`, `ultrasharp-v2-lite`) y recibes la imagen escalada
  4x en PNG. Opcionales: `scale` o `target_width`/`target_height` para una
  salida menor (solo reducen: nunca se estira por encima de lo que da el
  modelo; mira `X-Ocabra-Delivered-Resolution`), `output_format`
  (`png`/`jpeg`/`webp`), `output_quality` y `response_format=b64_json` si
  prefieres JSON al estilo OpenAI. La transparencia se conserva en PNG y WebP.
  Las fotos grandes se procesan por teselas; el límite es 25 MB y 3840×2160 px
  de entrada (413 si te pasas). Un modelo que no es de escalado da 400
  `model_not_capable` sin llegar a cargarse.

### Cambiado

- **Un modelo que comparte GPU con otro ya no se desaloja entre dos peticiones
  seguidas de un mismo cliente.** Cuando un modelo espera la GPU que ocupa
  otro, el cargado aguarda ahora hasta **4 s** tras su última petición por si
  llega la siguiente, y la atiende. Lo hace como mucho durante **60 s**
  desde que el otro la pidió; pasado ese tiempo cede la GPU aunque siga
  llegando trabajo. Antes se desalojaba en cuanto quedaba libre: dos agentes
  que encadenaban peticiones a modelos distintos de la misma tarjeta
  provocaron 12 cargas en 10 minutos, y cada petición siguiente esperaba
  1-2 minutos una recarga para hacer un segundo de trabajo. Los dos tiempos
  se ajustan en Ajustes → GPU (gracia 0 = el comportamiento anterior).

### Arreglos

- **La petición que llegaba justo cuando su modelo iba a desalojarse se
  quedaba 30 s parada y luego esperaba a que el modelo se recargara.** El
  servidor intentaba antes encoger el contexto del modelo y, al deshacerlo,
  esperaba a que terminase esa petición, que a su vez esperaba al desalojo.
  Ya no espera a las peticiones aparcadas, no deshace el encogimiento de un
  modelo que va a descargarse y no vuelve a intentarlo cuando ya midió que no
  libera bastante memoria.

---

## 0.119.0 — 2026-11-19

### Corregido

- **El gateway de servicios (ComfyUI, ACE-Step, Unsloth, TRELLIS.2,
  Automatic1111) no enrutaba nada desde la 0.108.0**: al dejar de pasarle el
  `.env` entero se quedó sin las direcciones de los servicios. Vuelve a
  recibirlas.
- **El listado de batches** (`GET /v1/batches`) hace una sola consulta por
  página en vez de una por lote. La respuesta es la misma.

### Cambiado

- La documentación servida y la del repositorio están al día: `docs/API.md`
  lista las 284 rutas (y un test impide que se quede atrás), `docs/INSTALL.md`
  describe la instalación actual, `.env.example` documenta todos los ajustes con
  sus valores por defecto reales, y todos los endpoints tienen descripción.

---

## 0.118.2 — 2026-11-19

### Arreglos

- **Una transcripción larga ya no sale como parada mientras se prepara.**
  Antes de transcribir, el worker de whisper decodifica el audio entero y
  busca los tramos con voz, y en ese rato no hay avance que contar: con 1 h
  35 min de MP3 fueron 4 minutos en los que `/ocabra/models/activity` la
  marcaba como «sin actividad del worker» (`stalled: true`). Ahora
  `audio.phase` vale `preparing` durante esa fase (valor nuevo, antes de
  `transcribing` y `diarizing`) y no cuenta como parada. Y `audio.realtimeFactor`
  se mide desde que empieza a transcribir: incluía la preparación y salía
  más bajo de lo real.

---

## 0.118.1 — 2026-11-19

### Arreglos

- **`POST /ocabra/estimate` ya no confunde una carga en frío con una petición
  lenta.** La duración que aprende cada modelo incluía la espera a que se
  cargase: una sola petición de 1 s tras una carga de 17 s dejaba la mediana
  de ese modelo en 18 s. Solo el chat con streaming descontaba la carga; las
  peticiones sin streaming, las líneas de batch y el resto de endpoints no.
  Ahora se descuentan la carga y la cola en todas, y `expected_s` vuelve a ser
  lo que tarda la petición con el modelo ya cargado (las filas antiguas se van
  saliendo de la ventana de 7 días).
- **`cold_start_seconds` daba siempre 0** y `expected_s` de un modelo
  descargado no sumaba su carga: las cargas se guardan por modelo base y las
  peticiones por perfil, y no se casaban. Ahora casan, y se suma la mediana
  de sus cargas (el p95 solo a `p95_s`). La duración esperada de las
  peticiones en `/ocabra/models/activity` hereda el arreglo cuando el modelo
  aún no tiene ritmos medidos.
- **`DELETE /v1/files/{id}` del fichero de entrada de un batch terminado
  daba 500.** Ahora lo borra (`"deleted": true`) y el batch conserva su
  `input_file_id`, como en OpenAI. Mientras el batch está en curso sigue
  negándose con 409 `file_in_use`.

---

## 0.118.0 — 2026-11-19

### Cambiado

- **Un solo orden para expulsar modelos cuando hace falta memoria.** Hasta
  ahora había dos: el de las cargas normales (primero los modelos bajo demanda,
  luego los que se mantienen calientes, y los fijados en último lugar) y el de
  las cargas con GPU fija (por antigüedad de uso). Ahora todas usan el primero,
  también el vigilante de memoria. En la práctica: una carga con GPU fija puede
  expulsar antes un modelo bajo demanda usado hace un momento que uno caliente
  que lleva horas sin usarse, y si ese modelo bajo demanda está atendiendo
  peticiones, la carga espera a que terminen antes de pasar al siguiente. Los
  modelos fijados siguen siendo los últimos, y los fijados de forma estricta y
  las sesiones de voz en curso nunca se expulsan.

---

## 0.117.0 — 2026-11-19

### Retirado

- **Acción** — Se elimina la integración con **LiteLLM**: desaparece
  `POST /ocabra/config/litellm/sync` y los campos `litellm*` de
  `GET /ocabra/config`. `PATCH /ocabra/config` sigue aceptando esas claves y las
  ignora. Las variables `LITELLM_*` ya no hacen nada.
- **Acción** — Se elimina la integración con **Langfuse** (trazas). Las
  variables `LANGFUSE_*` ya no hacen nada.
- **Acción** — Se elimina `GET /ocabra/stats/by-model`, que no usaba nada (las
  estadísticas por modelo están en `GET /ocabra/stats/performance`).
- Se retira **Hunyuan3D** de los servicios y del gateway (ya no estaba en el
  compose). Sus claves en `PATCH /ocabra/config` se aceptan y se ignoran.

### Corregido

- **Ráfagas de subidas**: varias transcripciones o ediciones de imagen
  simultáneas podían superar el límite de peticiones a la vez del modelo,
  porque se contaban después de leer el fichero. Ahora se cuentan al entrar:
  las que sobran reciben `429` con `Retry-After`.
- **`/v1/images/generations` contaba cada petición dos veces** contra ese
  límite, así que un modelo de imagen admitía la mitad de lo configurado.
- **Cargas que expulsan otro modelo** terminan antes y expulsan menos: la espera
  a que se libere la memoria acaba en cuanto la tarjeta lo muestra (antes podía
  esperar 12 s de más), y ya no se expulsa un segundo modelo cuando bastaba con
  uno.
- **Un modelo podía quedarse «descargándose» para siempre** si se cancelaba su
  descarga a mitad; ahora vuelve a su estado anterior y la siguiente petición lo
  carga.
- **Una transcripción con diarización** ya no pierde la memoria que necesita
  porque otro modelo se cargue a la vez en la misma tarjeta.
- `POST /ocabra/models/{id}/context` espera a una carga en curso del mismo
  modelo y responde `409` (reintentable) si no puede hacerlo; los `503`
  `context_temporarily_reduced` llegan al momento en vez de tras una espera
  larga.

---

## 0.116.0 — 2026-11-19

### Nuevo

- `/v1/batches`: las respuestas tienen su esquema en OpenAPI (`Batch`,
  `BatchList`, con `status`, `request_counts`, `output_file_id`,
  `error_file_id`…). Antes las cuatro rutas declaraban `{"type": "object"}` y
  un cliente generado del esquema no podía leer el resultado de un lote.
- Un lote sin terminar trae `ocabra`: en qué fase está y por qué espera
  (`queued` con `queue_position`, `loading_model` con
  `models[].expected_wait_seconds`, `running`, `yielding_to_interactive` con
  `retry_in_seconds`, `waiting_for_vram`, `finalizing`, `cancelling`), con una
  frase en `detail`. Un lote a 0/144 durante un minuto ya no es indistinguible
  de uno lento.
- Límite de peticiones por lote: 50 000 (`BATCH_MAX_REQUESTS`). Por encima,
  `POST /v1/batches` da 400 con `code: "batch_too_large"` (comprobado antes de
  validar las líneas); un fichero sin peticiones, `code: "empty_batch"`. Los
  límites están documentados en `/docs`.

---

## 0.115.0 — 2026-11-19

### Cambiado

- **Las subidas (multipart) y `/v1/realtime` respetan el modo mantenimiento y
  el límite de peticiones simultáneas**, como el resto. `/v1/audio/transcriptions`,
  `/v1/images/edits` y demás peticiones multipart reciben `503`
  `model_maintenance` (`Retry-After: 30`) si el modelo está en mantenimiento y
  `429` `model_at_capacity` con `Retry-After` si está lleno; antes pasaban
  siempre. `/v1/realtime` rechaza la conexión con esos mismos códigos antes de
  abrir la sesión.
- **Subagentes**: cada delegación tiene un tope de 600 s; al agotarlo, el agente
  padre recibe un error de herramienta `subagent_timeout`. Una respuesta de
  subagente de más de 32 KiB le llega recortada, con una marca visible.

### Corregido

- **Las herramientas de un agente seguían ejecutándose** después de que el
  cliente se desconectara a mitad de un stream. Ahora se cancelan.
- **llama.cpp podía servir otro fichero GGUF sin avisar**: un modelo cuyo
  fichero había desaparecido, o cuyo nombre coincidía con varios ficheros (o
  con otro más corto: `qwen3.8:27b` con `qwen3.gguf`), cargaba el que
  encontrara. Ahora falla con un error que lo explica. Lo mismo en BitNet.
- **whisper, TTS y ACE-Step**: si el proceso del modelo moría al arrancar, la
  carga esperaba todo el tiempo máximo y fallaba sin motivo. Ahora falla al
  instante con el código de salida y el final del log.
- **Carreras de memoria de GPU**: la descarga manual de un modelo, hacer sitio
  para la visión o para whisper, la reserva de memoria de un servicio y el
  vigilante de VRAM ya no se cruzan con una carga en curso; esperan su turno
  (o devuelven `503` reintentable si no pueden). Antes podían dejar la cuenta
  de memoria desfasada y acabar en errores de memoria de CUDA.
- **Latencia**: `/api/tags`, `/api/ps` y la lista de modelos ya no recorren el
  disco en cada petición (los tamaños pueden tardar hasta 30 s en reflejar un
  cambio dentro de una subcarpeta), y el servidor ya no vuelve a analizar los
  streams largos en cada trozo ni lee en memoria los ficheros subidos para las
  estadísticas.

---

## 0.114.0 — 2026-11-19

### Nuevo

- Modelos llama.cpp con adaptadores LoRA: `extra_config.llama_cpp.lora` (una
  ruta a un LoRA en GGUF, o una lista de rutas o de `{path, scale}`) los aplica
  sobre los pesos del modelo al cargar (`--lora` / `--lora-scaled`). Sirve
  para correr un fine-tune publicado como adaptador sobre el GGUF que ya está
  instalado, por ejemplo en un perfil propio (`load_overrides`) sin duplicar
  el modelo base en disco. La memoria del adaptador entra en la proyección.

---

## 0.113.2 — 2026-11-18

### Corregido

- **Los modelos de whisper.cpp se perdían al reinstalar o actualizar el
  backend.** Vivían dentro de su carpeta, que una reinstalación o una
  actualización sustituye entera. Ahora están en `<MODELS_DIR>/whisper_cpp` (el
  modelo de transcripción y el de VAD). La carpeta antigua se sigue leyendo,
  para los nodos que aún no se han movido.

---

## 0.113.1 — 2026-11-18

### Corregido

- **Las transcripciones con whisper.cpp fallaban todas** con `500` «FFmpeg
  conversion failed» desde la 0.106.2. Es el motor por CPU al que recurre
  `recomendado-stt` cuando las tarjetas están ocupadas, y también el de los
  modelos `-cpu`. whisper.cpp dejaba el audio convertido en una carpeta en la
  que el servidor ya no puede escribir. Ahora usa la carpeta temporal.
- **whisper.cpp tiene VAD.** Está instalado el modelo Silero v6.2.0, así que
  `vad_filter` funciona y viene activado por defecto, igual que en
  faster-whisper: los tramos sin voz se descartan antes de transcribir. Manda
  `vad_filter=false` para desactivarlo.

---

## 0.113.0 — 2026-11-18

### Corregido

- **Un modelo que estaba en «cargando» detrás de otra carga larga** acababa en
  `ERROR` (y respondía `503` sin `Retry-After`) aunque nada se hubiera roto.
  Ahora, si la espera se agota, el modelo vuelve a su estado anterior y la
  petición recibe un `503` con `Retry-After`, que se puede reintentar.
- **Una petición colgada más allá del tiempo máximo** dejaba el modelo en
  `ERROR` con su proceso vivo y su memoria ocupada: la siguiente petición
  arrancaba un segundo proceso al lado. Ahora se para el proceso y se libera la
  memoria; la siguiente petición lo vuelve a cargar. Si otras peticiones de ese
  mismo modelo siguen avanzando, el proceso se mantiene.
- **Descargar un modelo mientras se estaba cargando** podía cruzarse con la
  carga. Ahora la descarga espera a que termine.
- **Al descargar voxtral, vibeasr o whisper.cpp** quedaban vivos procesos que
  ellos habían arrancado (por ejemplo, el `vllm serve` de voxtral), con memoria
  de GPU ocupada hasta reiniciar. Ahora se paran con el modelo.
- **whisper.cpp** respeta `condition_on_previous_text=false` (salvo que mandes
  `prompt`, que comparte el mismo presupuesto y gana). Acepta `vad_filter`,
  pero solo lo aplica si hay un modelo VAD instalado; hoy no lo hay.
- **Un modelo que solo existe en un nodo federado** cuyo nodo no responde daba
  `404` «el modelo no existe». Ahora da `503` `peer_unavailable` con
  `Retry-After`, nombrando el nodo.
- **`speed` y `temperature`** con `nan` o `inf` dan `400` `invalid_value`.
- **`/ocabra/stats/recent`** contaba como «en cola» peticiones que estaban
  ejecutándose en modelos llama.cpp sin presets: ahora usa los slots reales.
- **Modelos vLLM configurados con claves en camelCase** tenían una estimación
  de memoria que no contaba la caché KV.
- **El escalado de vídeo** aparece en la tabla de rendimiento en fotogramas por
  segundo.

### Cambiado

- `routerFallbackDelayMs` ya no aparece en `GET /ocabra/config`: nunca tuvo
  efecto. `PATCH` lo sigue aceptando y lo ignora.

---

## 0.112.0 — 2026-11-18

### Corregido

- **Errores del cliente que daban `500`.** Un cuerpo que no es JSON válido, o
  que es JSON pero no un objeto (`[]`, `"texto"`), da `400` con
  `code: "invalid_json"` en todos los endpoints de `/v1` y `/api` que leen
  JSON. Un `temperature` (transcripciones) o un `speed` (voz) que no sea un
  número da `400` con `code: "invalid_value"` y `param`; `speed: null` usa el
  valor por defecto.
- **Streaming hacia un nodo federado.** En `/v1/chat/completions`,
  `/v1/completions`, `/api/chat` y `/v1/audio/speech` con `stream: true`, si el
  nodo remoto rechazaba la petición recibías un `200` con un stream que se
  cortaba. Ahora, si el rechazo admite respaldo (`401`, `403`, `404`, `5xx`)
  o el nodo no responde, la petición se sirve aquí; si no, recibes el código
  y el mensaje del nodo. `/api/chat` además manda al nodo el modelo que eligió
  el router, no el nombre del router.
- **Routers en `/v1/completions` con streaming.** Un router sin destino
  disponible daba `500`; ahora da lo mismo que en chat: `400`
  `model_not_capable` si ningún destino completa texto, `503`
  `router_no_target` si no hay ninguno habilitado. Los destinos que no
  completan texto ya no se eligen.
- **`/v1/audio/speech`.** Un worker de voz inalcanzable da `503`
  `worker_unavailable` con `Retry-After` (antes `500`), y uno que no contesta a
  tiempo da `504` `generation_timeout`. En WAV, un `4xx` del worker tiene tipo
  `invalid_request_error` (antes `server_error`). Con federación, los nombres
  de OpenAI (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`) llegan al nodo remoto
  como el modelo al que este servidor los traduce.
- **`/v1/audio/transcriptions` en un nodo federado** respeta ya `vad_filter`
  y `condition_on_previous_text`.
- **`Retry-After`.** Los errores que vienen del motor conservan su
  `Retry-After` (un `429` o `503` del propio motor lo perdía). El `409`
  `insufficient_vram` lo lleva ahora (30 s): la memoria se libera cuando otros
  modelos quedan ociosos.
- **`/api/chat`, `/api/generate` y `/api/embed` con nombres antiguos de
  Ollama.** Si la carga del modelo fallaba, la petición esperaba el timeout
  de carga completo y respondía «did not finish loading in time». Ahora
  responde enseguida con el error real.
- **`/v1/realtime`** con un nombre que comparten dos despliegues (la etiqueta
  de Ollama y el de llama.cpp de `gemma4:26b`) elegía uno según el orden
  interno; ahora elige el despliegue por defecto, como los endpoints REST.
- **`/ocabra/video/upscale`** acepta routers y aparece en las estadísticas.

---

## 0.111.0 — 2026-11-18

Arreglos en la API compatible con Ollama (`/api/chat`, `/api/generate`,
`/api/embed`), para modelos servidos por llama.cpp, vLLM y el resto de motores
que no son el propio Ollama.

### Corregido

- **`/api/chat` y `/api/generate` daban `500` ante errores del motor.** Un
  timeout, un modelo que se descargó mientras la petición esperaba, un worker
  degradado o un `400` del motor llegaban como `500 Internal Server Error`.
  Ahora tienen el mismo código que en `/v1`: `504` con `Retry-After` por
  timeout, `503` con `Retry-After` si el worker no está disponible, y el `4xx`
  del motor con su mensaje.
- **En streaming, un fallo cortaba la conexión sin avisar.** Si el motor
  rechaza la petición antes de empezar, ahora recibes el código HTTP de error
  en vez de un `200` vacío. Si falla a mitad, el stream termina con una línea
  `{"error": "...", "code": "..."}`, como hace Ollama, en vez de cortarse sin
  `done: true`.
- **`think` no hacía nada fuera de los modelos de Ollama.** Ahora
  `"think": false` desactiva el razonamiento y `"think": "low"|"medium"|"high"`
  fija el nivel, en todos los motores. Sin `think` se aplica el nivel por
  defecto del modelo, igual que en `/v1/chat/completions`. También se respetan
  ya `reasoning_effort` y el resto de ajustes de razonamiento de los perfiles.
- **El razonamiento se perdía.** Ahora llega en `message.thinking`, en
  streaming y sin streaming, como en Ollama.
- **`/api/embed` con varias entradas podía devolver un solo vector.** Si la
  petición fallaba, se repetía en silencio con la primera entrada y respondía
  `200` con un vector en lugar de N. Ahora un fallo devuelve su error. Solo
  un Ollama demasiado antiguo para tener `/api/embed` pasa a la vía antigua, y
  esa vía calcula un vector por entrada.
- **`/api/embed` ignoraba `dimensions`.** Ahora se pasa al motor y, si este lo
  ignora (llama.cpp), los vectores se recortan y renormalizan, como en
  `/v1/embeddings`. Un valor que no sea un entero positivo da `400`.
- **`/api/embed` no aceptaba routers** (`recomendado-embedding`) y convertía
  un `403` (sin acceso al modelo) en un `404` engañoso.
- **`/api/embeddings` (la versión antigua) no tenía el campo de Ollama.** Ahora
  responde también `embedding` con un solo vector, como Ollama; `embeddings`
  se mantiene.

---

## 0.110.2 — 2026-11-18

### Corregido

- Desde la 0.109.0, una clave API no podía crear claves (`403` en
  `POST /ocabra/auth/keys`), y eso rompía a quien da de alta claves
  automáticamente: Coder crea una por workspace con su propia clave. Ahora
  vuelve a poder hacerlo, siempre que la clave nueva caduque en 90 días o
  menos (`expires_in_days` entre 1 y 90). Una clave sin caducidad, o crear
  claves para otro usuario (`POST /ocabra/users/{id}/keys`), sigue exigiendo
  la sesión iniciada.

---

## 0.110.1 — 2026-11-18

### Corregido

- Un modelo podía quedarse en «cargando» para siempre, sin que nada volviera a
  cargarlo hasta reiniciar el servidor. Pasaba si la carga se cancelaba justo
  al empezar, por ejemplo cuando una sesión de `/v1/realtime` se cerraba en
  el momento en que arrancaba la carga de su modelo de voz. Las peticiones a
  ese modelo esperaban a una carga que no iba a terminar. Ahora una
  cancelación en ese punto deja el modelo descargado y la siguiente petición
  lo vuelve a cargar.

---

## 0.110.0 — 2026-11-18

### Nuevo

- `GET /ocabra/stats/performance`: cada modelo trae `workSpeeds`, la velocidad
  de lo que no se mide en tokens generados, en su unidad: transcripción y voz
  en «×tiempo real» (segundos de audio por segundo), TTS también en caracteres
  por segundo, embeddings en textos y tokens por segundo, rerank/score/classify
  en documentos por segundo, imágenes en segundos por imagen y el chat de
  `/v1/realtime` en tokens por segundo.
- Las estadísticas de cada petición guardan el trabajo hecho en
  `work_size_meta`: número de textos (embeddings), de documentos (rerank y
  afines) y de imágenes, segundos de audio generados (TTS, música) y, en
  `/v1/realtime`, los segundos de audio transcritos y hablados.

### Corregido

- La energía de cada petición faltaba en la gran mayoría de las que sirve un
  peldaño (el 86 % de los chats): se buscaba la GPU en el modelo base, que no
  está cargado. Ahora sale de las GPUs del worker que la sirvió, todas si el
  modelo está repartido.
- `/v1/responses` y `/v1/audio/generate` se registraban como tipo `other`;
  ahora son `responses` (con sus tokens) y `music`. La duración pedida de una
  pieza de música no se guardaba nunca.
- TTS: el texto enviado como `text` en lugar de `input` no contaba caracteres.

---

## 0.109.0 — 2026-11-18

### Cambiado

- **Acción** — `/v1/realtime` ya no lee la clave de `?api_key=` en la URL: las
  URLs acaban en los logs de los proxies. Manda `Authorization: Bearer <clave>`
  o, desde un navegador (que no puede poner cabeceras en un WebSocket), el
  subprotocolo `openai-insecure-api-key.<clave>`, como el cliente de OpenAI.
  El servidor responde con el subprotocolo `realtime` si lo ofreces.
- **Acción** — Una clave API ya no puede crear claves API
  (`POST /ocabra/auth/keys`, `POST /ocabra/users/{id}/keys`): da `403`. Hay
  que hacerlo con la sesión iniciada en el panel. Así una clave filtrada no
  puede dejarse sustitutas. (Suavizado en la 0.110.2: con caducidad de hasta
  90 días, sí.)
- Cambiar la contraseña (la tuya o, como admin, la de otro) cierra todas las
  demás sesiones de ese usuario. La sesión desde la que la cambias sigue
  abierta. Las claves API no se revocan: son de integraciones, revócalas una
  a una si hace falta.
- La cookie de sesión lleva `Secure` cuando entras por HTTPS, también detrás
  del proxy. Por HTTP en la LAN sigue funcionando como antes.

---

## 0.108.0 — 2026-11-18

### Cambiado

- **Acción** — Las URLs `http(s)` en partes `image_url`, `video_url` y
  `audio_url` de `/v1/chat/completions` (y de los Batches que lo usan) ya no
  pueden apuntar a direcciones privadas: localhost, la LAN, la red de Docker o
  la de metadatos. Antes el backend descargaba lo que le pidieras desde dentro
  de la red del servidor. Ahora esa petición da `400` con
  `code: "media_url_not_allowed"`. Las URLs públicas y las `data:` funcionan
  igual. Si tus imágenes están en un servidor interno, mándalas como `data:`
  o pide al administrador que añada el host a `MEDIA_URL_ALLOWED_HOSTS`.
- **Acción** — Las opciones de modelo que ejecutan código (`trust_remote_code`,
  `extra_args`, `tool_parser_plugin`, y `python_bin`/`serve_bin`/`serve_module`
  de TensorRT-LLM) solo las puede activar un `system_admin`. Con el rol
  `model_manager`, una petición a `/ocabra/*` que las active da `403` y dice
  cuáles son. Desactivarlas o editar otros campos de un modelo que ya las
  tiene no cambia.
- **Acción** — El token de servicio (`X-Gateway-Token`) ya no permite actuar
  como otro usuario con `X-Internal-User-Id`: se ignora la cabecera y la
  petición sigue con la identidad del servicio. Solo el procesador de Batches,
  dentro del propio servidor, puede hacerlo.

### Corregido

- Un Batch cuyo dueño se había borrado ya no se ejecuta con permisos de
  servicio, que eran más amplios que los suyos: ahora sus peticiones dan `403`.

---

## 0.107.2 — 2026-11-18

### Corregido

- La vista previa podía decir «cabe» para una carga que luego fallaba por
  memoria. Si otro modelo se cargaba a la vez (un embedding que vuelve con una
  petición), su proceso podía atribuirse al que se medía, y la calibración
  aprendía que ocupaba 1 GB menos. Ahora solo se asocia un proceso cuya VRAM
  se parece a la de esa carga, y una medida que se aleja más de un 15 % (o
  400 MB) de la proyección se descarta. El ajuste queda limitado a ±512 MB.
- El sondeo en modo «contexto objetivo» (`target_context`) ya no exige
  `LLAMA_CPP_LIVE_CONTEXT_ENABLED`: carga cada candidato a ese contexto y no
  necesita ampliarlo en caliente.

---

## 0.107.1 — 2026-11-18

### Corregido

- La imagen `ocabra-node` y los backends nativos compilados en el nodo
  (`llama_cpp`, `llamacpp_mks`) ya no fallan al compilar: llama.cpp intentaba
  descargar su interfaz web de una versión que no existe. oCabra no la usa, así
  que ya no se descarga.

---

## 0.107.0 — 2026-11-18

### Nuevo

- Sondeo de contexto en modo «contexto objetivo, lo más rápido» para modelos
  repartidos entre GPU y RAM: `POST /ocabra/profiles/{id}/runtime-presets/{key}/context-probe`
  con `target_context` busca las menos capas en RAM (expertos MoE, o capas
  enteras en modelos densos) con las que cabe ese contexto, cargando de verdad
  cada candidato. El resultado (`mode: "offload"`, `in_ram`, `placement`,
  `used_mb`, `ram_need_mb`) se aplica con `.../context-probe/apply`, que deja el
  peldaño con ese contexto y ese reparto.
- `POST /ocabra/models/{id}/preview` devuelve `host` para modelos llama.cpp: la
  RAM total y libre del nodo, la que el borrador deja residente (`needMb`: lo
  que va a CPU; con `mlock`, el fichero entero) y `fits`; y
  `estimate.fits_host_ram`.

### Corregido

- La memoria proyectada de llama.cpp se ajusta con lo medido en cargas
  anteriores del mismo modelo (parte `calibration` del desglose): en una GTX
  1660, qwen3:30b quedaba ~150 MB por debajo de su proyección en todas las
  configuraciones y se rechazaban cargas que cabían.
- En nodos cuyo driver da PID del host dentro del contenedor, oCabra no veía
  la VRAM real de sus propios modelos (contaba la estimación). Ahora asocia el
  proceso nuevo de cada carga a su worker.
- El máximo de contexto por petición de la vista previa contaba 1 petición
  simultánea en los modelos `llamacpp_mks` (leía mal su configuración).
- El alta automática avisaba de la RAM contra el 85 % de la total; ahora
  contra la libre y teniendo en cuenta `mlock`.

---

## 0.106.4 — 2026-11-17

### Correcciones

- En una instalación nueva no se podía instalar ningún backend. La api corre
  sin root desde la 0.106.2, y el volumen de backends se creaba con dueño
  root. Además, `setup.sh` no arrancaba el agente de Docker, del que
  dependen los servicios, Ollama y las actualizaciones.


---

## 0.106.2 — 2026-11-17

### Correcciones

- Tras cada reinicio del servidor, el modelo de Faster-Whisper volvía a
  descargarse (1,6 GB) en una caché que no se conservaba, y la transcripción
  tardaba más en estar disponible. Ahora se queda en la caché persistente.


---

## 0.106.1 — 2026-11-17

### Correcciones

- Los motores de TensorRT-LLM servidos en modo `docker` usan ahora la GPU
  asignada al modelo. Antes el contenedor arrancaba con `--gpus all` y el
  motor se ejecutaba en la primera GPU, aunque oCabra le hubiera asignado
  otra.
- Cancelar una compilación de TensorRT-LLM, o descargar un motor servido en
  modo `docker`, para ahora el contenedor. Antes podía quedar en marcha,
  compilando o con su puerto ocupado.


---

## 0.106.0 — 2026-11-17

### Cambios de comportamiento

- **Acción:** `POST /ocabra/trtllm/compile` valida `engine_name`: solo letras,
  dígitos, `.`, `_` y `-`, empezando por letra o dígito. Antes se aceptaba
  cualquier cadena, y un `../` escribía el motor fuera de su carpeta. Un
  nombre no válido responde `422`.
- En los modelos de TensorRT-LLM con `launch_mode: docker`, la imagen, el
  binario de Docker y los montajes salen ahora solo de la configuración del
  servidor (`TENSORRT_LLM_DOCKER_*`). Ya no se pueden fijar por modelo, y el
  editor ya no los ofrece. Ningún modelo los tenía fijados.

### Novedades

- Las operaciones de Docker de oCabra pasan por un servicio aparte,
  `docker-agent`, que es el único que tiene el socket de Docker y solo
  ejecuta las operaciones de oCabra: arrancar y parar servicios, actualizar
  Ollama, la autoactualización y la compilación de TensorRT-LLM. Para quien
  usa la API no cambia nada.


---

## 0.105.2 — 2026-11-17

### Cambios

- Backends de este servidor actualizados a sus últimas versiones y probados
  con peticiones reales:
  - **Whisper:** transformers 5.23, NeMo 3.0, pyannote 4.0.7 y torch 2.15.
    La transcripción y la diarización (`*-diarized`) funcionan igual.
  - **diffusers:** su rama principal de hoy (0.44.dev), transformers 5.23 y
    torch 2.15.
  - **vLLM** 0.34.0, **Piper** con dependencias renovadas y **upscaler** con
    torch 2.15.
  - **llama.cpp** (upstream) y **whisper.cpp:** sus últimos commits.
  - **ACE-Step:** último `main` de upstream.
- Los nodos en la nube que instalan estos backends desde GHCR reciben las
  mismas versiones.

---

## 0.105.1 — 2026-11-17

### Correcciones

- **Cancelar la carga de un modelo de imagen, vídeo o escalado** (diffusers,
  Mage-Flow, FlashVSR, upscaler) dejaba el proceso cargando en la GPU, sin
  que oCabra lo tuviera registrado: hasta 20 GB de VRAM ocupados que no
  liberaba nada. Ahora el proceso se para al cancelar.
- **Un worker colgado** (dejaba de responder a `/health` pero seguía vivo) no
  se paraba antes del reinicio automático. Según el backend, se arrancaba un
  segundo worker al lado del primero o se recibía de vuelta el mismo colgado.
  Ahora se para primero.
- Si Redis fallaba durante la carga o la descarga, el modelo podía quedarse
  atascado en `loading` o `unloading` con el worker vivo, y la siguiente
  petición arrancaba otro. Ya no pasa.
- Una carga que fallaba o se cancelaba después de reservar la VRAM dejaba esa
  reserva puesta: la GPU parecía más llena de lo que estaba hasta que el
  modelo volvía a cargarse en ella o se reiniciaba el servidor.
- Editar un servidor MCP de tipo stdio (por ejemplo, solo cambiarle el
  nombre) borraba sus variables de entorno cifradas. Ahora se conservan; para
  borrarlas hay una casilla explícita.

---

## 0.105.0 — 2026-11-17

### Cambios

- Backends a sus últimas versiones:
  - **vLLM** 0.32.0 → **0.34.0** (torch 2.14). Afecta a los modelos que sirve
    vLLM, hoy el reranker bge.
  - Fuera de este servidor (no instalados aquí, sin probar en ejecución):
    **SGLang** 0.5.9 → 0.5.23, **Voxtral** (vllm-omni) 0.18 → 0.32 y
    **Mage-Flow** a diffusers 0.43, transformers 5.23 y torch 2.14.

---

## 0.104.0 — 2026-11-17

### Nuevo

- `POST /ocabra/cloud/nodes` (y `.../nodes/{id}/models`) acepta `backends`:
  los backends que se instalan en el nodo al unirse, antes que cualquier
  modelo. Los que necesitan los modelos elegidos se añaden siempre, y
  llama.cpp ya viene en la imagen del nodo. En la precarga salen como
  elementos `kind: "backend"` (`model_id: "backend:<tipo>"`), que acaban en
  `ready` cuando quedan instalados.
- Las recetas ya no son solo de llama.cpp: `POST /ocabra/cloud/recipes` acepta
  `backend_type` (Whisper, diffusers, vLLM…). Para esos backends se descarga
  el repo entero de Hugging Face. Un tag de Ollama sigue siendo llama.cpp. Un
  repo para llama.cpp sin `.gguf`, o Ollama como backend, responden **422**.

---

## 0.103.0 — 2026-11-17

### Nuevo

- `POST /ocabra/cloud/recipes` añade un modelo para los nodos en la nube desde
  Hugging Face (un fichero `.gguf`) o desde Ollama (un tag), con su contexto,
  slots y política de carga. Se sirve con llama.cpp. Si el modelo ya tenía
  receta, se actualiza. Un repo de HF sin `.gguf` responde **422**.
- `PATCH /ocabra/cloud/recipes/{id}` cambia el contexto, los slots, la política
  de carga o el nombre que tendrá ese modelo en el próximo nodo. Contexto o
  slots a `0` dejan el valor por defecto del nodo.
- Las recetas devuelven también su `load_policy`.

---

## 0.102.0 — 2026-11-17

### Nuevo

- **`GET /ocabra/backends` y `GET /ocabra/backends/{backend_type}` dicen cuántos
  modelos tiene cargados cada backend**, en el campo nuevo `models_loaded`.
  Cada modelo cuenta una vez aunque tenga cargados varios workers de peldaño.

### Cambios

- **`installed_version` ya es una versión.** Antes muchos backends devolvían el
  requisito con el que se instalaron (`"torch>=2.5"`, `"piper-tts>=1.8.0"`) o
  `"source"`. Ahora es la versión instalada de su paquete principal (`"0.32.0"`
  para vLLM; con el nombre delante cuando no es el del backend, como
  `"faster-whisper 1.2.1"`), o `null` si no tiene una (los nativos y los que se
  instalan desde git: su rama o commit siguen en `git_ref`).
- **Ollama** aparece con nombre (`"Ollama"`), descripción y etiquetas propias en
  vez de la descripción genérica de los backends integrados.

**Acción:** si leías `installed_version` esperando un requisito de pip o
`"source"`, usa `packages` o `git_ref`.

---

## 0.101.0 — 2026-11-17

### Nuevo

- **Recetas de modelos para nodos en la nube.** El servidor recuerda los
  modelos que han tenido los pods (también los descargados y configurados a
  mano en su panel): de dónde salen y con qué configuración. Los guarda cada
  10 minutos y justo antes de terminar un pod o cambiarle la imagen.
  `GET /ocabra/cloud/recipes` los lista (los más usados recientemente,
  primero) y `DELETE /ocabra/cloud/recipes/{id}` olvida uno.
- `POST /ocabra/cloud/nodes` y `POST /ocabra/cloud/nodes/{id}/models` aceptan
  `recipes` (ids de esas recetas) y `download_only` (ids de modelos o recetas
  que solo se descargan, sin cargarlos al arrancar; se cargan cuando se
  piden). La precarga de esos queda en el estado nuevo `ready`.
- Los nodos dicen con qué modo de privacidad se lanzaron (`privacy_mode`).

### Cambios

- Al cambiar la imagen de un pod (`update-image`), vuelve a tener también los
  modelos que se le pusieron a mano, no solo los de su precarga.

---

## 0.100.0 — 2026-11-17

### Cambios

- `strata` y `tensorrt_llm` se instalan como el resto de backends
  (`POST /ocabra/backends/{tipo}/install`); antes figuraban como integrados.
  Strata instala su Terminal 0.1.59.1 fijado y verificado por SHA-256 (el motor
  y los pesos siguen llegando con la configuración de cada modelo).
  TensorRT-LLM descarga su imagen NGC (`TENSORRT_LLM_DOCKER_IMAGE`) y necesita
  Docker, así que no se puede instalar en la imagen `ocabra-node`. **Acción**
  para un nodo que ya los usaba: instalarlos una vez desde Ajustes → Backends;
  hasta entonces aparecen como no instalados y sus modelos no cargan.

---

## 0.99.0 — 2026-11-17

### Nuevo

- En un nodo en modo **0 data**, `GET /ocabra/stats/recent` ya no sale vacío:
  devuelve las **últimas 10 peticiones**, guardadas solo en memoria y sin
  quién las hizo (usuario, clave, grupo, IP, user agent) ni el texto del error.
  Trae además un campo `zeroData` con su resumen: cuántas, errores, duración
  media, tokens y velocidad de generación. Se pierden al reiniciar y al
  cambiar el modo de privacidad. `GET /ocabra/stats/my-recent` devuelve la
  lista vacía con ese resumen, porque en 0 data no se sabe de quién es cada
  petición.

---

## 0.98.6 — 2026-11-17

### Errores corregidos

- Una petición a un router cuyo destino elegido estaba en otro nodo (por
  ejemplo, `recomendado-sota` con el modelo de un pod en primer lugar) se
  apuntaba al primer destino de **este** nodo: en las estadísticas salía como
  ese modelo y sin el router, y en la admisión contaba contra su límite de
  peticiones simultáneas, así que podía dar **429** a quien usaba ese modelo
  directamente. Ahora cuenta como el modelo que la sirvió y con su router.

---

## 0.98.5 — 2026-11-17

### Errores corregidos

- Desde la 0.97.0, cada nodo anunciaba **0 modelos** a la federación (o solo
  parte de ellos): las capacidades que se añadieron al heartbeat traían también
  números, como el contexto, y la validación tumbaba la lista entera. Los
  vecinos no veían sus modelos y no les mandaban peticiones. Cada nodo tiene
  que estar en 0.98.5 para que sus vecinos vuelvan a ver sus modelos.

---

## 0.98.4 — 2026-11-17

### Errores corregidos

- Un modelo que solo tiene otro nodo de la federación (un pod) salía en
  `GET /v1/models` con su nombre público, pero pedirlo por ese nombre daba
  **404** `model_not_found`: solo funcionaba con el id completo
  (`llamacpp_mks/...`). Ahora valen los dos.

---

## 0.98.2 — 2026-11-17

### Corregido

- Actualización automática: cuando fallaba, `GET /ocabra/system/update` se
  quedaba en `phase: updating` para siempre y el registro (`updater.log`) no
  traía el error de `docker compose`, que solo iba a stderr. Ahora pasa a
  `failed` con el motivo, el registro incluye stderr, y con la política `auto`
  una versión que falló no se reintenta sola (sí una más nueva o el botón).
- La actualización fallaba siempre con «pull falló»: el actualizador usaba el
  `docker compose` (v2.31) de la imagen en marcha, que no entiende `gpus: all`
  del `docker-compose.yml` actual. Ahora descarga primero la imagen de la
  versión nueva y la actualización corre con ella (y la imagen trae compose
  v5.7.1). Los servicios opcionales que la API arranca con `docker compose`
  (Unsloth, TRELLIS…) fallaban por lo mismo. **Acción** para los nodos en
  0.98.0 o 0.98.1: actualizar una vez a mano a 0.98.2 (cambiar las dos líneas
  `OCABRA_*_IMAGE` del `.env`, `git pull`, `docker compose pull api frontend`,
  `docker compose up -d api frontend`); desde ahí se actualizan solos.

---

## 0.98.0 — 2026-11-17

### Nuevo

- `POST /ocabra/cloud/nodes/{id}/update-image` (`{"version": "X.Y.Z"}` o vacío
  para la imagen configurada): reinicia un nodo de RunPod en marcha con otra
  imagen de `ocabra-node`, en la misma máquina y GPU. Se borra el disco del
  contenedor, así que el nodo vuelve a unirse como un peer nuevo con una
  invitación de un solo uso nueva; el resto de su entorno (privacidad, token de
  Hugging Face) se conserva, los modelos de su disco de red también y los que
  tenía en precarga se vuelven a encolar.

---

## 0.97.0 — 2026-11-17

### Nuevo

- Los nodos de una federación suman sus modelos: `GET /v1/models` y
  `GET /api/tags` listan también los modelos que un vecino ofrece **en frío**,
  no solo los que tiene cargados. Pedir uno funciona: el vecino lo carga.
- Un router (`recomendado-*`) puede tener como destino un modelo que solo
  existe en otro nodo, por ejemplo un pod con un modelo grande. Mientras ese
  nodo no está en línea, el router lo salta. Mientras lo está, sigue la regla
  de siempre: gana el primer destino de la lista que esté caliente.
- Estrategia de reparto por modelo, que configura quien administra el nodo:
  `auto` (lo de antes), `local_only` (nunca sale de este nodo) o `prefer` (una
  lista de nodos en orden; el primero en línea que lo tenga lo sirve, aunque
  esté en frío). Tu cliente no cambia nada: la respuesta es la misma, solo
  cambia dónde se calcula.

### Cambios

- Con `prefer` sin caída a otros nodos, si ninguno de los preferidos puede
  servir el modelo, la respuesta es **503** con `code:
  "preferred_nodes_unavailable"` y `Retry-After`. Reintenta más tarde.
- **Acción** (solo API interna): los peldaños (`/ocabra/profiles/*/presets`)
  ya no tienen `execution_policy`; lo sustituye la estrategia del modelo
  (`GET`/`PUT /ocabra/federation/strategies`). Un cliente que lo siga mandando
  recibe un 422.

### Errores corregidos

- Un chat o completion **en streaming** a un router que acababa en otro nodo
  le mandaba el nombre del router en vez del modelo elegido. Un nodo sin ese
  router (un pod) respondía 404 y la petición acababa sirviéndose aquí.
- `GET /v1/models` y `GET /api/tags` listaban los modelos que solo tienen los
  vecinos sin mirar los grupos de tu clave. Ahora solo ves los que tu grupo
  puede usar, como con los modelos de este nodo.

---

## 0.96.0 — 2026-11-17

### Nuevo

- Releases publicadas: cada versión es una etiqueta `vX.Y.Z` y se publican las
  imágenes `ghcr.io/successbyfailure/ocabra-api`, `ocabra-frontend` y
  `ocabra-node` con ese número (y `latest`), con sus notas como release de
  GitHub. Los backends nativos (llama.cpp, el fork, whisper.cpp) se publican
  desde CI cuando cambia su receta.
- oCabra se actualiza solo: `GET /ocabra/system/update` (versión, última
  publicada, política y estado), `POST /ocabra/system/update/check`,
  `POST /ocabra/system/update/apply` y `PATCH /ocabra/system/update/policy`
  (`off`, `notify` —por defecto— o `auto`, que instala cuando el nodo lleva
  `idleMinutes` sin peticiones, batches ni descargas). Un nodo con
  `OCABRA_API_IMAGE`/`OCABRA_FRONTEND_IMAGE` apuntando a GHCR avanza su copia
  de git hasta la etiqueta, descarga las imágenes, recrea api y frontend y, si
  la API nueva no arranca con esa versión, vuelve sola a la anterior. Durante
  la actualización la API está caída alrededor de un minuto.

---

## 0.95.6 — 2026-11-17

### Corregido

- En nodos donde oCabra corre en un contenedor (los PID de la GPU son los del
  host), los modelos de oCabra cargados contaban como memoria ajena que no se
  puede desalojar: `capacity[].budgetMb` de la vista previa y el alta
  automática (`POST /ocabra/models/{id}/auto-setup`) se quedaban en ~150 MB y
  no proponían ninguna configuración. Ahora se cuenta con la misma regla que
  usa la carga para desalojar.
- Apagar la decodificación especulativa (`speculative: null`) en un modelo
  instalado desde Ollama no quitaba el draft MTP del tag: se volvía a poner en
  cada carga.

---

## 0.95.5 — 2026-11-16

### Corregido

- `POST /ocabra/models/{id}/preview`: `estimate.estimated_max_context_length` y
  `estimated_max_total_context` se calculaban contra la memoria libre de la
  tarjeta (con el escritorio y el margen del planificador dentro) y salían
  hasta 4 veces más altos de lo que cabe: en una GTX 1660, 80k de contexto
  para un modelo que no cargaba con 12k. Ahora salen del mismo hueco que
  `capacity[].budgetMb` y nunca superan la cifra anterior.

---

## 0.95.4 — 2026-11-16

### Cambios

- **En un nodo `zero_data` los ficheros temporales van a memoria**
  (`/dev/shm`): las subidas de más de 1 MB, y el audio y el vídeo que procesan
  los workers, ya no tocan el disco del nodo. Si `/dev/shm` se queda corto,
  una subida grande falla en vez de escribirse a disco. `GET /ocabra/config`
  informa de dónde van y cuánto sitio queda (`privacyTempDir`).

---

## 0.95.3 — 2026-11-16

### Corregido

- El editor de modelos decía «cabe desalojando otros» para cargas que luego
  fallaban por falta de VRAM: contaba los programas del escritorio de la GPU
  como modelos que se pueden desalojar y no descontaba el margen que guarda el
  planificador. `POST /ocabra/models/{id}/preview` devuelve ahora `capacity`
  (por GPU: `availableMb` ahora, `budgetMb` desalojando los demás modelos de
  oCabra y `systemMb`, con la misma cuenta que la carga) y
  `estimate.fits_after_eviction`; `estimate.fits_current_gpu` usa esa misma
  cuenta.

---

## 0.95.2 — 2026-11-16

### Cambios

- Los discos de modelos de los nodos en la nube guardan solo los modelos (y
  la caché de Hugging Face y de Ollama), ya no los backends: en un disco de
  red, los miles de ficheros pequeños de un backend tardaban más de 10 minutos
  en desempaquetarse. Los backends se instalan precompilados en el disco local
  del nodo (~90 s).

---

## 0.95.0 — 2026-11-16

### Novedades

- **Federación «Mi servidor, mis reglas».** Un nodo puede servir a cada
  vecino en modo **enjambre** (todo permitido, como hasta ahora) o **mis
  reglas**: `PATCH /ocabra/federation/peers/{id}` con `mode: "rules"` y, si
  hace falta, `rules` con ajustes para ese nodo. Las reglas por defecto se
  leen y cambian en `GET/PUT /ocabra/federation/rules`:
  - *Yo primero*: las peticiones del vecino ceden el paso a las locales.
  - *Solo lo cargado*: no puede provocar cargas de modelos.
  - *Sin expulsar*: sus cargas no echan un modelo en uso aquí.
  - Límites de peticiones por hora y de Wh al mes, modelos ofrecidos, franja
    horaria, y panel remoto y descargas permitidos o no.
- Una petición que no cumple una regla se rechaza con **503**
  (`hourly_quota`, `energy_quota`, `model_not_warm`, `outside_hours`), **404**
  (`model_not_offered`) o **403** (`read_only`, `downloads_not_allowed`); el
  nodo que la envió la resuelve por su cuenta, como hasta ahora.
- El heartbeat de federación publica a cada nodo las reglas que se le aplican
  (`rules`) y le lista solo los modelos que se le ofrecen.
  `GET /ocabra/federation/peers/{id}/rules-usage` da el consumo de un vecino
  frente a sus límites.

---

## 0.94.1 — 2026-11-16

### Correcciones

- **La estimación de memoria de llama.cpp cuenta bien los modelos repartidos
  en modo `tensor` y los que usan decodificación especulativa MTP.** En la
  0.94.0 un modelo en modo `tensor` (como `qwen3.8:27b`) se estimaba muy por
  debajo de lo real, y a los MTP les faltaba ~1 GB. Contrastado con cargas
  reales: `qwen3.8:27b` proyecta 23,4 GB a 115k y ocupa 23,4; `gemma4:26b` con
  los expertos en RAM proyecta 3,9 GB y ocupa 3,75.

---

## 0.94.0 — 2026-11-16

### Correcciones

- **Los modelos llama.cpp ya no se rechazan por una estimación de memoria
  inflada.** oCabra calculaba la memoria con fórmulas propias y constantes
  calibradas a mano. Con parte del modelo en RAM se pasaba en GPU y se quedaba
  corto en RAM: `gemma4:26b` con sus expertos en CPU se estimaba en 4.371 MB de
  GPU y ocupa 3.752, así que en una tarjeta de 6 GB se negaba a cargar por
  22 MB. Ahora, si el nodo tiene la herramienta de llama.cpp instalada, la
  estimación es la de llama.cpp (la misma cuenta que hace al arrancar).

### Novedades

- **Peldaños automáticos según los dispositivos.** Al descargar un modelo
  llama.cpp se elige dónde correrá (qué GPU y si puede usar la CPU), y tras la
  descarga oCabra crea los peldaños con la proyección de memoria de llama.cpp:
  uno en la GPU grande con todo el contexto que cabe, otro en varias GPU con
  más contexto, o con los expertos MoE justos en RAM cuando no cabe en GPU.
  También se puede recalcular desde el editor del modelo.

---

## 0.93.0 — 2026-11-16

### Novedades

- **Dos modos de privacidad por nodo** (`privacyMode` en
  `PATCH /ocabra/config`): `functional` (el de siempre: lo más privado
  posible sin entorpecer el funcionamiento) y `zero_data`. En
  `zero_data` el nodo no guarda nada de las peticiones ni de quién las hace:
  sin filas en las estadísticas, sin nombres en el feed en vivo, sin contenido
  en los logs y sin trazas externas. Pensado para nodos en máquinas ajenas;
  no protege frente al dueño de la máquina.
- Cada respuesta lleva `X-Ocabra-Privacy: functional` o `zero-data`, y el
  heartbeat de federación publica `privacy_mode` (los peers lo enseñan).
- `POST /ocabra/cloud/nodes` acepta `privacy_mode: "zero_data"`: el pod va a
  Secure Cloud y no recibe el token de Hugging Face.

### Cambios

- **En un nodo `zero_data`, `/v1/batches` y `/v1/files` responden 403
  (`zero_data_mode`)** y las imágenes se devuelven siempre en `b64_json`
  aunque se pida `url`: las dos cosas exigen guardar datos.
- **Acción:** los ficheros de batches (entrada, salida y errores) y las
  subidas a `/v1/files` **se borran** una semana después de terminar su batch
  (o de subirse, si ningún batch los usa); antes no se borraban nunca.
  Configurable con `openaiFilesRetentionHours` (`0` = no borrar). Si recoges
  resultados de batches más tarde de una semana, súbelo o descárgalos antes.

---

## 0.92.0 — 2026-11-16

### Novedades

- **Discos de modelos para los nodos en la nube (RunPod network volumes).**
  `GET/POST /ocabra/cloud/volumes`, `PATCH /ocabra/cloud/volumes/{id}` (solo
  crecer), `DELETE /ocabra/cloud/volumes/{id}` y
  `GET /ocabra/cloud/datacenters` (centros que admiten discos, con las GPUs en
  stock). Un nodo lanzado con `volume_id` guarda allí modelos, caché de
  Hugging Face y backends, y el siguiente arranca con ellos ya descargados.
  Nunca guarda datos ni peticiones. El nodo queda en el centro de datos del
  disco y en Secure Cloud; un disco lo usa un solo nodo a la vez y no se
  puede borrar mientras lo usa. Su precio mensual cuenta en el gasto del mes.

---

## 0.91.2 — 2026-11-16

### Correcciones

- El editor de capacidad mantiene ahora seleccionada la GPU al enviar los
  expertos MoE a la RAM. Guarda explícitamente todas las capas normales en
  GPU, también al administrar nodos antiguos cuyo valor por defecto era
  `gpu_layers=0`; antes la GPU se desmarcaba y no se podía volver a activar.

## 0.91.1 — 2026-11-16

### Correcciones

- La gráfica de potencia reconoce ahora el TDP de 105 W de los nodos con
  Ryzen 7 3800X y 3800XT. En estos AMD el kernel normalmente no publica RAPL,
  por lo que antes el máximo quedaba desconocido y la escala era incorrecta.

---

## 0.91.0 — 2026-11-16

### Despliegue

- oCabra puede instalarse desde Ubuntu en Windows 11/WSL2 mediante Docker
  Desktop. El instalador comprueba CUDA dentro de un contenedor, usa rutas
  escribibles dentro de WSL y desactiva allí los controles de potencia que el
  puente NVML de Windows no soporta.
- Las rutas host de modelos y ACE-Step se configuran ahora con
  `AI_MODELS_ROOT` y `ACESTEP_ROOT`; ya no es necesario editar el Compose.
- El arranque básico incluye el monitor de hardware y deja ComfyUI, A1111,
  TRELLIS.2, ACE-Step y Unsloth bajo la opción `setup.sh --with-services`.

---

## 0.90.1 — 2026-11-16

### Correcciones

- **Peticiones cortadas a mitad (502 "incomplete chunked read") cuando el
  servidor expulsaba un modelo para cargar otro.** Una petición que llegaba
  justo cuando el modelo se iba a descargar se mandaba al proceso que se
  estaba cerrando y moría a los ~1 s. Ahora espera a que termine la descarga
  y el modelo se vuelve a cargar para ella (tarda lo que tarde la carga, pero
  no falla).
- **503 "Failed to load model" por memoria de GPU agotada al recargar un
  modelo justo después de cargar otro.** La admisión de algunos modelos de
  varias GPUs leía una cifra de VRAM libre ligeramente atrasada y daba por
  libre memoria que otro modelo acababa de ocupar; ahora la lee en el momento.

---

## 0.90.0 — 2026-11-16

Versión de seguridad. Cierra accesos que no respetaban los grupos ni los roles.

### Cambios de comportamiento

- **Acción:** `POST /api/pull` y `DELETE /api/delete` (API de Ollama) exigen
  ahora el rol `model_manager`. Antes bastaba cualquier clave, de modo que
  cualquiera podía borrar modelos de todos o llenar el disco con descargas.
  Con una clave de rol `user` responden `403` con `{"error": "..."}`.
- **Acción:** `POST /ocabra/models/{id}/unload` exige `model_manager`.
  Descargar un modelo corta las peticiones de todos sus usuarios.
- `POST /ocabra/models/{id}/load` solo carga modelos concedidos al usuario a
  través de sus grupos. Uno fuera de sus grupos responde `404`, igual que uno
  que no existe.
- `/v1/realtime` aplica los grupos como el resto de endpoints, tanto al
  conectar como al cambiar de STT o de TTS con `session.update`. Un modelo no
  concedido al conectar da el error `model_not_found`. En `session.update` se
  ignora, igual que un modelo desconocido. Las voces de OpenAI (`echo`,
  `alloy`…) siguen funcionando: su destino lo configura el servidor.
- Con la federación activa, un modelo fuera de tus grupos que esté cargado en
  otro nodo ya no se sirve: responde `404 model_not_found`. Antes la decisión
  de enviarlo a otro nodo se tomaba antes de comprobar los grupos.
- `POST /ocabra/auth/login` responde `429` con `Retry-After` tras 10 intentos
  fallidos de un mismo usuario, o 30 desde una misma IP, en 15 minutos.
- Las contraseñas nuevas, tanto la propia como las que pone un administrador,
  deben tener al menos 8 caracteres y no pueden ser el nombre de usuario ni
  una contraseña por defecto conocida (`ocabra`…). Si no, responden `400`.
- El WebSocket `/ocabra/ws` rechaza una sesión cerrada o un usuario
  desactivado. Antes seguía enviando eventos hasta que caducaba el token.
- `GET /metrics` solo responde desde redes privadas (`403` desde fuera).

### Correcciones

- El puerto de Ollama (`7869`) se publica ahora solo en `127.0.0.1`. Estaba
  abierto a toda la red sin autenticación, saltándose claves, grupos y
  estadísticas. Si un cliente tuyo usaba Ollama directamente en ese puerto,
  pásalo a la API de oCabra (`/api/*` con tu clave).

---

## 0.89.1 — 2026-11-16

### Correcciones

- **Modelos MoE con los expertos en RAM** (`llama_cpp.n_cpu_moe`) en
  llama.cpp: tras cargar, el modelo figuraba con la VRAM del fichero entero.
  Por ejemplo, unos 18 GB para Qwen3-30B-A3B con `gpu_layers: 99`, que en
  realidad ocupa 2–3 GB en la GPU. Esa cifra era la que veían el
  planificador, las estadísticas y la recarga de los modelos fijados. En una
  tarjeta pequeña el modelo parecía ocupar la GPU entera y un modelo fijado no
  volvía a cargarse. Ahora la cifra tras la carga sale del mismo cálculo que
  antes de cargar: pesos en GPU, KV del contexto configurado y expertos fuera.
- Los expertos que se quedan en RAM se miden ahora en el propio GGUF, en lugar
  de suponer una proporción fija. En Qwen3-30B-A3B la proporción fija dejaba
  ~1,7 GB de más en la GPU.
- `llama_cpp.override_tensor` que manda los expertos a CPU (`exps=CPU`, o solo
  los de las N primeras capas) cuenta ahora como `n_cpu_moe`. Antes el modelo
  se estimaba entero en la GPU. Para otras reglas, fija `vram_estimate_mb` en
  el modelo.

---

## 0.89.0 — 2026-11-16

### Novedades

- **Elegir el nodo de la federación por petición:** cabecera
  `X-Ocabra-Node: <nombre del nodo>` (o `local` para este servidor) en
  cualquier endpoint de inferencia. Es una elección explícita para esa
  petición; si el nodo no está en línea se enruta como siempre. El Playground
  tiene un selector de nodo que la usa.
- `GET /ocabra/stats/federation` da mucho más: por nodo, lo **enviado**
  (peticiones, errores, tokens, p50/p95, modelos y, en nodos en la nube, el
  **coste** en el periodo, por petición y por 1.000 tokens) y lo **recibido**
  de otros nodos; por qué fue cada petición donde fue (`routes`) y los
  respaldos locales (`fallbackCount`, `fallbacksByModel`). `byNode` se
  mantiene y gana `nodeName`.
- `GET /ocabra/stats/recent` y el feed en vivo traen `node` (nodo que la
  sirvió), `inboundFrom` (nodo que nos la mandó) y `route` (por qué).
- `GET /ocabra/models`: `federation` trae `remote` y, en cada nodo, `loaded`.

### Correcciones

- Las peticiones servidas por un nodo que después se borró quedaban en las
  estadísticas solo con su id; ahora se guarda también el nombre (y se ha
  rellenado para los nodos que siguen existiendo).
- Cuando un nodo elegido fallaba y la petición se servía aquí, no quedaba
  rastro; ahora se registra como `fallback_local`.

---

## 0.88.0 — 2026-11-16

### Novedades

- **Backends precompilados.** `POST /ocabra/backends/{tipo}/install` con
  `"method": "oci"` ya funciona (antes respondía 501): descarga el backend
  publicado en el repositorio OCI (`ghcr.io/successbyfailure/ocabra-backends`,
  una etiqueta por backend), con **exactamente las versiones** del servidor
  que lo publicó, sin pip ni compilar. Los nativos (llama.cpp, whisper.cpp) se
  publican compilados para todas las GPUs recientes y CPUs x86-64 con AVX2, y
  llevan su propio runtime de CUDA (basta el driver ≥ 12.8).
- `"method": "auto"` es ahora **el valor por defecto**: precompilado si está
  publicado, desde el código si no (o si la descarga falla). **Acción:** si
  dependías de que sin `method` se instalara desde el código, manda
  `"method": "source"`.
- **Actualizar un backend:** instalarlo encima de uno instalado lo actualiza
  (se descarga al lado y se intercambian las carpetas; si falla, queda el
  anterior). Responde 409 mientras haya modelos de ese backend cargados.
- `GET /ocabra/backends/{tipo}/published`: qué hay publicado (referencia,
  digest, tamaño, versiones de los paquetes clave, arquitecturas de GPU) y
  `update_available`.
- El estado de cada backend (`GET /ocabra/backends`) trae `oci_ref`,
  `oci_digest`, `packages` (versiones instaladas de los paquetes que declara),
  `declared_packages` y `git_ref`, para comparar lo declarado con lo instalado.
- Los nodos en la nube instalan los backends que necesitan con `auto`.

---

## 0.87.3 — 2026-11-16

### Correcciones

- **Editar con `recomendado-imagen` cuando el modelo de generar estaba cargado
  en otro nodo de la federación.** La selección federada de un router no
  miraba qué sabe hacer cada destino: un `POST /v1/images/edits` se mandaba al
  nodo que tenía cargado el modelo de *generar*, y además con el nombre del
  router, que ese nodo no tenía; respondía 404 y la edición acababa cargando
  el modelo de editar aquí (expulsando a otro). Ahora el router solo considera
  destinos capaces de lo que pide el endpoint (generar o editar) y al nodo se
  le manda el modelo elegido, no el router.

---

## 0.87.1 — 2026-11-16

### Correcciones

- `GET /ocabra/cloud/replicable-models` devolvía `size_bytes: null` en todos
  los modelos llama.cpp; ahora trae lo que ocupan en disco (lo mismo que
  `disk_size_bytes` en `GET /ocabra/models`).

---

## 0.87.0 — 2026-11-16

### Novedades

- **Los nodos en la nube también llevan modelos de imagen (diffusers).** Al
  lanzar un nodo, o con `POST /ocabra/cloud/nodes/{id}/models`, ya se pueden
  elegir modelos diffusers cuyos ficheros vienen de Hugging Face (p. ej.
  `diffusers/qwen-image-2.1`, `diffusers/qwen-image-edit-plus`). El nodo
  instala el backend diffusers si no lo tiene, descarga **solo los ficheros
  que usamos aquí** (el pipeline sin los pesos del transformer que no se usan,
  y el transformer GGUF de su repo), lo registra con el mismo id y lo carga.
  Requiere un nodo con la imagen 0.87.0 o posterior; en uno anterior el
  modelo queda en error con el motivo.
- `GET /ocabra/cloud/replicable-models` incluye los modelos diffusers y un
  campo nuevo `backend_type`. Para estos, `size_bytes` es lo que el nodo va a
  descargar.
- Cada elemento de `preload_models` de un nodo puede pasar ahora por
  `installing` (instalando el backend) y lleva `steps`, `step` y `phase` (qué
  repo se está descargando).
- `POST /ocabra/downloads` acepta un modo de ficheros exactos para Hugging
  Face: `allow_patterns`, `local_dir` (relativo al directorio de modelos),
  `revision` y `register: false` para no autorregistrar el modelo al acabar.
- En diffusers, `gguf_transformer_path` puede ser relativo al directorio de
  modelos.

### Correcciones

- Los modelos que se preparan en un nodo en la nube ya no cuentan como
  inactividad: una copia grande podía hacer que el nodo se apagara por
  inactividad a mitad de la descarga.

---

## 0.86.1 — 2026-11-16

### Novedades

- **Llevar modelos a un nodo en la nube que ya está en marcha:**
  `POST /ocabra/cloud/nodes/{id}/models` con `{"models": [...]}`.
- `GET /ocabra/cloud/replicable-models` lista los modelos que un nodo puede
  llevar, con su tamaño, si están cargados aquí y, si no se pueden replicar,
  por qué (p. ej. ficheros que solo existen en este servidor).

### Correcciones

- **`GET /ocabra/models` volvía error 500 para todos los modelos** en cuanto
  uno había cambiado de despliegue por defecto (p. ej. al borrar el despliegue
  de llama.cpp de `qwen3:32b`): su perfil guardaba la capacidad validada del
  despliegue anterior, para recuperarla si se vuelve a él, en un campo que la
  respuesta no admitía.
- **`GET /ocabra/models/storage` ya no falla en un servidor recién
  instalado** ("No such file or directory: '/data/models'"): mide el disco
  donde se crearán los modelos aunque la carpeta aún no exista.

---

## 0.86.0 — 2026-11-16

### Novedades

- **Panel remoto.** Desde la página Federación, "Abrir panel" en un nodo
  muestra en este mismo interfaz el dashboard, los modelos, las estadísticas,
  etc. de ese nodo, y permite gestionarlo; funciona también con nodos sin URL
  pública ni interfaz propio (como los de la nube). Por debajo es el nuevo
  `/ocabra/federation/peers/{id}/proxy/{ruta}`, que reenvía la petición a ese
  nodo por el enlace de la federación.

### Cambios de comportamiento

- **Un nodo "enjambre" deja gestionar más cosas al otro.** Con acceso
  completo, el otro nodo puede llegar a toda la API de gestión salvo usuarios,
  grupos, claves, la propia federación, la nube y los cambios de ajustes (que
  sí puede leer). Antes solo podía cargar, descargar y bajar modelos. Hace
  falta que el nodo remoto tenga esta versión.

---

## 0.85.1 — 2026-11-16

### Correcciones

- **`/v1/realtime` ya no transcribe el ruido como texto inventado.** Un
  ruido corto que activaba la detección de voz se mandaba a Whisper sin el
  filtro de alucinaciones, y volvía como frases inventadas, a menudo en otro
  alfabeto si no se fijaba el idioma. Además el asistente las contestaba.
  Ahora se descartan los segmentos que el propio Whisper considera no-voz, y
  un turno sin nada útil no genera respuesta.
- **Una imagen ya no deja al modelo con el contexto recortado.** Al cargar el
  proyector de visión, el contexto se reduce para hacerle sitio (en
  `ornith-1.5:35b`, de 229k a 143k), y el proyector no se puede descargar. Las
  peticiones de texto que necesitaban más contexto recibían
  `503 context_temporarily_reduced` una y otra vez. Ahora, si el modelo está
  libre, se recarga sin proyector con su contexto completo: la petición
  espera esa recarga en vez de fallar.
- **`nemotron-3.5-lightning:30b` ya no se recarga cada poco.** Arrancaba
  dejando en la 3060 menos sitio del que ocupa whisper más el margen de
  seguridad, y el vigilante de memoria lo sacaba cada minuto o dos. La
  siguiente petición pagaba la recarga. Ahora arranca con un contexto algo
  menor y crece cuando una petición lo pide.

---

## 0.85.0 — 2026-11-16

### Novedades

- **Los nodos en la nube llegan con modelos.** `POST /ocabra/cloud/nodes`
  acepta `models`: identificadores de modelos de este servidor (de llama.cpp)
  que el nodo descarga y carga en cuanto se une, con la misma configuración y
  el mismo identificador, así que las peticiones a esos modelos pueden ir a él.
  Si la reserva de memoria de un modelo no cabe en la GPU del nodo, se ajusta
  (y se reduce su contexto) en vez de fallar. El progreso de cada modelo
  aparece en `preload_models` de `GET /ocabra/cloud/nodes`.

### Correcciones

- **"No hay máquina libre" en RunPod se explica.** Cuando RunPod no tiene una
  máquina disponible para la GPU elegida, el error dice eso y qué probar, con
  código 409, en lugar de un 400 genérico.

---

## 0.84.3 — 2026-11-16

### Correcciones

- **Un nodo en la nube con la GPU averiada se termina solo.** RunPod puede
  dar una máquina cuya GPU se ve (nombre y memoria) pero no sirve para
  calcular; los modelos cargaban "bien" y respondían a velocidad de CPU
  mientras se pagaba la GPU. Cada nodo comprueba ahora que CUDA funciona de
  verdad, lo publica en su heartbeat (`GET /ocabra/federation/heartbeat`,
  campo nuevo `cuda`) y el servidor que lo lanzó lo termina con el motivo
  `gpu_broken`.

---

## 0.84.2 — 2026-11-16

### Correcciones

- **Los nodos sin Ollama pueden instalar modelos de Ollama.** Una descarga con
  `source: "ollama"` en un servidor sin el servicio de Ollama (como los nodos
  en la nube) fallaba; ahora oCabra baja la etiqueta directamente del registro
  de Ollama, comprobando cada fichero, y el modelo funciona igual (con su
  proyector de visión y su capa de borrador si la etiqueta los trae).

---

## 0.84.1 — 2026-11-16

### Correcciones

- **`/v1/realtime` envía la transcripción de lo que dice el usuario.** Con
  un modelo sin entrada de audio nativa (Whisper → texto → LLM), el texto
  solo iba dentro de `conversation.item.created` y nunca llegaba
  `conversation.item.input_audio_transcription.completed`, que es el evento
  que escuchan los clientes de OpenAI para mostrarlo.

---

## 0.84.0 — 2026-11-16

### Novedades

- **`/v1/realtime` entiende el dialecto GA de la Realtime API.** Además del
  `session.update` beta (ajustes de audio en la raíz), acepta el GA:
  `session.type: "realtime"`, `session.audio.input.{format, transcription,
  turn_detection}`, `session.audio.output.{format, voice}` y
  `output_modalities`. Antes esa parte se ignoraba en silencio: la voz, el
  idioma y los parámetros de turno no tenían efecto. Con GA,
  `session.updated` devuelve también la forma GA y los eventos de audio usan
  los nombres GA (`response.output_audio.delta`,
  `response.output_audio_transcript.delta`…).
- **G.711 µ-law y A-law** de entrada y salida (`audio/pcmu`, `audio/pcma`,
  `g711_ulaw`, `g711_alaw`, 8 kHz), para telefonía.
- **La voz se elige con los nombres de OpenAI.** `alloy`, `echo`… usan la voz
  fijada para cada uno en la compatibilidad de TTS (la misma que
  `/v1/audio/speech` con `tts-1`).
- **`create_response` e `interrupt_response`** de `turn_detection` se
  respetan: con `create_response: false` el turno se confirma sin contestar;
  con `interrupt_response: false` hablar no corta la respuesta en curso.

### Correcciones

- **La entrada se trata a la frecuencia que declara el cliente.** Todo el
  audio se leía como 16 kHz: a 24 kHz llegaba a VAD y transcripción a 0,67×
  de velocidad, con los tiempos de `speech_started`/`speech_stopped`
  desplazados. Ahora se remuestrea a 16 kHz y los tiempos son de audio real.
- **Las respuestas habladas ya no suenan un 9 % rápidas y agudas.** El TTS
  produce 22.050 Hz y se enviaba como si fuera 24 kHz.
- **El umbral de `turn_detection` usa la escala de OpenAI** (0–1, 0,5 por
  defecto). Se aplicaba tal cual a un detector de energía, y 0,5 exigía 25
  veces el volumen normal: con `threshold` el VAD de servidor casi nunca
  detectaba la voz.
- Un `transcription.model` que oCabra no tiene (`gpt-4o-transcribe`) se
  ignora en vez de dejar la sesión sin transcripción.

**Acción:** si usas el dialecto beta, nada cambia: `pcm16` sigue siendo
16 kHz. Si mandabas 24 kHz con `pcm16`, declara `input_audio_sample_rate:
24000` o usa el formato GA `{"type": "audio/pcm", "rate": 24000}`.

---

## 0.83.1 — 2026-11-16

### Correcciones

- **Cargar y descargar un modelo en otro nodo de la federación funciona.**
  `POST /ocabra/federation/peers/{id}/models/{model_id}/load` (y `…/unload`)
  respondían 404 con cualquier modelo, porque todos los identificadores llevan
  una `/` (`llamacpp_mks/…`) y la ruta no la admitía.
- **Descarga remota.** `POST /ocabra/federation/peers/{id}/downloads` devuelve
  ya el trabajo de descarga en JSON (antes lo envolvía como un stream de
  eventos que nunca llegaba, y quien reintentaba lanzaba la descarga dos
  veces). Su progreso se consulta con el nuevo
  `GET /ocabra/federation/peers/{id}/downloads/{job_id}`. **Acción:** si leías
  líneas `data:` de ese endpoint, lee el JSON.
- **`GET /ocabra/federation/peers/{id}/gpus` funciona** (preguntaba al otro
  nodo con el método equivocado y siempre fallaba).
- **Un servidor sin la herramienta `docker` arranca.** Antes la API no llegaba
  a arrancar en un sistema sin ella.
- **Los nodos en la nube solo se crean en máquinas con CUDA 12.8 o superior**,
  las únicas en las que su imagen puede usar la GPU.

---

## 0.83.0 — 2026-11-15

### Novedades

- **Nodos temporales en RunPod.** Desde la página Federación (o con
  `POST /ocabra/cloud/nodes`) se alquila una GPU en RunPod que arranca oCabra,
  se une sola a la federación con una invitación de un uso y empieza a atender
  peticiones como un nodo más. No necesita puertos abiertos.
- **Límites de gasto que se cumplen solos:** cada nodo tiene tiempo máximo y
  apagado por inactividad, hay un tope de gasto mensual, y un nodo que no
  consigue unirse en 30 minutos se termina. Si el servidor que lo lanzó
  desaparece, el propio nodo se apaga tras 20 minutos sin conexión. Los pods
  `ocabra-…` que RunPod tenga y oCabra no conozca se muestran para poder
  terminarlos.

---

## 0.82.2 — 2026-11-15

### Correcciones

- **Las solicitudes de unión a la federación muestran la IP real de quien
  las envía**, no la del proxy que hay delante del servidor, y el límite de
  solicitudes por IP ya no se puede esquivar falseando `X-Forwarded-For`.

---

## 0.82.1 — 2026-11-15

### Correcciones

- **`GET /health` devuelve la versión real del servidor.** Un `APP_VERSION`
  olvidado en la configuración de un despliegue pisaba la versión del código,
  así que un nodo podía anunciar una versión vieja (y lo mismo veían los nodos
  federados con él). Ahora esa variable se ignora.
- **Cambiar el nombre del nodo de federación surte efecto al momento.** Se
  guardaba, pero el nodo seguía anunciándose (y mostrándose) con el nombre
  anterior hasta reiniciar la API.

---

## 0.82.0 — 2026-11-15

### Novedades

- **Federación sin URL pública en uno de los lados.** Un oCabra puede unirse
  al de otra persona (o a uno en la nube) aunque no tenga puertos abiertos:
  abre él una conexión saliente `wss://…/ocabra/federation/tunnel` y por ella
  viajan las peticiones en los dos sentidos, streaming y subidas de audio
  incluidos. Solo el nodo al que se une necesita URL alcanzable.
- **Unirse se pide.** `POST /ocabra/federation/join` (admin) envía la
  solicitud al otro nodo; allí aparece en
  `GET /ocabra/federation/join-requests` para aceptarla o denegarla, con un
  código de verificación que ambos lados ven. Con una invitación
  (`POST /ocabra/federation/invites`) se entra sin esperar aprobación. Los
  endpoints públicos que usa el nodo que pide
  (`POST /ocabra/federation/join-requests` y su `…/status`) están
  desactivados salvo invitación o `federationAcceptJoinRequests: true`.
- **Modo "Soy parte del enjambre":** federación mutua con control completo.
  Cada lado crea un usuario de servicio `federation:<nodo>` con el que ejecuta
  lo que le pide el otro, así que sus peticiones aparecen en tus estadísticas
  con ese usuario.

### Cambios de comportamiento

- **Una petición que llega desde otro nodo de la federación ya no se reenvía
  a un tercero.** Los nodos marcan lo que se mandan con la cabecera
  `x-ocabra-federation-hop`; una petición con ella se atiende en local o no se
  atiende, para que no pueda rebotar entre nodos cuyos heartbeats no
  coinciden. Si tu cliente la envía (no debería), fuerzas el mismo efecto.
- **El identificador del nodo es estable.** Si `FEDERATION_NODE_ID` está vacío
  se genera uno la primera vez y se guarda; antes cambiaba en cada reinicio.

---

## 0.81.6 — 2026-11-15

### Correcciones

- **Un modelo desalojado ya no vuelve a cargarse solo al instante.** Cuando
  oCabra sacaba un modelo de la GPU para hacer sitio a otro, la vigilancia
  de workers podía confundir ese cierre con una caída y "reiniciar" el
  modelo recién sacado, que a su vez echaba al que acababa de entrar. Desde
  fuera se veía como dos modelos quitándose la tarjeta y peticiones que
  esperaban una recarga tras otra.

---

## 0.81.5 — 2026-11-15

### Correcciones

- **Menos recargas en frío de `flint:40b`, `ravenx`, los `nemotron`,
  `qwen3-coder:30b`, `qwen3:8b` y `qwen3:32b`.** Cuando falta memoria en una
  GPU, oCabra primero encoge el contexto de un modelo cargado en vez de
  sacarlo entero. En los modelos sin visión eso nunca funcionaba: el motor
  arrancaba sin la opción que permite cambiar el contexto en caliente, así
  que se desalojaban enteros y la siguiente petición pagaba la recarga.

---

## 0.81.2 — 2026-11-15

### Correcciones

- **Una imagen que el modelo no puede atender ya no da 500.** Un modelo con
  visión necesita sitio en la GPU para su proyector de imágenes. Cuando la
  tarjeta está llena y encoger el contexto no lo libera, la petición se
  escapaba como `500 Internal Server Error`. Pasaba con
  `qwen3.8:27b-abliterated`, cuyo preset por defecto llena la tarjeta. Ahora
  responde `503` con código `vision_no_room` y un mensaje que lo explica.
- **Por un router, esa imagen va a otro destino con visión.** Si el destino
  elegido no tiene sitio para el proyector, el router reintenta una vez sin
  él (con streaming o sin él) en vez de devolver el error.

---

## 0.81.1 — 2026-11-15

### Correcciones

- **Los modelos con presets anuncian lo que de verdad saben hacer.** Un
  modelo con presets lo sirve siempre el worker de su preset, nunca el modelo
  base, y `/v1/models` lee el base: si nunca se había cargado tal cual, sus
  capacidades salían deducidas del nombre. `ornith-1.5:35b` aparecía sin
  visión aunque responde a imágenes, y por eso `recomendado-sota` no lo usaba
  para ellas. Ahora, al cargar el worker de un preset, sus capacidades pasan
  al modelo y se conservan tras un reinicio.

---

## 0.81.0 — 2026-11-15

### Correcciones

- **Las imágenes a `recomendado-sota` van a un modelo con visión.** El router
  elegía destino sin mirar si la petición llevaba imágenes: si el modelo
  cargado en ese momento no tenía visión (`ornith-1.5:35b`, `flint:40b`, los
  `nemotron`…), respondía 400 aunque `gemma4:26b` o `qwen3.8:27b`, en la
  misma lista, sí podían. Ahora un router solo elige entre los destinos que
  tienen lo que pide la petición: visión con imágenes, herramientas con
  `tools`, audio y vídeo. Vale para `/v1/chat/completions` y para
  `/api/chat` y `/api/generate` de Ollama.
- **`recomendado-sota` anuncia visión en `/v1/models`.** Antes copiaba las
  capacidades de su primer destino, que no la tiene, y un cliente como
  OpenWebUI no ofrecía adjuntar imágenes. Un router anuncia `vision`,
  `tools`, `audio_input` y `video_input` si alguno de sus destinos la tiene.
- **Router sin ningún destino capaz: 400, no 500.** Con streaming, una
  petición a un router que no podía servirla se escapaba como error 500.
  Ahora responde 400 `model_not_capable`, como un modelo concreto sin esa
  capacidad. Si el motivo es otro (todos los destinos deshabilitados), sigue
  siendo 503 `router_no_target`.

---

## 0.80.1 — 2026-11-15

### Correcciones

- **`recomendado-imagen` anuncia en `/v1/models` que también edita.** Un
  router copiaba las capacidades de su primer destino, que solo genera, así
  que publicaba `image_editing: false` aunque `/v1/images/edits` funcionaba.
  Ahora un router tiene `image_generation` o `image_editing` si alguno de sus
  destinos la tiene, porque son las que elige según el endpoint.

---

## 0.80.0 — 2026-11-15

### Novedades

- **Router `recomendado-imagen` para generar y editar.** Un solo nombre para
  los dos endpoints: `/v1/images/generations` va a `qwen-image-2.1` y
  `/v1/images/edits` a `qwen-image-edit-plus`. Un router ahora solo elige
  entre los destinos que saben hacer lo que pide el endpoint; antes elegía el
  primero que estuviera cargado, y con modelos que hacen una sola cosa una
  edición podía acabar en el que solo genera.
- **Los nombres de imagen de OpenAI funcionan.** `gpt-image-1`,
  `gpt-image-1-mini`, `dall-e-3` y `dall-e-2` responden con
  `recomendado-imagen`, así que un cliente escrito para OpenAI solo tiene que
  cambiar la URL base y la clave. Sin `model` se trata como `dall-e-2`, igual
  que OpenAI. Se configura en el modo compatibilidad, junto a las voces TTS.
- **`quality` hace algo.** `low` usa la mitad de pasos de inferencia que el
  perfil y `high`/`hd` la vez y media. Antes se ignoraba. Si mandas
  `num_inference_steps`, gana ese.

### Correcciones

- **Las ediciones aceptan `image[]`**, que es como manda las imágenes el SDK
  de OpenAI (y su ejemplo con `curl`). Antes la petición daba 422 por falta de
  `image`. La primera imagen es la base y el resto, referencias.

### Cambios

- **Con `dall-e-*` (o sin `model`) la respuesta es `url` por defecto**, como
  en OpenAI. Con los nombres propios de oCabra y con `gpt-image-*` sigue
  siendo `b64_json`.
  **Acción:** solo si llamabas a oCabra con un nombre `dall-e-*` sin
  `response_format` y leías `b64_json`; pide `response_format: "b64_json"`.

---

## 0.79.0 — 2026-11-14

### Cambios

- **`recomendado-sota` anuncia 262.144 tokens de contexto, no 1.048.576.**
  Un router anunciaba el mayor contexto de todos sus destinos. En
  `recomendado-sota` ese millón lo pone un único destino
  (`nemotron-3-nano:30b`); los otros ocho llegan a 262k como mucho. Un agente
  que compactaba según `context_length` o `recommended_input_tokens` dejaba
  crecer la conversación hasta el millón y, pasados los 262k, el router la
  mandaba siempre al modelo más flojo de la lista sin avisar. Ahora anuncia lo
  que alcanzan la mayoría de sus destinos. El enrutado no cambia: una petición
  más grande sigue funcionando y va al destino donde quepa.
  **Acción:** si tu agente compacta según el contexto anunciado, lo hará antes,
  a 262k. Si de verdad necesitas más de 262k, pide `nemotron-3-nano:30b`
  directamente.
- **Cada router elige qué contexto anuncia** en `/v1/models` y `/api/tags`: el
  mayor de sus destinos (`max`, el de siempre), el que alcanzan la mayoría
  (`common`) o uno fijo (`fixed`, nunca por encima del mayor). Se configura en
  `router_context` del perfil (`PATCH /ocabra/profiles/{id}`) o en la página
  de routers. `ocabra.context.router_policy` dice cuál usa cada uno.

---

## 0.78.6 — 2026-11-13

### Correcciones

- **Tras un reinicio, gemma4:26b ya no se corta unos 20 s con la primera
  petición.** Los modelos fijados (PIN) se autocargaban con su configuración
  base, pero las peticiones de un modelo con presets usan el worker de su
  preset por defecto. La primera petición tenía que cargar ese worker y
  desalojar el del PIN, los mismos pesos dos veces en la tarjeta, y mientras
  tanto esperaba. Ahora la autocarga (y la recarga de un PIN desalojado) carga
  directamente el worker del preset, como ya hacía el botón «Cargar».

---

## 0.78.5 — 2026-11-13

### Correcciones

- **gemma4:26b y gemma4:12b responden antes en conversaciones largas de
  agente.** La plantilla de chat de gemma4 recorría la conversación entera
  por cada mensaje con llamadas a herramientas, así que su coste crecía con
  el cuadrado del número de mensajes: con ~110k tokens de un agente, cada
  turno pasaba 4–6 s antes de empezar a procesar el prompt, aunque casi todo
  saliera de la cache. Ahora se usa una versión corregida que produce
  exactamente el mismo prompt: con 782 mensajes la plantilla pasa de 10,9 s a
  0,9 s.
- **Descargar un modelo con presets (`POST /ocabra/models/{id}/unload`)
  descarga también su worker.** Antes paraba el proceso pero dejaba el
  worker del preset marcado como cargado, y a los pocos segundos se
  reiniciaba solo con la configuración anterior a cualquier cambio.

---

## 0.78.4 — 2026-11-13

### Correcciones

- **Los nombres TTS oficiales de OpenAI respetan permisos como routers
  virtuales.** Conceder acceso a `tts-1`, `tts-1-hd` o
  `gpt-4o-mini-tts` permite usar los perfiles asociados a sus voces sin
  conceder por separado cada perfil interno. Antes, la comprobación se hacía
  después de traducir el alias y una clave autorizada recibía `404` para los
  destinos configurados. Un error de acceso sigue ocultando qué perfil había
  detrás del alias.
- **TTS ya no devuelve un `200` vacío cuando la petición o el worker la
  rechazan.** `/v1/audio/speech` valida que el texto no esté vacío y abre la
  respuesta del worker antes de enviar las cabeceras al cliente. Los errores
  `4xx`/`5xx` conservan ahora su estado en vez de aparecer como un fichero de
  audio vacío con estado exitoso.
- **Los clientes TTS que envían `text` también funcionan.** El nombre OpenAI
  estándar sigue siendo `input`, pero `text` se acepta como alias de
  compatibilidad y se normaliza antes de llamar al modelo.

**Acción:** concede al grupo cliente los nombres OpenAI que deba poder usar;
los perfiles internos no necesitan grants adicionales. Si tu cliente dependía
de interpretar una respuesta vacía como error, comprueba ahora el estado HTTP
y el cuerpo de error.

---

## 0.78.3 — 2026-11-13

### Novedades

- **`GET /ocabra/stats/recent` dice qué tipo de petición está en curso.**
  Las filas con `inFlight: true` traen ahora `requestKind` (`chat`,
  `embedding`, `audio_transcription`…), que antes venía siempre a `null`
  aunque el servidor ya lo sabía al admitirlas. El resto de campos de las
  peticiones en curso no cambia.
- Si calculas la velocidad de prefill a partir de esa respuesta, ten en
  cuenta lo mismo que en la 0.60.1: `promptTokensPerSecond` lo mide el motor
  solo sobre los tokens que no estaban en caché. Con casi todo el prompt en
  caché (p. ej. 1 token nuevo de 3.003) la cifra no representa la velocidad
  real; compárala con `inputTokens - cachedTokens`.

**Acción:** ninguna.

---

## 0.78.2 — 2026-11-13

### Rendimiento

- **Una conversación larga ya no frena a las demás del mismo modelo.** En
  los modelos de llama.cpp, el contexto de una conversación terminada se
  quedaba en la caché que comparten los 4 huecos del modelo, y las otras
  peticiones lo recorrían en cada token. Con 122.000 tokens de una
  conversación ociosa, `qwen3.8:27b` generaba a 72 t/s en vez de 91 (un 25 %
  menos). Ahora esa conversación se guarda en la RAM del servidor y sale de
  la caché compartida; si vuelve, se recupera de RAM en un par de segundos
  en vez de reprocesar el prompt entero. Aplicado a los 13 modelos de chat
  de llama.cpp; cada uno lo coge en su siguiente carga.
- **Prefill algo más rápido en los presets que reparten el modelo entre las
  dos GPU** (`qwen3.8:27b` con `context_preset: "extended"`): un 9 % con
  35.000 tokens y un 6 % con 122.000, cuando el prompt no está en caché. La
  velocidad de generación no cambia.

**Acción:** ninguna.

---

## 0.78.0 — 2026-11-13

### Novedades

- **Compatibilidad de voces OpenAI para clientes que solo saben sus
  nombres oficiales.** Un cliente de TTS que solo sabe mandar los nombres
  de OpenAI (modelo `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`; voz `alloy`,
  `echo`...) no puede apuntar a ningún nombre propio de oCabra. Ahora un
  admin puede fijar, por cada par `(modelo, voz)`, a qué perfil propio
  responde: `PUT /ocabra/openai-tts-voice-map/{openai_model}/{voice_name}`
  con `{"target_profile_id": "..."}` (p. ej. `tts-1` + `alloy` →
  `piper-es-mx-claude-high`). `GET /ocabra/openai-tts-voice-map` lista las
  asignaciones; `DELETE .../{openai_model}/{voice_name}` las quita. Sin
  asignación, esos nombres se resuelven literal — un cliente que ya manda
  un nombre propio de oCabra no nota nada. Independiente de los routers:
  no toca `routing_targets` ni `RouterResolver` en absoluto.

- **`GET /v1/audio/voices` acepta un `voice` opcional para verificar una
  asignación sin sintetizar audio.** `GET /v1/audio/voices?model=tts-1&voice=alloy`
  resuelve exactamente como lo haría `POST /v1/audio/speech` con esa
  misma voz, y devuelve a qué perfil real resolvió. La respuesta también
  trae un campo nuevo, `openai_voice_map`: un objeto `{"alloy": "...",
  ...}` cuando el backend que responda distingue por sí mismo los nombres
  clásicos de OpenAI (familias kokoro, bark y qwen3-tts CustomVoice), o
  `null` cuando los ignora por completo y sirve siempre la misma voz
  (piper, chatterbox, y qwen3-tts Base en modo clon) — en ese caso usa la
  tabla de compatibilidad si necesitas que cada nombre suene distinto.

---

## 0.77.4 — 2026-11-13

### Correcciones

- **El planificador de presets usa la VRAM medida de cada modelo.** Para
  decidir hasta qué contexto cabe un preset, oCabra estimaba la memoria con
  una fórmula y constantes calibradas en otros modelos o configuraciones.
  Apiladas, se pasaban mucho: gemma4:26b corre a 262.144 tokens en la 3090
  con 2 GB libres, y la estimación decía que solo cabía hasta 129.564. Al
  crear un preset por encima de esa cifra, la API lo rechazaba con "Ningún
  preset habilitado cabe". Ahora cada carga y cada cambio de contexto en
  vivo apuntan lo que ocupa el modelo de verdad. Con dos o más medidas de la
  misma configuración, el planificador usa esa curva medida, con margen y sin
  pasar del mayor contexto medido. Nunca la usa para bajar lo que la fórmula
  ya admitía. Afecta a los presets de una GPU con la KV unificada (los de
  llama.cpp en oCabra). En el editor, la huella estimada de esos presets sale
  como `vram_source: "measured"`.

---

## 0.77.2 — 2026-11-13

### Correcciones

- **Una petición a un router ya no falla si su modelo se descarga mientras
  espera turno.** El router elige destino al entrar la petición. Si ese
  modelo se descargaba mientras la petición esperaba en la cola (descarga
  manual, caída del proceso, un servicio que reclama la GPU), la petición
  fallaba con `500` sin streaming o con `502` con streaming. Ahora vuelve a
  elegir destino una vez y se sirve con el modelo que haya cargado. Con
  streaming llega antes de la respuesta un comentario SSE con
  `"event": "ocabra.rerouted"`.
  Las peticiones a un perfil concreto reciben `503` con
  `code: "worker_unloaded"` y `Retry-After`, en vez del 500 o el 502.
- **Batches: menos líneas perdidas cuando el tráfico interactivo les quita
  el modelo.** Cuando una línea se expulsaba para dejar paso a peticiones
  interactivas, el plazo de reintento contaba desde el inicio del lote y
  podía vencer antes de que terminase la pausa de esa expulsión. Las líneas
  acababan en `batch_priority_preempted` sin haber vuelto a intentarlo. Ahora
  cada expulsión deja por delante el plazo completo, con la ventana del lote
  (`completion_window`) como límite. Además, una línea a la que se le va el
  worker a mitad de envío ("All connection attempts failed") se reintenta en
  vez de fallar con `internal_error`.

---

## 0.77.1 — 2026-11-13

### Novedades

- **`GET /ocabra/models` incluye `maintenance`** (booleano) en cada modelo
  local: `true` cuando el modelo está en modo mantenimiento y las peticiones
  nuevas reciben 503. Antes solo se sabía mirando `…/insights` modelo a modelo.

### Corregido

- **Los modelos llama.cpp con `n_cpu_moe` reservaban VRAM de más al cargar.**
  La admisión contaba los expertos MoE que se quedan en RAM como si fueran a la
  GPU, así que un modelo que cabía podía esperar o desalojar a otros sin
  necesidad. Ahora la estimación de carga y `…/preview` descuentan esa parte
  (aproximada, ~85 % de los pesos de cada capa MoE) y la vista previa lo avisa.

---

## 0.77.0 — 2026-11-12

### Novedades

- **Cada preset de contexto puede llevar su propio reparto entre GPU y CPU**
  en el campo nuevo `placement` de `…/runtime-presets` (crear, `PATCH`, listar
  y `…/preview`):
  `{"split_policy", "split_mode", "tensor_split", "gpu_layers", "n_cpu_moe"}`,
  todos opcionales. `split_policy` decide qué pasa si no cabe en una GPU:
  `single` (solo una, lo de siempre), `when_needed` (si no cabe en ninguna, se
  reparte entre todas) o `always` (se reparte siempre). `split_mode`
  (`layer`/`row`/`tensor`) y `tensor_split` (proporciones en el orden de
  `preferred_gpus`) solo se aplican cuando el preset usa varias GPUs;
  `gpu_layers` deja en GPU solo esas capas (null = todas) y `n_cpu_moe` deja en
  RAM los expertos de esas capas MoE. Solo modelos llama.cpp. Los presets que
  ya existían quedan con `placement: null` y cargan exactamente igual.
  `PATCH` con `"placement": null` lo quita.
- **`…/runtime-presets/resolved` y `…/preview` informan de la RAM** que un
  preset deja en CPU en `cpu_mb` (solo cuando hay algo en CPU), y
  `vram_breakdown_by_gpu` cuenta ya solo lo que queda en GPU. Con `n_cpu_moe`
  la cifra es aproximada y el preset lleva el aviso
  "estimación aproximada: expertos MoE en CPU".
- **`POST /ocabra/models/{model_id}/smoke-test`** (rol `model_manager`): carga
  el modelo con su configuración actual (o su preset por defecto, como
  *Cargar*) y le manda una petición mínima según lo que sabe hacer —chat,
  embeddings, rerank, transcripción de 1 s de silencio o voz—. Body
  `{"cold": false}`; con `true` lo descarga antes para medir la carga en frío.
  Devuelve `{"loaded", "load_ms", "request_kind", "request_ok", "request_ms",
  "tokens_per_second", "error"}` y los fallos van en `error`, nunca como 500.
- **Opción nueva `llama_cpp.split_policy` en la configuración base** de los
  modelos llama.cpp (los que cargan sin preset), con los mismos tres valores.
  Si no la pones, se comporta como hasta ahora: con `split_mode: "none"`
  nunca se reparte y, si no, solo cuando no cabe en una GPU.

### Errores corregidos

- **Un modelo llama.cpp con `disabled_gpus` podía acabar asignado a la GPU que
  excluye** cuando la otra estaba llena, y arrancaba sin ninguna GPU visible.
  Ahora solo se eligen las GPUs que permite, y para hacer sitio se descarga en
  ellas.
- **Un modelo con `split_mode: "none"` que no cabía en ninguna GPU se
  repartía igualmente** y llama.cpp, con `--split-mode none`, usaba solo una:
  fallaba al cargar por memoria. Ahora se rechaza con el error de VRAM
  insuficiente de siempre; si quieres que se reparta, pon `split_policy`.

---

## 0.76.1 — 2026-11-12

### Novedades

- **`…/runtime-presets/resolved` y `…/runtime-presets/preview` desglosan la
  VRAM de cada preset por GPU** en `vram_breakdown_by_gpu` (pesos, KV,
  estado recurrente, fijos y buffers), además del total por GPU que ya daban.

---

## 0.76.0 — 2026-11-12

### Cambios

- **Todo modelo de lenguaje tiene al menos un preset de contexto.** A los que
  no tenían ninguno se les crea `normal` (por defecto), copia exacta de lo que
  ya cargaban: mismo contexto, peticiones simultáneas, precisión de la KV y
  GPUs. No cambia cómo responden, pero ahora `normal` aparece en
  `…/runtime-presets` y se puede pedir con `ocabra.context_preset`. Si el
  planificador calcula que ese preset no cabe, no se crea y el modelo sigue
  como estaba. Embeddings, rerank y Ollama no tienen presets.
- **No se puede borrar el último preset de un modelo de lenguaje**
  (`DELETE /ocabra/profiles/{id}/runtime-presets/{key}` responde `400`): se
  edita en su lugar.

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
