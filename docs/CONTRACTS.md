# Contratos de interfaces entre módulos

Este documento resume las interfaces entre módulos que conviene conocer antes
de integrar algo nuevo. **El código es la fuente de verdad**: cada sección
indica el fichero que define el contrato, y donde el código y este documento
discrepen, manda el código (y hay que corregir el documento). Las firmas no se
copian enteras a propósito: se quedaban viejas en cuanto alguien añadía un
campo.

Última verificación contra el código: **2026-11-19**.

Para la lista de endpoints HTTP, ver `docs/API.md` y el Swagger en `/docs`.

---

## 1. BackendInterface — Contrato de backends de inferencia

Fuente: `backend/ocabra/backends/base.py`. Registro de backends:
`backend/ocabra/main.py` (`worker_pool.register_backend(...)`) y, para los que
se instalan por separado, `backend/ocabra/core/backend_installer.py`.

Backends registrados hoy (prefijo del `model_id` = `backend_type`):
`acestep`, `bitnet`, `chatterbox`, `diffusers`, `flashvsr`, `llama_cpp`,
`llamacpp_mks`, `mage`, `ollama`, `piper`, `sglang`, `strata`,
`tensorrt_llm` (solo si su runtime está disponible; si no, se registra como
deshabilitado), `tts`, `upscaler`, `vibeasr`, `vllm`, `voxtral`, `whisper`,
`whisper_cpp`.

Todo backend hereda de `BackendInterface` e implementa estos métodos
abstractos:

| Método | Contrato |
|--------|----------|
| `load(model_id, gpu_indices, **kwargs) -> WorkerInfo` | Arranca el proceso del worker y devuelve su `WorkerInfo`. |
| `unload(model_id)` | Para el worker y libera recursos. |
| `health_check(model_id) -> bool` | `True` si el worker acepta peticiones. |
| `get_capabilities(model_id) -> BackendCapabilities` | Capacidades detectadas del modelo. |
| `get_vram_estimate_mb(model_id, extra_config=None) -> int` | VRAM estimada antes de cargar. `extra_config` es el `ModelState.extra_config` cuando se tiene; `None` debe funcionar igualmente. |
| `forward_request(model_id, path, body) -> Any` | Reenvía una petición al worker. |
| `forward_stream(model_id, path, body) -> AsyncIterator[bytes]` | Reenvía una petición streaming. |

Ganchos opcionales con implementación por defecto:

- `install_spec -> BackendInstallSpec | None` (propiedad): cómo instalar el
  backend con el instalador modular (pip, apt, git, binarios extra, tamaño
  estimado…). `None` = viene en la imagen o es un servicio externo (Ollama).
- `prepare_request(model_id, path, body)`: transición previa a una petición
  (p. ej. activar el proyector de visión de llama.cpp bajo demanda).
- `request_headers(model_id) -> dict[str, str]`: cabeceras privadas que
  `WorkerPool` añade al hablar con el worker (Strata, ver 1.1).
- `attach_port_allocator(assign, release)` / `_acquire_aux_port()` /
  `_release_aux_port()`: puertos para procesos auxiliares que arranca el propio
  backend (sondas, sidecars), sacados del mismo rango que los workers
  (`WORKER_PORT_RANGE_START`–`WORKER_PORT_RANGE_END`).
- `supported_modalities()` y los métodos multimodales (`generate_text`,
  `stream_text`, `generate_embeddings`, `generate_image`, `transcribe`,
  `synthesize_speech`, `rerank`, `upscale_video`, `upscale_image`), que por defecto lanzan
  `NotImplementedError`.

`BackendCapabilities` es un dataclass de flags booleanos (`chat`,
`completion`, `tools`, `vision`, `embeddings`, `pooling`, `rerank`,
`classification`, `score`, `reasoning`, `image_generation`, `image_editing`,
`audio_transcription`, `tts`, `music_generation`, `audio_input`,
`video_input`, `audio_output`, `video_upscaling`, `image_upscaling`, `streaming`) más campos
numéricos donde `0` significa "desconocido": `context_length` (ventana
efectiva que servirá el backend), `native_context_length`,
`max_output_tokens`, `parallel_slots`, `context_per_request`, y el booleano
`kv_cache_shared` (un único pool KV para todos los slots). Se serializa con
`to_dict()` / `from_dict()`; consulta `base.py` para el significado exacto de
cada campo.

`WorkerInfo`: `backend_type`, `model_id`, `gpu_indices`, `port`, `pid`,
`vram_used_mb` y `endpoint_url` opcional (worker que no escucha en
`127.0.0.1:port`).

No todo backend necesita un worker Python propio: cuando el proceso nativo
ya expone HTTP (`whisper-server`, `llama-server`), `load()` lo lanza
directamente como el worker y `forward_request`/`forward_stream` apuntan a
su puerto — ver `whispercpp_backend.py` (usa `--inference-path /transcribe`
para que el binario sirva exactamente el path que `audio.py` golpea) frente
a `whisper_backend.py` (sí necesita `workers/whisper_worker.py` porque
envuelve faster-whisper, una librería Python, no un servidor HTTP nativo).

### 1.1 Backend Strata

Fuente: `backend/ocabra/backends/strata_backend.py`.

Un despliegue `strata` sirve un run config creado por Strata Terminal. Su
artefacto es `{"kind":"strata_config","source":"local","path":"/ruta/strata-*.json","ref":"modelo"}`;
`materialize()` deriva `extra_config.strata.config_path` solo en memoria.

El backend lanza `serve/server.py` en localhost como grupo de procesos,
inyecta una clave interna aleatoria en una copia temporal y exige que
`/health` responda con `loaded=true`. Al descargar termina el grupo completo,
incluido el hijo nativo que posee la VRAM.

La clave no se expone al cliente: `request_headers()` entrega el Bearer
efímero a `WorkerPool`, que lo aplica tanto a JSON como a SSE al conectar con
el worker localhost.

`_runtime.capacity_vram_by_gpu_mb` declara reservas asimétricas validadas.
ModelManager usa ese mapa para admisión y locks, sin dividir el total al 50 %.

### 1.2 Escalado de vídeo (VIDEO_UPSCALING)

Los backends de escalado (`upscaler`, `flashvsr`) implementan
`upscale_video(model_id, video, **kwargs)` y devuelven
`(bytes del segmento, cabeceras x-ocabra-* de estadísticas)`. (La firma por
defecto de `base.py` todavía anota `-> bytes`; los dos backends reales
devuelven la tupla y es lo que espera `api/internal/video.py`.)

Contrato deliberadamente pobre: **entra un segmento autocontenido sin audio y
sale el segmento escalado**, sin estado entre peticiones. Así el planificador
sigue siendo libre de expulsar el modelo entre segmentos según la política del
administrador, en vez de que el trabajo imponga un `pin`.

Quien trocea, remezcla el audio y reensambla es el cliente (VHS), que ya tiene
ffmpeg y NVENC. Expuesto en `POST /ocabra/video/upscale` (multipart: `file`,
`model`, `target_height`, `crf`) — va en la API interna y no en `/v1/*` porque
no existe endpoint OpenAI estándar para esto.

Capacidad: `video_upscaling`. No confundir con `video_input`, que significa
aceptar partes `video_url` dentro de un chat.

### 1.3 Escalado de fotos (`image_upscaling`)

Solo el backend `upscaler` implementa
`upscale_image(model_id, image, *, filename, content_type, **kwargs)`, que
devuelve `(bytes de la imagen, content-type, cabeceras x-ocabra-*)`. Los
`kwargs` (`scale`, `target_width`, `target_height`, `output_format`,
`quality`) van al worker como query de `POST /upscale/image` (multipart
`file`). El worker decodifica con Pillow (orientación EXIF aplicada), pasa la
red por teselas de `UPSCALER_IMAGE_TILE` px (512) con `UPSCALER_IMAGE_TILE_PAD`
(32) de contexto, reintenta con teselas a la mitad si no hay VRAM (hasta 64,
luego 507), reescala el alfa aparte con Lanczos y solo reduce la salida del
modelo, nunca la amplía. Límite de entrada `UPSCALER_IMAGE_MAX_PIXELS`
(3840×2160, 413 si se pasa).

Expuesto en `POST /v1/images/upscale` (`api/openai/image_upscale.py`), con la
resolución de modelos, routers, federación, admisión y estadísticas
(`request_kind=image_upscale`) de las demás rutas de imagen. FlashVSR no tiene
`image_upscaling`: es temporal y solo escala vídeo.

---

## 2. GPUManager — Contrato de estado de GPU

Fuente: `backend/ocabra/core/gpu_manager.py` (`GPUState`, `GPUProcessInfo`).

`GPUState`: `index`, `name`, `total_vram_mb`, `free_vram_mb`,
`used_vram_mb`, `utilization_pct` (0-100), `temperature_c`, `power_draw_w`,
`power_limit_w`, `locked_vram_mb` (VRAM reservada por oCabra: modelos
cargados y reservas previas a una asignación) y `processes:
list[GPUProcessInfo]` (`pid`, `process_name`, `process_type`
`"compute"|"graphics"`, `used_vram_mb`).

El bucle de sondeo (cada 2 s) guarda cada GPU en la clave Redis
`gpu:state:{index}` (TTL 5 s) y, cuando oCabra es la fuente de verdad,
publica la lista completa en el canal `gpu:stats`. Los cambios de power limit
no se aplican desde la api: se publican en `gpu:set_power_limit`
(`{"gpu_index", "limit_w"}`) y los aplica `hw-monitor`, que es el contenedor
con privilegios.

---

## 3. ModelManager — Contrato de estado de modelo

Fuente: `backend/ocabra/core/model_manager.py` (`ModelStatus`, `LoadPolicy`,
`ModelState`).

`ModelStatus`: `discovered`, `configured`, `loading`, `loaded`,
`unloading`, `unloaded`, `error`.

`LoadPolicy`:

- `pin` — precargado al arranque, inmune a la expulsión por inactividad y
  excluido de la expulsión "educada" (`_get_eviction_candidates`). Con
  `pin_permanent=False` (por defecto, *soft pin*) sigue siendo el último
  recurso de la cascada de presión (`_get_pressure_eviction_candidates`, orden
  decidido en `core/vram_arbiter.rank_pressure_candidates`) y se recarga solo
  cuando vuelve a haber VRAM. Con `pin_permanent=True` (*hard pin*) tampoco
  entra en esa cascada: solo se mueve con un unload administrativo o un
  schedule `evict_all`.
- `warm` — se carga bajo demanda, no baja por inactividad, candidato normal de
  expulsión por presión. No se recarga solo tras una expulsión.
- `on_demand` — se carga bajo demanda, baja por inactividad, primera prioridad
  de expulsión por presión.

`ModelState` (campos principales; la lista completa y `to_dict()` están en el
código):

- Identidad: `model_id` (despliegue `<backend>/<model_key>`, p. ej.
  `llamacpp_mks/gemma4:26b`), `model_key` (modelo lógico, tabla `models`;
  `None` en estados solo en memoria), `backend_model_id` (lo que recibe el
  backend: tag de Ollama, repo HF, voz…), `backend_type`, `display_name`,
  `artifact` (qué fichero sirve, ver `core/artifacts.py`; las rutas se derivan
  en memoria y no se persisten).
- Política: `load_policy`, `pin_permanent`, `preferred_gpu`, `extra_config`.
- Estado: `status`, `current_gpu`, `vram_used_mb` (max(medido, reservado)),
  `vram_estimate_mb` (lo que reservó el scheduler), `ram_used_mb` (RSS en
  cargas solo-CPU), `capabilities`, `worker_info`, `error_message`,
  `last_request_at`, `loaded_at`, `created_at`, `updated_at`,
  `reload_suppressed_until`, `live_context` (contexto cambiado en caliente en
  el fork de llama.cpp; no se persiste).
- Vista runtime agregada (para modelos con presets, `model_id` sigue siendo el
  canónico aunque el proceso real use `model_id::<hash>`): `runtime_phase`
  (`configured|loading|loaded|upsizing|downsizing|unloading|unloaded|error`,
  o `None`), `runtime_worker_id`, `runtime_slots` (>= 1), `can_cancel_load`,
  `can_downsize`.

Cada transición se publica en el canal Redis `model:events` con
`{"event", "model_id", "canonical_model_id", "new_status"}` (eventos:
`status_changed`, `load_failed`, `load_cancelled`, `unload_failed`,
`busy_timeout`, `metadata_updated`, `runtime_downsize_complete`) y el estado
se guarda en `model:state:{model_id}`. Un fallo de Redis al publicar se
registra y no rompe la máquina de estados.

La API administrativa agrega cualquier worker derivado sobre su entrada
canónica también durante transiciones, no solo cuando ya está cargado. El
campo legado `status` conserva el ciclo de vida del worker seleccionado;
`runtime_phase` distingue un cambio de capacidad de una carga/descarga
normal.

La identidad derivada se resuelve contra los ids canónicos del catálogo. No se
puede separar por el primer `::`: en llama.cpp el propio id GGUF usa
`repositorio::artefacto` y el worker añade después `::<hash>`. Los eventos
incluyen `canonical_model_id` para que los clientes no tengan que inferirlo.

Las operaciones administrativas son:

- `POST /ocabra/models/{model_id}/cancel-load`: cancela las tareas de carga
  del canónico o de sus derivados y ejecuta la limpieza normal del backend.
- `POST /ocabra/models/{model_id}/downsize`: si no hay peticiones activas,
  descarga el runtime multi-GPU y restaura el preset por defecto de una GPU.
  Devuelve `409` si el runtime está ocupado o no existe una variante que
  pueda reducirse.

---

## 4. WorkerPool — Contrato de registro de workers

Fuente: `backend/ocabra/core/worker_pool.py`.

- Registro: `register_backend(backend_type, backend)`,
  `register_disabled_backend(backend_type, reason)`, `registered_backends()`,
  `await get_backend(backend_type)` (falla si no existe o está deshabilitado).
- Workers: `get_worker(model_id)`, `set_worker(model_id, info)`,
  `remove_worker(model_id)` (libera su puerto), `await assign_port()` /
  `release_port(port)` sobre el rango `WORKER_PORT_RANGE_START`–`END`
  (por defecto 18000–19000).
- Reenvío: `forward_request(model_id, path, body, *, user_key=None, slots=1,
  is_batch=False, kv_capacity_tokens=None, kv_tokens=0, telemetry=None)` y
  `forward_stream(...)` con los mismos argumentos. Con `user_key` la petición
  pasa por la cola justa (`core/fair_slot_gate.FairSlotGate`) del worker; sin
  él va directa. Lanzan `InferenceTimeoutError` cuando el upstream no responde
  (o deja de producir datos) dentro de `INFERENCE_REQUEST_TIMEOUT_SECONDS`; la
  capa API lo traduce a HTTP 504 y cierra la conexión upstream para cancelar
  el trabajo. Otros errores tipados: `WorkerGoneError`,
  `WorkerDegradedError` (circuit breaker), `VisionNoRoomError`.

Las peticiones se contabilizan contra la clave real del worker, derivada de
`profile.base_model_id + profile.load_overrides`
(`core/model_manager_helpers.compute_worker_key`), aunque el cliente use un
`profile_id`. El alias público se conserva por separado para estadísticas.
Esto impide eviction y admisión incoherentes entre alias y modelo canónico.

---

## 5. API Interna oCabra — Contratos REST

Base URL: `/ocabra/`. La lista completa de rutas está en `docs/API.md` y los
esquemas en el Swagger (`/docs`). Aquí solo se recogen las semánticas que no
se deducen del esquema.

Errores internos de `/ocabra/*` siguen el formato estándar de FastAPI
`{"detail": ...}`. Los endpoints de compatibilidad OpenAI y Ollama usan sus
propios envelopes de error.

### 5.1 Modelos

Nota OpenAI `/v1/*` y Ollama `/api/*`: el campo `model` acepta, por este orden, un alias oculto, un `profile_id` (el nombre público de un modelo es el `profile_id` de su perfil por defecto, igual a su `model_key`), un id de despliegue `<backend>/<model_key>` (resuelve al perfil de ese despliegue) y, en la vía legacy (`LEGACY_MODEL_ID_FALLBACK`), un `backend_model_id`/`model_key`; si varios despliegues coinciden, gana el **despliegue por defecto** del modelo.

`GET /ocabra/models` y `GET /ocabra/models/{model_id}` devuelven
`ModelState.to_dict()` (ver §3) más `disk_size_bytes` cuando se puede estimar.

`PATCH /ocabra/models/{model_id}` (`ModelPatch` en
`api/internal/models.py`, `extra="forbid"`: una clave desconocida es 422)
acepta `displayName`, `backendType`, `loadPolicy`, `pinPermanent`,
`preferredGpu`, `vramEstimateMb`, `extraConfig` y `schedules` (también en
snake_case). `extraConfig` se **fusiona en profundidad** con el existente
(`_deep_merge_dicts`): enviar un subárbol no borra el resto.

`DELETE /ocabra/models/{model_id}` elimina la configuración y los ficheros, y
responde `{"ok": true, "deleted_path": str|null}`.

Unload con el `model_id` canónico de un perfil con runtime presets: el worker
realmente cargado vive en una clave derivada (`model_id::<hash>`), nunca en la
canónica; `unload()` lo detecta y descarga todos los derivados cargados.

`POST /ocabra/models/{model_id}/memory-estimate` (body: `preferred_gpu?`,
`extra_config?`, `run_probe?`) evalúa una configuración sin persistir nada:

- `source="heuristic"`: cálculo rápido sin arrancar runtime.
- `source="runtime_probe"`: validación real con el engine (sobre todo vLLM).
- Campos: `backend_type`, `preferred_gpu`, `source`, `estimated_total_mb`,
  `estimated_weights_mb`, `estimated_engine_mb_per_gpu` y `engine_present`
  (TensorRT-LLM), `estimated_kv_cache_mb`, `estimated_max_context_length`,
  `maximum_concurrency`, `status` (`ok|warning|error`, si hay probe),
  `warning`, `error`, `notes`.

### 5.1.1 Modelos lógicos (unificación modelo/backends)

Un **modelo** (`models.model_key`, nombre público) tiene un **despliegue** por backend
(`model_configs`, id `<backend>/<model_key>`, único por `(model_key, backend_type)`) y un
`default_backend`. Invariantes (ver `ocabra/core/model_deployments.py`):

- Los perfiles públicos (`backend IS NULL`) del modelo apuntan al despliegue por defecto.
- Todo despliegue tiene al menos un perfil; los de despliegues no-default están **fijados**
  (`model_profiles.backend`) y son **ocultos**: resuelven, no se listan en `/v1/models` ni `/api/tags`.
- Solo el despliegue por defecto lleva la `load_policy`/`pin` del modelo.
- `runtime_policy.validated_*` describe siempre el backend al que apunta el perfil; al cambiar de
  backend se aparca en `runtime_policy.validated_by_backend[<backend>]`.
- Permisos: `group_models` guarda `model_key` (o ids de router / perfiles sueltos); conceder un
  `model_key` concede sus despliegues, perfiles y alias (`_expand_model_grants`).

Endpoints: `/ocabra/logical-models[/{model_key}[/compatibility|/deployments[/{backend}]]]`
(ver `api/internal/logical_models.py`). `LogicalModel` = `{modelKey,
displayName, category, defaultBackend, deployments[], compatibleBackends[],
publicProfiles, hiddenProfiles, registryMetadata}`; `CompatibleBackend` =
`{backend, status: "compatible"|"probable", installed, reason}`.

Las respuestas de búsqueda del registro (HF, BitNet, Ollama: tarjetas, detalle y variantes)
llevan `compatible_backends: list[CompatibleBackend]` (reglas en `ocabra/core/compatibility.py`).
La descarga acepta en `register_config`: `model_key`, `default_backend` y `backends` (despliegues
extra del mismo fichero).

### 5.2 Descargas

`POST /ocabra/downloads` (`DownloadCreateRequest` en
`api/internal/downloads.py`): `source` (`huggingface|ollama|bitnet`),
`model_ref`, `artifact?`, `register_config?` y, solo para Hugging Face, el
modo de ficheros exactos (`allow_patterns`, `local_dir`, `revision`,
`register`). El trabajo se encola en la lista Redis `queue:download`, su
estado vive en `download:job:{job_id}` y cada cambio se publica en
`download:progress` y `download:progress:{job_id}` con el `DownloadJob`
completo (`schemas/registry.py`: `status`
`queued|downloading|completed|failed|cancelled`, `progress_pct`,
`speed_mb_s`, `eta_seconds`, `error`…). `GET /ocabra/downloads/{job_id}/stream`
es el mismo flujo en SSE.

### 5.3 Stats

Rutas en `api/internal/stats.py`; agregaciones en `stats/aggregator.py`.
Filtros habituales: `from`, `to` (ISO 8601) y `model_id`.

Una respuesta streaming cancelada después de enviar sus cabeceras se registra
como `status_code = 499` y `error = "stream_cancelled"`, aunque el estado HTTP
que alcanzó a ver el cliente fuese inicialmente 200. Los tokens parciales que
ya hayan atravesado el middleware se conservan; si el upstream no emitió ningún
chunk, `output_tokens` permanece `NULL`.

#### Estadísticas de OpenAI Realtime

Las conexiones `GET /v1/realtime` conservan el `UserContext` resuelto durante
el handshake y escriben en la tabla `request_stats`; no se persiste
audio, texto de conversación ni credenciales.

| `request_kind` | Unidad registrada | `model_id` |
|---|---|---|
| `realtime_session` | Una fila por WebSocket abierto, al cerrarse | Modelo solicitado/resuelto |
| `realtime_transcription` | Cada llamada STT, incluidas hipótesis parciales | Modelo Whisper/STT |
| `realtime_chat` | Cada generación LLM iniciada por la sesión | Modelo LLM |
| `realtime_tts` | Cada síntesis de audio | Modelo TTS |

Todas usan `endpoint_path=/v1/realtime`, guardan duración, estado/error,
`user_id`, `group_id`, `api_key_name`, IP y user-agent. Una sesión cerrada
normalmente usa `status_code=101`; las operaciones internas usan `200` en éxito
y un estado 4xx/5xx representativo en error.

### 5.4 Config

`GET /ocabra/config` y `PATCH /ocabra/config` (`api/internal/config.py`,
`ServerConfigPatch`, `extra="forbid"`) usan claves camelCase.

- Cada clave modificada se aplica en caliente a `settings` **y** se persiste
  en la tabla `server_config` (`db/server_config.save_override`). Al arrancar,
  `main.py` aplica esos overrides sobre los valores del `.env` antes de lanzar
  ningún bucle. Consecuencia: un proceso Python suelto (sin pasar por el
  arranque del api) ve los valores del `.env`, no los guardados.
- `modelsDir` es de solo lectura (sale de `MODELS_DIR`); intentar cambiarlo da 400.
- `globalSchedules` se guarda en `eviction_schedules` como dos filas por
  ventana: `evict_all` al inicio y `reload` al final. `check_schedule_reloads()`
  (`core/scheduler.py`) recarga los modelos `pin`/`warm` a los que apunta un
  schedule `reload`.
- `newUserGroupNames`: grupos que se asignan al crear un usuario (por defecto
  `["default", "Makers"]`); los nombres que ya no existan se ignoran y el alta
  nunca crea grupos.

### 5.5 Autorización por grupos y API keys

- La pertenencia a grupos es propiedad del usuario. Todas sus API keys heredan
  exactamente la misma ACL; `api_keys.group_id` es legado y no participa en
  autorización.
- El grupo `default` se aplica implícitamente a cualquier usuario no admin y al
  acceso anónimo cuando está habilitado.
- `groups.grants_all_models=true` concede acceso dinámico a cualquier modelo o
  perfil, incluidos los que se registren después. No depende del nombre del
  grupo ni de filas manuales en `group_models`.
- Autorizar un perfil `router` autoriza su uso indirecto de targets, pero no
  concede acceso directo a esos targets ni los añade por separado a `/v1/models`
  o `/api/tags`.

### 5.6 Servicios interactivos

Fuente: `backend/ocabra/core/service_manager.py` (`ServiceState`) y
`api/internal/services.py`.

Servicios definidos: `trellis2`, `comfyui`, `a1111`, `acestep`, `unsloth`.
`ServiceState.to_dict()` incluye identidad (`service_id`, `service_type`,
`display_name`, `base_url`, `ui_url`, `health_path`, `unload_path`),
política (`preferred_gpu`, `idle_unload_after_seconds`, `idle_action`
`stop|restart`, `generation_grace_period_s`, `enabled`), estado
(`service_alive`, `runtime_loaded`, `status`
`unknown|idle|active|unreachable|disabled`, `active_model_ref`,
`last_*_at`, `detail`, `extra`) y métricas (`is_generating`, `queue_depth`,
`vram_used_mb`, `gpu_util_pct`, `cpu_pct`, `mem_used_mb`, `mem_limit_mb`).
Cada cambio se publica en `service:events`
(`{"event", "service_id", "status", "service"}`) y se guarda en
`service:state:{service_id}`; el enable/disable persiste en
`service:overrides`.

`PATCH /ocabra/services/{service_id}/runtime` (body `runtime_loaded`,
`active_model_ref?`, `detail?`) lo llama el propio servicio para declarar si
tiene pesos en VRAM.

`POST /ocabra/services/{service_id}/ensure_vram` (body `vram_needed_mb`,
`suppress_reload_seconds = 0`) es para servicios que gestionan su propia VRAM
fuera del scheduler (TRELLIS.2): expulsa modelos on-demand/warm/soft-pin
(nunca hard-pin) de la GPU preferida del servicio con la misma cascada que
una carga normal (`ModelManager.ensure_vram_free`) y después **reserva** la
VRAM liberada (`ServiceManager.reserve_gpu_vram` → `GPUManager.lock_vram`), de
modo que ni el scheduler ni `RouterResolver.pick()` mandan otra carga a esa
GPU mientras el servicio genera. `suppress_reload_seconds` impide que un PIN
expulsado se recargue solo durante ese tiempo.

La reserva se libera al llamar a `/unload`, en el siguiente health check con
`runtime_loaded=false`, si el contenedor deja de responder, al pasar de
generando a ocioso (cuando `/runtime/status` expone `in_flight_requests`) o
tras `vram_reservation_max_age_s` (20 min) como red de seguridad. Por eso el
servicio debe pedir `ensure_vram` antes de **cada** generación, no solo en la
carga en frío.

### 5.7 Tiempo real (WebSocket)

`WS /ocabra/ws` (`api/internal/ws.py`). Exige cookie `ocabra_session` válida
(misma comprobación que HTTP) y mismo origen; si no, cierra con 1008.

Cada mensaje de un canal Redis se reenvía como `{"type": ..., "data": ...}`
con `data` igual al payload publicado:

| Canal Redis | `type` | `data` |
|-------------|--------|--------|
| `gpu:stats` | `gpu_stats` | `list[GPUState]` |
| `model:events` | `model_event` | `{event, model_id, canonical_model_id, new_status}` |
| `service:events` | `service_event` | `{event, service_id, status, service}` |
| `download:progress` | `download_progress` | `DownloadJob` |
| `system:alerts` | `system_alert` | `{level, message}` (`level` = `info`/`warning`/`error`) |

Además, el servidor envía `{"type": "ping"}` tras 30 s sin mensajes del
cliente, y un cliente que manda `{"subscribe": "models_activity"}` recibe
cada segundo un frame `models_activity` (mismo payload que
`GET /ocabra/models/activity`) hasta que manda
`{"unsubscribe": "models_activity"}`.

### 5.8 OpenAI Files API

Fuente: `backend/ocabra/api/openai/files.py`.

- `purpose` válidos al subir: `batch`, `batch_output`, `user_data`,
  `batch_input_audio` y `batch_input_image` (estos dos, extensión oCabra, son
  los ficheros a los que apuntan las líneas de lote de
  `/v1/audio/transcriptions` y `/v1/images/edits`, ver 5.9). `batch_error` y
  `batch_artifact` los crea solo el `BatchProcessor`. Tope: 512 MB por fichero.
- `GET /v1/files`: `purpose?`, `limit` (1-10000, por defecto 100).
- `file_id` acepta `file-<uuid>` o el UUID directo.
- `DELETE /v1/files/{file_id}` responde 409 `file_in_use` si un batch activo
  lo usa como entrada.

### 5.9 OpenAI Batches API

Fuente: `backend/ocabra/api/openai/batches.py` (`ALLOWED_ENDPOINTS`) y
`backend/ocabra/core/batch_processor.py`.

Endpoints admitidos en `endpoint`: los de `ALLOWED_ENDPOINTS` (chat,
completions, embeddings, images generations/edits, audio speech/
transcriptions/generate, pooling, score, rerank, classify). Cada línea del
JSONL de entrada: `{"custom_id", "method": "POST", "url", "body"}`.
`GET /v1/batches` pagina con `after` y `limit` (1-100, por defecto 20).

- `/v1/audio/transcriptions` y `/v1/images/edits` (extensión oCabra: los
  endpoints reales son multipart y no caben en una línea JSON): el `body`
  lleva `file_id` (transcripción) o `image_file_id` y opcionalmente
  `mask_file_id`, `image_ref_{1,2,3}_file_id` (edición), que deben apuntar a
  ficheros propios con `purpose` `batch_input_audio` / `batch_input_image`.
  Si no, la creación se rechaza con 400 (`missing_file_id` |
  `invalid_file_id`, con el número de línea). El procesador reconstruye la
  petición multipart al despachar.
- Salida: `output_file_id` solo éxitos, `error_file_id` solo fallos (`null`
  si no hubo). Línea:
  `{"id": "batch_req_<uuid>", "custom_id", "response": {"status_code", "request_id", "body"}, "error"}`.
- Respuestas binarias (`/v1/audio/speech`, `/v1/audio/generate`): el `body`
  no lleva el audio, sino `{"file_id", "bytes", "content_type"}` de un fichero
  `purpose="batch_artifact"` descargable por `GET /v1/files/{id}/content`.
- Un lote cuyo `completion_window` vence pasa a `expired` conservando lo ya
  escrito; las líneas no despachadas van a `error_file_id` con
  `error.code="batch_expired"`.
- Prioridad frente a tráfico interactivo: una petición interactiva puede
  cancelar una línea de lote en curso (ver `batchPreemptMode` en
  `/ocabra/config`); la línea se reintenta dentro del mismo batch y solo cuenta
  como fallo si se agotan los reintentos (`error.code="batch_priority_preempted"`).
- Configuración (`config.py`): `OPENAI_FILES_DIR` (por defecto
  `/data/openai_files`), `BATCH_MAX_CONCURRENCY` (4),
  `BATCH_REQUEST_TIMEOUT_SECONDS` (960; el efectivo se mantiene por encima del
  timeout de inferencia), `BATCH_POLL_INTERVAL_SECONDS` (5) y las de
  prioridad `BATCH_PRIORITY_ENABLED`, `BATCH_PREEMPT_MODE`
  (`never|instant|conditional`), `BATCH_PREEMPT_MIN_ELAPSED_S`,
  `BATCH_PREEMPT_GRACE_S`, `BATCH_FREE_SLOTS_PCT`,
  `BATCH_RESUME_COOLDOWN_SLOTS_S`, `BATCH_RESUME_COOLDOWN_EVICTION_S`,
  `BATCH_INTER_JOB_COOLDOWN_S`.

### 5.10 Benchmarks de aplicaciones

Rutas internas (rol `model_manager`) en `api/internal/benchmarks.py`:
`GET /ocabra/benchmarks/catalog`, `POST/GET /ocabra/benchmarks/runs`, detalle,
log y acciones `pause|resume|cancel|retry|baseline`. `retry` solo admite ejecuciones
fallidas y continúa en el modelo actual reutilizando su checkpoint. Los estados persistidos son
`queued|running|pausing|paused|cancelling|completed|failed|cancelled`.
`phase` detalla el trabajo dentro del estado (`checkout|preparing|running|finalizing`
y las fases terminales o de espera) para no presentar una preparación larga como
si el proceso estuviera bloqueado. El payload de detalle incluye además
`testProgress` cuando el runner publica avance estructurado: `completed`, `total`,
`unit`, `stage` y `updatedAt`. Es un snapshot efímero del intento actual, no una
métrica oficial del resultado.

`BenchmarkManager` ejecuta una sola unidad `app + suite + model` a la vez. Usa
un usuario/API key internos para distinguir sus peticiones en
`ModelManager.active_requests_snapshot()` y `request_stats`. Tráfico de otra
identidad termina el subprocess, marca el intento y pausa la ejecución; tras
`benchmark_quiet_period_s` sin tráfico la reencola. Un intento terminado con
solape se repite desde un checkout limpio, incluso si `auto_pause=false` (en
ese caso se reencola inmediatamente). Un intento interrumpido o fallido
reutiliza su workspace y los argumentos de reanudación del runner.

En un pod propio con varios modelos base, el hub prepara hasta dos en paralelo.
Tras registrar un modelo del run, el primer intento disponible entra en la cola.
El orden de `models` refleja la ejecución efectiva. El cursor conserva el prefijo
procesado y los resultados siguen identificados por perfil. Las suites de una
batería comparten las descargas y el pod.

Cada `benchmark_attempt` conserva `environment`: versión de oCabra, Python,
scorer cuando existe, commit real del dataset, revisión solicitada, perfil,
backend y versión instalada del módulo (o commit exacto de fuentes para
backends nativos y digest para OCI), preset/contexto efectivos, nodo y
todas las GPU físicas que atendieron sus peticiones (índice, modelo y VRAM),
además de recuentos de peticiones, errores y tokens. El JSON descargado desde
la UI incluye el resultado bruto y esta evidencia reproducible de todos los
intentos, no solo la puntuación aceptada.

El catálogo combina suites mantenidas por aplicaciones (Escriba y Corrala)
con benchmarks públicos fijados a revisión (BFCL e IFEval). Estos últimos usan
un checkout parcial y un `local_runner` de oCabra para adaptar la API OpenAI y
normalizar la salida, pero ejecutan el dataset y el scorer de la revisión
oficial. Cada adaptador escribe checkpoint por caso antes de atender una pausa.
El manager les pasa `OCABRA_BENCHMARK_PROGRESS_PATH`; el runner actualiza esa
ruta de forma atómica con un JSON acotado
`{"completed": N, "total": M, "unit": "casos", "stage": "generating|scoring|completed"}`.
IFEval y BFCL implementan este contrato; los runners externos que todavía no lo
hacen se presentan como progreso indeterminado.

Workspaces y logs viven bajo `benchmark_dir` (`/data/benchmarks` por defecto,
volumen persistente). Corrala usa una base separada cuyo nombre contiene
`bench/test/scratch/tmp`; nunca la base de producción.

Los planes reutilizables viven en `benchmark_schedules` y se gestionan con
`GET/POST /ocabra/benchmarks/schedules`, `PUT/DELETE
/ocabra/benchmarks/schedules/{id}` y `POST .../{id}/run`. Guardan una copia de
aplicación, suite, revisión, modelos/backends, argumentos y pausa automática,
además de días de la semana, hora y zona IANA. Al vencer, el worker avanza
`next_run_at` y crea una ejecución con snapshot del plan; un bloqueo de fila y
la comprobación de una ejecución no terminal evitan solapes del mismo plan.
`schedule_id` conserva la procedencia histórica aunque después se edite el
plan. Solo puede existir un `benchmark_run.is_baseline` por `app + suite`; las
comparaciones de la UI calculan deltas contra esa referencia sin modificar el
resultado bruto.

---

## 6. Redis — Keys y canales

Todo pasa por `backend/ocabra/redis_client.py` (`publish`, `set_key`,
`get_key`, `lpush`/`rpop`, `subscribe`; JSON en ambos sentidos). No hay colas
de carga/descarga de modelos en Redis: las cargas las coordina
`ModelManager` en memoria.

Canales pub/sub:

| Canal | Productor | Payload |
|-------|-----------|---------|
| `gpu:stats` | `core/gpu_manager.py` (cada 2 s) | `list[GPUState]` |
| `gpu:set_power_limit` | `gpu_manager` (throttling térmico), `api/internal/gpus.py`, `db/gpu_power_settings.py` | `{gpu_index, limit_w}`; lo consume `hw-monitor` |
| `model:events` | `core/model_manager.py` | ver §3 |
| `service:events` | `core/service_manager.py` | ver §5.6 |
| `download:progress`, `download:progress:{job_id}` | `api/internal/downloads.py` | `DownloadJob` |
| `system:alerts` | `redis_client.publish_system_alert` | `{level, message}` |
| `trtllm:compile:{job_id}` | `core/trtllm_compile_manager.py` | progreso de compilación TensorRT-LLM |

Keys:

| Key | Contenido |
|-----|-----------|
| `gpu:state:{index}` | `GPUState` (TTL 5 s) |
| `model:state:{model_id}` | `ModelState.to_dict()` (sin TTL; `ModelManager` purga los de modelos que ya no existen) |
| `service:state:{service_id}` | `ServiceState.to_dict()` |
| `service:overrides` | enable/disable persistido de servicios |
| `download:job:{job_id}` | `DownloadJob` |
| `queue:download` | lista Redis: cola de descargas `{job_id}` |
| `server:power`, `hw-monitor:heartbeat` | escritas por `hw-monitor` (TTL 5 s y 10 s) |
| `jwt_revoked:{jti}` | lista de revocación de sesiones al cerrar sesión (TTL = vida restante del token). Ojo: a 2026-11-19 la escritura (`api/internal/auth.py`) y la lectura (`api/_deps_auth.py`) hacen `await get_redis()` sobre una función síncrona y fallan en silencio, así que la key no se llega a escribir |
| `auth:login_fail:user:{username}`, `auth:login_fail:ip:{ip}` | contadores de `core/login_guard.py` |
| `ocabra:realtime_session:{session_id}` | sesiones Realtime vivas (`core/session_registry.py`) |
| `llama_cpp:observed_vram:*`, `llama_cpp:vram_curve:*` | VRAM medida de workers llama.cpp (`core/observed_vram.py`) |

---

## 7. Esquema de base de datos

Fuente: los modelos SQLAlchemy de `backend/ocabra/db/` y las migraciones de
`backend/alembic/versions/`. Este índice dice qué tabla vive dónde; las
columnas, en el código.

| Tabla | Fichero | Contenido |
|-------|---------|-----------|
| `models` | `db/model_config.py` | Modelo lógico (`model_key`, `default_backend`) |
| `model_configs` | `db/model_config.py` | Despliegue por backend (`<backend>/<model_key>`), política de carga, `extra_config` |
| `model_profiles` | `db/model_config.py` | Perfiles públicos / ocultos / routers (§8) |
| `model_profile_aliases` | `db/model_config.py` | Alias ocultos de perfiles (y preset fijado) |
| `profile_runtime_presets` | `db/model_config.py` | Presets de runtime por perfil (§8.9) |
| `eviction_schedules` | `db/model_config.py` | Ventanas `evict_all` / `reload` (globales o por modelo) |
| `openai_tts_voice_map`, `openai_image_model_map` | `db/model_config.py` | Nombres OpenAI → perfil propio (§8.10) |
| `users`, `groups`, `user_groups`, `group_models`, `api_keys` | `db/auth.py` | Usuarios, grupos, ACL de modelos, API keys |
| `request_stats`, `model_load_stats`, `gpu_stats`, `server_stats`, `service_generation_stats`, `tool_call_stats` | `db/stats.py` | Estadísticas (sin contenido de prompts ni respuestas) |
| `server_config` | `db/server_config.py` | Overrides de ajustes guardados desde la UI (§5.4) |
| `gpu_power_settings` | `db/gpu_power_settings.py` | Power caps y persistence mode, reaplicados al arrancar |
| `openai_files`, `openai_batches` | `db/openai_batches.py` | Files y Batches API. Desde la migración 0049, `openai_batches.input_file_id` ya no es FK: borrar el fichero de entrada de un batch terminado está permitido |
| `federation_peers`, `federation_invites`, `federation_join_requests` | `db/federation.py` | Federación |
| `cloud_providers`, `cloud_nodes`, `cloud_model_recipes` | `db/cloud.py` | Nodos en la nube y recetas de precarga |
| `agents`, `agent_mcp_servers` | `db/agents.py` | Agentes |
| `mcp_servers` | `db/mcp.py` | Servidores MCP |
| `trtllm_compile_jobs` | `db/trtllm.py` | Compilaciones TensorRT-LLM |
| `benchmark_runs`, `benchmark_attempts`, `benchmark_schedules` | `db/benchmarks.py` | Cola, planes semanales, checkpoints, contaminación, referencias y resultados de benchmarks |

---

## 8. Model Profiles — Contrato de perfiles de modelo

### 8.1 ModelProfile (DB → ProfileRegistry → API)

```python
# backend/ocabra/db/model_config.py

class ModelProfile(Base):
    __tablename__ = "model_profiles"

    profile_id: str           # PK, slug sin /, lowercase, solo guiones y alfanuméricos
    profile_kind: str         # "execution" | "router"
    base_model_id: str | None # requerido en execution (id de despliegue); NULL en router
    backend: str | None       # fija el perfil a un despliegue no-default → perfil oculto (no listado)
    display_name: str
    description: str | None
    category: str             # "llm" | "tts" | "stt" | "image" | "music"
    load_overrides: dict      # JSONB, merge con extra_config del modelo base al cargar
    request_defaults: dict    # JSONB, inyectados como defaults en body de request
    assets: dict              # JSONB, {"voice_ref": "/data/profiles/{profile_id}/reference.wav", ...}
    routing_targets: list | None  # solo routers: profile_id de destino
    runtime_policy: dict | None   # política de presets (8.9; esquema en schemas/profiles.py)
    router_context: dict | None   # solo routers
    enabled: bool             # default True
    is_default: bool          # default False, max 1 por base_model_id
    created_at: datetime
    updated_at: datetime
```

Validaciones:
- `profile_id`: regex `^[a-z0-9]+(-[a-z0-9]+)*$` (slug, no `/`, no espacios, no mayúsculas).
- `is_default`: si se marca True, cualquier otro perfil del mismo `base_model_id` con `is_default=True` se desactiva automáticamente.
- Un perfil ``execution`` requiere ``base_model_id`` existente en
  ``model_configs.model_id`` y no puede declarar ``routing_targets``.
- Un perfil ``router`` no tiene ``base_model_id`` ni presets de runtime y debe
  declarar al menos un ``routing_target`` válido. Los destinos siempre son
  ``profile_id`` estables de perfiles de ejecución u otros routers.
- Borrar un modelo sólo cascada sus perfiles de ejecución. Los routers son
  independientes del artefacto físico y se validan antes de borrar destinos.

### 8.2 Resolución de perfiles (ProfileRegistry → OpenAI/Ollama API)

Fuente: `backend/ocabra/api/openai/_deps.py` (`resolve_profile`,
`merge_profile_defaults`, `ensure_loaded`); el registro en memoria es
`core/profile_registry.ProfileRegistry`.

`resolve_profile(profile_id, model_manager, profile_registry, *, user=None, ...)
-> (ModelProfile, ModelState)` resuelve en este orden:

0. Alias oculto (`model_profile_aliases`) → su perfil (+ preset fijado).
1. `profile_id` exacto en `ProfileRegistry` (debe estar habilitado).
2. Id de despliegue (`<backend>/<model_key>`): su perfil default habilitado o,
   si no hay, el primer perfil habilitado (todo despliegue tiene al menos uno).
3. Si nada coincide → 404. Si el usuario no tiene acceso, también 404 (no se
   filtra la existencia).

`merge_profile_defaults(profile, body)` aplica, de menor a mayor prioridad,
los `request_defaults` del perfil, los del router por el que entró la
petición y el body del cliente; después inyecta los assets (`voice_ref`), que
el cliente no puede sobrescribir.

### 8.3 Worker key (ModelManager)

Fuente: `core/model_manager_helpers.compute_worker_key`.

```
worker_key = base_model_id                                   # load_overrides vacío
worker_key = f"{base_model_id}::{sha256(json(load_overrides, sort_keys))[:12]}"
```

El interruptor de diarización no forma parte de la clave
(`strip_diarization_overrides`).

- `load_overrides` vacío o idéntico entre perfiles → mismo worker (compartido).
- `load_overrides` diferente → worker separado (dedicado).
- Esto permite que perfiles como `chat` y `chat-creative` (que solo difieren en `request_defaults`) compartan worker, mientras que `chat` y `chat-long` (que difieren en `load_overrides`) usen workers separados.

### 8.4 Assets de perfil (ProfileRegistry → filesystem)

```
Asset storage: /data/profiles/{profile_id}/{filename}
Upload: POST /ocabra/profiles/{profile_id}/assets (multipart)
Delete: DELETE /ocabra/profiles/{profile_id}/assets/{asset_key}
```

Seguridad:
- Path traversal: validar que la ruta resultante esté dentro de `/data/profiles/{profile_id}/`.
- `profile_id` ya está validado como slug (sin `/`, sin `..`).
- Filenames sanitizados antes de escribir al disco.

### 8.5 Backend Chatterbox

```python
# backend/ocabra/backends/chatterbox_backend.py

class ChatterboxBackend(BackendInterface):
    backend_type = "chatterbox"
    # Capabilities: tts=True, streaming=True
    # VRAM estimate: 4096 MB (Turbo) | 8192 MB (Full)
```

Worker endpoints (`backend/workers/chatterbox_worker.py`):
```
GET  /health                       → {"status": "ok"}
GET  /info                         → {"model": str, "languages": list[str], "supports_voice_clone": bool}
GET  /voices                       → {"voices": [...], "supports_voice_clone": true}
POST /synthesize                   → audio bytes (Content-Type según formato solicitado)
POST /synthesize/stream            → chunked audio (Transfer-Encoding: chunked)
```

Contratos de seguridad:
- `voice_ref`: path controlado por oCabra (asset del perfil), NO ruta libre del cliente.
- El worker recibe la ruta resuelta desde el backend, nunca un path enviado directamente por el usuario final.

### 8.6 Exposición pública de perfiles

```
/v1/models                          → solo lista perfiles habilitados (profile_id como "id")
/v1/models/{id}                     → profile_id (identificador público anunciado)
/v1/chat/completions, /v1/audio/speech, etc. → model= acepta profile_id o model_id canónico
/ocabra/models                      → lista modelos internos con profiles[] anidado (admin only)
```

Resolución de identificadores:
- Si `model=` coincide exactamente con un `model_id` canónico:
  1. Buscar perfil default del modelo.
  2. Si no existe default, usar el primer perfil habilitado.
  3. Si no hay perfiles habilitados → 404 `profile_not_configured`.
- Esta compatibilidad no depende de `LEGACY_MODEL_ID_FALLBACK`: el ID canónico
  es estable y lo usa la API interna. La opción legacy queda reservada para
  alias históricos no canónicos y podrá retirarse en una migración posterior.

### 8.7 Endpoints REST de perfiles

Rutas en `api/internal/profiles.py` (CRUD, `rename`, `clone`, `assets`,
`aliases`, `runtime-presets`); lista completa en `docs/API.md`, esquemas
(`ProfileCreate`, `ProfileUpdate`, `ProfileOut`) en `schemas/profiles.py` y en
el Swagger.

### 8.8 Caché de perfiles

No hay keys ni canales Redis de perfiles: `ProfileRegistry` los mantiene en
memoria, los carga de BD al arrancar (`load_all`) y los recarga tras cada
escritura.

### 8.9 Presets de runtime por perfil

El contexto deja de representarse mediante perfiles o modelos duplicados con
sufijos ``-ctx*``. Un perfil público estable puede ofrecer varios presets de
runtime; cada preset agrupa todos los parámetros que cambian la identidad del
runner (contexto, paralelismo, KV cache y colocación).

```python
class ProfileRuntimePreset(Base):
    __tablename__ = "profile_runtime_presets"

    id: UUID                    # PK
    profile_id: str             # FK model_profiles.profile_id, ON DELETE CASCADE
    preset_key: str             # slug único dentro del perfil ("normal", "max")
    label: str
    context_mode: str           # "fixed" | "max"
    context_tokens: int | None  # requerido para fixed; NULL para max
    parallelism: int            # paralelismo solicitado, >= 1 (default 2)
    kv_cache_dtype: str         # "fp16" | "bf16" | "fp8" | "q8" | "q4" (default "q8")
    placement_policy: str       # "auto" | "preferred" | "forced"
    preferred_gpus: list[int] | None
    placement: dict | None      # colocación propia del preset (split, tensor split,
                                # capas en GPU, expertos MoE en CPU); NULL = la del
                                # modelo. Esquema: schemas.profiles.PresetPlacement
    enabled: bool
    is_default: bool
    priority: int
    created_at: datetime
    updated_at: datetime
```

Restricciones:

- ``(profile_id, preset_key)`` es único.
- Solo puede existir un preset ``is_default=true`` por perfil.
- El preset default debe estar habilitado.
- ``context_mode=fixed`` requiere ``context_tokens > 0``.
- ``context_mode=max`` no persiste un máximo como fuente de verdad; se resuelve
  contra metadatos del modelo, backend, hardware, paralelismo y límites de
  administración.
- La configuración que identifica un runner se canonicaliza en un
  ``runtime_key`` independiente del nombre visible del preset.
- Los presets **no** tienen ``load_policy`` propio (se quitó el 2026-10-30 —
  antes tenía uno y pisaba silenciosamente el del modelo base en cuanto se
  resolvía una petición vía perfil, haciendo que marcar el modelo base como
  ``pin`` no tuviera efecto real). ``RuntimeCoordinator.plan_profile_runtime``
  calcula el ``load_policy`` efectivo del worker derivado como el del modelo
  base **solo si** el preset elegido es el ``is_default``; cualquier otro
  preset (una excursión puntual a más contexto del habitual) siempre carga
  como ``on_demand``, independientemente de la política del modelo base.

El perfil conserva además una política de selección JSONB:

```python
runtime_policy = {
    "selection": "prefer_loaded",              # prefer_loaded | prefer_default
    "restore_default_when_idle": True,
    "downgrade_grace_seconds": 600,
    "minimum_residency_seconds": 120,
    "small_requests_extend_large_ttl": False,
    "remote_warm_threshold_seconds": 10,      # sin efecto: se acepta por compatibilidad
    "max_context_tokens": None,               # techo validado opcional
    "max_output_tokens": None,                # límite duro de generación opcional
    "validated_context_tokens": {"normal": 98304}, # pools fijos medidos
    "validated_vram_mb": {"normal": 24000}, # pico medido + margen
    # además: validated_by_backend, context_probe y otros; ver
    # schemas/profiles.RuntimePolicy (extra="forbid")
}
```

``validated_context_tokens`` solo puede confirmar el tamaño exacto de un preset
``fixed`` del mismo nombre. Permite que una prueba empírica reproducible
prevalezca sobre un estimador conservador para esa colocación concreta; no
eleva ``max_context_tokens`` ni el contexto nativo del modelo y no se aplica a
otros presets o GPU.

``validated_vram_mb`` es opcional y solo se aplica cuando el contexto exacto
del mismo preset figura también en ``validated_context_tokens``. Sustituye la
estimación genérica durante admisión y reserva por un pico empírico con margen;
permite representar runtimes que caben realmente aunque el estimador del GGUF
y del grafo supere la capacidad física. No afecta presets ni colocaciones no
validados.

Selección de preset para chat/completions:

1. Estimar ``prompt_tokens + reserved_output_tokens``.
2. Si un preset cargado cubre la petición, reutilizar el menor cargado que
   cubra el trabajo (salvo override explícito).
3. Sin candidato cargado, usar el default si cubre la petición; si no, usar el
   menor preset habilitado suficiente.
4. El reparto local/remoto lo decide la estrategia de federación del modelo
   (`core/federation_policy.py`), no esta política.
5. Si el perfil configura ``runtime_policy.max_output_tokens`` y un
   ``max_tokens`` o ``max_completion_tokens`` explícito lo supera, responder
   HTTP 400 con
   ``param=<campo>`` y ``code=max_tokens_exceeded`` antes de admission.
6. Si la salida solicitada respeta ese límite pero la suma de prompt y salida
   no cabe en ningún preset, responder ``context_length_exceeded`` antes de
   arrancar un runner.
7. El escalado hacia arriba es inmediato; la vuelta al default se difiere por
   ``downgrade_grace_seconds`` y nunca interrumpe peticiones activas.

**Presets ``max``.** El preset con ``context_mode="max"`` pide el mayor
contexto que el planner cree que cabe en la colocación del preset. Los
backends llama.cpp reparten entre varias GPU (``tensor_split``); Ollama no
tiene presets de contexto (ver el final de esta sección).

**Margen de seguridad del budget KV.** ``max_context_tokens`` aplica un
``DEFAULT_KV_SAFETY_FACTOR=0.90`` (``core/vram_planner.py``) al presupuesto (``free - weights -
overhead``) para absorber buffers no modelados (CUDA graphs,
Flash-Attention workspace, scratch de Ollama/llama.cpp, chunked prefill).
Sin esto el planner reportaba ``fits=True`` para 262k en 24 GB cuando
Ollama lo rechazaba con HTTP 500. vLLM no lo necesita porque ya reserva
VRAM via ``gpu_memory_utilization``.

**Pool KV unificado de llama.cpp.** ``extra_config.llama_cpp`` admite cuatro
campos opcionales y tri-state; si no aparecen, oCabra conserva el comportamiento
por defecto del build fijado:

```python
{
    "kv_unified": True,              # --kv-unified / --no-kv-unified
    "kv_unified_per_slot": 131072,   # cap opcional por secuencia, >= 1
    "cache_ram_mib": 0,              # -1 sin límite, 0 desactivada, >0 MiB
    "cache_idle_slots": False,       # --cache-idle-slots / --no-cache-idle-slots
}
```

El backend acepta tanto ``snake_case`` como ``camelCase`` por compatibilidad
con la UI. Estos campos configuran ``llama-server`` al crear el runtime; no
autorizan una mutación en caliente.

**Admission control ponderado del pool KV.** La cola justa existente es la
única autoridad de admisión delante del worker. No se crea una segunda cola
para KV: ``FairSlotGate`` aplica atómicamente los dos límites al conceder una
reserva:

```python
async with gate.slot(
    user_key,
    is_batch=False,
    kv_tokens=required_tokens,       # prompt estimado + salida reservada
    prompt_tokens=prompt_tokens,
) as admission:
    ...
```

(``core/fair_slot_gate.py``; ``WorkerPool.forward_request``/``forward_stream``
lo usan cuando reciben ``user_key``.)

- ``slots`` limita peticiones concurrentes físicas.
- ``kv_capacity_tokens`` es el pool agregado del runtime; ``None`` desactiva
  la dimensión KV y conserva exactamente el comportamiento anterior.

**Admisión computacional adaptativa de llama.cpp.** Los slots declarados por
el runtime siguen siendo físicos y no se redimensiona el pool KV. Antes de
conceder cada lease, el mismo ``FairSlotGate`` calcula un límite efectivo a
partir del prompt estimado:

- prompt menor de ``LLAMA_CPP_MEDIUM_PREFILL_TOKENS`` (32768): todos los slots físicos;
- desde ese umbral: máximo ``LLAMA_CPP_MEDIUM_PREFILL_MAX_CONCURRENCY`` (3) peticiones activas;
- desde ``LLAMA_CPP_LARGE_PREFILL_TOKENS`` (65536): máximo ``LLAMA_CPP_LARGE_PREFILL_MAX_CONCURRENCY`` (2).

Mientras una petición ponderada está activa, el límite también se aplica a
las peticiones pequeñas que lleguen después. Así un prefill grande no queda
rodeado por tres admisiones nuevas. Los umbrales sólo se aplican a workers
llama.cpp. La dimensión computacional se concede atómicamente junto con
slot, KV y prioridad justa; no existe una segunda cola con orden diferente.

En streaming, el gate conoce además la frontera real entre prefill y decode:
si el candidato o un prefill activo supera el umbral medio, no inicia otro
prefill hasta que el stream activo entrega su primer chunk. En ese momento el
lease cambia a decode y los slots restantes vuelven a estar disponibles. Los
prefills pequeños pueden solaparse entre sí. El camino non-streaming conserva
la admisión anterior porque no expone una frontera TTFT intermedia.

Cada lease contextual tiene un identificador monotónico por worker. Su
liberación es idempotente y queda ligada a la tarea propietaria; antes de una
nueva admisión se recuperan leases cuya tarea haya terminado sin ejecutar el
``finally`` normal. Además, el middleware de stats cierra explícitamente el
iterador interior ante desconexión para propagar la cancelación hasta el
backend y liberar slot, KV y conexión HTTP.

Cada fila completada puede persistir ``runtime_telemetry`` con estas claves
opcionales:

```json
{
  "prompt_tokens_estimate": 68240,
  "kv_tokens_reserved": 98304,
  "queue_ms": 1270,
  "load_ms": 30171,
  "ttft_ms": 3240,
  "prefill_ms": 1850,
  "slot_id": 2,
  "physical_slots": 4,
  "admission_limit": 2,
  "admission_concurrency": 2,
  "admission_lease_id": 37,
  "gpu_indices": [0, 1]
}
```

``queue_ms`` mide exclusivamente la espera dentro del gate; ``load_ms`` la
espera explícita del runtime en los caminos que cargan desde el stream;
``ttft_ms`` se mide desde la entrada HTTP hasta el primer token visible.
``prefill_ms`` y ``slot_id`` sólo aparecen cuando llama.cpp los entrega en su
bloque ``timings``. ``gpu_indices`` contiene todas las tarjetas que sirvieron
la petición (incluidos modelos partidos); `gpu_index` conserva la primera por
compatibilidad. ``GET /ocabra/stats/recent`` expone el objeto sin
reinterpretarlo.

Detalles adicionales del gate KV:

- ``kv_tokens`` procede de ``RuntimePlan.required_tokens``. El coordinador lo
  calcula también para perfiles sin presets; una petición sin estimación usa
  cero y nunca activa por accidente un gate ponderado.
- La capacidad usa primero el contexto efectivo del preset y, para perfiles
  fijos sin presets, ``llama_cpp.ctx_size``/``use_case.context``. Del mismo
  modo, los slots físicos usan ``_runtime.parallelism`` o el
  ``parallel_slots``/``use_case.slots`` configurado.
- Una reserva mayor que el pool se rechaza de inmediato; no puede quedar
  encolada para siempre.
- Una reserva que cabe individualmente pero no en el remanente espera en la
  misma cola justa. El orden sigue siendo interactiva antes que batch y,
  dentro de cada tier, usuario menos recientemente servido entre las
  cabeceras que caben en ese momento.
- La liberación es idempotente desde el punto de vista del contexto de
  petición y ocurre al completar, fallar o cancelar, incluido streaming.
- Solo los runtimes llama.cpp con ``kv_unified=true`` anuncian capacidad KV a
  este gate. El resto de backends conserva el gate exclusivamente por slots.
- En el modo productivo preferido, sin presets, ``llama_cpp.ctx_size`` es el
  pool físico agregado y ``parallel_slots`` es únicamente el límite de
  peticiones simultáneas. Si ``kv_unified_per_slot`` no está configurado, cada
  slot puede ver hasta el pool completo; ``FairSlotGate`` impide que la suma de
  reservas lógicas lo exceda. No se multiplica ``ctx_size`` por los slots.

El snapshot de diagnóstico debe distinguir ``slot_saturated``,
``compute_saturated`` y ``kv_saturated`` y publicar capacidad, reservado y
libre. Estas cifras son
reservas lógicas conservadoras de oCabra hasta que llama.cpp exponga ocupación
física inequívoca; no se presentarán como celdas KV observadas.

``GET /ocabra/models/{model_id}/status`` incluye ``admission=null`` hasta que
el worker haya creado su gate. Si se consulta el ID canónico y existe un único
runtime derivado cargado, resuelve automáticamente su gate; con varios runtimes
derivados simultáneos el cliente debe consultar el ID derivado inequívoco.
En ese caso también publica ``loaded/status/gpu/vramUsedMb`` del runtime activo
manteniendo el ``modelId`` canónico solicitado. Después expone:

```json
{
  "admission": {
    "slots": 4,
    "activeRequests": 2,
    "queuedRequests": 1,
    "slotSaturated": false,
    "computeSaturated": true,
    "admissionLimit": 2,
    "kvCapacityTokens": 98304,
    "kvReservedTokens": 72000,
    "kvFreeTokens": 26304
  }
}
```

**Visión opcional en llama.cpp.** ``extra_config.llama_cpp`` acepta
``mmproj_model_id`` junto a ``vision_mode``:

```python
{
    "mmproj_model_id": "/data/models/mmproj.gguf",
    "vision_mode": "enabled",  # enabled | disabled | on_demand
}
```

- ``enabled`` carga ``--mmproj`` al arrancar, como antes de existir el campo.
- ``disabled`` no carga el projector y anuncia ``vision=false``.
- ``on_demand`` arranca sin projector, anuncia que la visión está disponible
  y la activa al recibir la primera petición que contiene una imagen. Tras la
  activación permanece cargada durante la vida del worker.
- Si se omite ``vision_mode``, un ``mmproj_model_id`` presente implica
  ``enabled`` para mantener compatibilidad; sin projector implica ``disabled``.
- ``enabled`` y ``on_demand`` requieren ``mmproj_model_id``.

El modo bajo demanda es exclusivo del runtime custom de oCabra. El worker
arranca con ``--props`` y oCabra invoca ``POST /props`` sobre loopback con:

```json
{"vision": {"enabled": true, "mmproj": "/data/models/mmproj.gguf"}}
```

La respuesta de éxito es
``{"success": true, "vision": {"enabled": true, "active": true}}``. oCabra
exige ``vision.active=true`` para no confundir la respuesta genérica de un
``POST /props`` upstream con una activación real. El runtime serializa la
transición, espera a que sea seguro modificar el projector y no acepta una
petición multimodal hasta completarla. Un llama.cpp upstream sin esta extensión
debe producir un error explícito; oCabra no reinicia silenciosamente el modelo.

**Razonamiento en llama.cpp.** ``extra_config.llama_cpp`` admite controles de
arranque independientes del parser de respuesta:

```python
{
    "jinja": True,
    "reasoning": "off",          # on | off | auto
    "reasoning_format": "none", # none | deepseek | deepseek-legacy | auto
    "reasoning_budget": -1,    # sin tope; permite activación por petición
}
```

``reasoning_format=none`` solo evita extraer el pensamiento a
``reasoning_content``; no desactiva por sí mismo la generación razonada.
Para desactivarla por defecto se usa ``reasoning=off``. Este valor configura el
arranque, pero no elimina la capacidad del modelo: una petición puede activar
thinking sin recargar el worker mediante ``reasoning_effort=low|medium|high``.
En llama.cpp oCabra lo traduce conjuntamente a
``chat_template_kwargs.enable_thinking=true`` y al nivel solicitado. Un valor
ausente o ``default`` conserva el valor de arranque; ``off``/``none`` fuerza
``enable_thinking=false`` para esa petición. ``reasoning_budget=0`` queda
disponible para modelos o plantillas que necesiten además cerrar el bloque de
pensamiento inmediatamente, pero no debe configurarse en un perfil que quiera
permitir activación bajo demanda.

``capabilities.reasoning`` expresa soporte, no que thinking esté activo por
defecto. El catálogo expone el valor de arranque separadamente como
``ocabra.reasoning_default`` cuando está configurado explícitamente.

La compatibilidad OpenAI elimina de ``message.content`` y de los deltas SSE
únicamente un bloque inicial ``<think>`` vacío. Un bloque con contenido se
preserva; cuando thinking se solicita explícitamente, el parser lo publica como
``reasoning_content``.

El fork de Makespace es desde 2026-11-06 un backend propio, ``llamacpp_mks``,
que instala su propio binario (``git_repo`` + ``git_ref`` del install_spec).
``LLAMA_CPP_CUSTOM_SERVER_BIN`` se eliminó con él: era el apaño previo para que
este backend apuntase a un árbol construido a mano, y mantenerlo habría dejado
una segunda vía silenciosa de cambiar el binario bajo cualquiera de los dos
backends. Cada uno resuelve ahora el suyo por su propia instalación modular
(``metadata.json`` → ``extra_bins.server``); ``llama_cpp`` cae a
``LLAMA_CPP_SERVER_BIN`` como último recurso y ``llamacpp_mks`` falla explícito,
porque servir el binario de upstream bajo el nombre del fork anunciaría
capacidades —visión on-demand— que el proceso no tiene.

Con KV unificada, ``ctx_size`` dimensiona un único pool físico y cada slot puede
direccionar hasta todo el pool. ``kv_unified_per_slot`` no reserva memoria por
slot: solo reduce ese máximo lógico. Para reparto completamente flexible se
omite el campo o se iguala a ``ctx_size``; oCabra debe controlar que las reservas
agregadas de las peticiones admitidas no superen el pool.

Endpoints internos (lista completa en ``docs/API.md``):
``/ocabra/profiles/{profile_id}/runtime-presets`` (CRUD), ``/resolved``,
``/preview``, ``/{preset_key}/benchmark`` y ``/{preset_key}/context-probe``
(``GET``/``POST`` y ``/apply``).

``runtime-presets/resolved`` calcula una instantánea administrativa contra la
arquitectura, backend y VRAM total actuales. Devuelve por preset
``effective_context``, ``model_max``, ``hardware_max``, GPUs resueltas,
``requested_parallelism``, ``effective_parallelism``, ``fits`` y advertencias.
El paralelismo efectivo aplica límites conocidos del runtime antes de calcular
KV y ``runtime_key``; nunca se anuncia ni reserva memoria para slots que el
backend va a reducir. No carga ni descarga el modelo y no se usa directamente
en ``/v1/models``; la API pública consumirá la misma información desde una
caché de capacidades para no convertir el listado en una sonda bloqueante.
En presets legacy, ``effective_context`` y ``hardware_max`` son contexto por
petición. Para llama.cpp sin KV unificada, el runner materializa ese reparto
como ``--ctx-size = effective_context * effective_parallelism``. Cuando el
modelo usa ``llama_cpp.kv_unified=true``, ambos valores expresan en cambio el
pool agregado real: ``--ctx-size = effective_context`` y los cuatro slots
pueden direccionar todo el pool. El ``FairSlotGate`` impide que la suma de
reservas concurrentes lo exceda. Esta distinción se persiste en el runtime como
``_runtime.context_is_pool=true`` y evita multiplicar accidentalmente por el
número de slots durante la admisión o el arranque.

Los perfiles adaptativos llama.cpp pueden combinar un preset default forzado a
una GPU y otro preset extendido forzado a dos. La petición selecciona
el menor pool que cubre ``prompt + salida reservada``; el salto a dos GPU drena
y desaloja los runtimes que ocupen cualquiera de sus shards. El retorno al
default ocurre únicamente tras ``minimum_residency_seconds`` y
``downgrade_grace_seconds``, nunca con peticiones activas.

``runtime-presets/{preset_key}/benchmark`` admite una carga fría opcional y
devuelve ``load_ms``, ``generation_ms``, tokens de salida y tokens/segundo. No
se ejecuta automáticamente al guardar: es una acción administrativa explícita.

``ProfileOut`` incluye ``runtime_presets`` y ``runtime_policy``. La creación de
presets no cambia por sí sola el runtime cargado; aplicar un cambio al runner es
una operación explícita posterior (inmediata, drenada o en la siguiente carga).

Exposición pública:

- ``/v1/models`` y ``/v1/models/{id}`` anuncian el máximo efectivo de los
  presets habilitados, no el contexto del runner activo.
- La extensión ``ocabra.context`` diferencia ``model_max``, ``effective_max``,
  ``default`` y ``active``.
- Los alias legacy ``-ctx*`` pueden seguir aceptándose durante la migración,
  pero quedan ocultos del listado y resuelven al perfil canónico + preset. Un
  alias puede superponerse temporalmente a un perfil legacy existente: el
  alias tiene precedencia en resolución y oculta ese perfil de los listados.

Selección y observabilidad:

- Un cuerpo OpenAI puede fijar preset mediante la extensión opcional
  ``ocabra.context_preset``. oCabra la elimina antes de llamar al backend.
- ``context_length`` público equivale a ``effective_max``. El contexto activo
  se expone aparte y nunca reduce la capacidad anunciada.
- ``ocabra.capabilities.max_output_tokens`` es un límite duro exigible y solo
  se publica cuando el perfil o el backend declaran uno independiente de la
  ventana total. Nunca se infiere de ``context_length``. Si el límite no está
  configurado, el campo se omite y admission solo valida que entrada más
  salida quepan en el preset seleccionado.
- ``recommended_output_tokens`` es el presupuesto operativo derivado de la
  reserva por defecto y del tráfico real. ``recommended_input_tokens`` es la
  parte de la ventana que queda al reservar ese presupuesto. Ambos son
  recomendaciones para planificación y nunca límites duros.
- Las estadísticas conservan el ``profile_id`` público y atribuyen también el
  preset, runtime key y contexto efectivos cuando existan.

Ollama (desde 2026-11-12):

- Un único daemon (``OLLAMA_BASE_URL``) para todos los modelos de Ollama; los
  pools por GPU y ``OLLAMA_POOLS_JSON`` se retiraron. Ollama coloca las capas;
  oCabra no fuerza GPU.
- Paralelismo y contexto máximo son globales (``settings.ollama_num_parallel``
  y ``settings.ollama_max_context``, editables en ``PATCH /ocabra/config``).
  ``POST /ocabra/ollama/server/apply-settings`` los escribe en
  ``data/ollama/ollama.env`` (``env_file`` del servicio) y recrea el daemon.
- El ``num_ctx`` de un cliente se recorta al máximo global (o al
  ``ollama.num_ctx`` del modelo si es menor). Los modelos de Ollama no tienen
  presets de contexto: la API los rechaza.
- Para perfiles con contexto dinámico, las rutas OpenAI se traducen dentro del
  worker pool a ``/api/chat`` o ``/api/generate`` y la respuesta vuelve a
  normalizarse como OpenAI. El shim ``/v1`` nativo de Ollama no se usa porque
  ignora ``options.num_ctx`` y recargaría el contexto global del daemon.

### 8.10 Compatibilidad de voces OpenAI (TTS) — independiente de routers

Un cliente que solo sabe los nombres oficiales de TTS de OpenAI (modelo:
``tts-1``, ``tts-1-hd``, ``gpt-4o-mini-tts``; voz: ``alloy``, ``echo``...) no
puede mandar ningún nombre propio de oCabra, ni siquiera un router. Esta tabla
traduce cada par ``(openai_model, voice_name)`` directamente a un
``profile_id`` propio — ejecución o router, sin restricción — **sin tocar**
``routing_targets``/``RouterResolver``: un router sigue decidiendo por su
propia lógica de caliente/prioridad, ajeno del todo a esta traducción.

```python
class OpenAITTSVoiceMap(Base):
    __tablename__ = "openai_tts_voice_map"

    openai_model: str        # PK; "tts-1", "tts-1-hd", "gpt-4o-mini-tts"...
    voice_name: str          # PK; "alloy", "echo"... (sin validar contra una
                              # lista cerrada — OpenAI puede añadir nombres)
    target_profile_id: str   # FK model_profiles.profile_id, ON DELETE CASCADE
    created_at: datetime
    updated_at: datetime
```

Endpoints (``/ocabra/``, rol ``model_manager``):

```
GET    /ocabra/openai-tts-voice-map                              → list[OpenAITTSVoiceMap]
PUT    /ocabra/openai-tts-voice-map/{openai_model}/{voice_name}   → OpenAITTSVoiceMap
  body: { target_profile_id: str }
DELETE /ocabra/openai-tts-voice-map/{openai_model}/{voice_name}   → {"ok": true}
```

Resolución, en ``POST /v1/audio/speech`` y ``GET /v1/audio/voices``:

- Sin fila para ``(model, voice)``: ``model`` se resuelve literal, igual que
  hoy — un cliente que ya manda un nombre propio de oCabra no nota nada.
- Con fila: ``model`` se sustituye por ``target_profile_id`` antes de
  cualquier otra resolución (incluida la federación), como si el cliente
  hubiera mandado ese nombre desde el principio.
- ACL: el permiso se comprueba contra ``openai_model`` (por ejemplo,
  ``tts-1``), no contra cada ``target_profile_id``. Tras autorizar el nombre
  virtual, el destino se ejecuta con la misma semántica que el target interno
  de un router: no necesita un grant adicional y no se filtra su identidad en
  errores de acceso. Los aliases OpenAI se conceden mediante ``group_models``
  como cualquier otro model id virtual.
- ``target_profile_id`` debe existir y estar ``enabled`` en el momento de
  guardar el pin; no se revalida en cada petición.
- Renombrar o borrar el perfil destino recarga la tabla entera desde BD
  (barato, no hay lógica de reescritura cruzada entre registries).

---

## 9. Configuración (variables de entorno)

Fuente: `backend/ocabra/config.py` (`Settings`, pydantic-settings; cada campo
se lee de la variable de entorno del mismo nombre en mayúsculas). Las variables
que un operador puede necesitar, con sus valores por defecto, están en
`.env.example`.

Invariantes entre módulos que conviene conocer:

- `APP_VERSION` no se lee del entorno: es una constante del código (ver
  `CLAUDE.md`).
- Los ajustes editables desde la UI (`PATCH /ocabra/config`) se guardan en la
  tabla `server_config` y **pisan** al `.env` en cada arranque (§5.4). Cambiar
  el `.env` no tiene efecto sobre un ajuste que alguien guardó desde la UI.
- `settings` se modifica en caliente: el código que copia un valor al arrancar
  no ve los cambios posteriores; léelo de `settings` en el momento de usarlo.
- `MODELS_DIR` es de solo lectura en runtime; `DOWNLOAD_DIR` vacío significa
  `{MODELS_DIR}/downloads`.
- `WORKER_PORT_RANGE_START`/`END` es el único rango de puertos de workers y
  procesos auxiliares (ver §1 y §4).
