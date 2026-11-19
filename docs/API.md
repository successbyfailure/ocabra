# oCabra — índice de la API

Una línea por ruta: método, ruta, qué hace y quién puede llamarla. Los
esquemas de petición y respuesta **no** están aquí:

- **API pública** (`/v1/*`, `/api/*`, sistema): Swagger en `/docs` (y
  `/openapi.json`), con la descripción general, los routers `recomendado-*`,
  Batches, las cabeceras `X-Ocabra-*` y los formatos de error.
- **Cambios entre versiones**: `/docs/changelog` (fuente
  `backend/ocabra/docs_content/changelog.md`). **Rendimiento por modelo**:
  `/docs/models`. Las dos también en Markdown crudo añadiendo `.md`.
- **API interna** (`/ocabra/*`, la que usa el panel): se registra con
  `include_in_schema=False`, así que **no sale en Swagger**. Sus esquemas son
  los modelos Pydantic de `backend/ocabra/schemas/` y de cada módulo de
  `backend/ocabra/api/internal/`, y cada endpoint lleva docstring.

La versión servida la devuelven `GET /version` y `GET /health`; vive en
`APP_VERSION` (`backend/ocabra/config.py`).

**Mantenimiento.** `backend/tests/test_api_md.py` falla si una ruta de la
aplicación no aparece aquí o si aquí aparece una ruta que ya no existe. La
columna «Qué hace» es el `summary` (o la primera línea del docstring) del
endpoint en el código.

---

## Acceso

Todo pasa por Caddy en el puerto 8484 del host (`/v1`, `/api`, `/ocabra`,
`/health`, `/ready`, `/version`, `/docs`, `/metrics`; el resto es el panel).

Cómo se identifica una petición (`get_current_user` en
`backend/ocabra/api/_deps_auth.py`), por orden:

1. `X-Ocabra-Internal-Token`: llamadas del propio proceso (el procesador de
   Batches), las únicas que pueden actuar como otro usuario.
2. `X-Gateway-Token` = `GATEWAY_SERVICE_TOKEN`: servicios acompañantes
   (gateway, TRELLIS.2). Identidad `model_manager`, sin suplantación.
3. Cookie `ocabra_session` (JWT del panel; se obtiene con
   `POST /ocabra/auth/login`).
4. `Authorization: Bearer <api key>`. Un JWT en esta cabecera se rechaza: los
   JWT solo viajan en la cookie.
5. Anónimo, solo si `REQUIRE_API_KEY_OPENAI` / `REQUIRE_API_KEY_OLLAMA` están
   a `false` (afecta a `/v1` y `/api` respectivamente).

Valores de la columna **Acceso**:

| Valor | Significado |
|---|---|
| pública | Sin autenticación |
| clave | API key o cookie (anónimo si la clave no es obligatoria); además, el modelo pedido tiene que estar en los grupos del usuario |
| `user` | Cualquier usuario autenticado (401 si es anónimo) |
| `model_manager` | Rol `model_manager` o superior (403 si no) |
| `system_admin` | Solo administradores |
| cookie de sesión / token del túnel | WebSockets que validan dentro del handler |

Los roles son jerárquicos: `user` < `model_manager` < `system_admin`.

```bash
# Panel / API interna: login con cookie
curl -c cookies.txt -X POST http://localhost:8484/ocabra/auth/login \
  -H "Content-Type: application/json" -d '{"username":"…","password":"…"}'
curl -b cookies.txt http://localhost:8484/ocabra/models

# API pública: clave
curl -H "Authorization: Bearer sk-…" http://localhost:8484/v1/models
```

## Convenciones comunes

- **Cabeceras de respuesta.** Las rutas de `/v1` que llevan a un modelo
  devuelven `X-Ocabra-Model-Status`, `X-Ocabra-Model-Id` y, según el caso,
  `X-Ocabra-Expected-Wait-Seconds`, `X-Ocabra-Load-Queue-Depth`,
  `X-Ocabra-Load-Active`, `X-Ocabra-Was-Cold-Start`,
  `X-Ocabra-Load-Duration-Ms`, `X-Ocabra-Router`, `X-Ocabra-Router-Target`,
  `X-Ocabra-Slot-Queue-Saturated` / `-In-Flight` / `-Count` y
  `X-Ocabra-KV-Queue-Saturated` / `-Capacity-Tokens` / `-Reserved-Tokens`.
  Toda respuesta lleva `X-Ocabra-Privacy`. Definición de cada una:
  `OCABRA_MODEL_HEADERS` en `backend/ocabra/api/openai/_deps.py`, que es
  lo que publica Swagger.
- **Cabeceras de petición.** `X-Ocabra-Node: <nodo>|local` fuerza dónde se
  sirve con federación; `X-Ocabra-Stream-Events: true`,
  `X-Ocabra-Allowed-Tools` y `X-Ocabra-Require-Approval` afectan a los
  agentes (`agent/<slug>`).
- **Streaming.** Los SSE intercalan comentarios `: {"event":
  "ocabra.model_loading" | "ocabra.model_ready" | "ocabra.slot_queued", …}`
  que los parsers estrictos ignoran. Un rechazo antes del primer byte llega
  como HTTP 4xx/5xx normal; un fallo a mitad, como `data: {"error": …}`.
- **Errores.** `/v1` y `/api` devuelven un objeto `error` (`message`, `type`,
  `code`), casi siempre dentro de `detail` (`{"detail": {"error": {…}}}`,
  que es como lo emite `HTTPException`); `/ocabra` devuelve
  `{"detail": "…"}`. Las condiciones transitorias (cargando, al límite de
  concurrencia → 429 `model_at_capacity`, circuit breaker, mantenimiento)
  llevan `Retry-After`; las permanentes no.

## Sistema y documentación

`/metrics` solo responde desde redes privadas: Caddy devuelve 403 a cualquier otra IP.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/openapi.json` | Esquema OpenAPI de la API pública | pública |
| GET | `/docs` | Swagger UI | pública |
| GET | `/docs/oauth2-redirect` | Redirección OAuth2 de Swagger UI | pública |
| GET | `/redoc` | ReDoc | pública |
| GET | `/health` | Quick liveness check. Always returns 200 if the process is running | pública |
| GET | `/version` | Version of the running server, for clients that probe `/version` | pública |
| GET | `/ready` | Readiness check. Returns 200 only when all dependencies are healthy | pública |
| GET | `/docs/{pagina}.md` | Devuelve la fuente en Markdown de una página de documentación | pública |
| GET | `/docs/{pagina}` | Renderiza una página de documentación en prosa como HTML | pública |
| GET | `/metrics` | Prometheus metrics | pública |

## OpenAI — modelos, texto y embeddings (`/v1`)

`model` acepta un `profile_id` (recomendado), un router `recomendado-*`, un agente `agent/<slug>` o, mientras `LEGACY_MODEL_ID_FALLBACK` siga activo, un model_id canónico.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/v1/models` | List models | clave |
| GET | `/v1/models/{model_id:path}` | Retrieve a model | clave |
| POST | `/v1/chat/completions` | Create chat completion | clave |
| POST | `/v1/completions` | Create text completion | clave |
| POST | `/v1/embeddings` | Create embeddings | clave |
| POST | `/v1/pooling` | Run pooling on a model | clave |
| POST | `/v1/score` | Score text pairs | clave |
| POST | `/v1/rerank` | Rerank documents for a query | clave |
| POST | `/v1/classify` | Classify inputs with a classification model | clave |

## OpenAI — imagen y audio (`/v1`)

`/v1/images/files/{name}` es pública a propósito: es la URL que devuelve `response_format=url`, con nombre aleatorio y caducidad (`IMAGE_URL_TTL_SECONDS`).

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| POST | `/v1/images/generations` | Create image | clave |
| POST | `/v1/images/edits` | Edit image | clave |
| GET | `/v1/images/files/{name}` | Retrieve a generated image | pública |
| POST | `/v1/audio/transcriptions` | Transcribe audio | clave |
| GET | `/v1/audio/voices` | List TTS voices for a model | clave |
| POST | `/v1/audio/speech` | Generate speech | clave |
| POST | `/v1/audio/generate` | Generate music | clave |

## OpenAI — Realtime (`/v1/realtime`)

WebSocket con el protocolo de eventos de OpenAI Realtime. La clave va en `Authorization: Bearer` o en el subprotocolo que usa OpenAI.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| WS | `/v1/realtime` | OpenAI Realtime API WebSocket endpoint | clave |

## OpenAI — Files y Batches (`/v1`)

Desactivados en modo de privacidad `zero_data`. Qué endpoints admite un lote: tabla de Batches en `/docs` (`ALLOWED_ENDPOINTS` en `api/openai/batches.py`).

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| POST | `/v1/files` | Upload a file | clave |
| GET | `/v1/files` | List uploaded files | clave |
| GET | `/v1/files/{file_id}` | Retrieve file metadata | clave |
| GET | `/v1/files/{file_id}/content` | Download file content | clave |
| DELETE | `/v1/files/{file_id}` | Delete a file | clave |
| POST | `/v1/batches` | Create a batch | clave |
| GET | `/v1/batches/{batch_id}` | Retrieve a batch | clave |
| GET | `/v1/batches` | List batches | clave |
| POST | `/v1/batches/{batch_id}/cancel` | Cancel a batch | clave |

## Ollama (`/api`)

Formato de Ollama sobre cualquier backend de oCabra, no solo sobre el daemon Ollama.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/api/tags` | List models | clave |
| GET | `/api/ps` | List running models | clave |
| POST | `/api/show` | Show model details | clave |
| POST | `/api/pull` | Pull a model | clave · `model_manager` |
| POST | `/api/generate` | Generate completion | clave |
| POST | `/api/chat` | Create chat completion | clave |
| POST | `/api/embeddings` | Create embeddings (legacy) | clave |
| POST | `/api/embed` | Create embeddings | clave |
| DELETE | `/api/delete` | Delete a model | clave · `model_manager` |
| GET | `/api/version` | Server version (Ollama-compatible) | pública |

## Autenticación, usuarios y grupos

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| POST | `/ocabra/auth/login` | Login | pública |
| POST | `/ocabra/auth/logout` | Logout | pública |
| GET | `/ocabra/auth/me` | Get current user | `user` |
| PUT | `/ocabra/auth/password` | Change own password | `user` |
| GET | `/ocabra/auth/keys` | List own API keys | `user` |
| POST | `/ocabra/auth/keys` | Create API key | `user` |
| DELETE | `/ocabra/auth/keys/revoked` | Delete own revoked API keys | `user` |
| DELETE | `/ocabra/auth/keys/{key_id}` | Revoke API key | `user` |
| GET | `/ocabra/users` | List all users | `system_admin` |
| POST | `/ocabra/users` | Create a new user | `system_admin` |
| GET | `/ocabra/users/{user_id}` | Get a user by ID | `system_admin` |
| PATCH | `/ocabra/users/{user_id}` | Update a user's role, active status, or email | `system_admin` |
| DELETE | `/ocabra/users/{user_id}` | Delete a user | `system_admin` |
| POST | `/ocabra/users/{user_id}/reset-password` | Reset a user's password | `system_admin` |
| GET | `/ocabra/users/{user_id}/keys` | List all API keys for a user (without the raw key value) | `system_admin` |
| GET | `/ocabra/api-keys` | List every API key across all users, with owner info | `system_admin` |
| DELETE | `/ocabra/api-keys/revoked` | Permanently delete every revoked API key across all users | `system_admin` |
| POST | `/ocabra/api-keys/reassign` | Reassign one or more API keys to a different owner (bulk action) | `system_admin` |
| DELETE | `/ocabra/users/{user_id}/keys/{key_id}` | Revoke a specific API key belonging to a user | `system_admin` |
| POST | `/ocabra/users/{user_id}/keys` | Create an API key for a specific user (admin only) | `system_admin` |
| GET | `/ocabra/groups` | List all groups with member and model counts | `system_admin` |
| POST | `/ocabra/groups` | Create a new group | `system_admin` |
| PATCH | `/ocabra/groups/{group_id}` | Update a group's name or description | `system_admin` |
| DELETE | `/ocabra/groups/{group_id}` | Delete a group | `system_admin` |
| GET | `/ocabra/groups/{group_id}/models` | List all model IDs assigned to a group | `system_admin` |
| POST | `/ocabra/groups/{group_id}/models` | Add a model to a group | `system_admin` |
| DELETE | `/ocabra/groups/{group_id}/models/{model_id:path}` | Remove a model from a group | `system_admin` |
| GET | `/ocabra/groups/{group_id}/members` | List all members of a group | `system_admin` |
| POST | `/ocabra/groups/{group_id}/members` | Add a user to a group | `system_admin` |
| DELETE | `/ocabra/groups/{group_id}/members/{user_id}` | Remove a user from a group | `system_admin` |

## Modelos

`{model_id:path}` admite `/` (ids como `llama_cpp/gemma4-26b`). `POST …/context` cambia en caliente el contexto de un worker llama.cpp (solo el fork `llamacpp_mks`, con `LLAMA_CPP_LIVE_CONTEXT_ENABLED`). Los modelos lógicos agrupan un mismo fichero servido por varios backends.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/models` | List all models | `user` |
| GET | `/ocabra/models/storage` | Get models storage usage | `user` |
| GET | `/ocabra/models/activity` | Per-model in-flight activity | `user` |
| GET | `/ocabra/models/ollama-runtime` | Ollama runtime placement | `user` |
| GET | `/ocabra/models/{model_id:path}/status` | Model load status | `user` |
| GET | `/ocabra/models/{model_id:path}/capacity` | Model VRAM capacity plan | `user` |
| GET | `/ocabra/models/{model_id:path}/speculative-candidates` | List speculative-decoding draft candidates | `user` |
| GET | `/ocabra/models/{model_id:path}/insights` | Measured behaviour of a model, for the model editor | `user` |
| GET | `/ocabra/models/{model_id:path}/config-schema` | Editable options of a model's backend | `user` |
| GET | `/ocabra/models/{model_id:path}` | Get model state | `user` |
| POST | `/ocabra/models` | Register a new model | `model_manager` |
| POST | `/ocabra/models/{model_id:path}/load` | Load model onto GPU | `user` |
| POST | `/ocabra/models/{model_id:path}/smoke-test` | Load a model and send it one minimal request | `model_manager` |
| POST | `/ocabra/models/{model_id:path}/unload` | Unload model from GPU | `model_manager` |
| POST | `/ocabra/models/{model_id:path}/cancel-load` | Cancel an in-progress model load | `model_manager` |
| POST | `/ocabra/models/{model_id:path}/downsize` | Restore a multi-GPU runtime to its one-GPU default | `model_manager` |
| POST | `/ocabra/models/{model_id:path}/context` | Change a loaded llama.cpp worker's context in place | `model_manager` |
| POST | `/ocabra/models/{model_id:path}/maintenance` | Pause or resume new requests for a model | `model_manager` |
| PATCH | `/ocabra/models/{model_id:path}` | Update model configuration | `model_manager` |
| POST | `/ocabra/models/{model_id:path}/preview` | Preview a draft configuration without saving or loading | `user` |
| POST | `/ocabra/models/{model_id:path}/auto-setup` | Plan (and optionally write) runtime presets for the chosen devices | `model_manager` |
| POST | `/ocabra/models/{model_id:path}/memory-estimate` | Estimate model memory requirements | `user` |
| POST | `/ocabra/models/refresh-tokenizer-fingerprints` | Backfill llama.cpp tokenizer fingerprints | `model_manager` |
| DELETE | `/ocabra/models/{model_id:path}` | Delete a model | `model_manager` |
| POST | `/ocabra/models/{model_id:path}/update` | Re-pull a single installed model | `model_manager` |
| POST | `/ocabra/models/refresh-metadata` | Refresh registry metadata for installed models | `model_manager` |
| POST | `/ocabra/models/update-all` | Re-pull every updatable model | `model_manager` |
| GET | `/ocabra/logical-models` | List logical models | `model_manager` |
| GET | `/ocabra/logical-models/{model_key:path}/compatibility` | Backends that can serve a model's file | `model_manager` |
| POST | `/ocabra/logical-models/{model_key:path}/deployments` | Add a deployment (backend) to a model | `model_manager` |
| DELETE | `/ocabra/logical-models/{model_key:path}/deployments/{backend}` | Remove a deployment from a model | `model_manager` |
| GET | `/ocabra/logical-models/{model_key:path}` | Get a logical model | `model_manager` |
| PATCH | `/ocabra/logical-models/{model_key:path}` | Update a logical model (default backend, display name) | `model_manager` |

## Perfiles, presets y alias de compatibilidad

Los presets (`runtime-presets`) son la escalera de contexto/colocación de un perfil. Los mapas de compatibilidad traducen nombres de OpenAI (`tts-1` + voz, `gpt-image-1`) a perfiles propios.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/models/{model_id:path}/profiles` | List profiles for a model | `user` |
| GET | `/ocabra/profiles` | List all profiles | `user` |
| POST | `/ocabra/models/{model_id:path}/profiles` | Create a profile for a model | `model_manager` |
| GET | `/ocabra/profiles/{profile_id}/runtime-presets` | List profile runtime presets | `user` |
| GET | `/ocabra/profiles/{profile_id}/runtime-presets/resolved` | Resolve profile runtime preset capacities | `user` |
| POST | `/ocabra/profiles/{profile_id}/runtime-presets/preview` | Preview a draft runtime preset's capacity | `user` |
| POST | `/ocabra/profiles/{profile_id}/runtime-presets/{preset_key}/context-probe` | Measure the largest context a runtime preset fits | `model_manager` |
| GET | `/ocabra/profiles/{profile_id}/runtime-presets/{preset_key}/context-probe` | Progress and result of the last context probe of a preset | `model_manager` |
| POST | `/ocabra/profiles/{profile_id}/runtime-presets/{preset_key}/context-probe/apply` | Apply the last context probe to the preset | `model_manager` |
| POST | `/ocabra/profiles/{profile_id}/runtime-presets/{preset_key}/benchmark` | Measure one runtime preset | `model_manager` |
| POST | `/ocabra/profiles/{profile_id}/runtime-presets` | Create profile runtime preset | `model_manager` |
| PATCH | `/ocabra/profiles/{profile_id}/runtime-presets/{preset_key}` | Update profile runtime preset | `model_manager` |
| DELETE | `/ocabra/profiles/{profile_id}/runtime-presets/{preset_key}` | Delete profile runtime preset | `model_manager` |
| POST | `/ocabra/profiles/{profile_id}/aliases` | Create hidden profile alias | `model_manager` |
| DELETE | `/ocabra/profiles/{profile_id}/aliases/{alias_id:path}` | Delete profile alias | `model_manager` |
| POST | `/ocabra/profiles` | Create a router profile | `model_manager` |
| GET | `/ocabra/profiles/{profile_id}` | Get profile detail | `user` |
| PATCH | `/ocabra/profiles/{profile_id}` | Update a profile | `model_manager` |
| POST | `/ocabra/profiles/{profile_id}/rename` | Rename a profile | `model_manager` |
| POST | `/ocabra/profiles/{profile_id}/clone` | Clone a profile | `model_manager` |
| DELETE | `/ocabra/profiles/{profile_id}` | Delete a profile | `model_manager` |
| POST | `/ocabra/profiles/{profile_id}/assets` | Upload an asset to a profile | `model_manager` |
| DELETE | `/ocabra/profiles/{profile_id}/assets/{asset_key}` | Delete a profile asset | `model_manager` |
| GET | `/ocabra/openai-tts-voice-map` | List OpenAI TTS voice-name compatibility pins | `model_manager` |
| PUT | `/ocabra/openai-tts-voice-map/{openai_model}/{voice_name}` | Pin an OpenAI (model, voice) pair to one of oCabra's own profiles | `model_manager` |
| DELETE | `/ocabra/openai-tts-voice-map/{openai_model}/{voice_name}` | Remove an OpenAI TTS voice-name compatibility pin | `model_manager` |
| GET | `/ocabra/openai-image-model-map` | List OpenAI image model-name compatibility pins | `model_manager` |
| PUT | `/ocabra/openai-image-model-map/{openai_model}` | Pin an OpenAI image model name to one of oCabra's own profiles | `model_manager` |
| DELETE | `/ocabra/openai-image-model-map/{openai_model}` | Remove an OpenAI image model-name compatibility pin | `model_manager` |

## Registro y descargas

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/registry/hf` | Search HuggingFace models | `model_manager` |
| GET | `/ocabra/registry/hf/search` | Search HuggingFace models | `model_manager` |
| GET | `/ocabra/registry/hf/{repo_id:path}/variants` | List HuggingFace model variants | `model_manager` |
| GET | `/ocabra/registry/hf/{repo_id:path}` | Get HuggingFace model detail | `model_manager` |
| GET | `/ocabra/registry/bitnet` | Search BitNet models | `model_manager` |
| GET | `/ocabra/registry/bitnet/search` | Search BitNet models | `model_manager` |
| GET | `/ocabra/registry/bitnet/{repo_id:path}/variants` | List BitNet model variants | `model_manager` |
| GET | `/ocabra/registry/ollama` | Search Ollama models | `model_manager` |
| GET | `/ocabra/registry/ollama/search` | Search Ollama models | `model_manager` |
| GET | `/ocabra/registry/ollama/{model_name}/variants` | List Ollama model variants | `model_manager` |
| GET | `/ocabra/registry/local` | List local models | `model_manager` |
| POST | `/ocabra/registry/local/register` | Register an already-downloaded model under a chosen backend | `model_manager` |
| GET | `/ocabra/downloads` | List download jobs | `user` |
| POST | `/ocabra/downloads` | Enqueue a download | `model_manager` |
| GET | `/ocabra/downloads/{job_id}` | Get download job | `model_manager` |
| DELETE | `/ocabra/downloads` | Clear download history | `model_manager` |
| DELETE | `/ocabra/downloads/{job_id}` | Cancel a download | `model_manager` |
| GET | `/ocabra/downloads/{job_id}/stream` | Stream download progress | `model_manager` |

## Backends

Instalación por backend en `/data/backends`, desde el registro OCI o desde fuente; `GET …/install` es el mismo flujo como SSE para `EventSource`.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/backends` | List modular backends | `model_manager` |
| GET | `/ocabra/backends/{backend_type}` | Get backend state | `model_manager` |
| POST | `/ocabra/backends/{backend_type}/install` | Install a backend | `model_manager` |
| GET | `/ocabra/backends/{backend_type}/install` | Install a backend (GET alias for EventSource) | `model_manager` |
| POST | `/ocabra/backends/{backend_type}/uninstall` | Uninstall a backend | `model_manager` |
| GET | `/ocabra/backends/{backend_type}/logs` | Fetch install logs | `model_manager` |
| GET | `/ocabra/backends/{backend_type}/published` | Prebuilt image on offer for a backend | `model_manager` |
| POST | `/ocabra/trtllm/compile` | Submit TRT-LLM compilation job | `model_manager` |
| GET | `/ocabra/trtllm/compile` | List TRT-LLM compilation jobs | `model_manager` |
| GET | `/ocabra/trtllm/compile/{job_id}/stream` | Stream compilation progress | `model_manager` |
| DELETE | `/ocabra/trtllm/compile/{job_id}` | Cancel compilation job | `model_manager` |
| DELETE | `/ocabra/trtllm/engines/{engine_name}` | Delete TRT-LLM engine | `model_manager` |
| GET | `/ocabra/trtllm/estimate` | Estimate TRT-LLM VRAM requirements | `user` |

## Daemon Ollama

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/ollama/version` | Get Ollama daemon version | `model_manager` |
| GET | `/ocabra/ollama/server/update` | Get Ollama server update status | `model_manager` |
| POST | `/ocabra/ollama/server/update` | Update the Ollama daemon container | `system_admin` |
| GET | `/ocabra/ollama/server/settings` | Global settings of the shared Ollama server | `user` |
| POST | `/ocabra/ollama/server/apply-settings` | Apply the global Ollama settings (recreates the daemon) | `system_admin` |

## Servicios interactivos

TRELLIS.2, ComfyUI, A1111, ACE-Step y Unsloth Studio. Los propios servicios llaman a `touch`, `runtime` y `ensure_vram` con `GATEWAY_SERVICE_TOKEN`.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/services` | List all services | `user` |
| GET | `/ocabra/services/{service_id}` | Get service state | `model_manager` |
| PATCH | `/ocabra/services/{service_id}` | Enable or disable a service | `model_manager` |
| POST | `/ocabra/services/{service_id}/refresh` | Refresh service state | `model_manager` |
| PATCH | `/ocabra/services/{service_id}/runtime` | Update service runtime state | `model_manager` |
| POST | `/ocabra/services/{service_id}/touch` | Touch service activity | `model_manager` |
| POST | `/ocabra/services/{service_id}/start` | Start a service | `model_manager` |
| GET | `/ocabra/services/{service_id}/generations` | List service generations | `model_manager` |
| POST | `/ocabra/services/{service_id}/ensure_vram` | Ensure free VRAM on a service's preferred GPU | `model_manager` |
| POST | `/ocabra/services/{service_id}/unload` | Unload service runtime | `model_manager` |

## GPUs, host, estado, sesiones y vídeo

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/gpus` | List all GPUs | `user` |
| GET | `/ocabra/gpus/{index}` | Get GPU state | `user` |
| GET | `/ocabra/gpus/{index}/stats` | Get GPU historical stats | `user` |
| GET | `/ocabra/gpus/{index}/power-limits` | Get GPU power limit constraints | `user` |
| PATCH | `/ocabra/gpus/{index}/power` | Set GPU power limit and persistence mode | `system_admin` |
| GET | `/ocabra/status` | Server load snapshot | `user` |
| GET | `/ocabra/host/stats` | Return host CPU and RAM stats. Cached for 2 seconds | `user` |
| POST | `/ocabra/estimate` | Estimate how long an inference request will take | `user` |
| GET | `/ocabra/sessions` | List live Realtime sessions | `system_admin` |
| POST | `/ocabra/sessions/{session_id}/kill` | Force-close a Realtime session | `system_admin` |
| POST | `/ocabra/video/upscale` | Upscale a video segment | `user` |

## Estadísticas

Las rutas `my*` devuelven solo lo del propio usuario; las agregadas por usuario, grupo o clave exigen `model_manager`.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/stats/requests` | Request statistics | `user` |
| GET | `/ocabra/stats/gpu` | GPU usage statistics | `user` |
| GET | `/ocabra/stats/energy` | Energy statistics | `user` |
| GET | `/ocabra/stats/performance` | Performance statistics | `model_manager` |
| GET | `/ocabra/stats/tokens` | Token statistics | `user` |
| GET | `/ocabra/stats/overview` | Overview statistics | `model_manager` |
| GET | `/ocabra/stats/recent` | Recent requests log | `user` |
| POST | `/ocabra/stats/requests/{request_id}/cancel` | Cancel an in-flight request | `user` |
| GET | `/ocabra/stats/my-recent` | Own recent requests log | `user` |
| GET | `/ocabra/stats/by-user` | Stats aggregated by user | `model_manager` |
| GET | `/ocabra/stats/by-group` | Stats aggregated by group | `model_manager` |
| GET | `/ocabra/stats/batch-vs-interactive` | Batches API vs. direct request volume | `model_manager` |
| GET | `/ocabra/stats/batch-priority-status` | Batch preemption/cooldown live status | `model_manager` |
| GET | `/ocabra/stats/my` | Own user stats | `user` |
| GET | `/ocabra/stats/my-group` | Own group stats | `user` |
| GET | `/ocabra/stats/by-api-key` | Stats aggregated by API key | `model_manager` |
| GET | `/ocabra/stats/user/{identifier}/detail` | Detailed stats for a user | `model_manager` |
| GET | `/ocabra/stats/federation` | Federation statistics | `model_manager` |
| GET | `/ocabra/stats/routing` | Router redirect distribution | `user` |
| GET | `/ocabra/stats/server-power` | Current server power readings | `user` |
| GET | `/ocabra/stats/by-agent` | Agent-aggregated statistics | `user` |
| GET | `/ocabra/stats/tool-calls` | Recent tool_call_stats rows | `user` |

## Agentes y MCP

Un agente se usa como modelo `agent/<slug>` en `/v1/chat/completions`.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/agents` | List agents accessible to the caller | `user` |
| POST | `/ocabra/agents` | Create an agent | `model_manager` |
| GET | `/ocabra/agents/{slug}` | Retrieve an agent | `user` |
| PATCH | `/ocabra/agents/{slug}` | Update an agent | `model_manager` |
| DELETE | `/ocabra/agents/{slug}` | Delete an agent | `model_manager` |
| POST | `/ocabra/agents/{slug}/test` | Dry-run an agent | `model_manager` |
| GET | `/ocabra/mcp-servers` | List MCP servers | `model_manager` |
| POST | `/ocabra/mcp-servers` | Register a new MCP server | `model_manager` |
| GET | `/ocabra/mcp-servers/{server_id}` | Retrieve an MCP server | `model_manager` |
| PATCH | `/ocabra/mcp-servers/{server_id}` | Update an MCP server | `model_manager` |
| DELETE | `/ocabra/mcp-servers/{server_id}` | Delete an MCP server | `model_manager` |
| POST | `/ocabra/mcp-servers/{server_id}/refresh` | Refresh tools/list for an MCP server | `model_manager` |
| GET | `/ocabra/mcp-servers/{server_id}/tools` | List cached tools for an MCP server | `model_manager` |
| POST | `/ocabra/mcp-servers/{server_id}/test` | Test connectivity to an MCP server | `model_manager` |
| POST | `/ocabra/mcp-servers/test-config` | Test a candidate MCP server config without persisting it | `model_manager` |

## Configuración y actualización

`PATCH /ocabra/config` guarda en `server_config`, que se aplica en cada arranque y pisa `.env`.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/config` | Get server configuration | `system_admin` |
| PATCH | `/ocabra/config` | Patch server configuration | `system_admin` |
| GET | `/ocabra/system/update` | Version, newest release and update policy | `system_admin` |
| POST | `/ocabra/system/update/check` | Look for a new release now | `system_admin` |
| POST | `/ocabra/system/update/apply` | Update this node to a release | `system_admin` |
| PATCH | `/ocabra/system/update/policy` | Set this node's update policy | `system_admin` |

## Federación

`join-requests` (POST y su `status`) son públicas para que un nodo sin cuenta pueda pedir unirse; la aceptación es siempre manual o con invitación. `…/proxy/{path}` reenvía el panel de un peer.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/federation/heartbeat` | Federation heartbeat | `user` |
| GET | `/ocabra/federation/peers` | List federation peers | `user` |
| POST | `/ocabra/federation/peers` | Add federation peer | `system_admin` |
| PATCH | `/ocabra/federation/peers/{peer_id}` | Update federation peer | `system_admin` |
| POST | `/ocabra/federation/peers/{peer_id}/rotate-token` | Rotate tunnel token | `system_admin` |
| WS | `/ocabra/federation/tunnel` | Reverse tunnel endpoint: a tunnel-mode peer dials in and serves our requests | token del túnel |
| DELETE | `/ocabra/federation/peers/{peer_id}` | Remove federation peer | `system_admin` |
| POST | `/ocabra/federation/peers/{peer_id}/test` | Test peer connection | `system_admin` |
| POST | `/ocabra/federation/peers/{peer_id}/models/{model_id:path}/load` | Remote model load | `system_admin` |
| POST | `/ocabra/federation/peers/{peer_id}/models/{model_id:path}/unload` | Remote model unload | `system_admin` |
| POST | `/ocabra/federation/peers/{peer_id}/downloads` | Remote model download | `system_admin` |
| GET | `/ocabra/federation/peers/{peer_id}/downloads/{job_id}` | Remote download status | `system_admin` |
| GET | `/ocabra/federation/peers/{peer_id}/gpus` | Remote GPU monitoring | `system_admin` |
| GET | `/ocabra/federation/node` | This node's federation identity | `system_admin` |
| POST | `/ocabra/federation/join-requests` | Ask to join this node's federation | pública |
| GET | `/ocabra/federation/join-requests/{request_id}/status` | Join request status (for the requesting node) | pública |
| GET | `/ocabra/federation/join-requests` | List join requests | `system_admin` |
| POST | `/ocabra/federation/join-requests/{request_id}/accept` | Accept a join request | `system_admin` |
| POST | `/ocabra/federation/join-requests/{request_id}/deny` | Deny a join request | `system_admin` |
| GET | `/ocabra/federation/invites` | List invites | `system_admin` |
| POST | `/ocabra/federation/invites` | Create invite | `system_admin` |
| DELETE | `/ocabra/federation/invites/{invite_id}` | Revoke invite | `system_admin` |
| POST | `/ocabra/federation/join` | Join another node's federation | `system_admin` |
| POST | `/ocabra/federation/peers/{peer_id}/join-check` | Re-check a pending join request | `system_admin` |
| GET, POST, PUT, PATCH, DELETE | `/ocabra/federation/peers/{peer_id}/proxy/{path:path}` | Remote panel proxy | `system_admin` |
| GET | `/ocabra/federation/rules` | Node rules for peers in rules mode | `system_admin` |
| PUT | `/ocabra/federation/rules` | Set node rules for peers in rules mode | `system_admin` |
| GET | `/ocabra/federation/peers/{peer_id}/rules-usage` | What a peer in rules mode has used | `system_admin` |
| GET | `/ocabra/federation/strategies` | Per-model federation strategies | `model_manager` |
| PUT | `/ocabra/federation/strategies` | Set a model's federation strategy | `model_manager` |

## Nube (RunPod)

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| GET | `/ocabra/cloud/runpod` | RunPod configuration | `system_admin` |
| PUT | `/ocabra/cloud/runpod` | Update RunPod configuration | `system_admin` |
| GET | `/ocabra/cloud/runpod/gpu-types` | RunPod GPU catalog | `system_admin` |
| GET | `/ocabra/cloud/nodes` | Cloud nodes | `system_admin` |
| POST | `/ocabra/cloud/nodes` | Launch a cloud node | `system_admin` |
| DELETE | `/ocabra/cloud/nodes/{node_id}` | Terminate a cloud node | `system_admin` |
| POST | `/ocabra/cloud/nodes/{node_id}/update-image` | Restart a cloud node on another ocabra-node image | `system_admin` |
| POST | `/ocabra/cloud/orphans/{pod_id}/terminate` | Terminate an orphan pod | `system_admin` |
| GET | `/ocabra/cloud/replicable-models` | Models a cloud node can carry | `system_admin` |
| POST | `/ocabra/cloud/nodes/{node_id}/models` | Add models to a running cloud node | `system_admin` |
| GET | `/ocabra/cloud/recipes` | Models captured from earlier cloud nodes | `system_admin` |
| POST | `/ocabra/cloud/recipes` | Add a model for cloud nodes from Hugging Face or Ollama | `system_admin` |
| PATCH | `/ocabra/cloud/recipes/{recipe_id}` | Change a model recipe | `system_admin` |
| DELETE | `/ocabra/cloud/recipes/{recipe_id}` | Forget a model recipe | `system_admin` |
| GET | `/ocabra/cloud/volumes` | Network volumes | `system_admin` |
| GET | `/ocabra/cloud/datacenters` | Data centers for volumes | `system_admin` |
| POST | `/ocabra/cloud/volumes` | Create a network volume | `system_admin` |
| PATCH | `/ocabra/cloud/volumes/{volume_id}` | Grow a network volume | `system_admin` |
| DELETE | `/ocabra/cloud/volumes/{volume_id}` | Delete a network volume | `system_admin` |

## Eventos en tiempo real

WebSocket del panel: exige la cookie `ocabra_session` y mismo origen. Tipos de evento: `docs/CONTRACTS.md` §5.

| Método | Ruta | Qué hace | Acceso |
|---|---|---|---|
| WS | `/ocabra/ws` | Real-time event stream over WebSocket | cookie de sesión |
