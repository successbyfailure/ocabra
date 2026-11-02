# oCabra — Roadmap

Última actualización: 2026-10-30

Fuente de verdad del trabajo pendiente. `docs/PLAN.md` documenta la arquitectura y las
fases completadas. `docs/REFACTOR_PLAN.md` recoge el estado del refactor (cerrado).

---

## Estado general

El sistema está en producción. Todas las fases del plan original (0–5) están completadas.
Los backends validados end-to-end son: `vllm`, `llama_cpp`, `sglang`, `tensorrt_llm`,
`ollama`, `diffusers`, `whisper`, `tts`, `bitnet`, `acestep`.
La UI de compilación de engines TRT-LLM también está implementada y cableada.

---

## ✅ Bloque 1 — Auth, usuarios y grupos (COMPLETADO)

Plan completo en `docs/tasks/auth-system-plan.md`.

### Implementado

- 3 roles: `user`, `model_manager`, `system_admin`
- Sesión JWT (cookie HTTP-only) para el dashboard
- API keys por usuario (`sk-ocabra-...`) para OpenAI y Ollama
- Modo sin key configurable por separado (anonymous → solo grupo default)
- Grupos de acceso a modelos con selector por backend en la UI
- Página de API Keys en el sidebar
- Auth interna reemplaza LiteLLM como capa de autenticación
- Gateway de servicios con directorio autenticado

---

## ✅ Bloque 2 — Fixes pendientes (COMPLETADO)

- [x] **[B6]** `ModelPatch` ya usa `model_dump(exclude_unset=True)` — permite resetear `preferred_gpu` a null
- [x] **[A4]** `_watch_and_reload` ya tiene `while not self._stopped` — check de shutdown implementado
- [x] **[M7]** `HFModelVariant.backend_type` Literal incluye los 11 backends
- [x] **[WS]** `system_alert` WebSocket event — emitido desde GPU manager (temperatura) y model manager (fallos de carga)

---

## ✅ Bloque 3 — Persistencia de settings en base de datos (COMPLETADO)

Implementado: tabla `server_config`, overrides cargados al arranque, `PATCH /ocabra/config` persiste en BD.

---

## ✅ Bloque 4 — Integración Langfuse (COMPLETADO)

**Opcional, desactivado por defecto.** Plan en `docs/tasks/langfuse-integration-plan.md`.

Implementado:
- `backend/ocabra/integrations/langfuse_tracer.py` — singleton, `trace_generation()`, `wrap_stream()`, `parse_sse_chunk()`
- Hook en `WorkerPool.forward_request` y `forward_stream`
- Settings `langfuse_*` en `config.py`, variables en `docker-compose.yml` y `.env.example`
- Dependencia `langfuse>=2.0` en `pyproject.toml`
- Tests: 11 casos en `backend/tests/integrations/test_langfuse_tracer.py`

---

## ✅ Bloque 5 — Cobertura de tests (COMPLETADO)

109 tests cubriendo paths críticos y flujos e2e:

| Test file | Función | Tests |
|-----------|---------|-------|
| `test_path_traversal.py` | `_delete_model_files()`, `_is_path_within_base()`, TRT-LLM engine deletion | 14 |
| `test_config_patch.py` | `patch_config()` schema + runtime | 12 |
| `test_model_manager_config.py` | `update_config()` whitelist | 17 |
| `test_worker_lifecycle.py` | Port lifecycle, worker registration, forward errors, backends | 15 |
| `test_langfuse_tracer.py` | Langfuse integration (disabled, streaming, content, sampling) | 11 |
| `test_load_unload_e2e.py` | Load/unload cycle, eviction, auto-reload, idle, policies | 17 |
| `test_trtllm_compile.py` | Compile jobs, Docker cmd, phases, cancel, failure handling | 23 |

---

## ✅ Bloque 6 — Operativo / deuda técnica menor (COMPLETADO)

- [x] Limpiar entrada `tensorrt_llm/Qwen3-32B-AWQ-fp16` del inventario (SQL directo)
- [x] Documentación OpenAPI enriquecida — summaries, descriptions y responses en todos los endpoints `/ocabra/*`
- [x] Script first-run (`setup.sh`) + guía de instalación (`docs/INSTALL.md`)
- [ ] Validación TRT-LLM con múltiples engines en producción (requiere prueba manual con hardware)

---

## ✅ Bloque 7 — Teams, stats ampliadas, admin UX (COMPLETADO)

Implementado:
- 7.1 `group_id` en ApiKeys y RequestStats (migración `0009`, collector, UserContext)
- 7.2 Admin crea keys para otros usuarios (`POST /ocabra/users/{user_id}/keys` + modal UI)
- 7.3 Endpoints de stats: `recent`, `by-user`, `by-group`, `my`, `my-group`
- 7.4 Settings en tabs (Radix tabs: General, GPUs, Backends, Almacenamiento, LiteLLM)
- 7.5 Stats en tabs (7 tabs con filtro por rol)
- 7.6 Dashboard log de últimas 20 peticiones (poll 30s)

---

## ✅ Bloque 8 — Voice Pipeline (COMPLETADO)

**Plan completo:** `docs/tasks/voice-pipeline-plan.md`

### Fase 1 — Tres endpoints oficiales funcionando correctamente (✅ COMPLETADA)

| Item | Archivo | Estado |
|------|---------|--------|
| [VP-1] TTS encoding real | `backend/workers/tts_worker.py` | ✅ MP3/WAV/PCM/FLAC via soundfile+ffmpeg |
| [VP-2] TTS streaming por frases | `backend/workers/tts_worker.py` + `api/openai/audio.py` | ✅ `/synthesize/stream` |
| [VP-3] STT verificar M4A | `backend/workers/whisper_worker.py` | ✅ faster-whisper acepta M4A |
| [VP-4] Voices endpoint | `backend/ocabra/api/openai/audio.py` | ✅ `GET /v1/audio/voices?model=` |

### Fase 1.5 — Voxtral TTS backend (vllm-omni) (✅ COMPLETADA)

| Item | Archivo | Estado |
|------|---------|--------|
| [VP-9] Voxtral worker | `backend/workers/voxtral_worker.py` | ✅ Wrapper FastAPI + vllm-omni subprocess |
| [VP-10] Voxtral backend | `backend/ocabra/backends/voxtral_backend.py` | ✅ BackendInterface |
| [VP-11] Registro backend | `backend/ocabra/main.py` + `config.py` | ✅ `voxtral_python_bin` |
| [VP-12] Instalar vllm-omni | `/opt/voxtral-venv/` vllm==0.18.0 + vllm-omni==0.18.0 | ✅ Instalado y testeado |

### Fase 2 — OpenAI Realtime API (`GET /v1/realtime`) (✅ COMPLETADA)

| Item | Archivo | Estado |
|------|---------|--------|
| [VP-5] Endpoint WebSocket | `backend/ocabra/api/openai/realtime.py` | ✅ WS `/v1/realtime?model=` con auth |
| [VP-6] RealtimeSession | `backend/ocabra/core/realtime_session.py` | ✅ Pipeline STT→LLM→TTS, protocolo completo |
| [VP-7] VAD servidor | `backend/ocabra/core/vad.py` | ✅ SimpleVAD por RMS |
| [VP-8] session.update | En `RealtimeSession` | ✅ Configuración de voz, modelo STT, VAD params |

---

## ✅ Bloque 9 — Model Profiles (Fase 6) (COMPLETADO)

Plan en `docs/tasks/model-profiles-chatterbox-swarm-plan.md`.

Implementado:
- `ModelProfile` DB + migración `0010`, `profile_registry.py` con CRUD y cache
- Endpoints REST `/ocabra/profiles/*` con upload de assets
- `resolve_profile()` en `/v1/*` y `/api/*` con fallback legacy configurable
- Worker sharing por `(base_model_id, load_overrides_hash)`
- UI de perfiles en Models (`ProfileModal.tsx`)
- Validación de `voice_ref` restringida a paths controlados
- Detección correcta de backends en local scanner

---

## ✅ Bloque 10 — Chatterbox TTS (Fase 7 parcial) (COMPLETADO)

Plan en `docs/tasks/model-profiles-chatterbox-swarm-plan.md`.

Implementado:
- `chatterbox_backend.py` — BackendInterface completo (304 líneas)
- `chatterbox_worker.py` — Worker FastAPI con synthesize/stream/voices (610 líneas)
- Registro en `main.py`, config con `chatterbox_python_bin`
- Detección en HuggingFace registry y local scanner
- Voice cloning via `voice_ref` con path traversal protection
- 28 idiomas, mapeo de voces OpenAI, formatos múltiples

---

## ✅ Bloque 11 — Resiliencia de backends (COMPLETADO)

Plan en `docs/tasks/backend-resilience-plan.md`.

### 11.1 — Evicción LRU + umbral de VRAM ✅
- `vram_eviction_threshold` en config, `_get_eviction_candidates()`, `_evict_for_space()`
- Pre-load VRAM check con evicción preventiva, `_vram_watchdog()` background loop
- 7 tests

### 11.2 — Aislamiento por proceso ✅
- `BackendProcessManager` con health loop, detección de PID muerto, auto-restart con backoff exponencial
- Settings: `worker_health_check_interval_seconds`, `auto_restart_workers`, `max_worker_restarts`
- Integrado en `main.py` lifespan. 7 tests

### 11.3 — Interfaz unificada multi-modal ✅
- `ModalityType` enum, métodos opcionales tipados en `BackendInterface`
- `supported_modalities()` en los 13 backends
- Helpers `get_backends_for_modality()` y `supports_modality()` en WorkerPool. 17 tests

### 11.4 — Busy timeout / health watchdog ✅
- `ActiveRequest` dataclass, `begin_request()` retorna request_id
- `_busy_watchdog()` loop, `_timeout_counts` metrics
- `busy_timeout_seconds` y `busy_timeout_action` configurables. 10 tests
- Alias/perfiles se resuelven a la clave canónica del worker antes de contabilizar
  peticiones activas; idle/pressure eviction ya no puede descargar un modelo que
  sigue atendiendo una petición.
- `inference_request_timeout_seconds` separa el timeout HTTP del watchdog:
  cancela el upstream y devuelve `504 generation_timeout` (también en batches);
  el timeout efectivo del watchdog y del batch processor siempre queda por encima.

---

## ✅ Bloque 12 — Federación P2P (COMPLETADO)

Plan en `docs/tasks/federation-plan.md`.

Implementado (5 fases):
- **Fase 1**: `FederationManager` con heartbeat, DB `federation_peers`, cifrado Fernet, CRUD endpoints, circuit breaker
- **Fase 2**: Proxy transparente (`proxy_request`/`proxy_stream`), `resolve_federated()`, load balancing con bias local, hooks en todos los endpoints OpenAI y Ollama
- **Fase 3**: Inventario federado en `/v1/models`, `/api/tags`, `/ocabra/models` con deduplicación
- **Fase 4**: UI completa — tab Federation en Settings, badges remotos en Models, panel en Dashboard, WebSocket events
- **Fase 5**: Operaciones remotas (load/unload/download/GPUs) para peers con `access_level=full`
- 54 tests de federación

---

## ✅ Bloque 13 — Observabilidad de potencia + stats ampliadas (COMPLETADO)

Implementado:
- Servicio `hw-monitor/` (contenedor dedicado) lee potencia CPU por RAPL (`/sys/class/powercap`) y GPU vía NVML, publica en Redis y persiste en `server_stats` (migración `0015`). GPU Manager consume snapshots de Redis con fallback a pynvml.
- Agregador de stats ampliado con `cost_calculator.py` (coste estimado por petición), endpoints `/ocabra/stats/by-api-key`, `/ocabra/stats/server-power`, `/ocabra/stats/federation`, detalle por usuario.
- Nuevos paneles frontend: `ApiKeyPanel`, `CostSavingsCard`, `FederationPanel`, `UserDetailPanel`, `EnergyPanel` rediseñado. Layout (Header/Sidebar/Layout) y página Stats rehechos.
- Componentes comunes reutilizables: `ConfirmDialog`, `Skeleton`, `StyledSelect`.
- Gateway con portal de servicios y páginas mejoradas.
- Benchmark harness en `benchmark/run_benchmark.py`.

---

## ✅ Bloque 14 — OpenAI Batches + Files API + ACL de modelos (COMPLETADO)

Implementado:
- **OpenAI Files API** (`/v1/files` · `POST`/`GET`/`DELETE`, `GET /v1/files/{id}/content`) con almacenamiento en disco (`/data/openai_files`, volumen `./data/openai_files`). Tabla `openai_files`.
- **OpenAI Batches API** (`/v1/batches` · `POST`/`GET`/`cancel`) con tabla `openai_batches` y migración `0016`. Soporta `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/images/generations`, `/v1/audio/speech` (añadidos 2026-10-30). Respuestas no-JSON (TTS) se guardan como fichero aparte (`purpose="batch_artifact"`) referenciado por `file_id` en vez de ir inline. `/v1/audio/transcriptions` también soportado (2026-10-30) pero como **extensión propia fuera del estándar OpenAI**: la línea referencia un `file_id` pre-subido (`purpose="batch_input_audio"`) en vez de llevar el audio inline, y `batch_processor` reconstruye la petición multipart real en tiempo de despacho — necesario porque el endpoint real recibe `multipart/form-data`, incompatible con una línea JSON. `/v1/images/edits` se queda fuera por el mismo motivo, sin extensión equivalente todavía.
- **Batches — `/v1/pooling`, `/v1/score`, `/v1/rerank`, `/v1/classify`, `/v1/audio/generate`** (2026-11-02): añadidos a `ALLOWED_ENDPOINTS`. Los cinco toman JSON puro de entrada y devuelven JSON o binario (música, igual que TTS) — encajan sin cambios en el `_dispatch()` genérico de `batch_processor.py`, a diferencia de transcripciones. Motivado por una auditoría de documentación que encontró que la matriz de soporte real no coincidía con lo documentado.
- **Batches — `/v1/images/edits`** (2026-11-02): ya no se queda fuera. Añadida la misma extensión oCabra que `/v1/audio/transcriptions` (referencia a fichero pre-subido en vez de binario inline): `image_file_id` (requerido) + `mask_file_id`/`image_ref_{1,2,3}_file_id` (opcionales), todos con `purpose="batch_input_image"` vía `POST /v1/files`. `batch_processor` reconstruye el multipart real (hasta 5 ficheros) en tiempo de despacho.
- **OpenAPI/Swagger discoverable, no solo prosa** (2026-11-02), a raíz de feedback de un agente externo tras la revisión de Batches: `BatchCreateBody.endpoint` y `purpose` (`POST /v1/files`) ahora llevan `enum` real en el schema en vez de ser un `string` cualquiera. Nuevos schemas `PoolingRequest`/`ScoreRequest`/`ClassifyRequest` (antes esos 3 endpoints no tenían ningún requestBody documentado) y `TranscriptionBatchLineBody`/`ImageEditsBatchLineBody` (describen el `body` de una línea de Batches con las extensiones oCabra; inyectados en `components.schemas` aunque no cuelgan de ningún requestBody real — su "petición" vive dentro de un fichero, no en el HTTP body). De paso, `MusicGenerationRequest` tenía un campo `duration_seconds` que nunca existió en el endpoint real (es `audio_duration`); corregido y completado con el resto de parámetros de ACE-Step. La creación de un batch ahora también valida la forma completa de cada línea de transcripción/edición de imagen (tipos, campos requeridos como `model`/`prompt`) contra esos mismos schemas, no solo la presencia del file_id.
- **BatchProcessor** (`backend/ocabra/core/batch_processor.py`): loop en background, dispatch in-process vía `httpx.ASGITransport`, impersonación del owner con `X-Gateway-Token` + `X-Internal-User-Id` (añadido soporte en `_deps_auth.py`). Concurrencia configurable (`batch_max_concurrency=4`), poll cada `batch_poll_interval_seconds=5`; el timeout efectivo por petición es `max(batch_request_timeout_seconds, inference_request_timeout_seconds + 30)`.
- **Evicción por presión ya no se queda atascada en `preferred_gpu`** (2026-11-02, `model_manager.py::_assign_gpus_for_load`), reportado en vivo: `qwen3-embedding:8b` (`preferred_gpu=0`, la 3060) falló con VRAM insuficiente aunque la 3090 tenía a `gemma4:26b` (soft-pin, `pin_permanent=False`) con 22GB evictables, porque la cascada de evicción solo miraba candidatos residentes en `preferred_gpu`. `preferred_gpu` es una **preferencia de orden**, no una restricción — `find_gpu_for_model` ya recorría todas las GPUs en el intento inicial; la evicción no lo hacía. Fix: prueba el pool de candidatos de cada GPU en orden de preferencia (`preferred_gpu` primero, luego el resto) hasta que uno libere suficiente, sin romper la propiedad que ya protegía el incidente del 2026-10-31 (nunca evictar en la GPU A para liberar la GPU B — cada intento sigue acotado a una sola GPU). Seguía sin evictar `gemma4:26b` si ningún pool tiene hueco realista (chequeo de factibilidad ahora también acotado a la GPU del pool, no sumado entre todas). Cubierto por `test_pressure_eviction_falls_back_to_other_gpu_pool` en `tests/test_model_manager.py`.
- **ACL: 5 modelos llama.cpp migrados desde Ollama sin su entrada nueva en ningún grupo** (2026-11-02): `mistral:7b`, `qwen3:8b`, `qwen3:32b`, `qwen3-embedding:0.6b`, `qwen3-embedding:4b` tenían la versión Ollama en `group_models` pero no la versión `llama_cpp/...` tras la migración — invisibles para cualquier usuario no-admin. Añadidas a `default`/`Makers`/`infra` vía `POST /ocabra/groups/{id}/models`, junto con `qwen3.8:27b-abliterated` (que solo estaba en `default`, inconsistente con el resto de modelos llama.cpp presentes en los 3 grupos). `group_models` es texto libre sin FK contra `model_configs`/`model_profiles` — vale la pena recordar añadir la entrada nueva explícitamente cada vez que se migra un modelo de backend, el sistema no lo hace solo.
- **Fix ACL `/ocabra/models`**: ahora filtra por `accessible_model_ids` para usuarios no-admin (igual que `/v1/models`). Verificado con usuario de grupo limitado: 1 modelo visible vs. todos antes.
- Smoke test e2e: upload JSONL → create batch → procesado completo con output JSONL descargable vía Files API.

---

## 🚧 Bloque 15 — Backends Modulares (Fase 2 cerrada al 10/11; pendiente Fase 3 OCI)

Plan en `docs/tasks/modular-backends-plan.md`.

Cada backend instalable/desinstalable en runtime desde la UI. Imagen Docker slim + distribución OCI.
5 fases: infraestructura → migrar backends → OCI → imagen slim → frontend.

### MVP mergeado (2026-04-24)

Equipo de agentes paralelos entregó Fases 1, 3 (draft) y 5 en el mismo día:

| Fase | Estado | Entregables |
|------|--------|-------------|
| **Fase 1 — Infra backend** | ✅ Merged | `BackendInstallSpec` dataclass, `BackendInstaller` con source install/uninstall/scan, router `/ocabra/backends` (POST+GET SSE, uninstall, logs), volumen `backends_data`, 13 tests en verde. Commits `77428d7`→`27d4b3e` + `2a8e65d` (GET alias para EventSource). |
| **Fase 3 — Dockerfiles (draft)** | ✅ Merged | 11 Dockerfiles OCI en `backends/dockerfiles/` con `ARG BASE_IMAGE` multi-variante (cuda12/cpu/rocm), `FROM scratch` final, OCI labels, README con CI matrix skeleton. |
| **Fase 5 — Frontend** | ✅ Merged | Página `/backends` (rol `system_admin`), `BackendCard` con 6 estados, `BackendStatusBadge`, `InstallProgressBar` SSE, `ConfirmUninstallDialog`, Zustand store con mock fallback, WebSocket live updates, `api.backends.*` en cliente, entrada en Sidebar. |

### Avance acumulado

- ✅ **Fase 1 — Infra backend**: `BackendInstallSpec`, `BackendInstaller`, router `/ocabra/backends`, SSE de instalación, tests.
- ✅ **Fase 4 — Imagen slim**: `Dockerfile.slim` (~987 MB) + `docker-compose.yml` por defecto a slim, `docker-compose.fat.yml` como override de rollback, `BACKENDS_FAT_IMAGE` flag.
- ✅ **Fase 5 — Frontend**: página `/backends` + cards + SSE + WebSocket events.
- ✅ **Fase 2 — 10 de 11 backends migrados a `install_spec`** (tensorrt_llm diferido a Fase 3 OCI):
  - Python-heavy (validados install end-to-end + algunos `load()`): `whisper`, `tts`, `diffusers`, `chatterbox`, `sglang`, `voxtral`, `vllm`.
  - Nativos (Ronda 1 cierre 2026-04-25, install runtime pendiente de validar): `llama_cpp` (cmake CUDA), `bitnet` (cmake CPU/CUDA), `acestep` (git + uv sync).
  - `BackendInstallSpec` ya soporta `apt_packages`, `git_repo`/`git_ref`/`git_recursive`, `post_install_script`, `extra_bins` (Deuda 9g cerrada).

### Pendiente del bloque 15

- **Validación runtime de los 3 nativos sobre slim** ✅ (2026-04-25): llama_cpp + bitnet validados load + chat OK; acestep instalado, generación pendiente de descarga de pesos. Durante la sesión también se cerraron D11 (vllm gcc), D13 (voxtral pin) y D14 (LD_LIBRARY_PATH genérico). D12 sglang queda parcial: el helper `venv_cuda_home` funciona pero `sgl-kernel` JIT requiere `nvcc` real (apt `cuda-nvcc-12-4` desde repo NVIDIA, ~2 GB).
- **diffusers e2e** ✅ (2026-04-25): worker extendido con `from_single_file()`; carga el `sd_xl_base_1.0.safetensors` ya presente y genera 1024×1024 PNG en ~117 s. Spec ahora pinea `transformers<5.0` para evitar el crash `CLIPTextModel.text_model`.
- **Federation hook + profile_id** ✅ (2026-04-25): `resolve_federated()` ahora resuelve `profile_id` → `base_model_id` antes de buscar local/remoto. Cableado en chat/images/audio/embeddings/completions. Validado e2e con `qwen2.5-0.5b-gguf` (chat) y `sdxl-base` (images).
- **acestep weights** ✅ (2026-04-25): bind-mount `/docker/ACE-Step-1.5/checkpoints` en el contenedor api evita re-descarga (~11 GB). Generación todavía bloqueada por `torchcodec/libavutil.so.56` mismatch en slim (deuda menor).
- **Fase 3 — CI pipeline + implementar `method="oci"`**: el endpoint devuelve `501` hasta que las imágenes OCI estén publicadas en `ghcr.io/ocabra/backend-*`. Resolver incógnitas: TensorRT-LLM (NGC vs wheel — bloquea su `install_spec`), ACE-Step pin, variantes CPU con torch CPU index, runners CI (disk cleanup o self-hosted).
- **Deudas Ronda 2 ✅** (cerradas 2026-04-25): `_derive_version` ya prefiere el paquete del backend (9h); `WorkerPool.registered_backends()` ya era público (#1); mock fallback frontend solo con flag explícita (#4); nuevo `BackendBadge` en Models con estado/tooltip (#5); `_run_pip_install` parsea stdout y emite SSE por wheel (#8).
- **Deuda #6 ruff sweep** (247 errores en `api/internal/`): pendiente, candidato a PR aparte para no enturbiar diffs.

---

## 🚧 Bloque 16 — Unsloth Studio (training/fine-tuning como servicio)

Capacidad nueva: fine-tuning de modelos abiertos sin código, integrado como un
servicio más con su propia UI (mismo patrón que ACE-Step / ComfyUI / A1111).
Cubre LoRA/QLoRA, FP8/FFT, datasets desde PDF/CSV/JSON/DOCX/TXT, chat local con
GGUF/safetensors, model arena y export a GGUF + 16-bit safetensors.

### Decisiones de arquitectura

- **Encaje como `ServiceState`, no como nuevo backend de inferencia.** Unsloth
  es principalmente librería de training; vLLM/SGLang ya cubren inferencia
  mejor. La imagen oficial `unsloth/unsloth` trae `unsloth studio` (UI Jupyter)
  en `:8888`, encaja con `ui_url` + iframe vía gateway.
- **GPU preferida 1** (la grande, RTX 3090 24G) — training es VRAM-hungry.
- **Idle unload deshabilitado por defecto** (`UNSLOTH_IDLE_UNLOAD_SECONDS=0`):
  un job de fine-tuning puede durar horas, evicción automática rompería la
  ejecución. `generation_grace_period_s=-1` complementa esto.
- **Salida = entrada del registry**: los GGUF exportados a
  `/docker/ai-models/ocabra/unsloth/exports/` quedan disponibles para que el
  scanner local + llama.cpp/vLLM los sirvan sin pasos manuales.
- **Licencia**: core Apache-2.0, UI AGPL-3.0 — irrelevante en self-host del
  usuario; revisar si en algún momento se distribuye la imagen como producto.

### Implementado (2026-04-25)

| Item | Archivo | Estado |
|------|---------|--------|
| Servicio Compose `unsloth-studio` | `docker-compose.yml` | ✅ Imagen `unsloth/unsloth`, GPU 1, volumen `work/` + `exports/`, healthcheck, profile `services` |
| Settings `unsloth_*` | `backend/ocabra/config.py` | ✅ `unsloth_base_url`, `unsloth_ui_url`, `unsloth_preferred_gpu`, `unsloth_idle_unload_seconds`, `unsloth_docker_container`, `unsloth_generation_grace_period_s` |
| Registro `ServiceState` | `backend/ocabra/core/service_manager.py` | ✅ `service_id="unsloth"`, `runtime_loaded_when_alive=True`, gestión de ciclo de vida idéntica al resto |
| Gateway routing | `gateway/config.py`, `gateway/pages.py` | ✅ Entrada en `_SERVICE_DEFS` + descripción en el portal |
| Variables de entorno | `.env.example` | ✅ Bloque `UNSLOTH_*` documentado |

### Pendiente

- **Validación e2e**: levantar `docker compose --profile services up -d unsloth-studio`, hacer un fine-tune corto LoRA sobre un modelo pequeño (Llama 3.2 1B / Qwen 0.5B), exportar GGUF y comprobar que el scanner local lo recoge.
- **Hot-import del adapter** desde Studio a vLLM sin reiniciar workers (depende de soporte vLLM para LoRA dinámicos — ya existe en versiones recientes).
- **Exposición opcional del puerto 8000** de Studio (endpoint OpenAI-compat interno) como backend de inferencia adicional para servir adapters sin re-cargar.
- **Watcher sobre `unsloth/exports/`** que auto-registre los GGUF/safetensors resultantes en el inventario de modelos.
- **Job tracking en oCabra**: panel mínimo en frontend que muestre training jobs activos leyendo de Studio (vía Jupyter API) o sólo enlaces directos a la UI — decidir tras validación e2e.

---

## ✅ Bloque 17 — llama.cpp loader parity con LM Studio (entregado 2026-05-05)

Plan completo en `docs/tasks/llama-cpp-parity-plan.md`. Schema compartido en
`backend/ocabra/schemas/backend_load.py` (`LlamaCppLoadConfig` + `SpeculativeConfig`).

Cobertura ahora: 100% de los flags que LM Studio expone (15+ campos) +
estimador VRAM determinístico desde header GGUF + multi-GPU granular +
speculative decoding + selector de runtime alterno. Entregado por equipo de 4
agentes en paralelo en worktrees (commits b5c7cd1, 23cb65a, 553abbb, 7a24163,
cb3f61b, 727987a).

### Pendiente del bloque 17

- **Tests speculative pendientes de container rebuild**: 3 tests en
  `test_llama_cpp_speculative.py` requieren `fastapi` (no está en el host
  Python). Funcionan en contenedor tras `docker compose build api`. Los
  66 tests restantes pasan en host (Sprint 17.1-17.4 + estimator + e2e).

### Sprint 17.1 — Tier 1: flags triviales en UI per-model

| Item | Detalle |
|------|---------|
| Mover per-model: `threads`, `batch_size`, `ubatch_size`, `mlock` | ya en backend, falta exposición UI |
| Nuevos: `mmap` (opt-out), `seed`, `no_kv_offload`, `rope_freq_base`, `rope_freq_scale` | extender `_build_options()` + args worker + UI |
| Crear `LlamaCppLoadConfig` Pydantic | base para Sprints 17.2-17.4 |

### Sprint 17.2 — KV-quant + estimación VRAM en vivo

| Item | Detalle |
|------|---------|
| `cache_type_k`, `cache_type_v` | enum `f16/q8_0/q5_1/q5_0/q4_1/q4_0` con gating `flash_attn` |
| Estimador determinístico VRAM | parsear header GGUF (n_layers, kv_heads, head_dim) + KV-quant + n_ctx |
| UI live update | recálculo on-change reusando `POST /memory-estimate` |

### Sprint 17.3 — Multi-GPU + MoE CPU offload

| Item | Detalle |
|------|---------|
| `main_gpu`, `tensor_split: list[float]`, `split_mode` | flags `--main-gpu`/`--tensor-split`/`--split-mode` |
| `n_cpu_moe`, `override_tensor` | `--n-cpu-moe N` + `--override-tensor "exps=CPU"` |
| `disabled_gpus`, `split_strategy` | composición sobre `CUDA_VISIBLE_DEVICES` actual |
| UI multi-GPU | selector con drag-and-drop de ratios y enable/disable per-GPU |

### Sprint 17.4 — Speculative decoding + runtime alterno + concurrent slots

| Item | Detalle |
|------|---------|
| `speculative.{draft_model_id, draft_n, draft_min, draft_p_min}` | flags `--model-draft`/`--draft-*` |
| Validación tokenizer compatible | indexar `vocab_size` + `bos/eos` en `local_scanner.py` |
| `runtime: cuda\|rocm\|vulkan\|cpu` | múltiples binarios `llama-server` por arch |
| `parallel_slots`, `cont_batching` | flags `--parallel N` + `--cont-batching` |
| `keep_alive_seconds` per-model | override sobre idle_eviction global |

---

## ✅ Bloque 18 — Fiabilidad de peticiones y backends ternarios

Abierto el 2026-08-23 tras revisar las peticiones reales de producción y los
cambios recientes de Bonsai/VibeASR. El objetivo es reducir fallos visibles al
cliente sin reabrir todavía la estrategia global de scheduling.

### 18.1 — Realtime STT y ciclo de vida

- [x] Contabilizar las transcripciones internas de Realtime como actividad del
  modelo mediante `begin_request`/`end_request`.
- [x] Actualizar `last_request_at` y exponer el origen `realtime_stt` en
  `/ocabra/models/activity`.
- [x] Esperar de forma acotada si el worker STT está en `loading` o `unloading`,
  en vez de intentar conectar a un puerto que está desapareciendo.
- [x] Añadir tests que impidan eviction idle/pressure mientras una transcripción
  Realtime está activa.

Motivo operativo: una sesión Realtime activa no aparecía en `request_stats` ni
actualizaba actividad. Whisper parecía ocioso, se descargaba y se recargaba al
siguiente fragmento, compitiendo con RavenX por VRAM.

### 18.2 — Transiciones de carga/descarga

- [x] Las peticiones que llegan durante `UNLOADING` deben esperar a que termine
  la descarga y activar la recarga bajo demanda.
- [x] Mantener el timeout acotado y devolver `503` con un código estable si la
  transición no converge; nunca convertir este estado transitorio en `500`.
- [x] Cubrir concurrencia entre eviction por presión y una nueva petición.

### 18.3 — IDs, perfiles y errores accionables

- [x] Aceptar siempre un `model_id` canónico registrado y resolverlo al perfil
  enabled/default correspondiente, conservando `profile_id` como ID público
  recomendado.
- [x] Generar un slug estable alternativo cuando dos perfiles por defecto
  colisionen, en vez de dejar el segundo modelo invisible.
- [x] Conservar en estadísticas el detalle estructurado de errores HTTP locales
  y upstream, no solo `HTTP 404`.
- [x] Completar `backend_type` cuando la petición utiliza un `profile_id`.

Motivo operativo: 12 peticiones de imagen usaron
`diffusers/flux2-klein-4b` mientras el perfil público era `flux2-klein`; las 12
terminaron en 404 sin detalle suficiente para distinguir resolución local,
federación o worker.

### 18.4 — Correcciones Bonsai, VibeASR y calidad

- [x] Unificar la resolución de `gpu_layers` de Bonsai entre scheduler y backend;
  el scheduler no puede tratarlo como CPU-only mientras PrismML usa 99 capas.
- [x] Estimar VRAM de Bonsai desde el GGUF, contexto y KV cuantizado, eliminando
  el fallback fijo de 400 MB para este modelo.
- [x] Resolver VibeASR desde el layout real
  `models/huggingface/<repo-aplanado>` y garantizar que VAE+LM proceden del mismo
  directorio.
- [x] Convertir los pesos ficticios de tests a ficheros sparse/mocks; la suite
  actual puede ocupar ~45 GiB de `/tmp`.
- [x] Corregir aserciones inertes, variables ausentes en `.env.example` y lint.

Correcciones adicionales entregadas: limpieza del sufijo interno de worker al
resolver repositorios vLLM, override explícito de `chat_template` en llama.cpp y
rehidratación estable del formulario de modelos ante broadcasts de GPU.

### Decisiones aplazadas

- **Estrategia global de recursos, concurrencia y timeouts**: revisar en un
  bloque separado prioridades, cooldowns, anti-thrashing, admission control,
  reparto heterogéneo de GPU y semántica de timeouts. No introducir cambios
  parciales de política en este bloque.
- **OpenAI Responses API (`/v1/responses`)**: evaluar compatibilidad, alcance y
  traducción a Chat Completions antes de comprometer contrato. Las peticiones
  actuales siguen recibiendo 404 de forma conocida.

---

## ✅ Bloque 19 — Observabilidad atribuible de OpenAI Realtime (COMPLETADO)

- [x] Resolver y conservar `UserContext` completo durante el handshake
  WebSocket, sin registrar tokens ni cookies.
- [x] Registrar apertura/cierre de sesión y cada operación STT, LLM y TTS en
  `request_stats`, diferenciadas mediante `request_kind`.
- [x] Añadir logs estructurados con `session_id`, usuario, etiqueta de API key,
  IP, modelo, duración, estado y error.
- [x] Cubrir autenticación por API key, cookie y anonymous, así como estadísticas
  de éxito/error sin persistir contenido de audio o conversación.

Validación dirigida: 30 tests de Realtime y del colector de estadísticas en
verde. La suite backend global sigue requiriendo el entorno completo descrito en
`docs/DOC_AUDIT.md`.

---

## 🚧 Bloque 20 — Router profiles, duration estimator y session registry

Objetivo: que oCabra decida automáticamente qué modelo sirve mejor una petición
según disponibilidad de GPU, ETA estimada, sesiones interactivas activas y
fallback por familia. Etiquetas virtuales tipo `gemma4:26b` que resuelven a
vLLM primary → 12B fallback → Ollama, con transparencia total en UI.

### Diseño consensuado

Cuatro componentes acoplados pero independientes:

- **DurationEstimator** — predice segundos totales y restantes por request; se
  calibra en vivo con `request_stats`. Percentiles como baseline, regresión
  lineal para LLM/chat, fórmulas específicas por familia (whisper, diffusers,
  flashvsr, tts, acestep, upscaler). Confidence 0..1 con `router_confidence_
  floor = 0.3`.
- **SessionRegistry** — registry en memoria + Redis (TTL 5 min heartbeat) de
  sesiones Realtime activas y qué workers reservan. Veta evict/router; auto-
  kill de zombies tras `session_max_idle_s = 900`.
- **RouterResolver** — un `profile_id` puede tener `routing_targets` (lista
  ordenada de perfiles). Elige el primero disponible según session veto,
  loaded state, evict impact y ETA comparada. Loops detectados.
- **Grace dinámico en `pressure_eviction`** — `drain_timeout` calculado por
  `estimator.in_flight_remaining()` + margen, con techo `max_drain_timeout_s
  = 900`. Session veto lo cortocircuita a -1 (nunca evict).

### Cambios de datos

- `model_profiles.routing_targets` (jsonb, nullable) — lista ordenada de
  profile_ids.
- `request_stats.work_size_meta` (jsonb, nullable) — tamaño de trabajo por
  familia (audio_seconds, frames, steps, etc.) para estimadores no-LLM.
- `request_stats.via_router_profile_id` (varchar 512, nullable, indexed) —
  qué router redirigió la petición, para audit y Stats/Routing.

### Endpoints nuevos

- `POST /ocabra/estimate` — devuelve `{expected_s, p50, p95, confidence,
  source, sample_count, cold_start_s}`. Rol `user`.
- `GET /ocabra/sessions` — sesiones Realtime activas. Rol `system_admin`.
- `POST /ocabra/sessions/{id}/kill` — matar sesión zombie. Rol `system_admin`.
- `GET /ocabra/stats/routing` — decisiones por router. Rol `user`.
- `PATCH /ocabra/profiles/{id}` — acepta `routing_targets` en el body.

### UI

- **Models page**: columna `Kind` (Model/Router), sub-fila expandible con
  targets ordenados y estado en vivo, chip `RESERVED-BY-SESSION`.
- **ModelConfigModal**: toggle "This profile is a Router" con drag&drop de
  targets y preview de decisión en vivo.
- **Playground**: chip ETA antes de enviar (con confidence), banner de
  attribution cuando el router redirige, barra de progreso vs ETA.
- **Sessions (nueva página)**: tabla de sesiones activas con botón Kill.
- **Stats/Routing (nueva vista)**: distribución por router → target con
  ratios y latencia media.
- **Settings**: sección Routing & Sessions con umbrales editables.

### Casos de uso cubiertos

1. Chat a `gemma4:26b` con 3090 libre y vLLM cargado → primary_loaded.
2. vLLM ocupado corto (~15s restantes) → espera con estimator.
3. vLLM ocupado largo (100k tokens, 5 min) → fallback a Ollama en 3060.
4. Sesión Realtime sobre vLLM → veto absoluto → fallback.
5. FlashVSR intenta 3090 con sesión Realtime activa → 409 con Retry-After.
6. Sesión Realtime idle 3 min → workers compartibles temporalmente.
7. Primary en ERROR → salta al siguiente sin ruido.
8. Loop de routers → detectado con `visited_set`, 500 claro.
9. Target ausente → skip silencioso + warning.
10. `ollama/gemma4:26b` con `/` → legacy fallback, no entra al router.

### Etapas (~11 días)

| Etapa | Contenido | Días | Estado |
|---|---|---:|---|
| 0 | Alembic: routing_targets + work_size_meta + via_router_profile_id | 0.5 | ✅ |
| 1 | DurationEstimator Nivel 1 (percentiles) + endpoint + Playground ETA | 1.5 | ✅ backend + UI |
| 2 | Regresión LLM en estimator | 0.5 | ✅ |
| 3 | Handlers rellenan work_size_meta | 0.5 | ✅ |
| 4 | SessionRegistry + hooks Realtime + veto en scheduler + Sessions UI | 2 | ✅ backend + UI |
| 5 | Estimadores específicos por familia + ETA no-LLM en Playground | 2 | ✅ backend (ETA reutiliza chip existente) |
| 6 | RouterResolver + resolve_profile + modo Router en ModelConfigModal | 2 | ✅ backend + UI |
| 7 | Grace dinámico en pressure_eviction | 0.5 | ✅ |
| 8 | Vista Stats/Routing + settings section | 1.5 | ✅ backend + UI |

Etapas UI extras completadas:

- Chip ETA con confianza en el playground (ChatInterface).
- Página `/sessions` (admin) con Kill + estado activa/pausada/zombie.
- Badge `ROUTER` + tooltip con la cadena de fallback en la lista de perfiles.
- Toggle "Este perfil es un Router" con textarea de targets en ModelConfigModal.
- Banner de attribution al recibir respuesta del router (headers
  `X-Ocabra-Router` / `X-Ocabra-Router-Target`).
- Tab `Routing` en Stats con desglose por router → target (peticiones, cuota,
  errores, duración media).
- Tab `Routing & Sessions` en Settings con los seis knobs hot-reloadables.

### Remap concreto planificado para Etapa 6

- Convertir `gemma4:26b` (2 536 reqs/30d, 9 clientes) en router con targets
  `[gemma4:26b-vllm-ctx64k, gemma4:12b-vllm-ctx128k, gemma4:26b-ollama]`
  para transición transparente.
- Considerar el mismo patrón para `gemma4:12b-vllm-ctx128k` cuando el cliente
  único lo valide.

### Decisiones cerradas

- Persistencia sesiones: Redis + memoria, no Postgres.
- UI router: toggle dentro de `ModelConfigModal`, no flujo separado.
- Retry-After en 409 por session veto: sí, calculado con p95 - elapsed.
- `/ocabra/estimate` público con auth `user`.
- Warmup del estimator sin datos: percentiles + `family_default`, sin
  benchmarks manuales.
- Feature flag `routing_enabled` (default `true`) por si hay que apagarlo.

---

## 🚧 Bloque 21 — Presets de contexto y runtimes adaptativos

Objetivo: exponer un único ``profile_id`` estable por modelo/uso y convertir
las variantes ``-ctx*`` en presets internos. oCabra anuncia el máximo efectivo
que puede servir, mantiene un contexto normal configurable y escala al preset
mínimo suficiente cuando llega una petición mayor.

> Infraestructura activa para los perfiles llama.cpp que cambian de
> colocación. Cada preset sigue usando un pool KV unificado: no vuelve la
> partición rígida por slot. Qwen usa un runtime normal en la 3090 y uno
> extendido en ambas GPU; Gemma y RavenX, cuyo máximo ya cabe en una tarjeta,
> conservan un único runtime.

### Decisiones cerradas

- ``MAX`` se calcula por modelo y despliegue; nunca es un número global.
- La API pública anuncia ``effective_max_context``.
- El contexto default es configurable por perfil y no limita la capacidad
  pública anunciada.
- El preset 64k puede usar paralelismo 2; la colocación se valida por
  modelo+contexto+paralelismo y capacidades reales del backend, no se deduce
  solo del contexto. Si el backend fuerza menos slots, la UI debe mostrar el
  paralelismo efectivo y no únicamente el solicitado.
- Un preset mayor ya cargado atiende peticiones pequeñas. El downgrade al
  default se difiere con histéresis y las peticiones pequeñas no prolongan por
  defecto el TTL de contexto grande.
- Los presets ``-ctx*`` existentes se migran a aliases ocultos antes de retirar
  tags/modelos duplicados.
- Warm/on-demand pasa a ser configurable por preset.

### Fases

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Contrato, spike Ollama con un tag y varios ``num_ctx``, pools de GPU | ✅ contrato y mediciones completados |
| 1 | Alembic + dominio + CRUD de ``profile_runtime_presets`` | ✅ implementado |
| 2 | Cálculo ``model_max/effective_max/default/active`` y generación inicial | ✅ resolución, caché y exposición pública implementadas |
| 3 | Selector por presupuesto de tokens + RuntimeKey + histéresis | ✅ integrado en OpenAI/Ollama con upgrade y downgrade drenados |
| 4 | Afinidad real Ollama mediante pools 3090/3060/dual | ✅ pools independientes por GPU implementados |
| 5 | OpenAI/Ollama, federación, aliases y estadísticas por preset | ✅ implementado |
| 6 | UI de presets, validación, benchmark y aplicación drenada | ✅ implementado |
| 7 | Migración Gemma4 12B → Gemma4 26B → Qwen3.8 27B | ✅ aplicada en el nodo principal; aliases legacy activos |

### Criterios de aceptación

- ``/v1/models`` lista un único ``gemma4:12b`` con contexto efectivo 262144.
- Las peticiones normales usan 64k/p2; las mayores escalan automáticamente.
- Gemma4 12B 64k y 256k/p2 se colocan solo en la RTX 3090 conforme a las
  mediciones validadas.
- Deshabilitar el preset MAX reduce el máximo anunciado.
- Un remoto caliente puede ganar frente a una carga local fría.
- Solicitudes concurrentes para presets diferentes no descargan el runner
  equivocado ni duplican una misma carga.
- La UI permite activar, editar, validar, medir y marcar el preset default sin
  editar JSON.
- Los IDs legacy ``-ctx*`` siguen funcionando durante la transición.

### Revalidación pendiente — paralelismo de Qwen 3.8 en Ollama

Ollama 0.40.6 identifica Qwen 3.8 como arquitectura ``qwen35`` y fuerza
``num_parallel=1`` aunque el pool esté configurado con
``OLLAMA_NUM_PARALLEL=2``. El runner confirma la reducción mediante ``-np 1``.
Es una limitación upstream de la gestión de varias secuencias sobre su estado
recurrente/híbrido, no una capacidad validada del despliegue actual.

- Mantener los presets efectivos de Qwen 3.8 en ``p1`` mientras Ollama emita
  ``model architecture does not currently support parallel requests``.
- Revisar esta restricción al actualizar Ollama y seguir las incidencias
  upstream de ``qwen35``/``qwen35moe``.
- Para considerar ``p2`` soportado deben cumplirse las tres comprobaciones:
  runner arrancado con ``-np 2``, dos peticiones solapadas realmente en el
  tiempo y ausencia de errores/corrupción en contexto, imágenes y tools.
- Tras habilitarlo, repetir las curvas de VRAM para 32k/64k/128k/MAX: el coste
  de KV se multiplica por los slots y puede reducir el máximo efectivo.
- No retirar el límite validado de 163840 ni anunciar un MAX mayor hasta
  repetir carga, inferencia y prueba sostenida con el nuevo runtime.

---

## 🔬 Investigación pendiente — vLLM sleep mode

Evaluar cómo integrar `--enable-sleep-mode` de vLLM con el ciclo de vida de
oCabra para reducir la latencia de reactivación sin mantener toda la VRAM
ocupada.

- Definir estados y transiciones `loaded → sleeping → loaded`, diferenciándolos
  de `unloaded`; actualizar contratos, API interna, eventos WebSocket y UI si se
  adopta un estado nuevo.
- Comparar sleep level 1 (pesos en RAM, KV descartada) y level 2 (memoria GPU
  descartada) frente a descarga/carga completa.
- Integrarlo con políticas `warm`, `on_demand`, evicción por presión, watchdog,
  health checks, auto-reload y workers compartidos por perfil.
- Medir VRAM/RAM liberada, tiempo de sleep/wake, primera respuesta tras wake y
  comportamiento con peticiones en curso.
- Validar que los endpoints sleep/wake del servidor vLLM 0.26.0 sean estables y
  restringirlos a llamadas internas.
- Criterio de decisión: adoptar sleep mode solo si libera VRAM suficiente para
  otro modelo y reduce claramente el tiempo de reactivación respecto a la carga
  en frío, sin introducir pérdida de peticiones ni estados inconsistentes.

---

## 🚧 Bloque 22 — KV elástica y paralelismo dinámico en llama.cpp

Plan activo en `docs/tasks/llama-cpp-elastic-kv-plan.md`.

Objetivo: medir Gemma 4 12B, Gemma 4 26B-A4B y Qwen3.8 27B en RTX 3060,
RTX 3090 y ambas; mantener los pesos residentes en la mejor colocación,
reservar el mayor pool KV seguro y repartirlo dinámicamente entre hasta cuatro
peticiones sin dividir el contexto en parcelas fijas por slot. oCabra aplicará
admission control por tokens y conservará el spill a ambas GPU para las cargas
que no quepan en una.

Decisiones iniciales:

- cuatro slots físicos como punto de partida, seguidos de pruebas con otros
  valores;
- medir hasta el límite nativo de 256k en cada colocación en vez de asumir un
  techo de 128k, como campaña oportunista y no como bloqueo inicial;
- empezar con capacidades upstream existentes antes de modificar llama.cpp;
- separar cambios genéricos candidatos a upstream de las políticas propias de
  oCabra;
- desplegar KV unificada de forma acotada en los perfiles llama.cpp principales
  tras superar aislamiento, MTP, visión y cancelación; el soak largo queda como
  validación posterior antes de ampliar pools o cambiar políticas globales;
- investigar después la recreación de `llama_context` conservando
  `llama_model` para cambios de pool en la misma colocación.

Estado: baseline de capacidad y wiring del pool KV documentados. Visión
``enabled``, ``disabled`` y ``on_demand`` está implementada y validada con
Gemma 12B/26B y Qwen3.8, incluida transición hot-mmproj, grounding,
concurrencia y soak acelerado. El admission control ponderado está cableado en
OpenAI/Ollama, cubierto por tests y validado en un canary real Gemma 26B
128k/4 con carga desigual, cancelación y rechazo oversized. Desde 2026-11-01,
Gemma 12B/26B mantienen 256k/4 en la 3090. Qwen estándar usa 96k/4 en la
3090 y escala a 256k/4 en ambas GPU con split 1:3; el abliterated usa 64k/1
en la 3090 y escala a 224k/4 con split 1:2,3. Qwen Coder usa 116k/4 en la
3090 y 256k/4 dual 1:3. Tras 300 s sin demanda grande vuelven al preset normal
y liberan completamente la 3060. Dentro de cada runtime, una petición puede
consumir todo el pool si las demás no lo están usando; el
``WeightedSlotGate`` controla la suma agregada. RavenX conserva 240k/4 en una
3090. La recreación caliente del
``llama_context`` conservando pesos queda como optimización posterior, no como
requisito del reparto flexible. 256k de Raven queda 74 MiB por encima del
presupuesto con el margen configurado.
Las variantes Ollama se conservan como ``*-ollama``. Quedan la tokenización
exacta cuando exista
una vía pre-load fiable y el soak largo solicitado por el operador. La UI ya
permite configurar KV unificada y el cap opcional por slot; dejar este último
vacío expone el pool completo. La admisión por presión también prevalida el
máximo recuperable y no desaloja modelos si el nuevo runtime seguiría sin
caber aun liberando todos los candidatos. La transición 1→2 reconcilia los
runners Ollama persistentes, drena peticiones activas, espera telemetría física
fresca y repite la evicción sobre el déficit real si una autocarga PIN cambia
el estado durante la operación. La vuelta 2→1 se validó de extremo a extremo
con Qwen abliterated 224k/4 → 64k/1.
Desde 2026-11-02 llama.cpp mantiene cuatro slots físicos pero aplica
admisión computacional adaptativa (4/<32k, 3/≥32k, 2/≥64k) para proteger decode
frente a prefills grandes. Las decisiones quedan persistidas junto con cola,
carga, TTFT, prefill y slot en ``request_stats.runtime_telemetry``. El barrido
104k–128k del perfil normal de Chaos terminó correctamente: queda validado en
128k/4 slots/Q8 sobre la RTX 3090, con ~20.990 MiB de pico y ~3.188 MiB libres.
Una ráfaga real de cuatro reservas de ~40k admitió tres y encoló la cuarta
62.986 ms, confirmando el límite adaptativo sin reducir slots físicos.
Desde 0.22.15, la API y la UI agregan los workers derivados durante todo su
ciclo de vida y señalizan ``loading``, ``loaded``, ``upsizing``,
``downsizing`` y ``unloading`` junto con sus slots físicos. Hay controles para
cancelar una carga (con limpieza del proceso parcial) y para restaurar un
runtime dual al preset single-GPU; este último solo se habilita si el modelo
está ocioso y existe un destino de una GPU verificable. Los eventos de un
worker ``::<hash>`` refrescan su card canónica inmediatamente.

---

## 🔬 Investigación pendiente — speculative en Nemotron-H híbrido

Revalidar speculative decoding para Nemotron 3 Nano cuando vLLM publique una
corrección para el estado Mamba con tokens especulativos:

- `ngram_gpu` de vLLM 0.26.0 carga, pero la primera inferencia termina el engine
  con una aserción en `mamba_mixer2.py`.
- `nemotron_h_mtp` requiere un checkpoint con pesos MTP; el Nano AWQ desplegado
  no contiene `mtp_num_layers` ni `num_nextn_predict_layers`.
- Mantener `speculative_config` disponible en el backend para modelos
  compatibles, pero no activarlo en esta receta hasta superar inferencia,
  calidad y benchmark sostenido.

---

## Orden de ejecución

```
[✅ Hecho]  Auth + gateway + settings persistence
[✅ Hecho]  Bloque 2 — Quick fixes
[✅ Hecho]  Bloque 3 — Settings en BD
[✅ Hecho]  Bloque 4 — Langfuse
[✅ Hecho]  Bloque 5 — Tests
[✅ Hecho]  Bloque 7 — Teams, stats, admin UX
[✅ Hecho]  Bloque 8 — Voice Pipeline
[✅ Hecho]  Bloque 9 — Model Profiles (Fase 6)
[✅ Hecho]  Bloque 10 — Chatterbox TTS (Fase 7 parcial)
[✅ Hecho]  Bloque 11 — Resiliencia de backends
[✅ Hecho]  Bloque 12 — Federación P2P
[✅ Hecho]  Bloque 13 — Observabilidad de potencia + stats ampliadas
[✅ Hecho]  Bloque 14 — OpenAI Batches + Files API + ACL de modelos
[🚧 En curso] Bloque 15 — Backends Modulares (Fases 1, 2 (10/11), 4, 5 hechas; pendientes Fase 3 CI/OCI + validación runtime nativos)
[🚧 En curso] Bloque 16 — Unsloth Studio (cableado completo; pendiente validación e2e y watcher de exports)
[✅ Hecho]    Bloque 17 — llama.cpp loader parity (4 sprints; pendiente wiring scanner→DB del tokenizer fingerprint)
[✅ Hecho]    Bloque 18 — Fiabilidad de peticiones y backends ternarios
[✅ Hecho]    Bloque 19 — Observabilidad atribuible de OpenAI Realtime
[🚧 En curso] Bloque 20 — Router profiles, duration estimator y session registry
[✅ Hecho]    Bloque 21 — Presets de contexto y runtimes adaptativos
[🚧 En curso] Bloque 22 — KV elástica y paralelismo dinámico en llama.cpp (pool fijo, admission, UI y canary completados; soak y context reinit pendientes)
[Pendiente]   Revalidar paralelismo Qwen 3.8 cuando Ollama soporte qwen35 con más de un slot
[Pendiente]   Investigación e integración de vLLM sleep mode
[Pendiente]   Revalidar speculative decoding en Nemotron-H híbrido
[Pendiente]   Validación manual TRT-LLM multi-engine en producción
[Pendiente]   UI para listar/descargar batches del usuario
```

---

## 🛠️ Deuda técnica upstream — vLLM 0.29.x

Fixes que quedaron pineados hasta que resolvan aguas arriba. Retirar cuando
el ecosistema se estabilice.

### Pin `openai>=2.25.0` en `install_spec` de vLLM

- **Síntoma:** al arrancar un worker, `ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'`.
- **Causa:** vLLM 0.29.0 importa `NamespaceTool` de `openai>=2.25.0` pero el wheel no pin explícito. Pip resuelve `openai==2.24.0` por dependencias transitivas.
- **Upstream:** [vllm-project/vllm#49103](https://github.com/vllm-project/vllm/issues/49103).
- **Fix aplicado:** `openai>=2.25.0` añadido a `pip_packages` en `backend/ocabra/backends/vllm_backend.py`.
- **Cuándo retirar:** cuando `vllm==0.29.x` publique con pin correcto o cuando subamos a la siguiente minor.

### `FLASHINFER_DISABLE_VERSION_CHECK=1` en el entorno del worker

- **Síntoma:** `RuntimeError: flashinfer-cubin version (0.6.6) does not match flashinfer version (0.6.18)` en el arranque del EngineCore.
- **Causa:** vLLM 0.29 arrastra `flashinfer-python==0.6.18` pero el último `flashinfer-cubin` publicado es `0.6.13`. Wheels desalineados en pypi.
- **Fix aplicado:** env var `FLASHINFER_DISABLE_VERSION_CHECK=1` inyectada por defecto desde `vllm_backend._build_cmd()`.
- **Cuándo retirar:** cuando `flashinfer-cubin==0.6.18` (o la versión que vLLM vaya pineando) aparezca en pypi.

### Migración `python -m vllm.entrypoints.openai.api_server` → `vllm serve`

- **Estado:** vLLM 0.29 emite `DeprecationWarning` en el arranque; el subproceso sigue funcionando.
- **Fix aplicado:** `_build_cmd()` intenta usar `<venv>/bin/vllm serve` cuando existe, con fallback al comando `-m` deprecado.
- **Cuándo retirar el fallback:** cuando `vllm serve` sea el único path (probablemente 0.30+) y no queden venvs de 0.26.x activos.
