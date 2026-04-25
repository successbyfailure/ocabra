# oCabra — Roadmap

Última actualización: 2026-04-25

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
- **OpenAI Batches API** (`/v1/batches` · `POST`/`GET`/`cancel`) con tabla `openai_batches` y migración `0016`. Soporta `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`.
- **BatchProcessor** (`backend/ocabra/core/batch_processor.py`): loop en background, dispatch in-process vía `httpx.ASGITransport`, impersonación del owner con `X-Gateway-Token` + `X-Internal-User-Id` (añadido soporte en `_deps_auth.py`). Concurrencia configurable (`batch_max_concurrency=4`), poll cada `batch_poll_interval_seconds=5`, timeout por petición `batch_request_timeout_seconds=600`.
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
[Pendiente]   Validación manual TRT-LLM multi-engine en producción
[Pendiente]   UI para listar/descargar batches del usuario
```
