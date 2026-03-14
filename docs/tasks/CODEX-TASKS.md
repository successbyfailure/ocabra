# Tareas para Codex

Antes de empezar cualquier tarea: lee `AGENTS.md`, `docs/CONTRACTS.md` y `docs/CONVENTIONS.md`.
Codex puede lanzar múltiples instancias en paralelo. Las tareas marcadas con el mismo nivel de FASE pueden ejecutarse simultáneamente.

---

## TAREA S-1 — Portal y proxy de servicios de generación IA MKS

**Estado:** EN PROGRESO
**Rama:** `feat/services-portal`
**Dependencias:** coordinación explícita por tocar ficheros compartidos (`docker-compose.yml`, `caddy/Caddyfile`)

### Objetivo

Añadir una segunda entrada HTTP para servicios interactivos de generación:
- landing `MKS AI Generation Services`
- proxy para `Hunyuan3D`, `ComfyUI` y `Automatic1111`
- seguimiento de actividad vía proxy para poder descargar VRAM por inactividad

Reglas cerradas para esta implementación:
- OpenAI `/v1/images/*` sigue resolviendo únicamente con `DiffusersBackend`
- `Hunyuan3D`, `ComfyUI` y `A1111` se ejecutarán como servicios separados
- el almacenamiento común parte de `/docker/ai-models`
- para oCabra se respetará la raíz existente `/docker/ai-models/ocabra`
- los pesos compartidos de imagen vivirán bajo `/docker/ai-models/ocabra/models/image/...`

### Layout de almacenamiento acordado

```text
/docker/ai-models/
├── ocabra/
│   ├── hf_cache/
│   └── models/
│       ├── huggingface/
│       ├── image/
│       │   ├── checkpoints/
│       │   ├── vae/
│       │   ├── loras/
│       │   ├── embeddings/
│       │   ├── controlnet/
│       │   └── upscalers/
│       ├── comfyui/
│       ├── a1111/
│       └── hunyuan3d/
├── ollama/
├── whisper/
└── localai-models/
```

### Plan por fases

**Fase 1 — Base compartida**
- [x] Añadir variables nuevas a `.env.example`
- [x] Añadir segundo puerto público en `docker-compose.yml`
- [x] Añadir landing y segundo site en `caddy/Caddyfile`
- [x] Crear carpeta estática para la landing del portal

**Fase 2 — Contratos y orquestación**
- [x] Documentar `ServiceState` y API interna `/ocabra/services/*` en `docs/CONTRACTS.md`
- [x] Implementar `ServiceManager` en backend
- [x] Registrar actividad por proxy y gestionar idle unload

**Fase 3 — Servicios interactivos**
- [x] Añadir servicio Docker `hunyuan`
- [x] Añadir servicio Docker `comfyui`
- [x] Añadir servicio Docker `a1111`
- [x] Montar rutas compartidas y específicas de modelos/config

**Fase 4 — Adaptadores de descarga de VRAM**
- [ ] Integrar unload de checkpoint para `A1111`
- [ ] Integrar free/unload para `ComfyUI`
- [x] Integrar unload de runtime para `Hunyuan3D`

### Orden de implementación

1. Portal y proxy base
2. Variables/configuración compartida
3. Contratos backend de servicios
4. Servicios Docker
5. Idle unload y scheduler

### Avances

- [x] Estructura real de `/docker/ai-models` revisada y plan ajustado a la organización actual
- [x] Variables y rutas base documentadas
- [x] Segundo puerto y landing implementados
- [x] Servicios interactivos añadidos a `docker-compose.yml`
- [x] Backend base de orquestación implementado
- [x] Endpoints de runtime añadidos a Hunyuan3D
- [x] Tracking automático de actividad HTTP desde el proxy
- [x] Tracking de actividad WebSocket para ComfyUI

### Validación

- [x] `docker compose config`
- [x] `Caddyfile` validado con imagen `caddy:2-alpine`
- [x] `Caddyfile.dev` validado con imagen `caddy:2-alpine`
- [x] `python3 -m compileall` sobre backend y Hunyuan tocados
- [x] `python3 -m compileall` sobre proxy HTTP de servicios
- [x] `python3 -m compileall` sobre proxy WebSocket de servicios
- [x] Portal en `:8485` accesible con landing `MKS AI Generation Services`
- [x] Proxy HTTP de Hunyuan validado extremo a extremo en runtime
- [x] UI raíz de Hunyuan accesible desde `:8485/hunyuan/`
- [x] `POST /runtime/unload` de Hunyuan sincroniza estado con `ServiceManager`
- [ ] ComfyUI validado en runtime
- [ ] A1111 validado en runtime

### Notas de implementación

- `DiffusersBackend` seguirá siendo el único backend para `/v1/images/*`.
- `ComfyUI` y `A1111` compartirán assets de imagen en disco, no memoria VRAM.
- `Hunyuan3D` irá separado en `/docker/ai-models/ocabra/models/hunyuan3d`.
- La UI de cada servicio debe permanecer accesible incluso cuando `runtime_loaded=false`.
- El tracking automático ya cubre tráfico HTTP proxied y el WebSocket `/comfy/ws*` pasando por oCabra.
- En validación real se corrigió la reescritura del portal para preservar subrutas `/hunyuan/*`, `/comfy/*` y `/a1111/*`.
- El proxy backend ya sanea cabeceras `Location` de upstream para no filtrar hosts internos como `http://hunyuan:8080`.
- En este host no se ha podido validar `ComfyUI` ni `A1111` porque sus contenedores no están levantados de forma funcional todavía.

---

## TAREA X-1 — Stream 1-C: Model Registry & Download Manager

**Estado:** ESPERAR a que `feat/0-foundation` esté mergeada en `main`
**Rama:** `feat/1-C-registry`
**Briefing completo:** `docs/agents/stream-1C-registry.md`
**Paralelo con:** X-2 (frontend base)

### Cuándo entrar
Cuando veas que `docs/agents/phase-0-foundation.md` tiene `[x] Completado`.

### Qué hacer, paso a paso

**Paso 1 — Schemas**
Crear `backend/ocabra/schemas/registry.py`:
- `HFModelCard`, `HFModelDetail`, `OllamaModelCard`, `LocalModel`, `DownloadJob`
- Exactamente los campos definidos en `docs/agents/stream-1C-registry.md §Schemas de datos`.

**Paso 2 — HuggingFace Registry**
Crear `backend/ocabra/registry/huggingface.py`:
- Clase `HuggingFaceRegistry`.
- `search()`: usa `huggingface_hub.list_models()` con filtros de query y task.
- `get_model_detail()`: usa `huggingface_hub.model_info()`. Infiere `suggested_backend`:
  - `pipeline_tag == "text-generation"` → `vllm`
  - `pipeline_tag == "image-to-image"` o `"text-to-image"` → `diffusers`
  - `pipeline_tag == "automatic-speech-recognition"` → `whisper`
  - `pipeline_tag == "text-to-speech"` → `tts`
- `download()`: usa `huggingface_hub.snapshot_download()` con progress callback.

**Paso 3 — Ollama Registry**
Crear `backend/ocabra/registry/ollama_registry.py`:
- Clase `OllamaRegistry`.
- `search()`: fetch de `https://ollama.com/api/models` (o scrape de `/library`).
- `pull()`: ejecuta `ollama pull {model_ref}` como subproceso y parsea el output NDJSON para el progress callback.

**Paso 4 — Local Scanner**
Crear `backend/ocabra/registry/local_scanner.py`:
- Clase `LocalScanner`.
- Escanea `MODELS_DIR` buscando: carpetas con `config.json` (HF format), ficheros `.gguf`, ficheros `Modelfile`.
- Retorna lista de `LocalModel` con metadata inferida.

**Paso 5 — Download Manager**
Crear `backend/ocabra/api/internal/downloads.py`:
- Clase `DownloadManager` con cola Redis (`queue:download`).
- `enqueue()`: crea `DownloadJob`, guarda en Redis key `download:job:{job_id}`, añade a cola.
- Worker asyncio que consume la cola y llama a HF o Ollama según `source`.
- Publica progreso en Redis canal `download:progress:{job_id}` cada 500ms.
- `cancel()`: marca el job como cancelled, interrumpe la descarga si está activa.

**Paso 6 — Endpoints**
Crear `backend/ocabra/api/internal/registry.py`:
- `GET /ocabra/registry/hf/search?q=&task=&limit=`
- `GET /ocabra/registry/hf/{repo_id}` (reemplazar `/` por `%2F` en el path)
- `GET /ocabra/registry/ollama/search?q=`
- `GET /ocabra/registry/local`

Crear `backend/ocabra/api/internal/downloads.py` (endpoints):
- `GET /ocabra/downloads`
- `POST /ocabra/downloads` — body: `{source, model_ref}`
- `DELETE /ocabra/downloads/{job_id}`
- `GET /ocabra/downloads/{job_id}/stream` — SSE suscrito al canal Redis

Registrar ambos routers en `backend/ocabra/main.py` en la sección `# ROUTERS`.

**Paso 7 — Tests**
Crear `backend/tests/test_registry.py` y `test_downloads.py`:
- Mock `huggingface_hub`: test de búsqueda, test de inferencia de backend.
- Mock del subprocess de ollama pull: test de progreso.
- Test de DownloadManager: enqueue → progress → complete.
- Test de LocalScanner: fixtures de directorio con modelos mock.

### Señal de finalización
Commit en rama `feat/1-C-registry`:
```
feat(1-C): implement model registry and download manager
```
Marcar `docs/agents/stream-1C-registry.md` → `[x] Completado`.

---

## TAREA X-2 — Stream 1-D: Frontend Base

**Estado:** ESPERAR a que `feat/0-foundation` esté mergeada
**Rama:** `feat/1-D-frontend-base`
**Briefing completo:** `docs/agents/stream-1D-frontend-base.md`
**Paralelo con:** X-1

### Cuándo entrar
Cuando `docs/agents/phase-0-foundation.md` tenga `[x] Completado`. Paralelo con X-1.

### Qué hacer, paso a paso

**Paso 1 — Tipos base**
Crear `frontend/src/api/types.ts` con exactamente los tipos de `docs/CONTRACTS.md §3` y §2:
`GPUState`, `ModelStatus`, `LoadPolicy`, `BackendType`, `ModelCapabilities`, `ModelState`, `DownloadJob`, `WSEvent` (union type discriminada).

**Paso 2 — API Client**
Crear `frontend/src/api/client.ts`:
- `const api = { gpus, models, downloads, registry, stats, config }`.
- Cada método hace `fetch` a la ruta correcta del contrato `docs/CONTRACTS.md §5`.
- Manejo de errores: si el servidor retorna error, lanzar con el mensaje del campo `detail`.
- Sin dependencias externas, solo `fetch` nativo.

**Paso 3 — Stores Zustand**
Crear `frontend/src/stores/gpuStore.ts`:
- `gpus: GPUState[]`, `setGpus()`, `lastUpdated: Date | null`.

Crear `frontend/src/stores/modelStore.ts`:
- `models: Record<string, ModelState>`, `setModels()`, `updateModel()`.
- Actions: `loadModel()`, `unloadModel()` que llaman al API client.

Crear `frontend/src/stores/downloadStore.ts`:
- `jobs: DownloadJob[]`, `addJob()`, `updateJob()`.

**Paso 4 — WebSocket Hook**
Crear `frontend/src/hooks/useWebSocket.ts`:
- Conecta a `ws://{host}/ocabra/ws`.
- Parsea eventos y los despacha al store correspondiente según `event.type`.
- Auto-reconexión: espera 1s, 2s, 4s, 8s, máximo 30s entre intentos.
- Retorna `{ connected: boolean, lastEvent: WSEvent | null }`.

Crear `frontend/src/hooks/useSSE.ts`:
- Helper para consumir EventSource (para progreso de descargas).
- `useSSE(url: string, onMessage: (data: any) => void)`.

**Paso 5 — Componentes GPU**
Crear `frontend/src/components/gpu/VramBar.tsx`:
- Barra de progreso con colores: verde (<70%), amarillo (70-90%), rojo (>90%).
- Props: `used`, `total`, `locked` (mostrar locked en color diferente).

Crear `frontend/src/components/gpu/PowerGauge.tsx`:
- Arco semicircular SVG que muestra `powerDrawW / powerLimitW` en %.
- Colores: verde (<50%), amarillo (50-80%), rojo (>80%).

Crear `frontend/src/components/gpu/GpuCard.tsx`:
- Card completo con: nombre GPU, VramBar, PowerGauge, temperatura, utilización %.
- Alerta visual si temperatura > 80°C o utilización > 80%.

**Paso 6 — Componentes de modelo**
Crear `frontend/src/components/models/ModelStatusBadge.tsx`:
- Badge de color según `ModelStatus`: loaded=verde, loading/unloading=amarillo animado, error=rojo, configured/unloaded=gris.

Crear `frontend/src/components/models/LoadPolicyBadge.tsx`:
- Badge: pin=púrpura, warm=azul, on_demand=gris.

Crear `frontend/src/components/common/CapabilityBadge.tsx`:
- Badges pequeños por capability: Chat, Tools, Vision, Reasoning, Embeddings, Image, Audio, TTS.

**Paso 7 — Dashboard completo**
Implementar `frontend/src/pages/Dashboard.tsx`:
- Sección GPU Cards: 2 GpuCard en grid (una por GPU).
- Sección modelos activos: lista de modelos con `status=loaded`, cada uno muestra nombre, LoadPolicyBadge, GPU chip, VRAM usada, botón Unload.
- Sección descargas activas: si hay jobs en progreso, mostrar barra animada con velocidad y ETA.
- Todos los datos se actualizan en tiempo real via `useWebSocket`.
- Carga inicial: `useEffect` que llama `api.gpus.list()` y `api.models.list()`.

**Paso 8 — Tests**
Crear `frontend/src/__tests__/useWebSocket.test.ts`:
- Mock de WebSocket global.
- Verificar que `gpu_stats` event actualiza `gpuStore`.
- Verificar auto-reconexión.

Crear `frontend/src/__tests__/GpuCard.test.tsx`:
- Render con datos mock, snapshot.
- Verificar clase de alerta cuando temperatura > 80.

### Señal de finalización
Commit en rama `feat/1-D-frontend-base`:
```
feat(1-D): implement frontend base infrastructure and dashboard
```
Marcar `docs/agents/stream-1D-frontend-base.md` → `[x] Completado`.

---

## TAREA X-3 — Stream 2-B: Diffusers Backend

**Estado:** ESPERAR a que `feat/1-B-model-manager` esté mergeada
**Rama:** `feat/2-B-diffusers`
**Briefing completo:** `docs/agents/stream-2B-diffusers.md`
**Paralelo con:** X-4 (audio)

### Cuándo entrar
Cuando `docs/agents/stream-1B-model-manager.md` tenga `[x] Completado`.

### Qué hacer, paso a paso

**Paso 1 — Worker**
Crear `workers/diffusers_worker.py`:
- FastAPI app con `POST /generate` y `GET /health` y `GET /info`.
- `detect_pipeline_class(model_path)`: lee `model_index.json` para decidir entre `FluxPipeline`, `StableDiffusionXLPipeline`, `StableDiffusionPipeline`.
- Carga el pipeline en startup con `torch.float16` (3060) o `torch.bfloat16` (3090 — detectar por GPU name).
- `POST /generate`: ejecutar en `run_in_executor` para no bloquear.
- Args: `--model-path`, `--port`, `--device` (e.g. `cuda:0`).

**Paso 2 — Backend**
Crear `backend/ocabra/backends/diffusers_backend.py`:
- `DiffusersBackend(BackendInterface)` completo.
- `load()`: lanza `workers/diffusers_worker.py`, espera healthcheck hasta 180s.
- `get_capabilities()`: retorna `BackendCapabilities(image_generation=True)`.
- `get_vram_estimate_mb()`: suma ficheros `.safetensors` × 1.3.
- `forward_request()`: traduce formato OpenAI `/v1/images/generations` → worker `/generate`.
  - Parsear `size` ("1024x1024") → `width`, `height`.
  - Pasar `n` como `num_images`.

**Paso 3 — Registrar backend**
En `backend/ocabra/main.py` (o en el startup del `ModelManager`), registrar `DiffusersBackend` en el `WorkerPool` para `backend_type="diffusers"`.

**Paso 4 — Tests**
Crear `backend/tests/test_diffusers_backend.py`:
- Mock subprocess: test de load/unload.
- Test de traducción de tamaños OpenAI → width/height.
- Test de detección de pipeline class (fixtures de `model_index.json`).

### Señal de finalización
Commit: `feat(2-B): implement diffusers image generation backend`
Marcar `docs/agents/stream-2B-diffusers.md` → `[x] Completado`.

---

## TAREA X-4 — Stream 2-C: Audio Backend (Whisper + TTS)

**Estado:** ESPERAR a que `feat/1-B-model-manager` esté mergeada
**Rama:** `feat/2-C-audio`
**Briefing completo:** `docs/agents/stream-2C-audio.md`
**Paralelo con:** X-3

### Cuándo entrar
Cuando `docs/agents/stream-1B-model-manager.md` tenga `[x] Completado`.

### Qué hacer, paso a paso

**Paso 1 — Whisper Worker**
Crear `workers/whisper_worker.py`:
- FastAPI app con `POST /transcribe` (multipart, campo `file`) y `GET /health`.
- Usa `faster_whisper.WhisperModel` con `compute_type="float16"`.
- Soporta `response_format`: `json`, `text`, `verbose_json`, `srt`, `vtt`.
- Procesar en `run_in_executor`.

**Paso 2 — Whisper Backend**
Crear `backend/ocabra/backends/whisper_backend.py`:
- `WhisperBackend(BackendInterface)` completo.
- `forward_request()`: reenvía el fichero de audio como multipart vía httpx.
- Tabla de VRAM estimada por nombre de modelo (ver briefing).

**Paso 3 — TTS Worker**
Crear `workers/tts_worker.py`:
- FastAPI app con `POST /synthesize` y `GET /voices` y `GET /health`.
- Soporte para Kokoro (`hexgrad/Kokoro-82M`) como implementación inicial.
- Retorna `StreamingResponse` con audio binario.
- Mapa de voces OpenAI → voces del modelo (ver briefing).

**Paso 4 — TTS Backend**
Crear `backend/ocabra/backends/tts_backend.py`:
- `TTSBackend(BackendInterface)` completo.
- `forward_request()`: traduce formato OpenAI `/v1/audio/speech` → worker `/synthesize`.
- Retorna audio binario con `Content-Type` correcto según `response_format`.

**Paso 5 — Tests**
Crear `backend/tests/test_whisper_backend.py` y `test_tts_backend.py`:
- Mock de worker: verificar traducción de formato.
- Test de Content-Type: mp3 → `audio/mpeg`, wav → `audio/wav`.
- Test de mapa de voces: todas las voces OpenAI tienen equivalente.

### Señal de finalización
Commit: `feat(2-C): implement whisper transcription and TTS backends`
Marcar `docs/agents/stream-2C-audio.md` → `[x] Completado`.

---

## TAREAS X-5, X-6, X-7 — Frontend Features (paralelo)

**Estado:** ESPERAR a que `feat/3-A-openai-api` Y `feat/1-D-frontend-base` estén mergeadas
**Briefing completo:** `docs/agents/stream-4-frontend-features.md`

Lanzar las 3 instancias simultáneamente en ramas separadas.

---

### TAREA X-5 — Stream 4-A: Models UI & Explore

**Rama:** `feat/4-A-models-ui`

**Qué hacer:**

1. **Models page completa** (`frontend/src/pages/Models.tsx`):
   - Tabla con columnas: nombre, backend type, LoadPolicyBadge, GPU chip, VRAM bar mini, ModelStatusBadge, acciones.
   - Filtros por status, tipo y GPU.
   - Botones: Load (si unloaded), Unload (si loaded), Configure, Delete (con confirm dialog).

2. **Modal de configuración** (`ModelConfigModal.tsx`):
   - Select de `load_policy` con descripción de cada opción.
   - Select de GPU preferida (0=3060, 1=3090, null=auto).
   - Toggle `auto_reload`.
   - Sección de schedules: lista de schedules existentes + botón Añadir.

3. **ScheduleEditor** (`ScheduleEditor.tsx`):
   - Checkboxes de días de la semana.
   - Time pickers de inicio y fin.
   - Preview en texto: "Los lunes y martes de 02:00 a 06:00 este modelo se descargará automáticamente."
   - Llama a `PATCH /ocabra/models/{id}` con el schedule actualizado.

4. **Explore page** (`frontend/src/pages/Explore.tsx`):
   - Tabs: HuggingFace | Ollama | Local.
   - Barra de búsqueda con debounce 300ms.
   - Filtros HF: task, tamaño estimado (GB), gated (sí/no).
   - Cards con: nombre, descripción, downloads, VRAM estimada, badge de backend sugerido, botón "Instalar".
   - Al instalar: Dialog con selector de load_policy → llama a `POST /ocabra/downloads`.

5. **DownloadQueue flotante** (`DownloadQueue.tsx`):
   - Panel fijo bottom-right visible si hay descargas activas.
   - Por cada job: nombre, barra de progreso animada, velocidad MB/s, ETA, botón cancelar.
   - Progreso vía `useSSE` al endpoint `/ocabra/downloads/{job_id}/stream`.

Commit: `feat(4-A): implement models management and explore pages`

---

### TAREA X-6 — Stream 4-B: Playground

**Rama:** `feat/4-B-playground`

**Qué hacer:**

1. **Playground page** (`Playground.tsx`):
   - ModelSelector: dropdown que filtra por capability según la tab activa (chat/imagen/audio).
   - Tabs según capability del modelo seleccionado: Chat / Imagen / Audio.
   - ParamsPanel lateral: temperature, max_tokens, top_p, system prompt, seed.

2. **ChatInterface** (`ChatInterface.tsx`):
   - Lista de mensajes con markdown rendering (`react-markdown` + `rehype-highlight`).
   - Streaming: añadir texto carácter a carácter al último mensaje del assistant.
   - Soporte de imágenes: drag & drop en el input sube imagen → añade como message content con `image_url`.
   - Tool calls: bloque especial con nombre de tool, args (JSON formateado), resultado.
   - Botón "Copiar como curl" que genera el comando equivalente.

3. **ImageInterface** (`ImageInterface.tsx`):
   - Textarea de prompt + textarea de negative prompt.
   - Presets de tamaño: 512×512, 1024×1024, 1792×1024, 1024×1792.
   - Slider steps (1-50), slider guidance (1-20), input seed con botón 🎲.
   - Galería de resultados con botón de descarga de cada imagen.

4. **AudioInterface** (`AudioInterface.tsx`):
   - Tab Transcripción: botón grabar (MediaRecorder) + botón subir fichero. Muestra transcripción con timestamps.
   - Tab TTS: textarea, selector de voz (lista de `/ocabra/models/{id}/voices` si existe), slider velocidad, botón sintetizar, reproductor de audio.

Commit: `feat(4-B): implement playground for chat, image and audio`

---

### TAREA X-7 — Stream 4-C: Stats UI

**Rama:** `feat/4-C-stats`

**Qué hacer:**

1. **Stats page** (`Stats.tsx`):
   - DateRangePicker: botones predefinidos (1h, 24h, 7d, 30d) + rango custom con date inputs.
   - Model selector: "Todos" o modelo específico.

2. **RequestsChart** (`RequestsChart.tsx`):
   - `LineChart` de Recharts: requests por minuto en el rango seleccionado.
   - Datos de `GET /ocabra/stats/requests`.

3. **TokensChart** (`TokensChart.tsx`):
   - `BarChart` apilado: tokens de entrada (azul) y salida (verde) por periodo.

4. **PerformanceTable** (`PerformanceTable.tsx`):
   - Tabla por modelo: total requests, avg latencia ms, tokens/s, errores, % uptime.
   - Ordenable haciendo clic en cabecera de columna.
   - Botón exportar CSV.

5. **EnergyPanel** (`EnergyPanel.tsx`):
   - Una card por GPU con: consumo actual W (actualizado por WS), kWh acumulados en el periodo, coste estimado en € (kWh × tarifa configurada).
   - Total combinado de ambas GPUs.

Commit: `feat(4-C): implement stats and energy monitoring UI`

### Señal de finalización (X-5, X-6, X-7)
- Marcar `docs/agents/stream-4-frontend-features.md` → `[x] 4-A/4-B/4-C completado`.
