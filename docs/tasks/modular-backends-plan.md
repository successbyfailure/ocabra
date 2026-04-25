# oCabra — Plan de Backends Modulares

Última actualización: 2026-04-25

**Estado**: Fase 1 ✅ · Fase 2 ✅ (10/11; tensorrt_llm diferido) · Fase 3 (draft de Dockerfiles) ✅ · Fase 4 ✅ · Fase 5 ✅ · Deudas Ronda 2 ✅ · Validación e2e en slim ✅ (8/12 funcionando; D11/D13/D14 cerradas, D12 sglang parcial: requiere nvcc real). Pendiente: Fase 3 CI + `method="oci"`, ruff sweep #6, sglang nvcc.
Ver sección "Registro de decisiones y deudas" al final.

---

## Concepto

Cada backend de inferencia (vllm, whisper, diffusers, llama_cpp, etc.) es un **módulo
instalable/desinstalable** en runtime desde la UI. La imagen Docker base ("slim") solo
contiene el core de oCabra (FastAPI, httpx, pydantic, etc.) sin backends pre-instalados.

Los backends se distribuyen como **imágenes OCI** que se extraen al filesystem local.
La ejecución sigue siendo por **subprocesos locales** (sin cambiar la arquitectura actual).

### Inspiración: LocalAI

Adaptamos el modelo de LocalAI (OCI como formato de distribución, extracción local,
aislamiento por venv/binarios) pero manteniendo:
- Workers HTTP/REST (no gRPC) — ya funciona, más simple de debuggear
- `BackendInterface` Python como contrato — no `run.sh`
- SSE para progreso de instalación (reutilizamos patrón de descargas)

---

## Arquitectura

```
┌─────────────────────────────────────────────┐
│  Imagen Docker "slim"                       │
│  Python 3.11 + FastAPI + core oCabra        │
│  cmake + CUDA toolkit (para builds nativos) │
│  docker CLI (para pull de imágenes OCI)     │
│                                             │
│  /data/backends/  (volumen persistente)      │
│    ├── vllm/                                │
│    │   ├── venv/         ← Python venv      │
│    │   ├── metadata.json ← versión, origen  │
│    │   └── bin/          ← si hay binarios  │
│    ├── llama_cpp/                           │
│    │   ├── bin/llama-server                 │
│    │   └── metadata.json                    │
│    └── whisper/                             │
│        ├── venv/                            │
│        └── metadata.json                    │
└─────────────────────────────────────────────┘
```

### Flujo de instalación

```
Usuario: "Instalar vllm"
    │
    ▼
BackendInstaller.install("vllm")
    │
    ├─ 1. docker pull ghcr.io/ocabra/backend-vllm:cuda12
    ├─ 2. docker create (contenedor temporal)
    ├─ 3. docker cp contenedor:/backend/ → /data/backends/vllm/
    ├─ 4. docker rm (contenedor temporal)
    ├─ 5. Escribir metadata.json
    ├─ 6. worker_pool.register_backend("vllm", VLLMBackend())
    └─ 7. Publicar evento WS "backend_installed"

    (todo con SSE de progreso al cliente)
```

### Alternativa: instalación desde source

Para desarrollo o hardware no soportado por las imágenes pre-built:

```
BackendInstaller.install_from_source("vllm")
    │
    ├─ 1. Crear venv en /data/backends/vllm/venv/
    ├─ 2. pip install (desde install_spec del backend)
    ├─ 3. Para backends nativos: cmake + make (llama_cpp, bitnet)
    └─ 4. Mismo registro posterior
```

---

## Diseño detallado

### 1. BackendInstallSpec — Declaración del módulo

Cada backend declara su spec de instalación. Esto vive en el propio `*_backend.py`:

```python
@dataclass
class BackendInstallSpec:
    # Identificador del paquete OCI para distribución pre-built
    oci_image: str                      # "ghcr.io/ocabra/backend-vllm"
    # Tags por variante de hardware (se elige automáticamente)
    oci_tags: dict[str, str]            # {"cuda12": "latest-cuda12", "cpu": "latest-cpu"}
    # Path dentro de la imagen OCI donde están los ficheros del backend
    oci_extract_path: str = "/backend"
    # Instalación desde source (fallback si no hay OCI)
    pip_packages: list[str] = field(default_factory=list)
    # Script post-install (compilación nativa, etc.)
    post_install_script: str | None = None
    # Tamaño estimado en disco (MB) para mostrar en UI
    estimated_size_mb: int = 0
    # Descripción y metadata para la UI
    display_name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)  # ["LLM", "GPU", "CUDA"]
    # Python version requirement (para venvs from source)
    python_version: str = "3.11"

# Ejemplo en VLLMBackend:
class VLLMBackend(BackendInterface):
    @property
    def install_spec(self) -> BackendInstallSpec:
        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-vllm",
            oci_tags={"cuda12": "latest-cuda12"},
            pip_packages=["vllm==0.17.1", "torch>=2.5"],
            estimated_size_mb=8000,
            display_name="vLLM",
            description="High-throughput LLM inference engine with PagedAttention",
            tags=["LLM", "GPU", "CUDA"],
        )

# Ejemplo en LlamaCppBackend (binario nativo):
class LlamaCppBackend(BackendInterface):
    @property
    def install_spec(self) -> BackendInstallSpec:
        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-llama-cpp",
            oci_tags={"cuda12": "latest-cuda12", "cpu": "latest-cpu"},
            pip_packages=[],  # no Python deps
            post_install_script="scripts/build_llama_cpp.sh",
            estimated_size_mb=500,
            display_name="llama.cpp",
            description="CPU/GPU inference for GGUF models",
            tags=["LLM", "CPU", "GPU", "GGUF"],
        )

# Backends sin install_spec (siempre disponibles):
class OllamaBackend(BackendInterface):
    @property
    def install_spec(self) -> None:
        return None  # Ollama es un servicio externo, no necesita instalación
```

### 2. BackendInstaller — Core del sistema

Nuevo módulo `backend/ocabra/core/backend_installer.py`:

```python
class BackendInstallStatus(StrEnum):
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    UNINSTALLING = "uninstalling"
    ERROR = "error"

@dataclass
class BackendModuleState:
    backend_type: str
    display_name: str
    description: str
    tags: list[str]
    install_status: BackendInstallStatus
    installed_version: str | None
    installed_at: datetime | None
    install_source: str | None      # "oci" | "source" | "pre-installed"
    estimated_size_mb: int
    actual_size_mb: int | None
    error: str | None
    # Progreso de instalación en curso
    install_progress: float | None  # 0.0 - 1.0
    install_detail: str | None      # "pulling image...", "extracting...", etc.

class BackendInstaller:
    def __init__(self, backends_dir: Path, worker_pool: WorkerPool):
        self._backends_dir = backends_dir  # /data/backends
        self._worker_pool = worker_pool
        self._states: dict[str, BackendModuleState] = {}
        self._install_locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        """Scan backends_dir, detect what's installed, register backends."""
        self._backends_dir.mkdir(parents=True, exist_ok=True)
        for backend_type, backend_class in ALL_BACKENDS.items():
            spec = backend_class.install_spec
            if spec is None:
                # Always-available backend (ollama, etc.)
                self._register(backend_type, backend_class)
                continue
            if self._is_installed(backend_type):
                self._register(backend_type, backend_class)
            else:
                self._worker_pool.register_disabled_backend(
                    backend_type, "not installed"
                )

    async def install(
        self, backend_type: str, method: str = "oci"
    ) -> AsyncIterator[BackendModuleState]:
        """Install a backend. Yields state updates for SSE streaming."""
        ...

    async def uninstall(self, backend_type: str) -> None:
        """Uninstall a backend: unload models, remove files, deregister."""
        ...

    def get_state(self, backend_type: str) -> BackendModuleState:
        ...

    def list_states(self) -> list[BackendModuleState]:
        ...

    def _is_installed(self, backend_type: str) -> bool:
        metadata_file = self._backends_dir / backend_type / "metadata.json"
        return metadata_file.exists()

    def _register(self, backend_type: str, backend_class: type) -> None:
        """Instantiate and register a backend with the worker pool."""
        ...
```

### 3. metadata.json — Estado persistido por backend

```json
{
  "backend_type": "vllm",
  "version": "0.17.1",
  "installed_at": "2026-04-12T10:30:00Z",
  "install_source": "oci",
  "oci_image": "ghcr.io/ocabra/backend-vllm",
  "oci_tag": "latest-cuda12",
  "oci_digest": "sha256:abc123...",
  "python_bin": "/data/backends/vllm/venv/bin/python",
  "extra_bins": {},
  "size_mb": 7850
}
```

Para backends con binarios nativos:

```json
{
  "backend_type": "llama_cpp",
  "version": "b5432",
  "install_source": "oci",
  "python_bin": null,
  "extra_bins": {
    "llama_server": "/data/backends/llama_cpp/bin/llama-server"
  },
  "size_mb": 480
}
```

### 4. Hardware Detection

Auto-detectar la variante de GPU para elegir el tag OCI correcto:

```python
def detect_gpu_variant() -> str:
    """Detect GPU and return the best OCI tag key."""
    # Usa pynvml (ya está en el proyecto)
    # NVIDIA + CUDA 12.x → "cuda12"
    # NVIDIA + CUDA 11.x → "cuda11"
    # AMD ROCm → "rocm"
    # Sin GPU → "cpu"
```

### 5. Adaptación de backends existentes

Cada backend debe saber encontrar su venv/binarios en `/data/backends/`:

```python
class VLLMBackend(BackendInterface):
    def _get_python_bin(self) -> str:
        metadata_file = Path(settings.backends_dir) / "vllm" / "metadata.json"
        if metadata_file.exists():
            meta = json.loads(metadata_file.read_text())
            return meta.get("python_bin", sys.executable)
        return sys.executable  # fallback: main env (imagen fat)
```

Los backends que hoy usan `settings.*_python_bin` (sglang, voxtral, chatterbox)
migran a leer de `metadata.json`, manteniendo el setting como override manual.

### 6. Config y volumen

```python
# config.py
backends_dir: str = "/data/backends"
```

```yaml
# docker-compose.yml — volumen persistente
api:
  volumes:
    - backends_data:/data/backends

volumes:
  backends_data:
```

---

## API Endpoints

### Backend Modules CRUD

```
GET    /ocabra/backends
       → lista de todos los backends con su estado de instalación

GET    /ocabra/backends/{backend_type}
       → estado detallado de un backend

POST   /ocabra/backends/{backend_type}/install
       Body: {"method": "oci" | "source"}
       → SSE stream con progreso de instalación

POST   /ocabra/backends/{backend_type}/uninstall
       → desinstala el backend (requiere que no haya modelos cargados)

GET    /ocabra/backends/{backend_type}/logs
       → logs de la última instalación/error
```

### Respuesta de GET /ocabra/backends

```json
[
  {
    "backend_type": "vllm",
    "display_name": "vLLM",
    "description": "High-throughput LLM inference with PagedAttention",
    "tags": ["LLM", "GPU", "CUDA"],
    "install_status": "installed",
    "installed_version": "0.17.1",
    "installed_at": "2026-04-12T10:30:00Z",
    "install_source": "oci",
    "estimated_size_mb": 8000,
    "actual_size_mb": 7850,
    "models_loaded": 1,
    "has_update": false
  },
  {
    "backend_type": "diffusers",
    "display_name": "Diffusers",
    "description": "Image generation with Stable Diffusion, SDXL, Flux",
    "tags": ["Image", "GPU", "CUDA"],
    "install_status": "not_installed",
    "estimated_size_mb": 3000,
    "models_loaded": 0
  },
  {
    "backend_type": "ollama",
    "display_name": "Ollama",
    "description": "External Ollama server integration",
    "tags": ["LLM", "External"],
    "install_status": "installed",
    "install_source": "built-in",
    "always_available": true
  }
]
```

---

## Imágenes OCI — Build en CI

Cada backend tiene un Dockerfile en `backends/dockerfiles/`:

```
backends/dockerfiles/
  ├── Dockerfile.vllm
  ├── Dockerfile.whisper
  ├── Dockerfile.diffusers
  ├── Dockerfile.tts
  ├── Dockerfile.llama_cpp
  ├── Dockerfile.bitnet
  ├── Dockerfile.sglang
  ├── Dockerfile.voxtral
  ├── Dockerfile.chatterbox
  └── Dockerfile.acestep
```

Ejemplo `Dockerfile.vllm`:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /backend/venv && \
    /backend/venv/bin/pip install --upgrade pip && \
    /backend/venv/bin/pip install vllm==0.17.1

# Copiar workers de oCabra que necesita este backend
COPY workers/vllm_worker.py /backend/workers/
```

Ejemplo `Dockerfile.llama_cpp`:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y cmake git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /build && \
    cd /build && mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70;75;80;86;89" && \
    cmake --build . --config Release -j$(nproc)

FROM scratch
COPY --from=builder /build/build/bin/llama-server /backend/bin/llama-server
COPY --from=builder /build/build/bin/lib*.so* /backend/lib/
```

**CI pipeline** (GitHub Actions):
- Build y push para cada backend × variante GPU
- Tags: `ghcr.io/ocabra/backend-{name}:latest-{variant}`
- Matrix: `[cuda12, cuda11, cpu]` × `[vllm, whisper, diffusers, ...]`

---

## Frontend — Página Backends

Nueva página o sección en Settings: **"Backends"**

### Vista principal

Cards por backend mostrando:
- Nombre + descripción + tags
- Estado: badge con color (igual que servicios)
- Si instalado: versión, tamaño, fecha
- Modelos actualmente cargados con este backend
- Botones: Instalar / Desinstalar / Actualizar

### Estados y badges

| Estado | Badge | Color | Acciones |
|--------|-------|-------|----------|
| No instalado | "No instalado" | Gris | Instalar |
| Instalando | "Instalando 45%" | Violeta, pulse | Cancelar |
| Instalado | "Instalado" | Verde | Desinstalar, Actualizar |
| Error | "Error" | Rojo | Reintentar, Ver log |
| Desinstalando | "Desinstalando" | Amber, pulse | - |
| Built-in | "Integrado" | Blue | - |

### Progreso de instalación

Al hacer click en "Instalar":
- El botón cambia a barra de progreso
- SSE muestra fase actual: "Pulling image...", "Extracting...", "Verifying..."
- Al completar: badge cambia a "Instalado", modelos del backend aparecen disponibles

---

## Fases de implementación

### Fase 1 — Infraestructura base ✅ (2026-04-24)

- [x] `BackendInstallSpec` dataclass en `backends/base.py`
- [x] `BackendInstaller` en `core/backend_installer.py`
  - [x] `start()`: escanear `/data/backends/`, detectar instalados, registrar
  - [x] `install()` desde source (venv + pip): flujo básico
  - [x] `uninstall()`: verificar que no hay modelos cargados, borrar directorio
  - [x] `is_installed()`, `list_states()`, `get_state()`
- [x] `install_spec` property en `BackendInterface` con default `None` (por backend aún pendiente → Fase 2)
- [x] Settings: `backends_dir` en `config.py` + `BACKENDS_DIR` en `.env.example`
- [x] Volumen `backends_data` en `docker-compose.yml`
- [x] Adaptar `main.py`: `BackendInstaller` instanciado en `lifespan` y registrado via `worker_pool._backends`
- [x] API endpoints: `GET /ocabra/backends`, `GET/POST .../install` (ambas variantes SSE), `POST .../uninstall`, `GET .../logs`
- [x] SSE de progreso de instalación
- [x] Tests: 13 en verde (`backend/tests/core/test_backend_installer.py`)

### Fase 2 — Migrar backends a install_spec (10/11 hechos; tensorrt_llm diferido a Fase 3)

Estado tras la sesión 2026-04-25 (Ronda 1 de cierre):

- [x] `whisper` — pip + cu124 (faster-whisper + pyannote + nemo). Install validado, **6.3 GB**. `load()` validado en GPU 1, 35 s.
- [x] `tts` — pip + cu124 (qwen-tts + kokoro + bark). Install validado, **5.7 GB**. `load(kokoro)` validado en GPU 1, 14 s.
- [x] `diffusers` — pip + cu124 (diffusers + accelerate + transformers). Install validado, **4.9 GB**.
- [x] `chatterbox` — pip + cu124 (chatterbox-tts). Install validado, **6.0 GB**. Resolver de python_bin con prioridad metadata > legacy `settings.chatterbox_python_bin` > sys.executable.
- [x] `sglang` — pip + cu124 (`sglang==0.5.9`). Install validado, **10.3 GB**. Resolver de python_bin para el server interno.
- [x] `voxtral` — pip + cu124 (vllm + vllm-omni). Install validado, **11.1 GB**.
- [x] `vllm` — pip + cu124 (`vllm==0.17.1` + torch). Install validado, **9.7 GB**. Launcher cambiado de `python` literal a `_resolve_python_bin()`.
- [x] `llama_cpp` — apt(build-essential cmake git ninja-build) + git(ggml-org/llama.cpp@master) + post_install_script `backend/scripts/install_llama_cpp.sh` (cmake CUDA build) + `extra_bins={"server": "bin/llama-server"}`. Sin venv (include_core_runtime=False). Resolver de server_bin con prioridad metadata > `settings.llama_cpp_server_bin`. **Ronda 1 — install end-to-end pendiente de validar en runtime con slim image (build CUDA tarda ~10 min).**
- [x] `bitnet` — apt(build-essential cmake git python3) + git(microsoft/BitNet@main `--recursive`) + post_install_script `backend/scripts/install_bitnet.sh` (kernel-header copy + cmake CPU/CUDA build) + `extra_bins={"server": "bin/bitnet-server"}`. Sin venv. **Ronda 1 — install end-to-end pendiente de validar.**
- [x] `acestep` — apt(git ffmpeg libsndfile1 curl) + pip(torch+torchaudio cu124) + git(ace-step/ACE-Step-1.5@main) + post_install_script `backend/scripts/install_acestep.sh` (instala uv si falta + `uv sync` contra el venv del install). `_resolve_project_paths()` lee metadata > `settings.acestep_project_dir`. **Ronda 1 — install end-to-end pendiente de validar.**
- [ ] `tensorrt_llm` — DIFERIDO a Fase 3 OCI. Sin `install_spec`: la distribución oficial es la imagen NGC de NVIDIA y los wheels de `pypi.nvidia.com` están demasiado acoplados a versiones específicas de CUDA/driver para el flujo source. Documentado en el docstring del backend.

Total disco source-installed en una instalación slim que active los 10: **~52.9 GB** Python-heavy + ~0.6 GB binarios nativos. En slim son pay-as-you-use.

Backends sin migración (siempre disponibles):
- `ollama` — servicio externo, no necesita instalación

### Fase 3 — Distribución OCI

- [x] Dockerfiles por backend en `backends/dockerfiles/` (11 ficheros + README + `.dockerignore`, 2026-04-24)
- [ ] CI pipeline para build y push a registry (skeleton GitHub Actions en `backends/dockerfiles/README.md`)
- [ ] `install()` desde OCI: pull → create → cp → rm (endpoint devuelve 501 hasta entonces)
- [ ] Hardware detection para elegir variante (cuda12/cpu)
- [ ] `metadata.json` con digest OCI para detección de updates
- [ ] Endpoint de check de actualizaciones

### Fase 4 — Imagen Docker slim

- [ ] Dockerfile base sin backends pre-instalados
- [ ] Solo incluye: Python 3.11, core oCabra, cmake, CUDA toolkit, docker CLI
- [ ] Mantener imagen "fat" como opción para quienes quieran todo pre-instalado
- [ ] Documentación de primer arranque (instalar backends desde la UI)

### Fase 5 — Frontend ✅ (2026-04-24)

- [x] Página `/backends` (no en Settings sino como ruta propia, rol `system_admin`)
- [x] Cards con estado, versión, tamaño, tags, conteo de modelos cargados
- [x] Botones install/uninstall con confirmación bloqueante si hay modelos cargados
- [x] Barra de progreso SSE durante instalación
- [x] Eventos WebSocket `backend_installed` / `backend_uninstalled` / `backend_progress` cableados en el store
- [ ] Indicadores en la página Models (qué backend usa cada modelo) — pendiente, fuera del MVP

---

## Impacto en código existente

| Módulo | Cambio |
|--------|--------|
| `backends/base.py` | Añadir `BackendInstallSpec`, `install_spec` property |
| `backends/*.py` | Añadir `install_spec` a cada backend |
| `core/backend_installer.py` | Nuevo módulo |
| `core/worker_pool.py` | Sin cambios (ya tiene register/disabled) |
| `config.py` | Añadir `backends_dir` |
| `main.py` | Reemplazar registro hardcodeado por BackendInstaller.start() |
| `api/internal/` | Nuevo router `backends.py` |
| `api/internal/ws.py` | Eventos `backend_installed`/`backend_uninstalled` |
| `docker-compose.yml` | Volumen `backends_data`, montar en `/data/backends` |
| `backend/Dockerfile` | Versión slim sin backends |
| `backends/dockerfiles/` | Nuevo directorio con Dockerfiles por backend |
| Frontend | Nueva página Backends |

**NO se modifican:** `worker_pool.py`, routers OpenAI/Ollama, GPU manager, scheduler,
service manager, model manager (solo el registro inicial cambia).

---

## Orden de ejecución recomendado

```
Fase 1 — Infraestructura          (BackendInstaller + API + install from source)
Fase 2 — Migrar backends          (uno a uno, empezando por tts/whisper)
Fase 3 — Distribución OCI         (Dockerfiles + CI + install from OCI)
Fase 4 — Imagen slim              (Dockerfile base sin backends)
Fase 5 — Frontend                  (UI de gestión de backends)
```

Fases 1-2 son el MVP: se puede instalar/desinstalar backends desde la API, la imagen
actual sigue funcionando. Fase 3 añade la distribución pre-compilada. Fase 4-5 son
la experiencia completa.

---

## Fuera de alcance (v1)

- Marketplace público de backends de terceros
- Backends en otros lenguajes (Go, Rust) — todos nuestros backends son Python/C++
- Hot-reload de backends sin reiniciar workers (los workers se reinician al load)
- Versionado semántico con rollback automático
- Backend plugins de terceros (system de plugins extensible)

---

## Registro de decisiones y deudas (2026-04-24)

Este registro se creó durante el merge del MVP (Fases 1 + 3-draft + 5). Cada entrada
debe resolverse o promoverse a issue antes de cerrar el bloque 15.

### Decisiones de diseño tomadas durante la integración

1. **Reconciliación blanda con la imagen "fat" actual.** Cuando `BackendInstaller.start()` encuentra un backend registrado que no tiene `metadata.json`, lo marca como `install_source="built-in"` + `install_status=INSTALLED` en lugar de `NOT_INSTALLED`. Evita que la imagen actual deje de funcionar mientras Fase 2 migra los backends uno a uno. Cuando cada backend empiece a escribir su propio `metadata.json` durante el install, la marca "built-in" se sobrescribe.

2. **`GET` además de `POST` en `/ocabra/backends/{type}/install`.** El navegador `EventSource` solo emite GET, y el frontend lo necesita para el SSE. La variante GET acepta `method` como query param y delega al handler POST. Ambas siguen disponibles — la POST queda para clientes no-browser.

3. **`always_available && status=="installed"` → badge "built-in" en UI.** Ollama y similares reportan `install_status=installed` pero no deben mostrar botón de desinstalar. El helper `toBackendModuleState` del cliente hace el mapeo.

4. **Mock fallback silencioso en el frontend.** Si `GET /ocabra/backends` devuelve 404 o fetch error, el store carga 5 backends mock y simula SSE con `setInterval`. Flag opcional `VITE_MOCK_BACKENDS=1` lo fuerza. Útil durante el desarrollo en paralelo con el backend; ahora que el backend está merged, el mock solo debería dispararse con la flag explícita — ver deuda #4 abajo.

5. **`method="oci"` devuelve 501.** El endpoint valida el string pero rechaza OCI hasta que Fase 3 publique las imágenes y el installer implemente `docker pull + cp`.

6. **Tests HTTP con app aislada.** `test_backend_installer.py` monta una mini-FastAPI app para los endpoints en lugar de importar `ocabra.main:app` (que dispara lifespan completo con DB/Redis/GPU). Patrón a replicar en otros tests de routers aislables.

### Deudas técnicas abiertas

1. **`WorkerPool._backends` accedido como privado desde `main.py`.** Agente A no añadió un método público para respetar el ownership del stream 1-B. Cuando alguien toque `worker_pool.py`, exponer `WorkerPool.registered_backends()` y migrar `main.py`.

2. **Fase 2 no arrancada.** Todos los backends aparecen como `built-in` hasta que se migren uno a uno a `install_spec` real. Orden sugerido en el plan: `tts → whisper → diffusers → vllm → sglang → chatterbox → voxtral → llama_cpp → bitnet → acestep → tensorrt_llm`.

3. **Incógnitas heredadas del draft de Dockerfiles (Agente B)** que bloquean la CI:
   - **TensorRT-LLM**: el distribution path real es la imagen NGC de NVIDIA, no un wheel limpio. Dockerfile marcado con TODO y excluido del matrix default. Decidir: repackage NGC vs pin wheel `pypi.nvidia.com`.
   - **ACE-Step**: `uv sync` contra upstream puede fallar si pinnean torch incompatible. Plan B: fijar `ACESTEP_REF` a commit conocido.
   - **Variantes CPU**: `BASE_IMAGE` parametrizado pero sin validar. Whisper/Diffusers/TTS necesitarán `--extra-index-url https://download.pytorch.org/whl/cpu`. Matrix CPU commentada en README hasta validar.
   - **GitHub runners ubuntu-22.04 (~14 GB libres)**: vllm/sglang/voxtral/tensorrt_llm no caben. Decidir: disk-cleanup step o self-hosted runner.

4. **Mock fallback del frontend debe apagarse por defecto.** Ahora mismo salta automáticamente si el backend responde error; con el backend ya mergeado, conviene que solo se active con `VITE_MOCK_BACKENDS=1` para que los errores reales sean visibles. Editar `backendsStore.ts` y quitar el `TODO: remove mock once Agent A merges`.

5. **Indicadores de backend en la página Models.** Fase 5 no incluyó badges "este modelo usa vLLM / Diffusers" en Models.tsx. Tarea menor, útil para que el usuario sepa qué pasa si desinstala un backend.

6. **Ruff en `api/internal/`**: 247 errores B008/ASYNC230/240/F401/I001 preexistentes. No tocados por Agente A para no divergir del estilo del repo. Candidato a sweep separado.

7. **Volumen `backends_data` nombrado.** `docker-compose.yml` lo declara como volumen docker nombrado, no bind mount. Si alguien quiere inspeccionar `/data/backends` desde el host, tendrá que añadir bind mount local — documentar en `docs/INSTALL.md` cuando se promocione a GA.

8. **`install_progress` granularidad.** El installer actual emite 4-5 pasos (`venv → pip → metadata → register`). Para pip con deps grandes (vllm, torch) el usuario ve "Installing pip packages..." durante varios minutos sin actualización. Mejora futura: parsear stdout de pip para reportar wheels individuales.

9a. **`BackendInstallSpec` no soporta `pip_extra_index_urls` (descubierto en Fase 2 piloto, 2026-04-24).** Whisper necesita `torch==2.x+cu124` desde `https://download.pytorch.org/whl/cu124`; sin ese índice el `pip install torch>=2.5` instala el wheel CPU-only de PyPI. Bloqueante para cualquier backend GPU que haga `install method=source`. Acción: extender el dataclass con `pip_extra_index_urls: list[str] = []` y propagarlo en `BackendInstaller._install_from_source()`.

9b. **Core runtime deps en venvs aislados.** Los workers (whisper, diffusers, etc.) importan `fastapi`, `uvicorn`, `httpx`, `pydantic` porque son FastAPI apps. Cuando el `install_spec` monta un venv aislado en `/data/backends/<name>/venv`, esas deps deben estar disponibles ahí. Opciones:
   - Añadirlas a cada `install_spec` (duplicación pero aislamiento real)
   - Añadir un campo `BackendInstallSpec.include_core_runtime: bool = True` y que el installer instale automáticamente `fastapi uvicorn httpx pydantic` antes de las `pip_packages` del backend.
   - Compartir `site-packages` del core con `venv --system-site-packages`. Pierde aislamiento pero es el atajo. **Recomendación**: segunda opción.

9c. **Apt packages no cubiertos por `install_spec`.** Whisper necesita `ffmpeg` y `libsndfile1`, TTS necesita `ffmpeg`, chatterbox también. Hoy están en el `Dockerfile` base. Cuando pasemos a slim: o los mantenemos en la imagen base (razonable — son dependencias de audio comunes), o extendemos `BackendInstallSpec` con `apt_packages: list[str]` + lógica de `apt-get install` en el installer (requiere container con apt disponible). **Recomendación**: imagen slim base incluye `ffmpeg + libsndfile1 + libsox` porque los comparten muchos backends.

9d. **Optional extras para backends con deps pesadas.** `nemo_toolkit[asr]` pesa ~1.2 GB y solo se usa para Parakeet/Canary, no para whisper estándar. Extender `BackendInstallSpec` con `pip_packages_optional: dict[str, list[str]]` (ej. `{"nemo": ["nemo_toolkit[asr]>=2.2"]}`) y que el UI permita al usuario tickear extras al instalar.

9e. **HF_TOKEN requerido post-migración para funcionalidades gated** (diarización en whisper, algunos TTS). El `install()` en sí no lo necesita, pero el primer `load()` del modelo gated fallará con mensaje poco claro si no está configurado. Añadir pre-check en `BackendInstaller.install()` o warning en el panel frontend cuando un backend declara `gated_models: bool = True`.

9. **Rebuild de imágenes Docker bloqueado (descubierto durante el smoke test 2026-04-24).** Dos fallos independientes impidieron el rebuild limpio de `ocabra-api` y `ocabra-frontend`:
   - `backend/Dockerfile` falla al compilar `llama-server` con `cmake --build ... -j4` (exit 2, stderr truncado). Cache-first builds lo evitan, pero `--no-cache` lo revela.
   - Frontend `node:20-alpine` falla al pullearse con `tls: certificate not valid` del mirror de Cloudflare (transitorio).
   Workaround usado para validar el MVP: `docker cp` de los ficheros nuevos al contenedor vivo + `docker compose restart api`, y `npx vite build --outDir /tmp/...` + `docker cp` del bundle al frontend. **Antes del próximo deploy real hay que diagnosticar el fallo de llama.cpp y reintentar el pull de node.**

### Validación end-to-end (2026-04-24)

Smoke script `scripts/smoke_bloque15.sh` pasa 5/5 casos contra la app real dentro del contenedor:
1. `GET /ocabra/backends` → 12 backends, todos `install_status=installed, install_source=built-in`
2. `GET /ocabra/backends/ollama` → detalle correcto
3. `GET /ocabra/backends/nope` → 404
4. `POST /ocabra/backends/whisper/uninstall` → 409 (built-in protegido)
5. `POST /ocabra/backends/whisper/install {"method":"oci"}` → 501 (Fase 3 pendiente)

Validado también a través de Caddy (`http://localhost:8484/ocabra/backends` con cookie JWT de `system_admin`): lista completa con los 12 backends esperados.

### Hito Fase 4 — slim image + install end-to-end validado (2026-04-25)

- Deuda #9 (llama.cpp build) resuelta: stub `libcuda.so.1` + `LIBRARY_PATH` + `-Wl,-rpath-link` en Dockerfile.
- `backend/Dockerfile.slim` entregado: `python:3.11-slim-bookworm` + FastAPI + ffmpeg/libsndfile1/sox + docker CLI. **987 MB vs 54 GB** fat (55×). Build ~2 min vs ~40 min.
- `docker-compose.slim.yml` override que arranca `ocabra-api:slim` sin tocar el `ocabra-api:latest` (→ fat, rollback).
- Deudas 9a/9b/psutil cerradas: `BackendInstallSpec.pip_extra_index_urls` + `include_core_runtime`; `settings.backends_fat_image` (default True, slim pone False) para que `start()` marque backends sin metadata como `not_installed` en slim.
- **Install end-to-end de whisper sobre slim verificado**: `POST /ocabra/backends/whisper/install method=source` dispara SSE con progreso real, crea `/data/backends/whisper/venv/`, ejecuta pip con los args correctos:
  ```
  venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch>=2.5 torchaudio>=2.5 faster-whisper>=1.1 soundfile>=0.12 \
      transformers>=4.47 pyannote.audio>=3.3 nemo_toolkit[asr]>=2.2 \
      librosa>=0.10 matplotlib>=3.10 numpy
  ```
  El contrato (spec → installer → venv → pip con índice CUDA + core runtime previo) está validado de punta a punta. Tests `tests/core/` = 24/24.

### Deuda nueva descubierta: install no se cancela al desconectar el cliente SSE

Si el cliente cierra la conexión SSE a `POST /install` mientras pip está descargando, el proceso pip sigue vivo en el contenedor (el subprocess sobrevive la cancelación del handler). Esto tiene dos caras: buena (el usuario puede cerrar la pestaña y el install termina) y mala (el `BackendModuleState.install_status` se queda en `"installing"` para siempre si el generator no llega a yieldear el estado final). Fix pendiente: detectar el pip terminado fuera del handler (task background o tick periódico) y actualizar estado a `installed`/`error`. Añadir a la lista de deudas como **9f**.

**Resuelto (2026-04-25)**: `install()` lanza la instalación como `asyncio.Task` detached y une al consumer SSE vía `asyncio.Queue`. Si el consumer es cancelado, el task sigue vivo y aterriza el estado final (`installed`/`error`) en `self._states`. Test `test_install_survives_consumer_cancellation` cubre el flujo. Commit `27dc571`.

### Decisiones y deudas nuevas (sesión 2026-04-25 — 7 backends migrados)

**Decisión #7 — Floor de `transformers` retirado del spec de tts.**
`qwen-tts==0.1.1` pinea `transformers==4.57.3` exacto. Listar `transformers>=5.0` en `pip_packages` rompía la resolución (`ResolutionImpossible`). Política: cuando una lib pinea transitivamente a una versión exacta, no añadirla como floor explícito; deja que la resolver decida. Aplicado también al `Dockerfile.tts`. Mismo cuidado con vllm-omni / sglang en futuras revisiones.

**Decisión #8 — Resolver de `python_bin` con prioridad triple para backends con venv legacy.**
Para `chatterbox`/`sglang`/`voxtral` (que ya tenían `settings.<name>_python_bin` apuntando a `/opt/<name>-venv` en la imagen fat), el `_resolve_python_bin()` consulta en orden:
1. `<backends_dir>/<name>/metadata.json` → `python_bin` (modular install)
2. `settings.<name>_python_bin` (fat legacy)
3. `sys.executable` (fallback)

Esto permite que el mismo `*_backend.py` funcione tanto en slim (modular) como en fat (legacy) sin condicionales por entorno.

**Decisión #9 — `vllm` cambia su launcher de `python` literal a `_resolve_python_bin()`.**
Antes hacía `cmd = ["python", "-m", "vllm.entrypoints..."]` — un `python` que solo existe en fat porque vllm está globalmente instalado. En slim el `python` del contenedor no tiene vllm, así que tira del venv en `/data/backends/vllm/venv/bin/python`. Mismo patrón aplicará a futuros backends LLM.

**Decisión #10 — Smoke test usa `ollama` para el caso "uninstall built-in".**
El test original probaba `POST /backends/whisper/uninstall` esperando 409 (built-in protegido). Tras migrar whisper, esa llamada DESINSTALA whisper (200) — eliminando el venv real (esto pasó accidentalmente en una corrida del smoke). Ahora el test usa `ollama` que es always-available y siempre rechaza uninstall. Lección: los smoke tests con side-effects deben apuntar a recursos verdaderamente inmutables.

**Deuda 9g — `BackendInstallSpec` necesita 4 campos para los nativos**:
- `apt_packages: list[str]` — para `ffmpeg`, `libsndfile1` y similares (hoy solo en imagen base).
- `extra_bins: dict[str, str]` — declara los binarios producidos (`{"llama_server": "bin/llama-server"}`); el installer los persiste en `metadata.json` y los backends los consumen.
- `git_repo: str | None` + `git_ref: str | None` — para `acestep` que parte de un repo upstream.
- `post_install_script: str | None` (ya existe en el dataclass pero el installer no lo ejecuta) — necesario para los builds nativos y el `uv sync` de acestep.

Bloquea la migración de `acestep`, `bitnet`, `llama_cpp` y `tensorrt_llm` (los 4 pendientes de Fase 2).

**Deuda 9h — `_derive_version()` cosmético.**
Para `tts` el state reporta `installed_version="torch>=2.5"` porque la heurística toma el primer paquete pinneado y `torch>=2.5` es el primero. No bloqueante, pero visualmente raro. Mejorar: preferir un paquete cuyo nombre coincida con el `backend_type` o el primer paquete que NO esté en la lista del core_runtime.

### Hito Validación e2e — Slim image runtime (2026-04-25)

Sesión completa de validación e2e de los 12 backends sobre slim. Estado tras
los fixes aplicados durante la validación:

| Backend | Install | Load | Request | Notas |
|---------|---------|------|---------|-------|
| ollama | built-in | n/a | ✅ chat | OK |
| whisper | ✅ source | ✅ GPU 1 (500 MB) | ✅ STT | Necesitó fix LD_LIBRARY_PATH (D14) |
| tts | ✅ source | ✅ GPU 1 (1 GB) | ✅ Kokoro WAV 24 kHz | Cableado D14 |
| chatterbox | ✅ source | ✅ GPU 1 (4 GB) | ✅ TTS WAV | Cableado D14 |
| diffusers | ✅ source | ⏭️ skip | ⏭️ skip | No hay modelo en formato diffusers en `/data/models` |
| vllm | ✅ source | ✅ GPU 1 (2 GB) | ✅ chat "pong" | D11 fix: apt gcc/g++ + D14 |
| voxtral | ✅ source | ✅ GPU 1 (4 GB) | ✅ TTS WAV 24 kHz | D13 fix: pin `vllm==0.18.0 vllm-omni==0.18.0` + D14 |
| llama_cpp | ✅ source (cmake CUDA, ~2 min) | ✅ GPU 1 | ✅ chat "pong" | Ronda 1 validada runtime |
| bitnet | ✅ source (cmake CPU, ~1 min) | ✅ CPU | ✅ chat "Pong" | Ronda 1 validada runtime |
| acestep | ✅ source (torch + uv sync, ~3 min) | ⏸️ pesos | ⏸️ pendiente | Install OK; generación requiere descargar `acestep-v15-turbo` (~7 GB) en primera carga |
| sglang | ✅ source | ❌ JIT | n/a | **D12 parcial**: helper `venv_cuda_home()` aplicado, `LD_LIBRARY_PATH` + `CUDA_HOME` cableados, pero `sgl-kernel` JIT requiere ``nvcc`` real (no incluido en wheels pip — necesita ``apt cuda-nvcc-12-4`` desde el repo NVIDIA, ~2 GB) |
| tensorrt_llm | ⚠️ built-in | ❌ n/a | n/a | Diferido a Fase 3 OCI (sin trtllm-serve en slim) |

**8/12 funcionando e2e**, **1 install OK pero sin modelo descargado** (diffusers), **1 install OK pero pesos pendientes** (acestep), **1 bloqueado por nvcc real** (sglang), **1 diferido** (tensorrt_llm).

#### Cambios de código aplicados durante la validación

1. **`venv_nvidia_ld_library_path(backends_dir, backend_type)`** — helper público
   en `core/backend_installer.py` que devuelve un `LD_LIBRARY_PATH` con todas
   las dirs `nvidia/*/lib/` del venv del backend. Cableado en
   `whisper`, `tts`, `diffusers`, `chatterbox`, `vllm`, `voxtral`, `sglang`
   antes de lanzar el subprocess (Deuda **D14**).
2. **`venv_cuda_home(backends_dir, backend_type)`** — construye un fake
   `CUDA_HOME` en `<backend>/cuda_home/` con symlinks (`lib64/` → wheel
   `cuda_runtime/lib`, `include/` → wheel `cuda_runtime/include`,
   opcionalmente `bin/` si hay `nvcc` REAL). Usado por `sglang` (Deuda **D12**).
3. **`_resolve_script_path()`** — añadidas más bases de búsqueda para que el
   `post_install_script` resuelva tanto en repo (`backend/scripts/...`) como
   en el contenedor (`/app/scripts/...`).
4. **`_derive_version()`** — heurística mejorada (Ronda 2, ya documentada).
5. **vllm spec**: `apt_packages=["gcc", "g++"]` para Triton JIT (Deuda **D11**).
6. **voxtral spec**: pin exacto `vllm==0.18.0 vllm-omni==0.18.0` (Deuda **D13**).
7. **sglang spec**: `apt_packages=["gcc","g++","ninja-build"]` + wheels
   `nvidia-cuda-{nvcc,runtime,cccl}-cu12` (Deuda **D12** parcial).
8. **`Dockerfile.slim`**: añadidos `build-essential`, `gcc`, `g++`, `cmake`,
   `git`, `ninja-build` (~250 MB, sigue siendo ~50× más pequeño que el fat).
   Permite que las JIT compilations y los cmake builds nativos no requieran
   apt-install runtime.

#### Deudas abiertas tras la validación

- **D12 sglang** — bloqueado por `nvcc` real. Solución conocida: añadir el
  repo NVIDIA al `Dockerfile.slim` y `apt install cuda-nvcc-12-4
  cuda-cudart-dev-12-4` (~2 GB, más cerca del slim "ligero" que del fat).
  No se aplicó en esta ronda porque añade dependencia del repo NVIDIA y
  reabre la decisión de Fase 3 OCI (donde el problema desaparece — la imagen
  OCI de sglang ya trae nvcc).
- **diffusers** — bloqueado por ausencia de modelo en formato diffusers en
  `/data/models`. Probar con descarga puntual de `stabilityai/sd-turbo`
  (~1.5 GB) o un SDXL del Hub.
- **acestep** — generación e2e pendiente; primera carga descargará el
  `acestep-v15-turbo` (~7 GB). No bloqueante.
- **`tensorrt_llm`** — diferido a Fase 3 OCI (sin solución en slim/source).

### Hito Ronda 2 — Deudas técnicas menores (2026-04-25)

**Avances**:
- **Deuda 9h** (`_derive_version` cosmético): nueva heurística que (1) prefiere un paquete cuyo nombre coincide con `backend_type`, (2) si no, el primer pin `==` que NO esté en una lista de paquetes core/comunes (`torch`, `numpy`, `transformers`, `fastapi`, etc.), (3) cae a la lista cruda. El spec de `tts` ya no reportará `installed_version="torch>=2.5"` sino el pin de qwen-tts/kokoro. Helpers `_req_name()` y constante `_CORE_RUNTIME_NAMES` añadidos. 3 tests nuevos.
- **Deuda #1** (`WorkerPool.registered_backends()` público): ya existía, `main.py:330` ya lo consume. Marcada como cerrada.
- **Deuda #4** (mock fallback frontend off-by-default): `backendsStore.ts` ya solo activa el mock con `VITE_MOCK_BACKENDS=1`; ningún fallback automático ante 404. El catch del fetch deja `usingMock: false` y un `error` real visible. Comentario aclarado.
- **Deuda #5** (badges de backend en Models): nuevo `BackendBadge.tsx` con pill por estado (built-in / installed / installing / not_installed / error) y tooltip con versión. Cableado en `ModelCard.tsx` reemplazando el texto plano de `model.backendType`. `Models.tsx` ahora pre-fetcha `backendsStore` en bootstrap para que los badges tengan datos en el primer paint.
- **Deuda #8** (granularidad SSE de pip): nuevo método async generator `_run_pip_install()` que ejecuta pip y parsea stdout para detectar líneas `Collecting <pkg>`, `Downloading <wheel>`, `Installing collected packages: …`, `Successfully installed`. Cada evento muta `state.install_detail` y emite snapshot al SSE. El progreso interpola con curva de retorno decreciente (`1 - 0.7^n`) entre `progress_start` y `progress_end` para no saturar al 100% antes de tiempo. Flag de instancia `_pip_progress_enabled` permite desactivarlo en tests que mockean `_run_subprocess`. 1 test nuevo.

**Tests**: 39/42 verde en local (los mismos 3 fallos preexistentes por `fastapi` ausente en el sandbox).

**Deudas que NO se tocan en Ronda 2** (decisión consciente):
- **Deuda #6** (ruff sweep en `api/internal/`, 247 errores B008/ASYNC230/240/F401/I001): merece un PR aparte. Mezclarlo con Ronda 2 enturbiaría el diff.
- **Deuda #7** (volumen `backends_data` documentado en `docs/INSTALL.md`): pendiente para promoción a GA, no bloquea nada hoy.
- **Validación runtime** de los 3 backends nativos sobre slim (Ronda 1): sigue pendiente porque cada cmake build tarda minutos y necesita la imagen slim arrancada con BD/Redis. Trabajo manual.

### Hito Ronda 1 — Cierre de Fase 2 con los 3 backends nativos (2026-04-25)

**Avances**:
- `BackendInstallSpec` extendido (cubre la **Deuda 9g** completa): `apt_packages`, `git_repo`, `git_ref`, `git_recursive`, `extra_bins`. El campo `post_install_script` que ya existía pero no se ejecutaba ahora se ejecuta de verdad.
- `BackendInstaller._install_from_source` reescrito para la nueva pipeline:
  apt → git clone → (opcional) venv → pip → post_install_script (con env `BACKEND_DIR`/`VENV_DIR`/`PYTHON_BIN`/`SRC_DIR`/`BIN_DIR`) → validación de `extra_bins` → metadata → register.
- `read_backend_metadata(backends_dir, backend_type)` exportado como helper público para que cada backend resuelva sus paths sin reimplementar el patrón.
- Backends nativos migrados:
  - `llama_cpp` — `_resolve_server_bin()` con prioridad metadata → settings legacy.
  - `bitnet` — mismo patrón. `LD_LIBRARY_PATH` ahora incluye también el bin dir modular.
  - `acestep` — `_resolve_project_paths()` devuelve `(src_dir, python_bin)` desde metadata o cae al `acestep_project_dir` legacy.
- Scripts post-install nuevos en `backend/scripts/`:
  - `install_llama_cpp.sh` — cmake CUDA build (auto-detecta toolkit, fallback CPU si no hay nvcc).
  - `install_bitnet.sh` — copia kernel headers pretuned + cmake. Reemplaza al `build_bitnet.sh` original (que clonaba solo); `BackendInstaller` ahora hace el clone.
  - `install_acestep.sh` — instala `uv` si falta + `uv sync` contra `${VENV_DIR}` con `UV_PROJECT_ENVIRONMENT`.
- `tensorrt_llm` deja explícito en su docstring por qué NO declara `install_spec` (NGC vendor-lock, esperar a `method=oci` de Fase 3).
- Tests:
  - `tests/core/test_backend_installer.py` +9 casos (apt avail/no-avail, git clone shape, post-install env vars, repo-relative path resolve, extra_bins missing/traversal, no-venv path, helper público).
  - `tests/core/test_native_install_specs.py` nuevo (13 casos): shape de cada spec + scripts presentes y ejecutables + resolver priority.
  - 100/103 verde en local; los 3 rojos preexistían (requieren `fastapi` que no está en el Python local del sandbox).

**Deudas técnicas**:
- **Validación end-to-end runtime no ejecutada en slim** para los 3 nuevos. Cada cmake build tarda 5–15 min (llama_cpp/bitnet) y `uv sync` de acestep descarga modelos pesados. Bloqueante para cerrar Fase 2 al 100%: `POST /ocabra/backends/llama_cpp/install method=source` desde el contenedor slim, mismo para bitnet y acestep, y verificar que `load()` levanta el worker.
- `apt_packages` mete `build-essential + cmake + git + ninja` en el contenedor slim en runtime (~500 MB en `/var`). Es idempotente y los paquetes desaparecen si se recrea el contenedor — está bien para v1, pero reduce la "slim-ness" temporalmente. Cuando `method=oci` esté operativo (Fase 3), llama_cpp/bitnet preferirán OCI y este coste desaparece.
- `_derive_version` sigue cosmético (Deuda 9h pendiente, prevista en Ronda 2).
- `WorkerPool._backends` accedido aún como privado desde `main.py` (Deuda #1, Ronda 2).

**Cuestiones pendientes**:
- ¿Variantes CUDA archs por defecto? `install_llama_cpp.sh` define `61;70;75;80;86;89` (RTX 3060 → RTX 4090, L4/L40S). Si el target es Hopper/Blackwell habrá que extender o exponer el override por env.
- `install_bitnet.sh` por defecto `BITNET_ENABLE_CUDA=false` (los kernels pretuned son CPU-first). Documentar en la UI cómo cambiar.

### Hito final Fase 4 — load() validado end-to-end en slim (2026-04-25)

Pipeline completo verificado:

1. Slim image arrancada (`ocabra-api:slim`, 987 MB) con `BACKENDS_FAT_IMAGE=false`.
2. `GET /ocabra/backends` → whisper marcado `not_installed` (los otros 11 siguen `built-in` hasta migrarse).
3. `POST /ocabra/backends/whisper/install method=source` → SSE con progreso, venv creado, pip instala torch CUDA + faster-whisper + nemo + pyannote. Tras descubrir que faltaba `python-multipart` en el core runtime (commit `2f921a3`), el venv quedó completo en 6.5 GB.
4. `metadata.json` escrito automáticamente por el installer; tras restart el state es `installed/source/1.2.1`.
5. `POST /ocabra/models/whisper/Systran/faster-whisper-base/load` → worker subprocess arrancado vía `/data/backends/whisper/venv/bin/python`, modelo cargado en GPU 1 con 500 MB VRAM en **35 segundos** desde slim. `audio_transcription: true`.

Tamaños comparativos:

| Imagen | Tamaño | Build con cache | Build limpio |
|--------|--------|-----------------|--------------|
| `ocabra-api:fat` (Dockerfile) | 51.1 GB | ~5 min | ~40 min |
| `ocabra-api:slim` (Dockerfile.slim) | 987 MB | ~30 s | ~2 min |
| Whisper venv on disk (slim runtime) | 6.5 GB | — | — |

La imagen "slim + whisper instalado" pesa 987 MB + 6.5 GB = 7.5 GB efectivos vs 51 GB del fat. Cuando se migren los demás backends, cada uno aporta su disco SOLO si el usuario lo instala — patrón pay-as-you-use.
