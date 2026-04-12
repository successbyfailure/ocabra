# oCabra — Plan de Backends Modulares

Última actualización: 2026-04-12

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

### Fase 1 — Infraestructura base

- [ ] `BackendInstallSpec` dataclass en `backends/base.py`
- [ ] `BackendInstaller` en `core/backend_installer.py`
  - `start()`: escanear `/data/backends/`, detectar instalados, registrar
  - `install()` desde source (venv + pip): flujo básico
  - `uninstall()`: verificar que no hay modelos cargados, borrar directorio
  - `is_installed()`, `list_states()`, `get_state()`
- [ ] `install_spec` property en cada backend (puede retornar None)
- [ ] Settings: `backends_dir` en `config.py`
- [ ] Volumen `backends_data` en `docker-compose.yml`
- [ ] Adaptar `main.py`: usar BackendInstaller en lugar de registro hardcodeado
- [ ] API endpoints: `GET /ocabra/backends`, `POST .../install`, `POST .../uninstall`
- [ ] SSE de progreso de instalación (reutilizar patrón de downloads)
- [ ] Tests: install, uninstall, scan, register/deregister

### Fase 2 — Migrar backends a install_spec

Migrar uno a uno, empezando por los más simples:

- [ ] `tts` — solo pip packages (transformers, kokoro)
- [ ] `whisper` — solo pip packages (faster-whisper, soundfile)
- [ ] `diffusers` — pip packages (diffusers, accelerate, Pillow)
- [ ] `vllm` — pip packages (vllm, torch)
- [ ] `sglang` — ya tiene venv aislado, migrar a nuevo patrón
- [ ] `chatterbox` — ya tiene venv aislado, migrar
- [ ] `voxtral` — ya tiene venv aislado, migrar
- [ ] `llama_cpp` — binario nativo, necesita build script
- [ ] `bitnet` — binario nativo, necesita build script
- [ ] `acestep` — pip packages
- [ ] `tensorrt_llm` — caso especial (Docker-based), adaptar

Backends sin migración (siempre disponibles):
- `ollama` — servicio externo, no necesita instalación

### Fase 3 — Distribución OCI

- [ ] Dockerfiles por backend en `backends/dockerfiles/`
- [ ] CI pipeline para build y push a registry
- [ ] `install()` desde OCI: pull → create → cp → rm
- [ ] Hardware detection para elegir variante (cuda12/cpu)
- [ ] `metadata.json` con digest OCI para detección de updates
- [ ] Endpoint de check de actualizaciones

### Fase 4 — Imagen Docker slim

- [ ] Dockerfile base sin backends pre-instalados
- [ ] Solo incluye: Python 3.11, core oCabra, cmake, CUDA toolkit, docker CLI
- [ ] Mantener imagen "fat" como opción para quienes quieran todo pre-instalado
- [ ] Documentación de primer arranque (instalar backends desde la UI)

### Fase 5 — Frontend

- [ ] Página/sección "Backends" en Settings
- [ ] Cards con estado, versión, tamaño
- [ ] Botones install/uninstall con confirmación
- [ ] Barra de progreso SSE durante instalación
- [ ] Eventos WebSocket para cambios de estado
- [ ] Indicadores en la página Models (qué backend usa cada modelo)

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
