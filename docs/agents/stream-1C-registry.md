# Briefing: Stream 1-C — Model Registry & Download Manager

**Prerequisito: Fase 0 completada.**
**Rama:** `feat/1-C-registry`

## Objetivo

Implementar la integración con HuggingFace Hub y el registro de Ollama,
más el gestor de descargas con progreso en tiempo real por SSE.

## Ficheros propios

```
backend/ocabra/registry/huggingface.py
backend/ocabra/registry/ollama_registry.py
backend/ocabra/registry/local_scanner.py
backend/ocabra/api/internal/downloads.py
backend/ocabra/api/internal/registry.py
backend/tests/test_registry.py
backend/tests/test_downloads.py
```

## Ficheros compartidos que tocas

- `backend/ocabra/main.py` — añade routers en sección `# ROUTERS`

## Contratos a implementar

Ver `docs/CONTRACTS.md`:
- §5.3: endpoints `/ocabra/downloads/*`
- §5.4: endpoints `/ocabra/registry/*`
- §6: keys Redis `download:job:{job_id}`, canal `download:progress:{job_id}`

## Schemas de datos

```python
# ocabra/schemas/registry.py

class HFModelCard(BaseModel):
    repo_id: str
    model_name: str
    task: str | None              # text-generation, image-generation, etc.
    downloads: int
    likes: int
    size_gb: float | None
    tags: list[str]
    gated: bool

class HFModelDetail(HFModelCard):
    siblings: list[dict]          # ficheros del repo
    readme_excerpt: str | None
    suggested_backend: str        # "vllm" | "diffusers" | "whisper" | "tts"
    estimated_vram_gb: float | None

class OllamaModelCard(BaseModel):
    name: str
    description: str
    tags: list[str]
    size_gb: float | None
    pulls: int

class DownloadJob(BaseModel):
    job_id: str
    source: Literal["huggingface", "ollama"]
    model_ref: str
    status: Literal["queued", "downloading", "completed", "failed", "cancelled"]
    progress_pct: float
    speed_mb_s: float | None
    eta_seconds: int | None
    error: str | None
    started_at: datetime
    completed_at: datetime | None
```

## Funcionalidades requeridas

### huggingface.py

```python
class HuggingFaceRegistry:
    async def search(self, query: str, task: str | None, limit: int) -> list[HFModelCard]:
        """Usa huggingface_hub.list_models() con filtros."""

    async def get_model_detail(self, repo_id: str) -> HFModelDetail:
        """
        Obtiene metadata completa.
        Infiere suggested_backend según pipeline_tag y ficheros presentes.
        Estima VRAM según tamaño de ficheros safetensors.
        """

    async def download(
        self,
        repo_id: str,
        target_dir: Path,
        progress_callback: Callable[[float, float], None]
    ) -> Path:
        """
        Descarga con huggingface_hub.snapshot_download().
        Llama progress_callback(pct, speed_mb_s) en cada chunk.
        """
```

### ollama_registry.py

```python
class OllamaRegistry:
    async def search(self, query: str) -> list[OllamaModelCard]:
        """Scrape o API de ollama.com/library."""

    async def pull(
        self,
        model_ref: str,
        progress_callback: Callable[[float, float], None]
    ) -> Path:
        """
        Usa el protocolo de descarga de Ollama (ollama pull via subprocess
        o API directa si está disponible).
        """
```

### Download Manager (en downloads.py)

El manager usa una cola Redis (`queue:download`) y workers asyncio:

```python
class DownloadManager:
    async def enqueue(self, source: str, model_ref: str) -> DownloadJob:

    async def cancel(self, job_id: str) -> None:

    async def get_job(self, job_id: str) -> DownloadJob | None:

    async def list_jobs(self) -> list[DownloadJob]:
```

### Endpoint SSE de progreso

```python
@router.get("/ocabra/downloads/{job_id}/stream")
async def stream_download_progress(job_id: str):
    """
    Server-Sent Events. Se suscribe al canal Redis download:progress:{job_id}
    y retransmite cada mensaje como evento SSE.
    """
```

### local_scanner.py

```python
class LocalScanner:
    async def scan(self, models_dir: Path) -> list[LocalModel]:
        """
        Escanea MODELS_DIR buscando:
        - Carpetas con config.json (HF format)
        - Ficheros .gguf
        - Ficheros Modelfile (Ollama format)
        Retorna lista de modelos locales con metadata inferida.
        """
```

## Tests requeridos

- Mock de huggingface_hub: test de búsqueda con filtros
- Test de inferencia de backend: repo con `pipeline_tag: text-generation` → `vllm`
- Test de DownloadManager: enqueue, progress updates en Redis, cancel
- Test de LocalScanner: directorio de fixtures con modelos de prueba

## Dependencias que consumes

- `settings.MODELS_DIR`, `settings.HF_TOKEN` de `ocabra/config.py`
- `redis_client` de `ocabra/redis_client.py`
- `huggingface_hub` (ya en dependencias)

## Estado

- [x] Completado
