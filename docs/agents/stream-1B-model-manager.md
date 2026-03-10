# Briefing: Stream 1-B — Model Manager & Worker Pool

**Prerequisito: Fase 0 completada.**
**Rama:** `feat/1-B-model-manager`

## Objetivo

Implementar la máquina de estados de modelos, el pool de workers (subprocesos),
y la lógica de carga/descarga/pin. Es el núcleo de oCabra.

## Ficheros propios

```
backend/ocabra/core/model_manager.py
backend/ocabra/core/worker_pool.py
backend/ocabra/backends/base.py
backend/ocabra/api/internal/models.py
backend/tests/test_model_manager.py
backend/tests/test_worker_pool.py
```

## Ficheros compartidos que tocas

- `backend/ocabra/main.py` — añade el router de models en la sección `# ROUTERS`
- `backend/ocabra/db/model_config.py` — usa el modelo existente, no lo cambies sin consenso

## Contratos a implementar

Ver `docs/CONTRACTS.md`:
- §1: `BackendInterface` y `WorkerInfo` (implementa la clase abstracta)
- §3: `ModelState`, `ModelStatus`, `LoadPolicy`
- §4: `WorkerPool`
- §5.1: endpoints `/ocabra/models/*`
- §6: publicar en Redis canal `model:events` y key `model:state:{model_id}`

## Funcionalidades requeridas

### model_manager.py

```python
class ModelManager:
    async def start(self) -> None:
        """Al arrancar: carga modelos con load_policy='pin' de la BD."""

    async def load(self, model_id: str, force_gpu: int | None = None) -> ModelState:
        """
        1. Determina backend_type del modelo
        2. Pide a GPUScheduler la GPU asignada
        3. Si no hay VRAM libre, orquesta evicción
        4. Llama a backend.load()
        5. Actualiza estado y publica evento en Redis
        """

    async def unload(self, model_id: str, reason: str = "manual") -> None:

    async def get_state(self, model_id: str) -> ModelState | None:

    async def list_states(self) -> list[ModelState]:

    async def update_config(self, model_id: str, patch: dict) -> ModelState:

    async def on_request(self, model_id: str) -> None:
        """Actualiza last_request_at. Si el modelo está UNLOADED, lo carga."""

    async def check_idle_evictions(self) -> None:
        """Llamado periódicamente. Descarga modelos on_demand con idle > timeout."""

    async def check_vram_pressure(self, gpu_index: int) -> None:
        """Si VRAM libre < threshold, inicia evicción de candidatos."""
```

### worker_pool.py

```python
class WorkerPool:
    def register_backend(self, backend_type: str, backend: BackendInterface) -> None:

    async def get_backend(self, backend_type: str) -> BackendInterface:

    async def get_worker(self, model_id: str) -> WorkerInfo | None:

    async def assign_port(self) -> int:

    async def release_port(self, port: int) -> None:

    async def forward_request(
        self, model_id: str, path: str, body: dict, stream: bool = False
    ) -> Any:
        """Proxy a httpx hacia el worker en localhost:{port}{path}."""
```

### Endpoints requeridos

```
GET    /ocabra/models
GET    /ocabra/models/{model_id}
POST   /ocabra/models/{model_id}/load
POST   /ocabra/models/{model_id}/unload
PATCH  /ocabra/models/{model_id}
DELETE /ocabra/models/{model_id}
```

### Tareas periódicas (APScheduler)

- Cada 30s: `check_idle_evictions()`
- Cada 5s: `check_vram_pressure()` para cada GPU
- Suscripción a Redis `gpu:stats` para detectar presión en tiempo real

## Importante: no implementes backends reales todavía

`BackendInterface` es abstracta. Crea un `MockBackend` para tests que simula carga/descarga sin procesos reales. Los backends reales (vLLM, etc.) los implementan los streams 2-A, 2-B, 2-C.

## Tests requeridos

- Ciclo completo con MockBackend: CONFIGURED → LOADING → LOADED → UNLOADING → UNLOADED
- Evicción por idle: modelo on_demand sin actividad > timeout se descarga
- Recarga automática de pin tras evicción con auto_reload=true
- WorkerPool: asignación de puertos sin colisiones

## Dependencias que consumes

- `GPUManager` y `GPUScheduler` de `core/gpu_manager.py` y `core/scheduler.py`
- `settings` de `ocabra/config.py`
- `redis_client` de `ocabra/redis_client.py`
- `AsyncSession` de `ocabra/database.py`

## Estado

- [ ] En progreso
- [ ] Completado
