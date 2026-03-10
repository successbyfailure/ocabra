# Briefing: Stream 1-A — GPU Manager & Scheduler

**Prerequisito: Fase 0 completada.**
**Rama:** `feat/1-A-gpu-manager`

## Objetivo

Implementar el sistema de monitoreo y asignación de GPUs. Es el cerebro que
decide qué modelo va en qué GPU, detecta presión de VRAM, y gestiona la
evicción por horario.

## Ficheros propios

```
backend/ocabra/core/gpu_manager.py
backend/ocabra/core/scheduler.py
backend/ocabra/api/internal/gpus.py
backend/tests/test_gpu_manager.py
backend/tests/test_scheduler.py
```

## Ficheros compartidos que tocas

- `backend/ocabra/main.py` — añade el router de gpus en la sección `# ROUTERS`
- `backend/ocabra/db/stats.py` — añade el modelo `GpuStat` si no existe

## Contratos a implementar

Ver `docs/CONTRACTS.md`:
- §2: `GPUState` dataclass
- §5.2: endpoints `/ocabra/gpus/*`
- §6: publicar en Redis canal `gpu:stats` y key `gpu:state:{index}`

## Funcionalidades requeridas

### gpu_manager.py

```python
class GPUManager:
    async def start(self) -> None:
        """Inicia polling de GPUs cada 2 segundos. Publica en Redis."""

    async def stop(self) -> None:

    async def get_all_states(self) -> list[GPUState]:

    async def get_state(self, index: int) -> GPUState:

    async def lock_vram(self, gpu_index: int, amount_mb: int, model_id: str) -> None:
        """Registra VRAM reservada por un modelo al cargarse."""

    async def unlock_vram(self, gpu_index: int, model_id: str) -> None:
        """Libera la VRAM al descargar el modelo."""

    async def get_free_vram(self, gpu_index: int) -> int:
        """VRAM real libre = total - used - locked - buffer(512MB)."""
```

### scheduler.py

```python
class GPUScheduler:
    async def find_gpu_for_model(
        self,
        vram_needed_mb: int,
        preferred_gpu: int | None
    ) -> list[int]:
        """
        Retorna lista de GPUs asignadas.
        - 1 GPU si cabe en la preferida o alternativa
        - [0, 1] si requiere tensor parallelism
        - Lanza InsufficientVRAMError si no hay forma
        """

    async def get_eviction_candidates(
        self,
        vram_needed_mb: int,
        target_gpu: int
    ) -> list[str]:
        """
        Retorna model_ids ordenados por prioridad de evicción:
        on_demand idle > on_demand reciente > warm > pin
        """

    async def check_schedule_evictions(self) -> None:
        """Llamado por APScheduler. Descarga modelos warm/pin en ventanas activas."""

    async def check_schedule_reloads(self) -> None:
        """Llamado por APScheduler. Recarga modelos pin/warm al salir de ventana."""
```

### Endpoints requeridos

```
GET /ocabra/gpus
GET /ocabra/gpus/{index}
GET /ocabra/gpus/{index}/stats?window=5m|1h|24h
```

El endpoint `/stats` lee de la tabla `gpu_stats` de PostgreSQL.

### Persistencia de stats

Cada minuto, agrega los últimos 30 polling y escribe una fila en `gpu_stats`.
Usar la función en `stats/aggregator.py` (si el agente de stats no existe aún, crea un helper mínimo).

### WebSocket events

Cuando el polling detecta cambio en VRAM > 100MB o temperatura > 5°C, emitir
evento `gpu_stats` al canal Redis `gpu:stats`. El WS handler (implementado en Fase 0 o
por el agente de Model Manager) lo retransmite a los clientes.

## Tests requeridos

- Mock de pynvml para simular lecturas de GPU
- Test de asignación: modelo de 8GB en sistema con 3060(2GB libre) + 3090(20GB libre)
- Test de evicción: orden correcto de candidatos
- Test de schedule: mock APScheduler, verificar que se llama check_schedule_evictions

## Dependencias que consumes

- `settings` de `ocabra/config.py` (DEFAULT_GPU_INDEX, WORKER_PORT_RANGE_*)
- `redis_client` de `ocabra/redis_client.py`
- `AsyncSession` de `ocabra/database.py`

## Estado

- [ ] En progreso
- [ ] Completado
