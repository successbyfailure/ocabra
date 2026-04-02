# Estado: Stream 1-A — GPU Manager & Scheduler

Estado: completado.

Área principal:
- `backend/ocabra/core/gpu_manager.py`
- `backend/ocabra/core/scheduler.py`
- `backend/ocabra/api/internal/gpus.py`

Resultado actual:
- monitorización NVML activa
- locking de VRAM y reconciliación con modelos cargados
- scheduler de idle eviction, pressure eviction y ventanas persistidas
- `/ocabra/gpus/*` y stats alineados con el runtime actual

Pendiente real:
- tuning fino de observabilidad y métricas de ejecución de schedules

Referencia viva:
- `docs/PLAN.md`
- `docs/CONTRACTS.md`
