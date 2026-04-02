# Estado: Stream 1-B — Model Manager & Worker Pool

Estado: completado.

Área principal:
- `backend/ocabra/core/model_manager.py`
- `backend/ocabra/core/model_manager_helpers.py`
- `backend/ocabra/core/worker_pool.py`
- `backend/ocabra/backends/base.py`

Resultado actual:
- lifecycle de carga y descarga estable
- `last_request_at` persistido
- forwarding y streaming endurecidos
- reconciliación de huérfanos en backends que lo requieren
- helpers extraídos para reducir tamaño y acoplamiento

Pendiente real:
- limpieza estructural adicional solo si vuelve a priorizarse deuda técnica

Referencia viva:
- `docs/PLAN.md`
- `docs/CONTRACTS.md`
