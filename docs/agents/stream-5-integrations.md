# Briefing: Stream 5 — Integrations & Polish

**Prerequisito: Todas las fases anteriores completadas.**
**Rama:** `feat/5-integrations`

## Objetivo

LiteLLM auto-sync, métricas Prometheus, logging estructurado, healthchecks,
tests de integración end-to-end, y documentación final.

## Ficheros propios

```
backend/ocabra/integrations/litellm_sync.py
backend/ocabra/stats/collector.py        (refinar)
backend/ocabra/stats/gpu_power.py
backend/ocabra/stats/aggregator.py
backend/ocabra/api/internal/config.py    (completar con sync endpoint)
backend/tests/integration/
backend/ocabra/api/metrics.py            (Prometheus)
```

## LiteLLM Sync

```python
class LiteLLMSync:
    async def sync_all(self) -> SyncResult:
        """
        1. Lista todos los modelos LOADED en oCabra
        2. Genera config LiteLLM para cada uno:
           {
             "model_name": model.display_name,
             "litellm_params": {
               "model": f"openai/{model.model_id}",
               "api_base": "http://ocabra:8000/v1",
               "api_key": "ocabra-internal"
             }
           }
        3. PATCH /config/update_config en LiteLLM proxy API
        4. Retorna { synced: int, errors: list }
        """

    async def on_model_event(self, event: ModelEvent) -> None:
        """
        Hook que se llama cuando un modelo cambia de estado.
        Si auto_sync=True y el evento es LOADED o UNLOADED, llama sync_all().
        """
```

## Prometheus Metrics

```
GET /metrics  → formato Prometheus text

# Métricas a exponer:
ocabra_requests_total{model, status}
ocabra_request_duration_seconds{model, quantile}
ocabra_tokens_total{model, type}  # type=input|output
ocabra_gpu_vram_used_bytes{gpu_index}
ocabra_gpu_utilization_percent{gpu_index}
ocabra_gpu_power_watts{gpu_index}
ocabra_gpu_temperature_celsius{gpu_index}
ocabra_models_loaded{backend_type}
ocabra_energy_joules_total{gpu_index}
```

Usar `prometheus-client` Python library.

## Logging estructurado

Configurar `structlog` en `main.py`:
- JSON en producción, colorized en desarrollo
- Context vars: `request_id`, `model_id`, `gpu_index` en cada log de request
- Nivel configurable via `LOG_LEVEL` env var

## Healthchecks

```
GET /health   → {"status": "ok", "version": "x.y.z"}  # siempre 200, rápido
GET /ready    → {"status": "ready", "checks": {...}}   # 200 si todo OK, 503 si no
  checks:
    postgres: "ok"|"error"
    redis: "ok"|"error"
    gpu_manager: "ok"|"error"
    models_loaded: 3
```

## Tests de integración

```
backend/tests/integration/
├── test_openai_chat.py      # OpenAI SDK → chat completo con modelo real o mock pesado
├── test_ollama_chat.py      # ollama client → chat
├── test_model_lifecycle.py  # load → request → idle → unload → reload
├── test_gpu_pressure.py     # llenar VRAM → verificar evicción correcta
└── test_litellm_sync.py     # mock LiteLLM API → verificar payload correcto
```

## Script de first-run

```bash
# scripts/setup.sh
# 1. Copia .env.example a .env si no existe
# 2. Genera SECRET_KEY random
# 3. docker compose up -d postgres redis
# 4. docker compose run --rm api alembic upgrade head
# 5. docker compose up -d
# 6. Imprime URL de acceso
```

## Estado

- [ ] En progreso
- [ ] Completado
