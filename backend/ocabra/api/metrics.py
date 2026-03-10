"""
GET /metrics — Prometheus metrics endpoint.

Exposes:
  ocabra_requests_total{model, status}
  ocabra_request_duration_seconds{model, quantile}
  ocabra_tokens_total{model, type}
  ocabra_gpu_vram_used_bytes{gpu_index}
  ocabra_gpu_utilization_percent{gpu_index}
  ocabra_gpu_power_watts{gpu_index}
  ocabra_gpu_temperature_celsius{gpu_index}
  ocabra_models_loaded{backend_type}
  ocabra_energy_joules_total{gpu_index}
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["metrics"])

# ── Metric definitions ─────────────────────────────────────

requests_total = Counter(
    "ocabra_requests_total",
    "Total requests served",
    ["model", "status"],
)

request_duration = Histogram(
    "ocabra_request_duration_seconds",
    "Request duration in seconds",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

tokens_total = Counter(
    "ocabra_tokens_total",
    "Total tokens processed",
    ["model", "type"],  # type=input|output
)

gpu_vram_used = Gauge(
    "ocabra_gpu_vram_used_bytes",
    "VRAM currently in use (bytes)",
    ["gpu_index"],
)

gpu_utilization = Gauge(
    "ocabra_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_index"],
)

gpu_power_watts = Gauge(
    "ocabra_gpu_power_watts",
    "GPU power draw in watts",
    ["gpu_index"],
)

gpu_temperature = Gauge(
    "ocabra_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_index"],
)

models_loaded = Gauge(
    "ocabra_models_loaded",
    "Number of currently loaded models",
    ["backend_type"],
)

energy_joules = Counter(
    "ocabra_energy_joules_total",
    "Total energy consumed in joules per GPU",
    ["gpu_index"],
)


@router.get("/metrics", summary="Prometheus metrics")
async def metrics(request: Request) -> Response:
    """
    Expose Prometheus-format metrics.

    Reads current state from GPUManager and ModelManager to update gauges
    before returning the metrics snapshot.
    """
    try:
        await _update_gauges(request)
    except Exception as exc:
        logger.warning("metrics_update_failed", error=str(exc))

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


async def _update_gauges(request: Request) -> None:
    """Refresh gauge metrics from live app state."""
    # GPU gauges
    try:
        gpu_manager = request.app.state.gpu_manager
        for gpu_state in gpu_manager.get_all_states():
            idx = str(gpu_state.index)
            gpu_vram_used.labels(gpu_index=idx).set(gpu_state.used_vram_mb * 1024 * 1024)
            gpu_utilization.labels(gpu_index=idx).set(gpu_state.utilization_pct)
            gpu_power_watts.labels(gpu_index=idx).set(gpu_state.power_draw_w)
            gpu_temperature.labels(gpu_index=idx).set(gpu_state.temperature_c)
    except Exception:
        pass

    # Model gauges
    try:
        from ocabra.core.model_manager import ModelStatus

        model_manager = request.app.state.model_manager
        states = await model_manager.list_states()
        loaded = [s for s in states if s.status == ModelStatus.LOADED]

        # Reset and recount by backend type
        by_backend: dict[str, int] = {}
        for s in loaded:
            by_backend[s.backend_type] = by_backend.get(s.backend_type, 0) + 1

        for backend, count in by_backend.items():
            models_loaded.labels(backend_type=backend).set(count)
    except Exception:
        pass


def record_request(model_id: str, duration_s: float, status: str = "ok") -> None:
    """Called from stats middleware to update request counters."""
    requests_total.labels(model=model_id, status=status).inc()
    request_duration.labels(model=model_id).observe(duration_s)


def record_tokens(model_id: str, input_tokens: int, output_tokens: int) -> None:
    """Record token usage for a completed request."""
    if input_tokens:
        tokens_total.labels(model=model_id, type="input").inc(input_tokens)
    if output_tokens:
        tokens_total.labels(model=model_id, type="output").inc(output_tokens)
