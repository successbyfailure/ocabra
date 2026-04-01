from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api.metrics import router as metrics_router
from ocabra.core.gpu_manager import GPUState


def test_metrics_endpoint_refreshes_gpu_gauges() -> None:
    app = FastAPI()
    app.include_router(metrics_router)
    app.state.gpu_manager = SimpleNamespace(
        get_all_states=AsyncMock(
            return_value=[
                GPUState(
                    index=0,
                    name="RTX 3090",
                    total_vram_mb=24576,
                    free_vram_mb=12288,
                    used_vram_mb=12288,
                    utilization_pct=42.0,
                    temperature_c=66.0,
                    power_draw_w=175.0,
                    power_limit_w=370.0,
                    locked_vram_mb=0,
                )
            ]
        )
    )
    app.state.model_manager = SimpleNamespace(list_states=AsyncMock(return_value=[]))

    client = TestClient(app)
    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert 'ocabra_gpu_vram_used_bytes{gpu_index="0"}' in resp.text
    assert 'ocabra_gpu_utilization_percent{gpu_index="0"}' in resp.text
    assert 'ocabra_gpu_temperature_celsius{gpu_index="0"}' in resp.text
