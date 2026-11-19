"""La espera a que cargue el modelo va a la telemetría de la petición.

Solo el chat con streaming la anotaba (``mark_load``): una petición sin
streaming o una línea de batch que esperaba 17 s a una carga en frío quedaba
como una petición de 18 s, y el estimador de duración aprendía de ella (19/11:
una sola dejó la mediana de qwen3:8b en 17,9 s para un servicio de 0,8 s).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.core.request_telemetry import RequestRuntimeTelemetry, current_runtime_telemetry


async def _ensure(waited_s: float, telemetry: RequestRuntimeTelemetry | None):
    from ocabra.api.openai import _deps

    state = MagicMock()
    token = current_runtime_telemetry.set(telemetry)
    try:
        with (
            patch.object(_deps, "_do_ensure_loaded_inner", AsyncMock(return_value=state)),
            patch.object(_deps.time, "monotonic", side_effect=[100.0, 100.0 + waited_s]),
        ):
            assert await _deps._do_ensure_loaded(MagicMock(), "m") is state
    finally:
        current_runtime_telemetry.reset(token)


@pytest.mark.asyncio
async def test_a_cold_load_wait_is_noted_as_load_ms():
    telemetry = RequestRuntimeTelemetry(started_monotonic=0.0)
    await _ensure(16.7, telemetry)
    assert telemetry.load_ms == 16700
    assert telemetry.as_dict()["load_ms"] == 16700


@pytest.mark.asyncio
async def test_a_warm_model_notes_nothing():
    telemetry = RequestRuntimeTelemetry(started_monotonic=0.0)
    await _ensure(0.01, telemetry)
    assert telemetry.load_ms is None


@pytest.mark.asyncio
async def test_the_streaming_measure_wins():
    telemetry = RequestRuntimeTelemetry(started_monotonic=0.0)
    telemetry.mark_load(17031)
    await _ensure(16.9, telemetry)
    assert telemetry.load_ms == 17031


@pytest.mark.asyncio
async def test_without_a_request_it_does_not_fail():
    await _ensure(5.0, None)


def test_the_estimator_learns_from_service_time_not_the_wait():
    from ocabra.core import duration_estimator as de

    assert "load_ms" in de._SERVICE_MS_SQL and "queue_ms" in de._SERVICE_MS_SQL
