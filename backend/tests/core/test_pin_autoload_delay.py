"""La autocarga de los PIN espera tras el arranque.

Al reiniciar con carga, la primera peticion y la autocarga de los PIN cargaban
a la vez y se peleaban por la VRAM. Ahora los PIN esperan
``pin_autoload_delay_s`` y despues a que acabe cualquier carga en curso, y solo
cargan los que siguen sin cargar.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.config import settings
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus


def _mm(*states: ModelState) -> ModelManager:
    mm = ModelManager(worker_pool=None)
    for st in states:
        mm._states[st.model_id] = st
    return mm


def _state(model_id: str, policy: LoadPolicy, status: ModelStatus) -> ModelState:
    return ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="llamacpp_mks",
        load_policy=policy,
        status=status,
    )


async def _run(mm: ModelManager, sleeps: list[float]) -> AsyncMock:
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds, *a, **k):
        sleeps.append(seconds)
        await real_sleep(0)

    load = AsyncMock()
    with (
        patch.object(mm, "_load_model", load),
        patch("ocabra.core.model_manager.asyncio.sleep", fake_sleep),
    ):
        await mm._autoload_pinned()
        await real_sleep(0)
        await real_sleep(0)
    return load


@pytest.mark.asyncio
async def test_los_pin_esperan_el_retraso_antes_de_cargar(monkeypatch):
    monkeypatch.setattr(settings, "pin_autoload_delay_s", 60)
    mm = _mm(_state("llamacpp_mks/gemma4:26b", LoadPolicy.PIN, ModelStatus.CONFIGURED))
    sleeps: list[float] = []

    load = await _run(mm, sleeps)

    assert sleeps[0] == 60
    load.assert_awaited_once_with("llamacpp_mks/gemma4:26b")


@pytest.mark.asyncio
async def test_un_pin_que_cargo_una_peticion_durante_el_retraso_no_se_recarga(monkeypatch):
    monkeypatch.setattr(settings, "pin_autoload_delay_s", 60)
    pin = _state("llamacpp_mks/gemma4:26b", LoadPolicy.PIN, ModelStatus.CONFIGURED)
    otro = _state("whisper/whisper-large-v3", LoadPolicy.PIN, ModelStatus.CONFIGURED)
    mm = _mm(pin, otro)
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds, *a, **k):
        if seconds == 60:  # durante el retraso llega una peticion
            pin.status = ModelStatus.LOADED
        await real_sleep(0)

    load = AsyncMock()
    with (
        patch.object(mm, "_load_model", load),
        patch("ocabra.core.model_manager.asyncio.sleep", fake_sleep),
    ):
        await mm._autoload_pinned()
        await real_sleep(0)
        await real_sleep(0)

    load.assert_awaited_once_with("whisper/whisper-large-v3")


@pytest.mark.asyncio
async def test_espera_a_que_acabe_la_carga_de_la_peticion(monkeypatch):
    monkeypatch.setattr(settings, "pin_autoload_delay_s", 0)
    pin = _state("llamacpp_mks/gemma4:26b", LoadPolicy.PIN, ModelStatus.CONFIGURED)
    mm = _mm(pin)
    mm._load_in_progress.add("llamacpp_mks/qwen3.8:27b")
    real_sleep = asyncio.sleep
    sleeps: list[float] = []

    async def fake_sleep(seconds, *a, **k):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            mm._load_in_progress.clear()
        await real_sleep(0)

    load = AsyncMock()
    with (
        patch.object(mm, "_load_model", load),
        patch("ocabra.core.model_manager.asyncio.sleep", fake_sleep),
    ):
        await mm._autoload_pinned()
        await real_sleep(0)
        await real_sleep(0)

    assert sleeps == [2, 2, 2]
    load.assert_awaited_once()


@pytest.mark.asyncio
async def test_warm_y_on_demand_no_se_autocargan(monkeypatch):
    monkeypatch.setattr(settings, "pin_autoload_delay_s", 0)
    mm = _mm(
        _state("a/warm", LoadPolicy.WARM, ModelStatus.CONFIGURED),
        _state("a/lazy", LoadPolicy.ON_DEMAND, ModelStatus.CONFIGURED),
    )
    load = await _run(mm, [])
    load.assert_not_awaited()


@pytest.mark.asyncio
async def test_sin_pin_pendientes_no_espera(monkeypatch):
    monkeypatch.setattr(settings, "pin_autoload_delay_s", 60)
    mm = _mm(_state("a/pin", LoadPolicy.PIN, ModelStatus.LOADED))
    sleeps: list[float] = []
    load = await _run(mm, sleeps)
    assert sleeps == []
    load.assert_not_awaited()
