"""Esperas de "VRAM liberada" del árbitro (REF-03): terminan en cuanto la
lectura muestra la memoria, nunca por un tiempo fijo, y siempre acotadas."""

from __future__ import annotations

import asyncio

import pytest

from ocabra.core.gpu_manager import GPUManager, GPUState
from ocabra.core.vram_arbiter import wait_for_reading


def _reader(values):
    it = iter(values)
    calls = {"n": 0}

    async def read():
        calls["n"] += 1
        return next(it)

    return read, calls


async def test_wait_ends_as_soon_as_the_reading_reaches_the_target():
    read, calls = _reader([1, 2, 10, 99])
    assert await wait_for_reading(read, target=10, max_s=5, poll_s=0) == 10
    assert calls["n"] == 3


async def test_wait_uses_the_initial_reading_first():
    read, calls = _reader([10])
    assert await wait_for_reading(read, target=10, max_s=5, poll_s=0, initial=3) == 10
    assert calls["n"] == 1


async def test_wait_ends_once_the_release_stops_moving():
    async def read():
        return 5

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await wait_for_reading(read, target=10, max_s=5, poll_s=0.01, settle_s=0.05) == 5
    assert loop.time() - started < 1


async def test_wait_is_bounded():
    n = {"v": 0}

    async def read():
        n["v"] += 1  # keeps moving, never enough
        return n["v"]

    loop = asyncio.get_running_loop()
    started = loop.time()
    await wait_for_reading(read, target=10**9, max_s=0.1, poll_s=0.01, settle_s=1)
    assert loop.time() - started < 1


async def test_no_reading_ends_the_wait():
    read, _ = _reader([None])
    assert await wait_for_reading(read, target=10, max_s=5, poll_s=0) is None


def _gm(live: int | None) -> GPUManager:
    gm = GPUManager()
    gm._states[1] = GPUState(
        index=1,
        name="3090",
        total_vram_mb=24_576,
        free_vram_mb=1_000,
        used_vram_mb=23_576,
        utilization_pct=0,
        temperature_c=40,
        power_draw_w=50,
        power_limit_w=350,
    )
    gm.live_free_vram_mb = lambda index: live  # type: ignore[method-assign]
    return gm


@pytest.mark.parametrize("live", [20_000, 0, 99_999])
async def test_refresh_from_live_brings_the_polled_figure_to_nvml(live):
    gm = _gm(live)
    expected = min(live, 24_576)
    assert gm.refresh_from_live(1) == expected
    state = await gm.get_state(1)
    assert state.free_vram_mb == expected
    assert state.used_vram_mb == 24_576 - expected


async def test_refresh_from_live_without_nvml_changes_nothing():
    gm = _gm(None)
    assert gm.refresh_from_live(1) is None
    assert (await gm.get_state(1)).free_vram_mb == 1_000
    assert gm.refresh_from_live(7) is None
