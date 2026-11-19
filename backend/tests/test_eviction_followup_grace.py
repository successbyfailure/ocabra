"""Gracia antes de desalojar: la petición siguiente de un agente llega 0,1-0,6 s
después de la anterior. Sin ella, gemma4:26b y ornith-1.5:35b se desalojaron
12 veces en 10 minutos el 2026-11-19, cada vez para un segundo de trabajo."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from tests.test_vram_eviction import _add_loaded_model, _make_mm


def _grace(monkeypatch, grace: float, chain: float) -> None:
    monkeypatch.setattr("ocabra.core.model_manager.settings.eviction_followup_grace_s", grace)
    monkeypatch.setattr("ocabra.core.model_manager.settings.eviction_followup_max_chain_s", chain)


def _finish(mm, model_id: str) -> None:
    rid = mm.try_begin_request(model_id, 4)
    mm.end_request(model_id, rid)


@pytest.mark.asyncio
async def test_a_follow_up_within_the_grace_is_served_before_the_eviction(monkeypatch):
    _grace(monkeypatch, 1.0, 60)
    mm = _make_mm()
    _add_loaded_model(mm, "a")
    _finish(mm, "a")
    served = []

    async def follow_up():
        await asyncio.sleep(0.2)
        rid = mm.try_begin_request("a", 4)
        served.append(mm.is_eviction_claimed("a"))
        await asyncio.sleep(0.2)
        mm.end_request("a", rid)

    mm.unload = AsyncMock()
    started = time.monotonic()
    task = asyncio.create_task(follow_up())
    await mm._evict_one("b", "a", gpu_index=None, needed_mb=1000, reason="vram_pressure")
    await task
    # It reached the worker unclaimed; the eviction came after it and its
    # own grace (~1 s after the follow-up ended).
    assert served == [False]
    mm.unload.assert_awaited_once()
    assert time.monotonic() - started >= 1.3


@pytest.mark.asyncio
async def test_no_follow_up_evicts_when_the_grace_runs_out(monkeypatch):
    _grace(monkeypatch, 0.3, 60)
    mm = _make_mm()
    _add_loaded_model(mm, "a")
    _finish(mm, "a")
    mm.unload = AsyncMock()
    started = time.monotonic()
    await mm._evict_one("b", "a", gpu_index=None, needed_mb=1000, reason="vram_pressure")
    assert 0.2 <= time.monotonic() - started < 1.0
    mm.unload.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_model_idle_for_longer_than_the_grace_goes_at_once(monkeypatch):
    _grace(monkeypatch, 4.0, 60)
    mm = _make_mm()
    _add_loaded_model(mm, "a")
    _finish(mm, "a")
    mm._went_idle_at["a"] -= 50  # its last request ended 50 s ago
    mm.unload = AsyncMock()
    started = time.monotonic()
    await mm._evict_one("b", "a", gpu_index=None, needed_mb=1000, reason="vram_pressure")
    assert time.monotonic() - started < 0.2


@pytest.mark.asyncio
async def test_no_grace_past_the_chain_cap(monkeypatch):
    _grace(monkeypatch, 4.0, 60)
    mm = _make_mm()
    _add_loaded_model(mm, "a")
    _finish(mm, "a")
    past = time.monotonic() - 1
    assert await mm._followup_grace("b", "a", past) is False


@pytest.mark.asyncio
async def test_disabled_grace_and_watchdog_evictions_do_not_wait(monkeypatch):
    _grace(monkeypatch, 0, 60)
    mm = _make_mm()
    _add_loaded_model(mm, "a")
    _finish(mm, "a")
    assert await mm._followup_grace("b", "a", time.monotonic() + 60) is False
    _grace(monkeypatch, 4.0, 60)
    mm.unload = AsyncMock()
    started = time.monotonic()
    await mm._evict_one("w", "a", gpu_index=None, needed_mb=1, reason="vram_pressure", drain=False)
    assert time.monotonic() - started < 0.2
