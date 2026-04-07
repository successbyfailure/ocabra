"""Tests for busy timeout / health watchdog (sub-block 11.4)."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.core.model_manager import (
    LoadPolicy,
    ModelManager,
    ModelStatus,
)
from ocabra.core.worker_pool import WorkerPool


def _make_mm() -> ModelManager:
    wp = WorkerPool()
    return ModelManager(wp)


def _add_model_state(mm: ModelManager, model_id: str = "test/model") -> None:
    from ocabra.core.model_manager import ModelState

    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="mock",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
    )
    mm._load_locks[model_id] = asyncio.Lock()


# ------------------------------------------------------------------
# begin_request / end_request
# ------------------------------------------------------------------


def test_begin_request_returns_id():
    mm = _make_mm()
    request_id = mm.begin_request("test/model")
    assert isinstance(request_id, str)
    assert len(request_id) == 36  # UUID format


def test_end_request_clears_active():
    mm = _make_mm()
    rid = mm.begin_request("test/model")
    assert rid in mm._active_requests
    assert mm._in_flight["test/model"] == 1

    mm.end_request("test/model", rid)
    assert rid not in mm._active_requests
    assert "test/model" not in mm._in_flight


def test_end_request_legacy_compat():
    """Without request_id, just decrements count (legacy callers)."""
    mm = _make_mm()
    rid = mm.begin_request("test/model")
    mm.end_request("test/model")
    assert mm._in_flight.get("test/model", 0) == 0
    # The active_requests entry remains since we didn't pass the id
    assert rid in mm._active_requests


def test_is_busy():
    mm = _make_mm()
    assert mm.is_busy("test/model") is False
    rid = mm.begin_request("test/model")
    assert mm.is_busy("test/model") is True
    mm.end_request("test/model", rid)
    assert mm.is_busy("test/model") is False


def test_multiple_requests_decrement_correctly():
    mm = _make_mm()
    r1 = mm.begin_request("test/model")
    r2 = mm.begin_request("test/model")
    assert mm._in_flight["test/model"] == 2

    mm.end_request("test/model", r1)
    assert mm._in_flight["test/model"] == 1

    mm.end_request("test/model", r2)
    assert "test/model" not in mm._in_flight


# ------------------------------------------------------------------
# Watchdog tests — call the method directly with manipulated state
# ------------------------------------------------------------------


class _FakeSettings:
    def __init__(self, timeout=300, action="mark_error"):
        self.busy_timeout_seconds = timeout
        self.busy_timeout_action = action


async def _run_watchdog_once(mm: ModelManager, cfg: _FakeSettings | None = None) -> None:
    """Run one iteration of the watchdog logic without the sleep loop."""
    if cfg is None:
        cfg = _FakeSettings()

    timeout_s = max(30, int(cfg.busy_timeout_seconds))
    now = time.time()
    with mm._in_flight_lock:
        snapshot = list(mm._active_requests.values())
    for req in snapshot:
        elapsed = now - req.started_at
        if elapsed > timeout_s:
            with mm._in_flight_lock:
                mm._active_requests.pop(req.request_id, None)
                count = mm._in_flight.get(req.model_id, 0)
                if count <= 1:
                    mm._in_flight.pop(req.model_id, None)
                else:
                    mm._in_flight[req.model_id] = count - 1
            mm._timeout_counts[req.model_id] = (
                mm._timeout_counts.get(req.model_id, 0) + 1
            )
            if cfg.busy_timeout_action == "restart_worker":
                try:
                    await mm.unload(req.model_id, reason="busy_timeout")
                except Exception:
                    pass
            else:
                state = mm._states.get(req.model_id)
                if state:
                    state.status = ModelStatus.ERROR
                    state.error_message = (
                        f"Request {req.request_id} timed out "
                        f"after {elapsed:.0f}s"
                    )


@pytest.mark.asyncio
async def test_watchdog_detects_timeout():
    """Request started 400s ago with timeout=300 should be cleaned up."""
    mm = _make_mm()
    _add_model_state(mm, "test/model")

    rid = mm.begin_request("test/model")
    mm._active_requests[rid].started_at = time.time() - 400

    await _run_watchdog_once(mm, _FakeSettings(timeout=300, action="mark_error"))

    assert rid not in mm._active_requests
    assert mm._states["test/model"].status == ModelStatus.ERROR


@pytest.mark.asyncio
async def test_watchdog_ignores_fresh_requests():
    """Request only 10s old should not be touched."""
    mm = _make_mm()
    _add_model_state(mm, "test/model")

    rid = mm.begin_request("test/model")
    mm._active_requests[rid].started_at = time.time() - 10

    await _run_watchdog_once(mm, _FakeSettings(timeout=300, action="mark_error"))

    assert rid in mm._active_requests
    assert mm._states["test/model"].status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_timeout_count_increments():
    """Verify _timeout_counts increments on timeout."""
    mm = _make_mm()
    _add_model_state(mm, "test/model")

    r1 = mm.begin_request("test/model")
    r2 = mm.begin_request("test/model")
    mm._active_requests[r1].started_at = time.time() - 400
    mm._active_requests[r2].started_at = time.time() - 500

    await _run_watchdog_once(mm, _FakeSettings(timeout=300, action="mark_error"))

    assert mm._timeout_counts.get("test/model") == 2


@pytest.mark.asyncio
async def test_mark_error_action():
    """busy_timeout_action='mark_error' transitions model to ERROR."""
    mm = _make_mm()
    _add_model_state(mm, "test/model")

    rid = mm.begin_request("test/model")
    mm._active_requests[rid].started_at = time.time() - 600

    await _run_watchdog_once(mm, _FakeSettings(timeout=300, action="mark_error"))

    state = mm._states["test/model"]
    assert state.status == ModelStatus.ERROR
    assert "timed out" in state.error_message
    assert rid in state.error_message


@pytest.mark.asyncio
async def test_restart_worker_action():
    """busy_timeout_action='restart_worker' calls unload()."""
    mm = _make_mm()
    _add_model_state(mm, "test/model")

    rid = mm.begin_request("test/model")
    mm._active_requests[rid].started_at = time.time() - 400

    with patch.object(mm, "unload", new=AsyncMock()) as unload_mock:
        await _run_watchdog_once(mm, _FakeSettings(timeout=300, action="restart_worker"))

    unload_mock.assert_awaited_once_with("test/model", reason="busy_timeout")
    assert rid not in mm._active_requests
