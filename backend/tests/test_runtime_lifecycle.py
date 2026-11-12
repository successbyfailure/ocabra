from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from ocabra.backends.ollama_backend import OllamaBackend
from ocabra.core.model_manager import (
    LoadPolicy,
    ModelManager,
    ModelState,
    ModelStatus,
)


def test_rebind_request_moves_inflight_to_runtime() -> None:
    manager = ModelManager(object())
    request_id = manager.begin_request("gemma4:12b")

    manager.rebind_request(request_id, "gemma4:12b", "gemma4:12b::64k")

    assert manager.inflight_count("gemma4:12b") == 0
    assert manager.inflight_count("gemma4:12b::64k") == 1
    manager.end_request("gemma4:12b::64k", request_id)
    assert manager.inflight_count("gemma4:12b::64k") == 0


@pytest.mark.asyncio
async def test_small_request_does_not_extend_large_runtime_ttl() -> None:
    manager = ModelManager(object())
    state = ModelState(
        model_id="ollama/gemma4:12b::max",
        display_name="Gemma MAX",
        backend_type="ollama",
        extra_config={
            "_runtime": {
                "effective_context": 262144,
                "default_context": 65536,
                "small_requests_extend_large_ttl": False,
            }
        },
    )
    manager._states[state.model_id] = state
    manager._persist_state = AsyncMock()  # type: ignore[method-assign]

    await manager.mark_runtime_request(state.model_id, required_tokens=32000)
    assert "last_large_request_at" not in state.extra_config["_runtime"]

    await manager.mark_runtime_request(state.model_id, required_tokens=70000)
    assert state.extra_config["_runtime"]["last_large_request_at"]


@pytest.mark.asyncio
async def test_idle_high_context_schedules_downgrade() -> None:
    manager = ModelManager(object())
    state = ModelState(
        model_id="ollama/gemma4:12b::max",
        display_name="Gemma MAX",
        backend_type="ollama",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.WARM,
        loaded_at=datetime.now(UTC) - timedelta(minutes=10),
        extra_config={
            "_runtime": {
                "effective_context": 262144,
                "default_context": 65536,
                "restore_default_when_idle": True,
                "downgrade_grace_seconds": 60,
                "minimum_residency_seconds": 30,
            }
        },
    )
    manager._states[state.model_id] = state
    scheduled: list[str] = []

    def capture(coro, *, task_name: str, model_id: str | None = None):
        scheduled.append(task_name)
        coro.close()
        return None

    manager._create_background_task = capture  # type: ignore[method-assign]
    await manager.check_idle_evictions()

    assert scheduled == [f"runtime-downgrade:{state.model_id}"]


def test_ollama_uses_the_single_daemon_and_caps_num_ctx(monkeypatch) -> None:
    """One shared daemon since 2026-11-12: no pool selection, and a model's
    num_ctx never above the global maximum."""
    monkeypatch.setattr("ocabra.backends.ollama_backend.settings.ollama_base_url", "http://ollama:11434/")
    monkeypatch.setattr("ocabra.backends.ollama_backend.settings.ollama_max_context", 65536)
    backend = OllamaBackend()
    assert backend._endpoint() == "http://ollama:11434"
    assert backend.context_cap(None) == 65536
    assert backend.context_cap(8192) == 8192
    assert backend.context_cap(262144) == 65536
