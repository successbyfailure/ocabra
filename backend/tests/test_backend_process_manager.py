"""Tests for BackendProcessManager — worker health monitoring and auto-restart."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ocabra.core.backend_process_manager import BackendProcessManager
from ocabra.core.model_manager import ModelStatus


@dataclass
class FakeWorkerInfo:
    backend_type: str = "vllm"
    model_id: str = "vllm/test-model"
    gpu_indices: list[int] | None = None
    port: int = 18001
    pid: int = 99999
    vram_used_mb: int = 4096

    def __post_init__(self):
        if self.gpu_indices is None:
            self.gpu_indices = [0]


@dataclass
class FakeModelState:
    model_id: str = "vllm/test-model"
    status: ModelStatus = ModelStatus.LOADED
    error_message: str | None = None
    current_gpu: list[int] | None = None
    vram_used_mb: int = 4096
    worker_info: object | None = None

    def __post_init__(self):
        if self.current_gpu is None:
            self.current_gpu = [0]


def make_settings(**overrides):
    s = MagicMock()
    s.worker_health_check_interval_seconds = 1
    s.auto_restart_workers = True
    s.max_worker_restarts = 3
    s.worker_restart_backoff_seconds = 0.01
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_fixtures(**settings_overrides):
    """Build model_manager, worker_pool, settings, and a BackendProcessManager."""
    model_manager = MagicMock()
    model_manager.get_state = AsyncMock()
    model_manager.load = AsyncMock()
    model_manager._publish_event = AsyncMock()
    model_manager._gpu_manager = MagicMock()
    model_manager._gpu_manager.unlock_vram = AsyncMock()

    worker_pool = MagicMock()
    worker_pool._workers = {}
    worker_pool.get_worker = MagicMock(return_value=None)
    worker_pool.remove_worker = MagicMock()

    settings = make_settings(**settings_overrides)

    bpm = BackendProcessManager(model_manager, worker_pool, settings)
    return bpm, model_manager, worker_pool, settings


@pytest.mark.asyncio
async def test_detects_dead_pid():
    """Mock a worker with PID that doesn't exist, verify model transitions to ERROR."""
    bpm, model_manager, worker_pool, _ = make_fixtures()

    worker = FakeWorkerInfo(pid=99999)
    state = FakeModelState()
    model_manager.get_state.return_value = state

    with (
        patch.object(BackendProcessManager, "_is_pid_alive", return_value=False),
        patch("ocabra.redis_client.publish_system_alert", new=AsyncMock()),
    ):
        await bpm._check_worker("vllm/test-model", worker)

    assert state.status == ModelStatus.ERROR
    assert "pid_dead" in (state.error_message or "")
    worker_pool.remove_worker.assert_called_once_with("vllm/test-model")


@pytest.mark.asyncio
async def test_detects_health_failure():
    """Mock HTTP timeout 3 times, verify transition to ERROR."""
    bpm, model_manager, worker_pool, _ = make_fixtures()

    worker = FakeWorkerInfo(pid=99999)
    state = FakeModelState()
    model_manager.get_state.return_value = state

    with (
        patch.object(BackendProcessManager, "_is_pid_alive", return_value=True),
        patch(
            "httpx.AsyncClient.get",
            new=AsyncMock(side_effect=httpx.ConnectError("refused")),
        ),
        patch("ocabra.redis_client.publish_system_alert", new=AsyncMock()),
    ):
        # First two failures -- no death yet
        await bpm._check_worker("vllm/test-model", worker)
        assert bpm._health_fail_counts.get("vllm/test-model") == 1
        assert state.status == ModelStatus.LOADED

        await bpm._check_worker("vllm/test-model", worker)
        assert bpm._health_fail_counts.get("vllm/test-model") == 2
        assert state.status == ModelStatus.LOADED

        # Third failure triggers death
        await bpm._check_worker("vllm/test-model", worker)

    assert state.status == ModelStatus.ERROR
    assert "health_check_failed" in (state.error_message or "")


@pytest.mark.asyncio
async def test_auto_restart_on_death():
    """Mock death + successful reload, verify model_manager.load is called."""
    bpm, model_manager, worker_pool, _ = make_fixtures()

    state = FakeModelState()
    model_manager.get_state.return_value = state

    with patch("ocabra.redis_client.publish_system_alert", new=AsyncMock()):
        await bpm._handle_worker_death("vllm/test-model", reason="pid_dead")

    model_manager.load.assert_awaited_once_with("vllm/test-model")
    assert bpm._restart_counts["vllm/test-model"] == 1


@pytest.mark.asyncio
async def test_max_restarts_exceeded():
    """4 consecutive deaths with max_worker_restarts=3, verify no 4th restart."""
    bpm, model_manager, worker_pool, _ = make_fixtures(max_worker_restarts=3)

    state = FakeModelState()
    model_manager.get_state.return_value = state

    with patch("ocabra.redis_client.publish_system_alert", new=AsyncMock()):
        model_manager.load = AsyncMock(side_effect=RuntimeError("still broken"))

        for _i in range(4):
            state.status = ModelStatus.LOADED
            state.current_gpu = [0]
            await bpm._handle_worker_death("vllm/test-model", reason="pid_dead")

    assert model_manager.load.await_count == 3
    assert bpm._restart_counts["vllm/test-model"] == 3


@pytest.mark.asyncio
async def test_backoff_increases():
    """Verify sleep duration grows exponentially between restarts."""
    bpm, model_manager, worker_pool, _ = make_fixtures(worker_restart_backoff_seconds=1.0)

    state = FakeModelState()
    model_manager.get_state.return_value = state

    sleep_durations: list[float] = []

    async def capture_sleep(duration):
        sleep_durations.append(duration)

    with (
        patch("ocabra.redis_client.publish_system_alert", new=AsyncMock()),
        patch(
            "ocabra.core.backend_process_manager.asyncio.sleep",
            side_effect=capture_sleep,
        ),
    ):
        model_manager.load = AsyncMock(side_effect=RuntimeError("still broken"))

        for _ in range(3):
            state.status = ModelStatus.LOADED
            state.current_gpu = [0]
            await bpm._handle_worker_death("vllm/test-model", reason="pid_dead")

    # backoff_base=1.0: 1*2^0=1.0, 1*2^1=2.0, 1*2^2=4.0
    assert sleep_durations == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_healthy_worker_resets_failure_count():
    """fail -> OK -> fail counts from 0."""
    bpm, model_manager, worker_pool, _ = make_fixtures()

    worker = FakeWorkerInfo(pid=99999)
    state = FakeModelState()
    model_manager.get_state.return_value = state

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    call_count = 0

    async def mock_get(url):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise httpx.ConnectError("refused")
        return mock_response

    with (
        patch.object(BackendProcessManager, "_is_pid_alive", return_value=True),
        patch("httpx.AsyncClient.get", new=AsyncMock(side_effect=mock_get)),
        patch("ocabra.redis_client.publish_system_alert", new=AsyncMock()),
    ):
        # Two failures
        await bpm._check_worker("vllm/test-model", worker)
        assert bpm._health_fail_counts.get("vllm/test-model") == 1

        await bpm._check_worker("vllm/test-model", worker)
        assert bpm._health_fail_counts.get("vllm/test-model") == 2

        # Success -- should reset
        await bpm._check_worker("vllm/test-model", worker)
        assert bpm._health_fail_counts.get("vllm/test-model") is None

        # Next failure starts from 0 again
        call_count = 0
        await bpm._check_worker("vllm/test-model", worker)
        assert bpm._health_fail_counts.get("vllm/test-model") == 1


@pytest.mark.asyncio
async def test_vram_freed_on_death():
    """Verify gpu_manager.unlock_vram is called for each GPU on worker death."""
    bpm, model_manager, worker_pool, _ = make_fixtures()

    state = FakeModelState(current_gpu=[0, 1])
    model_manager.get_state.return_value = state

    with patch("ocabra.redis_client.publish_system_alert", new=AsyncMock()):
        await bpm._handle_worker_death("vllm/test-model", reason="pid_dead")

    gpu_manager = model_manager._gpu_manager
    assert gpu_manager.unlock_vram.await_count == 2
    gpu_manager.unlock_vram.assert_any_await(0, "vllm/test-model")
    gpu_manager.unlock_vram.assert_any_await(1, "vllm/test-model")

    assert state.current_gpu == []
    assert state.vram_used_mb == 0
    assert state.worker_info is None
