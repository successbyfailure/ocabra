import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from ocabra.api.openai._deps import _do_ensure_loaded, keepalive_until_done
from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus


@pytest.mark.asyncio
async def test_keepalive_iterator_leaves_failed_task_for_caller() -> None:
    async def _fail() -> None:
        raise RuntimeError("load failed")

    task = asyncio.create_task(_fail())

    chunks = [chunk async for chunk in keepalive_until_done(task, interval=0.01)]

    assert chunks == []
    assert isinstance(task.exception(), RuntimeError)


@pytest.mark.asyncio
async def test_concurrent_load_wait_stops_when_model_enters_error() -> None:
    loading = ModelState(
        model_id="vllm/test-model",
        display_name="test-model",
        backend_type="vllm",
        status=ModelStatus.LOADING,
        load_policy=LoadPolicy.ON_DEMAND,
    )
    failed = ModelState(
        model_id="vllm/test-model",
        display_name="test-model",
        backend_type="vllm",
        status=ModelStatus.ERROR,
        load_policy=LoadPolicy.ON_DEMAND,
        error_message="insufficient VRAM",
    )
    model_manager = AsyncMock()
    model_manager.get_state = AsyncMock(side_effect=[loading, failed])

    with patch("ocabra.api.openai._deps.asyncio.sleep", new=AsyncMock()) as sleep:
        with pytest.raises(HTTPException) as exc_info:
            await _do_ensure_loaded(model_manager, "vllm/test-model")

    assert exc_info.value.status_code == 503
    assert "insufficient VRAM" in exc_info.value.detail["error"]["message"]
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_request_waits_for_unload_then_reloads_model() -> None:
    unloading = ModelState(
        model_id="vllm/test-model",
        display_name="test-model",
        backend_type="vllm",
        status=ModelStatus.UNLOADING,
        load_policy=LoadPolicy.ON_DEMAND,
    )
    unloaded = ModelState(
        model_id="vllm/test-model",
        display_name="test-model",
        backend_type="vllm",
        status=ModelStatus.UNLOADED,
        load_policy=LoadPolicy.ON_DEMAND,
    )
    loaded = ModelState(
        model_id="vllm/test-model",
        display_name="test-model",
        backend_type="vllm",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
    )
    model_manager = AsyncMock()
    model_manager.get_state = AsyncMock(side_effect=[unloading, unloaded, loaded])
    model_manager.load = AsyncMock()

    with patch("ocabra.api.openai._deps.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await _do_ensure_loaded(model_manager, "vllm/test-model")

    assert result is loaded
    model_manager.load.assert_awaited_once_with("vllm/test-model")
    sleep.assert_awaited_once_with(0.1)
