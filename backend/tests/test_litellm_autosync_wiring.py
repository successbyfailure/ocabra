from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.core.model_manager import ModelManager
from ocabra.core.worker_pool import WorkerPool
from ocabra.integrations.litellm_sync import LiteLLMSync


@pytest.fixture()
def worker_pool() -> WorkerPool:
    wp = WorkerPool()
    wp.register_backend("vllm", MockBackend(load_delay=0.0))
    return wp


@pytest.fixture()
def mock_session_factory():
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.__aexit__.return_value = False
    session.add = MagicMock()
    session.commit = AsyncMock()
    return lambda: session


@pytest.mark.asyncio
async def test_model_lifecycle_events_trigger_litellm_sync(worker_pool: WorkerPool, mock_session_factory) -> None:
    model_manager = ModelManager(worker_pool)
    syncer = LiteLLMSync(model_manager)
    syncer.sync_all = AsyncMock()
    model_manager.register_event_listener(syncer.handle_model_event)
    model_id = "vllm/test-autosync"

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch("ocabra.database.AsyncSessionLocal", new=mock_session_factory),
        patch("ocabra.integrations.litellm_sync.settings") as mock_settings,
    ):
        mock_settings.litellm_auto_sync = True

        await model_manager.add_model(model_id=model_id, backend_type="vllm")
        await asyncio.sleep(0)
        await model_manager.load(model_id)
        await asyncio.sleep(0)
        await model_manager.unload(model_id)
        await asyncio.sleep(0)

    assert 2 <= syncer.sync_all.await_count <= 3


@pytest.mark.asyncio
async def test_litellm_sync_does_not_block_model_load(worker_pool: WorkerPool, mock_session_factory) -> None:
    model_manager = ModelManager(worker_pool)
    model_id = "vllm/test-non-blocking-sync"
    sync_started = asyncio.Event()
    release_sync = asyncio.Event()

    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch("ocabra.database.AsyncSessionLocal", new=mock_session_factory),
        patch("ocabra.integrations.litellm_sync.settings") as mock_settings,
    ):
        mock_settings.litellm_auto_sync = True
        await model_manager.add_model(model_id=model_id, backend_type="vllm")

        syncer = LiteLLMSync(model_manager)

        async def slow_sync():
            sync_started.set()
            await release_sync.wait()

        syncer.sync_all = AsyncMock(side_effect=slow_sync)
        model_manager.register_event_listener(syncer.handle_model_event)

        await model_manager.load(model_id)
        await asyncio.wait_for(sync_started.wait(), timeout=1.0)
        state = await model_manager.get_state(model_id)
        assert state is not None
        assert state.status.value == "loaded"

        release_sync.set()
        await asyncio.sleep(0)
