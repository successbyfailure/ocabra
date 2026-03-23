"""
Integration test: model lifecycle — configure → load → request → idle → unload → reload.

Uses MockBackend (no real GPU/processes needed).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelStatus
from ocabra.core.worker_pool import WorkerPool


@pytest.fixture()
def worker_pool():
    wp = WorkerPool()
    wp.register_backend("vllm", MockBackend())
    return wp


@pytest.fixture()
def model_manager(worker_pool):
    return ModelManager(worker_pool)


@pytest.fixture()
def mock_session_factory():
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.__aexit__.return_value = False
    session.add = MagicMock()
    session.commit = AsyncMock()
    return lambda: session


class TestModelLifecycle:
    @pytest.mark.asyncio
    async def test_load_and_unload(self, model_manager: ModelManager, mock_session_factory):
        """Load a model and then unload it."""
        model_id = "vllm/test/model"
        with (
            patch("ocabra.core.model_manager.publish", new=AsyncMock()),
            patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
            patch("ocabra.database.AsyncSessionLocal", new=mock_session_factory),
        ):
            await model_manager.add_model(
                model_id=model_id,
                backend_type="vllm",
                display_name="Test Model",
            )
            state = await model_manager.load(model_id)
            assert state.status == ModelStatus.LOADED

            await model_manager.unload(model_id)
            state = await model_manager.get_state(model_id)
            assert state is not None
            assert state.status == ModelStatus.UNLOADED

    @pytest.mark.asyncio
    async def test_on_demand_load_via_on_request(self, model_manager: ModelManager, mock_session_factory):
        """on_request() should trigger load for CONFIGURED model."""
        model_id = "vllm/on-demand/model"
        with (
            patch("ocabra.core.model_manager.publish", new=AsyncMock()),
            patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
            patch("ocabra.database.AsyncSessionLocal", new=mock_session_factory),
        ):
            await model_manager.add_model(
                model_id=model_id,
                backend_type="vllm",
                load_policy="on_demand",
            )
            state = await model_manager.get_state(model_id)
            assert state is not None
            assert state.status == ModelStatus.CONFIGURED

            await model_manager.on_request(model_id)
            state = await model_manager.get_state(model_id)
            assert state is not None
            assert state.status == ModelStatus.LOADED

    @pytest.mark.asyncio
    async def test_idle_eviction(self, model_manager: ModelManager, mock_session_factory):
        """Models idle past timeout should be evicted."""
        model_id = "vllm/idle/model"
        with (
            patch("ocabra.core.model_manager.publish", new=AsyncMock()),
            patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
            patch("ocabra.database.AsyncSessionLocal", new=mock_session_factory),
            patch("ocabra.core.model_manager.settings") as mock_settings,
        ):
            mock_settings.idle_timeout_seconds = 1
            mock_settings.default_gpu_index = 0

            await model_manager.add_model(
                model_id=model_id,
                backend_type="vllm",
                load_policy=LoadPolicy.ON_DEMAND.value,
            )
            await model_manager.load(model_id)
            state = await model_manager.get_state(model_id)
            assert state is not None
            state.last_request_at = datetime(2020, 1, 1, tzinfo=timezone.utc)

            await model_manager.check_idle_evictions()
            await asyncio.sleep(0.05)

            state = await model_manager.get_state(model_id)
            assert state is not None
            assert state.status in (ModelStatus.UNLOADED, ModelStatus.UNLOADING)
