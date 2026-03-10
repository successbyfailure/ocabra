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
from ocabra.backends.base import BackendCapabilities
from ocabra.core.model_manager import ModelManager, ModelStatus, LoadPolicy
from ocabra.core.worker_pool import WorkerPool


@pytest.fixture()
def worker_pool():
    wp = WorkerPool()
    wp.register_backend("vllm", MockBackend())
    return wp


@pytest.fixture()
def model_manager(worker_pool):
    mm = ModelManager(worker_pool)
    return mm


class TestModelLifecycle:
    @pytest.mark.asyncio
    async def test_load_and_unload(self, model_manager: ModelManager):
        """Load a model and then unload it."""
        with (
            patch("ocabra.core.model_manager.publish", new=AsyncMock()),
            patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
            patch("ocabra.core.model_manager.AsyncSessionLocal"),
        ):
            await model_manager.add_model(
                model_id="test/model",
                backend_type="vllm",
                display_name="Test Model",
            )
            state = await model_manager.load("test/model")
            assert state.status == ModelStatus.LOADED

            await model_manager.unload("test/model")
            state = await model_manager.get_state("test/model")
            assert state.status == ModelStatus.UNLOADED

    @pytest.mark.asyncio
    async def test_on_demand_load_via_on_request(self, model_manager: ModelManager):
        """on_request() should trigger load for CONFIGURED model."""
        with (
            patch("ocabra.core.model_manager.publish", new=AsyncMock()),
            patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
            patch("ocabra.core.model_manager.AsyncSessionLocal"),
        ):
            await model_manager.add_model(
                model_id="on-demand/model",
                backend_type="vllm",
                load_policy="on_demand",
            )
            state = await model_manager.get_state("on-demand/model")
            assert state.status == ModelStatus.CONFIGURED

            await model_manager.on_request("on-demand/model")
            state = await model_manager.get_state("on-demand/model")
            assert state.status == ModelStatus.LOADED

    @pytest.mark.asyncio
    async def test_idle_eviction(self, model_manager: ModelManager):
        """Models idle past timeout should be evicted."""
        with (
            patch("ocabra.core.model_manager.publish", new=AsyncMock()),
            patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
            patch("ocabra.core.model_manager.AsyncSessionLocal"),
            patch("ocabra.core.model_manager.settings") as mock_settings,
        ):
            mock_settings.idle_timeout_seconds = 0  # immediate eviction
            mock_settings.default_gpu_index = 0

            await model_manager.add_model(
                model_id="idle/model",
                backend_type="vllm",
                load_policy="on_demand",
            )
            await model_manager.load("idle/model")
            state = await model_manager.get_state("idle/model")
            state.last_request_at = datetime(2020, 1, 1, tzinfo=timezone.utc)

            await model_manager.check_idle_evictions()
            await asyncio.sleep(0.05)  # let the task run

            state = await model_manager.get_state("idle/model")
            assert state.status in (ModelStatus.UNLOADED, ModelStatus.UNLOADING)
