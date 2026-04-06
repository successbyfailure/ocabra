"""Tests for ModelManager.update_config() whitelist enforcement.

Validates that only allowed fields (display_name, load_policy, auto_reload,
preferred_gpu, extra_config) can be updated, and all others are rejected.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus
from ocabra.core.worker_pool import WorkerPool


@pytest.fixture
def mm():
    wp = WorkerPool()
    wp.register_backend("mock", MockBackend())
    return ModelManager(wp)


async def _add_model(mm, model_id="test/model"):
    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="mock",
        load_policy=LoadPolicy.ON_DEMAND,
    )
    mm._load_locks[model_id] = asyncio.Lock()
    return mm._states[model_id]


class TestUpdateConfigWhitelist:
    @pytest.mark.asyncio
    async def test_display_name(self, mm):
        await _add_model(mm)
        updated = await mm.update_config("test/model", {"display_name": "New Name"})
        assert updated.display_name == "New Name"

    @pytest.mark.asyncio
    async def test_load_policy_valid_values(self, mm):
        await _add_model(mm)

        for policy in ("on_demand", "warm", "pin"):
            updated = await mm.update_config("test/model", {"load_policy": policy})
            assert updated.load_policy == LoadPolicy(policy)

    @pytest.mark.asyncio
    async def test_load_policy_invalid_value(self, mm):
        await _add_model(mm)
        with pytest.raises(ValueError):
            await mm.update_config("test/model", {"load_policy": "invalid_policy"})

    @pytest.mark.asyncio
    async def test_auto_reload(self, mm):
        await _add_model(mm)
        updated = await mm.update_config("test/model", {"auto_reload": True})
        assert updated.auto_reload is True

    @pytest.mark.asyncio
    async def test_preferred_gpu(self, mm):
        await _add_model(mm)
        updated = await mm.update_config("test/model", {"preferred_gpu": 2})
        assert updated.preferred_gpu == 2

    @pytest.mark.asyncio
    async def test_preferred_gpu_none(self, mm):
        await _add_model(mm)
        await mm.update_config("test/model", {"preferred_gpu": 2})
        updated = await mm.update_config("test/model", {"preferred_gpu": None})
        assert updated.preferred_gpu is None

    @pytest.mark.asyncio
    async def test_extra_config(self, mm):
        await _add_model(mm)
        updated = await mm.update_config(
            "test/model", {"extra_config": {"vllm": {"max_num_seqs": 8}}}
        )
        assert updated.extra_config == {"vllm": {"max_num_seqs": 8}}

    @pytest.mark.asyncio
    async def test_multiple_allowed_fields_at_once(self, mm):
        await _add_model(mm)
        updated = await mm.update_config("test/model", {
            "display_name": "Custom Name",
            "load_policy": "pin",
            "auto_reload": True,
            "preferred_gpu": 0,
            "extra_config": {"key": "val"},
        })
        assert updated.display_name == "Custom Name"
        assert updated.load_policy == LoadPolicy.PIN
        assert updated.auto_reload is True
        assert updated.preferred_gpu == 0
        assert updated.extra_config == {"key": "val"}


class TestUpdateConfigRejection:
    @pytest.mark.asyncio
    async def test_rejects_status_field(self, mm):
        await _add_model(mm)
        with pytest.raises(ValueError, match="Unsupported model config fields"):
            await mm.update_config("test/model", {"status": "loaded"})

    @pytest.mark.asyncio
    async def test_rejects_model_id_field(self, mm):
        await _add_model(mm)
        with pytest.raises(ValueError, match="Unsupported"):
            await mm.update_config("test/model", {"model_id": "new/id"})

    @pytest.mark.asyncio
    async def test_rejects_backend_type_field(self, mm):
        await _add_model(mm)
        with pytest.raises(ValueError, match="Unsupported"):
            await mm.update_config("test/model", {"backend_type": "vllm"})

    @pytest.mark.asyncio
    async def test_rejects_vram_used_mb(self, mm):
        await _add_model(mm)
        with pytest.raises(ValueError, match="Unsupported"):
            await mm.update_config("test/model", {"vram_used_mb": 9999})

    @pytest.mark.asyncio
    async def test_rejects_current_gpu(self, mm):
        await _add_model(mm)
        with pytest.raises(ValueError, match="Unsupported"):
            await mm.update_config("test/model", {"current_gpu": [0, 1]})

    @pytest.mark.asyncio
    async def test_rejects_mixed_valid_and_invalid(self, mm):
        """If any field is invalid, the whole update should be rejected."""
        await _add_model(mm)
        with pytest.raises(ValueError, match="Unsupported"):
            await mm.update_config("test/model", {
                "display_name": "ok",
                "status": "loaded",
            })

    @pytest.mark.asyncio
    async def test_rejects_arbitrary_field(self, mm):
        await _add_model(mm)
        with pytest.raises(ValueError, match="Unsupported"):
            await mm.update_config("test/model", {"hacked": True})

    @pytest.mark.asyncio
    async def test_nonexistent_model_raises_key_error(self, mm):
        with pytest.raises(KeyError, match="not found"):
            await mm.update_config("nonexistent/model", {"display_name": "x"})

    @pytest.mark.asyncio
    async def test_empty_patch_is_noop(self, mm):
        state = await _add_model(mm)
        original_name = state.display_name
        updated = await mm.update_config("test/model", {})
        assert updated.display_name == original_name
