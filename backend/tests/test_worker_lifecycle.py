"""Tests for worker pool lifecycle: port assignment, worker registration,
and request forwarding error handling.

Extends test_worker_pool.py with additional lifecycle and error path coverage.
"""
from unittest.mock import AsyncMock, patch

import pytest

from ocabra import config
from ocabra.backends._mock import MockBackend
from ocabra.backends.base import WorkerInfo
from ocabra.core.worker_pool import WorkerPool


@pytest.fixture
def pool():
    p = WorkerPool()
    p.register_backend("mock", MockBackend())
    return p


# ── Port assignment lifecycle ────────────────────────────────────


class TestPortLifecycle:
    @pytest.mark.asyncio
    async def test_ports_are_sequential_from_range_start(self):
        pool = WorkerPool()
        with patch.object(config.settings, "worker_port_range_start", 20000), \
             patch.object(config.settings, "worker_port_range_end", 20010):
            p1 = await pool.assign_port()
            assert p1 == 20000
            p2 = await pool.assign_port()
            assert p2 == 20001

    @pytest.mark.asyncio
    async def test_released_port_is_reused_before_next(self):
        pool = WorkerPool()
        with patch.object(config.settings, "worker_port_range_start", 20000), \
             patch.object(config.settings, "worker_port_range_end", 20010):
            p1 = await pool.assign_port()
            p2 = await pool.assign_port()
            pool.release_port(p1)
            p3 = await pool.assign_port()
            assert p3 == p1  # reuses the released port

    @pytest.mark.asyncio
    async def test_set_worker_reserves_port(self, pool):
        info = WorkerInfo(
            backend_type="mock", model_id="a/b", gpu_indices=[0],
            port=18500, pid=100, vram_used_mb=1024,
        )
        pool.set_worker("a/b", info)
        assert 18500 in pool._used_ports

    @pytest.mark.asyncio
    async def test_remove_worker_releases_port(self, pool):
        info = WorkerInfo(
            backend_type="mock", model_id="a/b", gpu_indices=[0],
            port=18500, pid=100, vram_used_mb=1024,
        )
        pool.set_worker("a/b", info)
        pool.remove_worker("a/b")
        assert 18500 not in pool._used_ports
        assert pool.get_worker("a/b") is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_worker_is_noop(self, pool):
        """Removing a worker that was never registered should not raise."""
        pool.remove_worker("nonexistent/model")


# ── Worker registration ──────────────────────────────────────────


class TestWorkerRegistration:
    def test_set_and_get_worker(self, pool):
        info = WorkerInfo(
            backend_type="mock", model_id="test/m", gpu_indices=[1],
            port=18100, pid=42, vram_used_mb=2048,
        )
        pool.set_worker("test/m", info)
        assert pool.get_worker("test/m") is info

    def test_overwrite_worker(self, pool):
        info1 = WorkerInfo(
            backend_type="mock", model_id="test/m", gpu_indices=[0],
            port=18100, pid=42, vram_used_mb=1024,
        )
        info2 = WorkerInfo(
            backend_type="mock", model_id="test/m", gpu_indices=[1],
            port=18200, pid=43, vram_used_mb=2048,
        )
        pool.set_worker("test/m", info1)
        pool.set_worker("test/m", info2)
        assert pool.get_worker("test/m") is info2
        # Both ports are used (old port not auto-released)
        assert 18100 in pool._used_ports
        assert 18200 in pool._used_ports

    def test_get_nonexistent_worker_returns_none(self, pool):
        assert pool.get_worker("missing/model") is None


# ── Forward request to unloaded model ────────────────────────────


class TestForwardRequestErrors:
    @pytest.mark.asyncio
    async def test_forward_request_unloaded_model_raises_key_error(self, pool):
        """Forwarding a request to a model with no worker should raise KeyError."""
        with pytest.raises(KeyError, match="No worker found"):
            await pool.forward_request("unloaded/model", "/v1/chat/completions", {})

    @pytest.mark.asyncio
    async def test_forward_stream_unloaded_model_raises_key_error(self, pool):
        """Forwarding a stream to a model with no worker should raise KeyError."""
        with pytest.raises(KeyError, match="No worker found"):
            async for _ in pool.forward_stream("unloaded/model", "/v1/chat/completions", {}):
                pass


# ── Backend registration ─────────────────────────────────────────


class TestBackendRegistration:
    @pytest.mark.asyncio
    async def test_get_registered_backend(self, pool):
        backend = await pool.get_backend("mock")
        assert isinstance(backend, MockBackend)

    @pytest.mark.asyncio
    async def test_get_unregistered_backend_raises(self, pool):
        with pytest.raises(KeyError, match="not registered"):
            await pool.get_backend("unknown_type")

    @pytest.mark.asyncio
    async def test_disabled_backend_raises_with_reason(self):
        pool = WorkerPool()
        pool.register_disabled_backend("vllm", "CUDA not available")

        with pytest.raises(RuntimeError, match="CUDA not available"):
            await pool.get_backend("vllm")

    @pytest.mark.asyncio
    async def test_register_backend_overrides_disabled(self):
        pool = WorkerPool()
        pool.register_disabled_backend("mock", "initially disabled")
        pool.register_backend("mock", MockBackend())

        backend = await pool.get_backend("mock")
        assert isinstance(backend, MockBackend)

    @pytest.mark.asyncio
    async def test_register_disabled_removes_active(self):
        pool = WorkerPool()
        pool.register_backend("mock", MockBackend())
        pool.register_disabled_backend("mock", "now disabled")

        with pytest.raises(RuntimeError, match="now disabled"):
            await pool.get_backend("mock")
