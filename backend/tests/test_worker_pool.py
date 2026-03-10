import pytest

from ocabra.backends._mock import MockBackend
from ocabra.core.worker_pool import WorkerPool


@pytest.fixture
def pool():
    p = WorkerPool()
    p.register_backend("mock", MockBackend())
    return p


@pytest.mark.asyncio
async def test_assign_ports_no_collision(pool):
    p1 = await pool.assign_port()
    p2 = await pool.assign_port()
    p3 = await pool.assign_port()
    assert len({p1, p2, p3}) == 3


@pytest.mark.asyncio
async def test_release_port_makes_it_available(pool):
    port = await pool.assign_port()
    pool.release_port(port)
    reused = await pool.assign_port()
    assert reused == port


@pytest.mark.asyncio
async def test_get_backend_not_registered(pool):
    with pytest.raises(KeyError):
        await pool.get_backend("nonexistent")


@pytest.mark.asyncio
async def test_set_get_remove_worker(pool):
    from ocabra.backends.base import WorkerInfo

    info = WorkerInfo(
        backend_type="mock",
        model_id="test/model",
        gpu_indices=[1],
        port=18050,
        pid=999,
        vram_used_mb=4096,
    )
    pool.set_worker("test/model", info)
    assert pool.get_worker("test/model") == info

    pool.remove_worker("test/model")
    assert pool.get_worker("test/model") is None


@pytest.mark.asyncio
async def test_assign_port_range_exhausted():
    from ocabra import config
    from unittest.mock import patch

    pool = WorkerPool()
    # Use a tiny range (only 2 ports)
    with patch.object(config.settings, "worker_port_range_start", 19000), \
         patch.object(config.settings, "worker_port_range_end", 19002):
        await pool.assign_port()
        await pool.assign_port()
        with pytest.raises(RuntimeError, match="No available ports"):
            await pool.assign_port()
