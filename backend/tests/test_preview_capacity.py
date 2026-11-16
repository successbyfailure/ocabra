"""Room per GPU in the model editor's preview, counted like the scheduler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocabra.api.internal.models import _preview_capacity
from ocabra.config import settings


class _Gpus:
    """Mapache's GTX 1660: a desktop (processes oCabra does not own) and the
    driver reservation; nothing of oCabra loaded."""

    def __init__(self, free: int = 4875):
        self.free = free
        self.released: list[dict] = []

    async def get_all_states(self):
        procs = [
            SimpleNamespace(pid=1865, used_vram_mb=428),
            SimpleNamespace(pid=3603882, used_vram_mb=452),
        ]
        return [
            SimpleNamespace(
                index=0,
                name="GTX 1660",
                total_vram_mb=6144,
                free_vram_mb=self.free,
                processes=procs,
            )
        ]

    async def get_free_vram(self, index: int) -> int:
        return max(0, self.free - settings.vram_buffer_mb)

    async def get_free_vram_if_released(self, index: int, released: dict) -> int:
        self.released.append(released)
        return max(0, self.free + 3700 - settings.vram_buffer_mb)


@pytest.mark.asyncio
async def test_a_desktop_is_system_not_evictable(monkeypatch):
    monkeypatch.setattr(settings, "vram_buffer_mb", 512)
    monkeypatch.setattr(settings, "vram_eviction_threshold", 0.96)
    gm = _Gpus()
    mm = SimpleNamespace(_gpu_manager=gm, _states={})
    state = SimpleNamespace(model_id="llamacpp_mks/gemma4:26b", current_gpu=[], worker_info=None)
    [cap] = await _preview_capacity(mm, gm, state)
    assert cap["availableMb"] == 4875 - 512
    # 6144 − 512 de margen − 389 sin atribuir − 880 de escritorio.
    assert cap["budgetMb"] == 4363
    assert cap["systemMb"] == 6144 - 4363
    # gemma4 on Mapache needed 4.486 MB: it does not fit, even evicting.
    assert 4486 > cap["budgetMb"]


@pytest.mark.asyncio
async def test_a_loaded_model_counts_its_own_copy_as_free(monkeypatch):
    monkeypatch.setattr(settings, "vram_buffer_mb", 512)
    gm = _Gpus(free=1175)
    mm = SimpleNamespace(_gpu_manager=gm, _states={})
    state = SimpleNamespace(model_id="m", current_gpu=[0], worker_info=SimpleNamespace(pid=3603882))
    [cap] = await _preview_capacity(mm, gm, state)
    assert gm.released == [{"m": 3603882}]
    assert cap["availableMb"] == 1175 + 3700 - 512
    assert cap["budgetMb"] >= cap["availableMb"]


@pytest.mark.asyncio
async def test_without_gpu_manager_there_is_no_capacity():
    assert await _preview_capacity(SimpleNamespace(), None, SimpleNamespace()) == []
