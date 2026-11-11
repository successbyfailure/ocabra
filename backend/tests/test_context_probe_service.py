"""The context probe job: what it touches and that it always puts it back."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.core.context_probe_service import ContextProbeService
from ocabra.core.model_manager import LoadPolicy, ModelStatus


class FakeManager:
    """Just what the probe uses of ModelManager, with a worker whose VRAM on
    GPU 1 follows its context (16 GB weights + 17 MB per 1k tokens)."""

    def __init__(self):
        self.maintenance: list[tuple[str, bool]] = []
        self.unloaded: list[str] = []
        self.loaded: list[str] = []
        self.ctx = 0
        self._states: dict = {}
        self.load_failures = 0
        self.unload_reasons: dict = {}
        self.waited_for_release: list[int] = []
        whisper = SimpleNamespace(
            model_id="whisper/turbo",
            status=ModelStatus.LOADED,
            current_gpu=[1],
            load_policy=LoadPolicy.PIN,
        )
        embed = SimpleNamespace(
            model_id="llamacpp_mks/qwen3-embedding:4b",
            status=ModelStatus.LOADED,
            current_gpu=[1],
            load_policy=LoadPolicy.ON_DEMAND,
        )
        other_gpu = SimpleNamespace(
            model_id="llamacpp_mks/gemma4:12b",
            status=ModelStatus.LOADED,
            current_gpu=[0],
            load_policy=LoadPolicy.PIN,
        )
        self.residents = [whisper, embed, other_gpu]
        gm = MagicMock()
        gm.admission_ceiling_mb = MagicMock(return_value=23600)
        gm.get_process_vram_mb = MagicMock(return_value={})
        self._gpu_manager = gm

    def set_maintenance(self, model_id, enabled):
        self.maintenance.append((model_id, enabled))

    async def list_states(self, include_derived=False):
        return list(self.residents)

    def is_busy(self, model_id):
        return False

    async def unload(self, model_id, reason=None):
        self.unloaded.append(model_id)
        self.unload_reasons[model_id] = reason
        self._states.pop(model_id, None)

    async def load(self, model_id):
        if self.load_failures:
            self.load_failures -= 1
            raise RuntimeError("Need 23600 MB VRAM")  # the driver had not freed it yet
        self.loaded.append(model_id)

    async def _get_vram_free_snapshot(self, gpus):
        return [100]

    async def _wait_for_vram_released(self, released, *, baseline_free_mb=None, gpu_indices=None):
        self.waited_for_release.append(released)

    async def get_state(self, model_id):
        return SimpleNamespace(backend_type="llamacpp_mks")

    async def resize_context(self, model_id, *, context):
        self.ctx = context

    def _worker_vram_by_gpu(self, state):
        return {1: int(16000 + 17.0 * self.ctx / 1024)}


def _service(fake):
    backend = SimpleNamespace(SUPPORTS_LIVE_CONTEXT=True, vision_projector_path=lambda _m: None)
    pool = SimpleNamespace(_backends={"llamacpp_mks": backend}, get_worker=lambda _k: None)
    profile = SimpleNamespace(
        profile_id="qwen3.8:27b",
        base_model_id="llamacpp_mks/qwen3.8:27b",
        runtime_presets=[SimpleNamespace(preset_key="normal")],
    )
    registry = SimpleNamespace(get=AsyncMock(return_value=profile))
    return ContextProbeService(fake, registry, pool)


def _plan():
    return SimpleNamespace(
        effective_context=98304,
        load_overrides={
            "llama_cpp": {"ctx_size": 98304, "parallel_slots": 4},
            "use_case": {"context": 98304, "slots": 4},
            "_runtime": {
                "gpu_indices": [1],
                "context_is_pool": True,
                "parallelism": 4,
                "default_worker_key": "x",
                "capacity_validated": True,
            },
        },
    )


async def _run(fake, service, **kw):
    async def ensure(mm, base, key, overrides, **_):
        fake.ctx = overrides["llama_cpp"]["ctx_size"]
        state = SimpleNamespace(
            worker_info=SimpleNamespace(pid=4242),
            backend_type="llamacpp_mks",
            backend_model_id="qwen3.8:27b",
            current_gpu=[1],
        )
        fake._states[key] = state
        fake.probe_overrides = overrides
        fake.probe_key = key
        return state

    with (
        patch(
            "ocabra.core.runtime_coordinator.plan_profile_runtime", AsyncMock(return_value=_plan())
        ),
        patch("ocabra.api.openai._deps.ensure_worker_loaded", ensure),
        patch("ocabra.core.context_probe_service.settings.llama_cpp_live_context_enabled", True),
    ):
        job = service.start(
            "qwen3.8:27b", "normal", full_validation=False, measure_vision=False, **kw
        )
        for _ in range(200):
            if job.finished_at:
                break
            await asyncio.sleep(0.01)
    return job


@pytest.mark.asyncio
async def test_probe_measures_and_puts_everything_back():
    fake = FakeManager()
    job = await _run(fake, _service(fake))

    assert job.status == "done", job.error
    # 16 GB + 17 MB/1k up to 23.6 GB → ~457k, capped by the 1M default
    assert job.result["max_context"] >= 450000
    assert job.result["used_mb"][1] <= 23600
    # Only what sat on GPU 1 was evicted. The pinned one left "under pressure",
    # so its own reload watcher brings it back; the probe does not reload it.
    assert job.evicted == ["whisper/turbo", "llamacpp_mks/qwen3-embedding:4b"]
    assert fake.unload_reasons["whisper/turbo"] == "pressure"
    assert fake.unload_reasons["llamacpp_mks/qwen3-embedding:4b"] == "context_probe"
    assert fake.loaded == []
    # The probe worker is gone and maintenance was lifted.
    assert fake.probe_key in fake.unloaded
    assert fake.maintenance == [
        ("llamacpp_mks/qwen3.8:27b", True),
        ("llamacpp_mks/qwen3.8:27b", False),
    ]
    # It started small, on its own key, without the preset's default bookkeeping.
    assert fake.probe_overrides["llama_cpp"]["ctx_size"] == 32768
    assert fake.probe_overrides["use_case"]["context"] == 32768
    assert "default_worker_key" not in fake.probe_overrides["_runtime"]


@pytest.mark.asyncio
async def test_a_failing_probe_still_cleans_up():
    fake = FakeManager()
    fake._gpu_manager.admission_ceiling_mb = MagicMock(return_value=None)  # NVML unreadable
    job = await _run(fake, _service(fake))

    assert job.status == "error"
    assert "reserved memory" in job.error
    assert fake.probe_key in fake.unloaded
    assert fake.maintenance[-1] == ("llamacpp_mks/qwen3.8:27b", False)


@pytest.mark.asyncio
async def test_only_one_probe_at_a_time():
    fake = FakeManager()
    service = _service(fake)
    await service._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="another context probe"):
            service.start("qwen3.8:27b", "normal", full_validation=False, measure_vision=False)
    finally:
        service._lock.release()


@pytest.mark.asyncio
async def test_the_probe_waits_for_its_worker_to_be_really_gone(monkeypatch):
    """2026-11-11 live: the card was not free yet right after unloading the
    probe worker; the next load (a pin's watcher, or a request) got a 409."""
    fake = FakeManager()
    job = await _run(fake, _service(fake))

    assert job.status == "done", job.error
    assert fake.waited_for_release and fake.waited_for_release[0] > 20000
