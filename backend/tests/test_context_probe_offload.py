"""«Contexto objetivo, lo más rápido»: fewest layers in RAM, validated by loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocabra.core import auto_setup, host_memory, llama_fit
from ocabra.core import vram_planner as vp
from ocabra.core.context_probe_service import ContextProbeService, ProbeError, ProbeJob

LAYERS = 48
LIMIT = 4600  # MB the worker may hold on the 1660


def _real_mb(n: int, ctx: int) -> int:
    """What a load really takes: 1500 MB with every expert in RAM at 8k, plus
    ~70 MB per expert layer kept on the GPU and the KV."""
    return 1500 + (LAYERS - n) * 70 + (ctx - 8192) // 1024 * 50


def _service(monkeypatch, *, ram_free=30000):
    llama_fit._offsets.clear()
    svc = ContextProbeService.__new__(ContextProbeService)
    state = SimpleNamespace(backend_type="llamacpp_mks", backend_model_id="qwen3:30b", extra_config={})
    svc._mm = SimpleNamespace(get_state=lambda _id: _aw(state))
    svc._pool = SimpleNamespace(_backends={"llamacpp_mks": SimpleNamespace(project_memory=lambda *a: None)})
    monkeypatch.setattr(vp, "arch_from_gguf", lambda f: SimpleNamespace(layers=LAYERS, expert_mb_by_layer=(1,)))
    monkeypatch.setattr(svc, "_free_limits", lambda gpus: {0: LIMIT})
    monkeypatch.setattr(host_memory, "read_host_memory", lambda: {"total_mb": 32000, "available_mb": ram_free})
    monkeypatch.setattr(host_memory, "file_mb", lambda f: 18000)

    class Projector:
        def __init__(self, *_):
            pass

        async def __call__(self, gpus, ov):
            n = ov.get("n_cpu_moe") or 0
            raw = _real_mb(n, ov["ctx_size"]) + 150  # projection 150 MB high (Mapache)
            off = llama_fit.calibration_offset_mb("/m.gguf", 0)
            return {"vram_by_gpu": {0: raw + off}, "raw_vram_by_gpu": {0: raw}, "host_mb": 14000 + n * 20}

    monkeypatch.setattr(auto_setup, "_Projector", Projector)
    loads: list[int] = []

    async def try_offload(job, base_id, ov, gpus, n, ctx, *, layers, moe):
        loads.append(n)
        used = _real_mb(n, ctx)
        return {"in_ram": n, "context": ctx, "used_mb": {"0": used}, "fits": used <= LIMIT}

    monkeypatch.setattr(svc, "_try_offload", try_offload)
    return svc, loads


async def _aw(v):
    return v


def _job(target: int) -> ProbeJob:
    return ProbeJob("j", "p", "normal", False, False, target_context=target, context_is_pool=True)


@pytest.mark.asyncio
async def test_finds_the_fewest_layers_in_ram_and_validates_them(monkeypatch):
    svc, loads = _service(monkeypatch)
    out = await svc._run_offload(_job(16384), "llamacpp_mks/qwen3:30b", {"llama_cpp": {"model_path": "/m.gguf"}}, [0])
    # Truth: the fewest n with _real_mb(n, 16k) <= 4600.
    truth = min(n for n in range(LAYERS + 1) if _real_mb(n, 16384) <= LIMIT)
    assert out["in_ram"] == truth and out["placement"] == {"n_cpu_moe": truth}
    # The projection alone (150 MB high) would have stopped two layers short;
    # the first real load calibrates it and the search goes on.
    assert out["first_guess"] > truth
    assert loads[0] == out["first_guess"] and truth in loads
    assert all(int(v) <= LIMIT for v in out["used_mb"].values())


@pytest.mark.asyncio
async def test_refuses_before_loading_when_the_ram_is_not_there(monkeypatch):
    svc, loads = _service(monkeypatch, ram_free=9000)
    with pytest.raises(ProbeError, match="RAM"):
        await svc._run_offload(_job(16384), "m", {"llama_cpp": {"model_path": "/m.gguf"}}, [0])
    assert loads == []


@pytest.mark.asyncio
async def test_says_so_when_not_even_everything_in_ram_fits(monkeypatch):
    svc, loads = _service(monkeypatch)
    with pytest.raises(ProbeError, match="no cabe"):
        await svc._run_offload(_job(262144), "m", {"llama_cpp": {"model_path": "/m.gguf"}}, [0])
    assert loads == []
