"""Alta automática de peldaños con la proyección de llama.cpp (2026-11-16)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ocabra.core import auto_setup as au


class _Backend:
    """Proyección de juguete: pesos + KV por 1k de contexto, repartidos por
    GPU según tensor_split; cada capa de expertos en CPU saca ``expert`` MB."""

    def __init__(self, weights=16000, kv_per_1k=40.0, compute=600, expert=400):
        self.w, self.kv, self.c, self.e = weights, kv_per_1k, compute, expert

    async def project_memory(self, model_id, gpus, extra):
        sec = extra.get("llama_cpp", {})
        if not gpus:
            return {"vram_by_gpu": {}, "estimated_vram_mb": 0, "host_mb": self.w}
        ctx = int(sec.get("ctx_size", 4096))
        n = int(sec.get("n_cpu_moe") or 0)
        gpu_w = max(0, self.w - n * self.e)
        layers = int(sec["gpu_layers"]) if sec.get("gpu_layers") is not None else 999
        if layers < 30:  # denso: solo las capas en GPU pesan en la GPU
            gpu_w = self.w * layers / 30
        total = gpu_w + self.kv * ctx / 1024
        split = sec.get("tensor_split") or [1.0] * len(gpus)
        per = {g: int(total * s / sum(split)) + self.c for g, s in zip(gpus, split, strict=True)}
        return {"vram_by_gpu": per, "estimated_vram_mb": sum(per.values()), "host_mb": n * self.e}


def _plan(budgets, backend, gpus, cpu, *, layers=30, moe=True, native=262144):
    state = SimpleNamespace(
        backend_model_id="m", extra_config={"llama_cpp": {"model_path": "/m.gguf"}}
    )
    arch = SimpleNamespace(
        layers=layers, expert_mb_by_layer=(1.0,) if moe else (), context_length=native
    )
    blist = [au.GpuBudget(i, n, b, b) for i, n, b in budgets if i in gpus]
    with (
        patch.object(au, "gpu_budgets", return_value=sorted(blist, key=lambda x: -x.budget_mb)),
        patch("ocabra.core.vram_planner.arch_from_gguf", return_value=arch),
    ):
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            au.plan_setup(None, backend, state, gpus=gpus, allow_cpu=cpu)
        )


RTX = [(0, "NVIDIA RTX 3060", 11000), (1, "NVIDIA RTX 3090", 22500)]


def test_cabe_en_una_gpu_y_se_eligen_dos_da_normal_y_extended():
    plan = _plan(RTX, _Backend(), [0, 1], False)
    keys = [r.key for r in plan.rungs]
    assert keys == ["normal", "extended"]
    normal, ext = plan.rungs
    assert normal.gpus == [1]  # la grande
    assert 8192 <= normal.context < 262144 and normal.context % 4096 == 0
    assert ext.gpus == [1, 0] and ext.context > normal.context
    assert ext.probe and ext.probe["kv_mb_per_1k"] and ext.probe["source"] == "llama_fit"


def test_con_una_sola_gpu_no_hay_extended():
    plan = _plan(RTX, _Backend(), [1], False)
    assert [r.key for r in plan.rungs] == ["normal"]


def test_solo_cabe_entre_las_dos():
    plan = _plan(RTX, _Backend(weights=28000, kv_per_1k=5), [0, 1], False)
    assert len(plan.rungs) == 1 and plan.rungs[0].gpus == [1, 0]
    assert plan.rungs[0].probe is not None


def test_no_cabe_sin_cpu_lo_dice():
    plan = _plan([(0, "GTX 1660", 4500)], _Backend(weights=15000), [0], False)
    assert not plan.rungs and plan.needs_cpu
    assert any("CPU" in w for w in plan.warnings)


def test_moe_con_cpu_saca_los_menos_expertos_posibles():
    backend = _Backend(weights=15000, kv_per_1k=20, compute=500, expert=450)
    plan = _plan([(0, "GTX 1660", 4500)], backend, [0], True)
    (rung,) = plan.rungs
    n = rung.placement["n_cpu_moe"]
    assert 0 < n <= 30 and rung.context >= 8192 and "RAM" in rung.label
    # Con un experto menos en RAM ya no llegaría al contexto objetivo.
    gpu_menos_uno = 15000 - (n - 1) * 450 + 20 * min(262144, au.TARGET_CONTEXT) / 1024 + 500
    assert gpu_menos_uno > 4500


def test_denso_con_cpu_baja_capas():
    plan = _plan([(0, "GTX 1660", 4500)], _Backend(weights=8000, expert=270), [0], True, moe=False)
    (rung,) = plan.rungs
    assert rung.placement["gpu_layers"] is not None and rung.placement["gpu_layers"] < 30


def test_solo_cpu():
    plan = _plan([], _Backend(), [], True)
    (rung,) = plan.rungs
    assert rung.gpus == [] and rung.placement == {"gpu_layers": 0}


@pytest.mark.asyncio
async def test_presupuesto_resta_escritorio_sistema_y_margen():
    gpu = SimpleNamespace(
        index=0,
        name="GTX 1660",
        total_vram_mb=6144,
        free_vram_mb=4888,
        processes=[
            SimpleNamespace(pid=1865, used_vram_mb=443),  # escritorio
            SimpleNamespace(pid=4242, used_vram_mb=700),  # worker de oCabra
        ],
    )
    gm = SimpleNamespace(get_all_states=lambda: _aw([gpu]))
    mm = SimpleNamespace(
        _gpu_manager=gm,
        _states={"x": SimpleNamespace(worker_info=SimpleNamespace(pid=4242))},
    )
    (b,) = await au.gpu_budgets(mm, [0])
    unowned = 6144 - 4888 - 1143
    from ocabra.config import settings

    esperado = min(6144 - settings.vram_buffer_mb, int(6144 * settings.vram_eviction_threshold))
    assert b.budget_mb == esperado - unowned - 443  # el worker de oCabra no resta


async def _aw(value):
    return value


def test_si_no_cabe_por_el_borrador_especulativo_lo_dice():
    backend = _Backend(weights=15000, expert=450)
    state = SimpleNamespace(
        backend_model_id="m",
        extra_config={
            "llama_cpp": {"model_path": "/m.gguf", "speculative": {"spec_type": "draft-mtp"}}
        },
    )
    arch = SimpleNamespace(layers=30, expert_mb_by_layer=(1.0,), context_length=262144)

    class _NoCabe(_Backend):
        async def project_memory(self, model_id, gpus, extra):
            r = await super().project_memory(model_id, gpus, extra)
            r["vram_by_gpu"] = {g: mb + 9000 for g, mb in r["vram_by_gpu"].items()}
            return r

    with (
        patch.object(au, "gpu_budgets", return_value=[au.GpuBudget(0, "GTX 1660", 4500, 4500)]),
        patch("ocabra.core.vram_planner.arch_from_gguf", return_value=arch),
    ):
        import asyncio

        plan = asyncio.get_event_loop().run_until_complete(
            au.plan_setup(None, _NoCabe(weights=15000, expert=450), state, gpus=[0], allow_cpu=True)
        )
    assert not plan.rungs
    assert any("especulativa" in w for w in plan.warnings)
    del backend


def test_avisa_si_la_ram_no_da():
    plan = au.SetupPlan(rungs=[au.Rung("normal", "1660 · 40k · RAM", [0], 40960, host_mb=60000)])
    with patch.object(au, "_host_ram_mb", return_value=32768):
        au.warn_ram(plan)
    assert plan.warnings and "RAM" in plan.warnings[0]
