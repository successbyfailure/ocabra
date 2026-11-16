"""Estimación de memoria con la proyección del propio llama.cpp (2026-11-16).

Referencia real (gemma4:26b, expertos en RAM, 10k, 4 slots, q8, 3060):
proyección 2459+755+541 = 3755 MiB en GPU y 14557+26 en RAM, frente a 3752
medidos; la fórmula de oCabra daba 4371 en GPU y 13809 en RAM.
"""

from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.core import llama_fit

SALIDA_REAL = (
    "0.00.065 I llama_fit_params: printing estimated memory in MiB to stdout ...\n"
    "CUDA0 2459 755 541 \n"
    "Host 14557 0 26\n"
)


def test_parse_mapea_cuda_k_a_la_gpu_fisica():
    p = llama_fit.parse_output("CUDA0 100 20 30\nCUDA1 200 40 50\nHost 900 0 5\n", [1, 3])
    assert p.gpus == {
        1: {"model": 100, "context": 20, "compute": 30},
        3: {"model": 200, "context": 40, "compute": 50},
    }
    assert p.vram_by_gpu == {1: 150, 3: 290}
    assert p.vram_total_mb == 440
    assert p.host_mb == 905


def test_parse_salida_real():
    p = llama_fit.parse_output(SALIDA_REAL, [0])
    assert p.vram_total_mb == 3755 and p.host_mb == 14583


def test_parse_sin_nada_util():
    assert llama_fit.parse_output("error: invalid argument\n", [0]) is None


def test_argumentos_como_el_arranque():
    args = llama_fit.args_from_options(
        "/m.gguf",
        {
            "ctx_size": 10240,
            "parallel_slots": 4,
            "gpu_layers": 999,
            "n_cpu_moe": 30,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "flash_attn": True,
            "tensor_split": [1.0, 3.0],
        },
    )
    joined = " ".join(args)
    for esperado in (
        "-m /m.gguf",
        "--fit-print on",
        "-c 10240",
        "-np 4",
        "-ngl 999",
        "--n-cpu-moe 30",
        "-ctk q8_0",
        "-ctv q8_0",
        "-fa on",
        "-ts 1,3",
    ):
        assert esperado in joined
    assert "--kv-unified" not in joined  # la herramienta no lo acepta


def _fake_bin(tmp_path: Path, salida: str, rc: int = 0) -> Path:
    server = tmp_path / "llama-server"
    server.write_text("")
    fit = tmp_path / "llama-fit-params"
    fit.write_text(
        "#!/bin/sh\n"
        f'echo "$CUDA_VISIBLE_DEVICES" > {tmp_path}/visibles\n'
        f"printf '%s' '{salida}'\n"
        f"exit {rc}\n"
    )
    fit.chmod(fit.stat().st_mode | stat.S_IEXEC)
    return server


@pytest.mark.asyncio
async def test_project_ejecuta_la_herramienta_con_las_gpus_visibles(tmp_path):
    server = _fake_bin(tmp_path, SALIDA_REAL)
    modelo = tmp_path / "m.gguf"
    modelo.write_bytes(b"GGUF")
    p = await llama_fit.project(server, str(modelo), [1], ["-m", str(modelo), "--fit-print", "on"])
    assert p is not None and p.vram_by_gpu == {1: 3755}
    assert (tmp_path / "visibles").read_text().strip() == "1"


@pytest.mark.asyncio
async def test_sin_herramienta_o_con_fallo_devuelve_none(tmp_path):
    modelo = tmp_path / "m.gguf"
    modelo.write_bytes(b"GGUF")
    sin = tmp_path / "otro" / "llama-server"
    sin.parent.mkdir()
    sin.write_text("")
    assert await llama_fit.project(sin, str(modelo), [0], ["x"]) is None
    server = _fake_bin(tmp_path, "", rc=1)
    assert await llama_fit.project(server, str(modelo), [0], ["y"]) is None


@pytest.mark.asyncio
async def test_la_carga_usa_la_proyeccion_cuando_hay():
    from ocabra.core.model_manager import ModelManager

    mm = ModelManager(worker_pool=None)
    state = SimpleNamespace(
        model_id="llamacpp_mks/gemma4:26b",
        backend_model_id="gemma4:26b",
        preferred_gpu=0,
        extra_config={"llama_cpp": {"n_cpu_moe": 30}},
    )
    backend = SimpleNamespace(
        project_memory=AsyncMock(return_value={"estimated_vram_mb": 3755, "vram_by_gpu": {0: 3755}})
    )
    assert await mm._project_llama_cpp_vram(state, backend) == 3755
    backend.project_memory.assert_awaited_once_with("gemma4:26b", [0], state.extra_config)
    sin = SimpleNamespace(project_memory=AsyncMock(return_value=None))
    assert await mm._project_llama_cpp_vram(state, sin) == 0


def test_la_vista_previa_traduce_las_partes_y_da_la_ram():
    from ocabra.api.internal.models import _apply_fit_projection

    gpus, est = _apply_fit_projection(
        {
            "vram_breakdown_by_gpu": {0: {"model": 2459, "context": 755, "compute": 541}},
            "estimated_vram_mb": 3755,
            "host_mb": 14583,
        },
        {"model_loading_memory_mb": 4371, "notes": ["Desglose en MB — x"]},
        {0: 6144},
    )
    assert gpus == [
        {"index": 0, "mb": 3755, "breakdown": {"weights": 2459, "kv": 755, "overhead": 541}}
    ]
    assert est["model_loading_memory_mb"] == 3755 and est["host_ram_mb"] == 14583
    assert est["source"] == "llama_fit" and est["fits_current_gpu"] is True
    assert est["formula"]["model_loading_memory_mb"] == 4371
