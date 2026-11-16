"""VRAM of a MoE with its experts in RAM (``n_cpu_moe`` / ``override_tensor``).

Case that motivated it (2026-11-16): Qwen3-30B-A3B and Gemma 4 26B on a 6 GB
GTX 1660 with ``gpu_layers: 99`` and the experts in RAM. Before the load the
planner already honoured ``n_cpu_moe``; after it the backend reported the whole
file (~18 GB), and the load keeps the larger figure.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from ocabra.core import vram_planner as vp

LAYERS = 48
EXPERT_MB_PER_LAYER = 348  # ~95 % of each layer, as measured on Qwen3-30B-A3B
OTHER_MB_PER_LAYER = 20
MB = 1024 * 1024


def _str(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _write_moe_gguf(path: Path) -> Path:
    """A sparse GGUF shaped like Qwen3-30B-A3B: real header and tensor table,
    no data on disk (the file just claims its size)."""
    kv = {
        "general.architecture": "qwen3moe",
        "qwen3moe.block_count": LAYERS,
        "qwen3moe.attention.head_count": 32,
        "qwen3moe.attention.head_count_kv": 4,
        "qwen3moe.attention.key_length": 128,
        "qwen3moe.attention.value_length": 128,
        "qwen3moe.embedding_length": 2048,
        "qwen3moe.context_length": 40960,
        "qwen3moe.expert_count": 128,
    }
    tensors: list[tuple[str, int]] = []
    for layer in range(LAYERS):
        tensors.append((f"blk.{layer}.attn_q.weight", OTHER_MB_PER_LAYER * MB))
        for kind in ("gate", "up", "down"):
            tensors.append((f"blk.{layer}.ffn_{kind}_exps.weight", EXPERT_MB_PER_LAYER * MB // 3))
    tensors.append(("token_embd.weight", 300 * MB))

    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, value in kv.items():
        out += _str(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + _str(value)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", value)
    offset = 0
    for name, size in tensors:
        out += _str(name) + struct.pack("<I", 2) + struct.pack("<QQ", 1, 1)
        out += struct.pack("<I", 0) + struct.pack("<Q", offset)
        offset += size
    data_start = -(-len(out) // 32) * 32
    with open(path, "wb") as f:
        f.write(out)
        f.truncate(data_start + offset)  # sparse: the size without the bytes
    return path


@pytest.fixture
def moe_gguf(tmp_path) -> Path:
    return _write_moe_gguf(tmp_path / "qwen3-30b-a3b.gguf")


def test_the_expert_size_of_each_layer_comes_from_the_tensor_table(moe_gguf):
    arch = vp.arch_from_gguf(moe_gguf)

    assert arch is not None and arch.layers == LAYERS
    assert len(arch.expert_mb_by_layer) == LAYERS
    assert arch.expert_mb_by_layer[0] == pytest.approx(EXPERT_MB_PER_LAYER, abs=1)


def test_all_experts_in_ram_leaves_only_the_rest_on_the_gpu(moe_gguf):
    """With the real table the experts moved out are their real size; the
    fixed 0.85 share left ~1.7 GB too many on the GPU for this model."""
    arch = vp.arch_from_gguf(moe_gguf)
    file_mb = moe_gguf.stat().st_size / MB

    measured = vp.moe_cpu_weights_mb(
        file_mb, LAYERS, 99, LAYERS, expert_mb_by_layer=arch.expert_mb_by_layer
    )
    guessed = vp.moe_cpu_weights_mb(file_mb, LAYERS, 99, LAYERS)

    assert measured == pytest.approx(LAYERS * EXPERT_MB_PER_LAYER, abs=LAYERS)
    assert measured - guessed > 1000


def test_n_cpu_moe_counts_from_the_first_layer_and_only_on_gpu_layers(moe_gguf):
    arch = vp.arch_from_gguf(moe_gguf)
    table = arch.expert_mb_by_layer

    first_ten = vp.moe_cpu_weights_mb(0, LAYERS, None, 10, expert_mb_by_layer=table)
    # 40 layers on the GPU (the last 40): layers 0-7 are already in RAM, so
    # n_cpu_moe=10 only moves the experts of layers 8 and 9.
    partial = vp.moe_cpu_weights_mb(0, LAYERS, 40, 10, expert_mb_by_layer=table)

    assert first_ten == pytest.approx(10 * EXPERT_MB_PER_LAYER, abs=10)
    assert partial == pytest.approx(2 * EXPERT_MB_PER_LAYER, abs=2)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("exps=CPU", LAYERS),
        (r"\.ffn_.*_exps\.=CPU", LAYERS),
        (r"blk\.([0-9]|1[0-9])\.ffn_.*_exps=CPU", 20),
        (r"blk\.(2|5)\.ffn_.*_exps=CPU", None),  # scattered layers: not modelled
        ("exps=CUDA0", None),  # pins to a GPU
        ("attn_q=CPU", None),  # not experts
        ("", None),
        (None, None),
    ],
)
def test_override_tensor_rules_that_move_experts_to_cpu(pattern, expected):
    assert vp.n_cpu_moe_from_override_tensor(pattern, LAYERS) == expected


def _config(moe_gguf: Path, **llama_cpp) -> dict:
    return {
        "llama_cpp": {
            "model_path": str(moe_gguf),
            "gpu_layers": 99,
            "ctx_size": 4096,
            **llama_cpp,
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "experts_in_ram", [{"n_cpu_moe": LAYERS}, {"override_tensor": "exps=CPU"}]
)
async def test_the_backend_reports_the_same_vram_before_and_after_the_load(
    moe_gguf, experts_in_ram
):
    """The bug: after the load the backend reported file × gpu_layers/total,
    i.e. the whole ~17 GB file for a model that needs ~2 GB on the GPU."""
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend

    backend = LlamaCppBackend()
    config = _config(moe_gguf, **experts_in_ram)
    before = await backend.get_vram_estimate_mb("qwen3-30b-a3b", extra_config=config)

    # What load() leaves behind: the resolved options and the config it used.
    backend._model_configs["qwen3-30b-a3b"] = {
        "gpu_layers": 99,
        "total_layers": LAYERS,
        "model_file": str(moe_gguf),
    }
    backend._load_extra_configs["qwen3-30b-a3b"] = config
    after = await backend.get_vram_estimate_mb("qwen3-30b-a3b")

    file_mb = moe_gguf.stat().st_size / MB
    assert before == after
    assert after < 3000 < file_mb


@pytest.mark.asyncio
async def test_without_experts_in_ram_the_whole_model_counts(moe_gguf):
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend

    estimate = await LlamaCppBackend().get_vram_estimate_mb(
        "qwen3-30b-a3b", extra_config=_config(moe_gguf)
    )

    assert estimate > moe_gguf.stat().st_size / MB


@pytest.mark.asyncio
async def test_after_the_load_an_explicit_vram_estimate_still_wins(moe_gguf):
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._model_configs["m"] = {"gpu_layers": 99, "model_file": str(moe_gguf)}
    backend._load_extra_configs["m"] = _config(moe_gguf, vram_estimate_mb=5000)

    assert await backend.get_vram_estimate_mb("m") == 5000


@pytest.mark.asyncio
async def test_a_model_found_by_id_uses_the_file_the_load_resolved(moe_gguf):
    """No ``model_path`` in the config (the model was resolved by id): the
    estimate needs the file the load found, or it falls back to nothing."""
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._model_configs["m"] = {"gpu_layers": 99, "model_file": str(moe_gguf)}
    backend._load_extra_configs["m"] = {"llama_cpp": {"n_cpu_moe": LAYERS}}

    assert 0 < await backend.get_vram_estimate_mb("m") < 3000
