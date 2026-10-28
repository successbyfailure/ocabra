from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends.llama_cpp_backend import LlamaCppBackend, _compose_visible_devices


def _fake_proc(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.pid = 5151
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    return proc


def _stub_settings(mock_settings: MagicMock, models_dir: str) -> None:
    """Apply the standard llama_cpp settings stub used across tests."""
    mock_settings.models_dir = models_dir
    mock_settings.llama_cpp_server_bin = "/usr/local/bin/llama-server"
    mock_settings.llama_cpp_gpu_layers = 16
    mock_settings.llama_cpp_ctx_size = 8192
    mock_settings.llama_cpp_threads = 8
    mock_settings.llama_cpp_batch_size = 256
    mock_settings.llama_cpp_ubatch_size = 64
    mock_settings.llama_cpp_flash_attn = True
    mock_settings.llama_cpp_mlock = True
    mock_settings.llama_cpp_embeddings = False
    mock_settings.llama_cpp_startup_timeout_s = 30
    mock_settings.cuda_device_order = "PCI_BUS_ID"


async def _capture_load_cmd(
    tmp_path: Path,
    extra_config: dict[str, Any],
) -> list[str]:
    """Run ``LlamaCppBackend.load`` with mocks and return the worker argv."""
    gguf = tmp_path / "demo.gguf"
    if not gguf.exists():
        gguf.write_bytes(b"GGUF" * 1024)

    proc = _fake_proc()
    backend = LlamaCppBackend()
    with (
        patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(LlamaCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        _stub_settings(mock_settings, str(tmp_path))
        await backend.load("demo", [1], port=18099, extra_config=extra_config)

    return list(create_proc.await_args.args)


@pytest.mark.asyncio
async def test_load_success(tmp_path: Path) -> None:
    gguf = tmp_path / "demo.gguf"
    gguf.write_bytes(b"GGUF" * 1024)

    proc = _fake_proc()
    backend = LlamaCppBackend()
    with (
        patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(LlamaCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        _stub_settings(mock_settings, str(tmp_path))
        info = await backend.load("demo", [1], port=18031, extra_config={})

    assert info.backend_type == "llama_cpp"
    assert info.port == 18031
    assert info.pid == 5151
    args = create_proc.await_args.args
    assert "llama_cpp_worker.py" in str(args[1])
    assert "--gpu-layers" in args
    assert "--flash-attn" in args
    assert info.vram_used_mb > 0
    # By default no Tier 1 flag is emitted.
    assert "--no-mmap" not in args
    assert "--seed" not in args
    assert "--no-kv-offload" not in args
    assert "--rope-freq-base" not in args
    assert "--rope-freq-scale" not in args


@pytest.mark.asyncio
async def test_capabilities_use_embedding_flag_and_context() -> None:
    backend = LlamaCppBackend()
    backend._model_configs["demo-embed"] = {"embedding": True, "ctx_size": 16384}

    caps = await backend.get_capabilities("demo-embed")

    assert caps.embeddings is True
    assert caps.chat is False
    assert caps.context_length == 16384


# --- Sprint 17.1 — Tier 1 flag wiring ---


@pytest.mark.asyncio
async def test_mmap_false_emits_no_mmap_flag(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {"llama_cpp": {"mmap": False}})
    assert "--no-mmap" in args


@pytest.mark.asyncio
async def test_mmap_true_does_not_emit_no_mmap_flag(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {"llama_cpp": {"mmap": True}})
    assert "--no-mmap" not in args


@pytest.mark.asyncio
async def test_mmap_unset_does_not_emit_no_mmap_flag(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {})
    assert "--no-mmap" not in args


@pytest.mark.asyncio
async def test_seed_emits_seed_flag(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {"llama_cpp": {"seed": 1234}})
    assert "--seed" in args
    idx = args.index("--seed")
    assert args[idx + 1] == "1234"


@pytest.mark.asyncio
async def test_seed_unset_does_not_emit_seed_flag(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {})
    assert "--seed" not in args


@pytest.mark.asyncio
async def test_no_kv_offload_emits_flag(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {"llama_cpp": {"no_kv_offload": True}})
    assert "--no-kv-offload" in args


@pytest.mark.asyncio
async def test_no_kv_offload_false_does_not_emit_flag(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {"llama_cpp": {"no_kv_offload": False}})
    assert "--no-kv-offload" not in args


@pytest.mark.asyncio
async def test_rope_freq_base_emits_flag_with_value(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {"llama_cpp": {"rope_freq_base": 10000.0}})
    assert "--rope-freq-base" in args
    idx = args.index("--rope-freq-base")
    assert float(args[idx + 1]) == pytest.approx(10000.0)


@pytest.mark.asyncio
async def test_rope_freq_scale_emits_flag_with_value(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {"llama_cpp": {"rope_freq_scale": 0.5}})
    assert "--rope-freq-scale" in args
    idx = args.index("--rope-freq-scale")
    assert float(args[idx + 1]) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_rope_freq_unset_does_not_emit_flag(tmp_path: Path) -> None:
    args = await _capture_load_cmd(tmp_path, {})
    assert "--rope-freq-base" not in args
    assert "--rope-freq-scale" not in args


@pytest.mark.asyncio
async def test_camel_case_keys_resolve_in_nested_section(tmp_path: Path) -> None:
    """Frontend persists camelCase keys; backend must accept both spellings."""
    args = await _capture_load_cmd(
        tmp_path,
        {"llama_cpp": {"noKvOffload": True, "ropeFreqBase": 8000.0}},
    )
    assert "--no-kv-offload" in args
    assert "--rope-freq-base" in args


@pytest.mark.asyncio
async def test_all_tier1_flags_combined(tmp_path: Path) -> None:
    args = await _capture_load_cmd(
        tmp_path,
        {
            "llama_cpp": {
                "mmap": False,
                "seed": 42,
                "no_kv_offload": True,
                "rope_freq_base": 12345.6,
                "rope_freq_scale": 0.25,
            }
        },
    )
    for flag in ("--no-mmap", "--seed", "--no-kv-offload", "--rope-freq-base", "--rope-freq-scale"):
        assert flag in args, f"missing {flag} in {args}"


# --- Sprint 17.2 — KV cache quantization ---


@pytest.mark.asyncio
async def test_load_passes_cache_type_flags(tmp_path: Path) -> None:
    gguf = tmp_path / "demo.gguf"
    gguf.write_bytes(b"GGUF" * 1024)

    proc = _fake_proc()
    backend = LlamaCppBackend()
    with (
        patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(LlamaCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        mock_settings.models_dir = str(tmp_path)
        mock_settings.llama_cpp_server_bin = "/usr/local/bin/llama-server"
        mock_settings.llama_cpp_gpu_layers = 16
        mock_settings.llama_cpp_ctx_size = 8192
        mock_settings.llama_cpp_threads = 8
        mock_settings.llama_cpp_batch_size = 256
        mock_settings.llama_cpp_ubatch_size = 64
        mock_settings.llama_cpp_flash_attn = True  # required for cache_type_v != f16
        mock_settings.llama_cpp_mlock = False
        mock_settings.llama_cpp_embeddings = False
        mock_settings.llama_cpp_startup_timeout_s = 30
        mock_settings.cuda_device_order = "PCI_BUS_ID"

        extra_config = {
            "llama_cpp": {
                "cache_type_k": "q8_0",
                "cache_type_v": "q4_0",
            }
        }
        await backend.load("demo", [0], port=18032, extra_config=extra_config)

    args = list(create_proc.await_args.args)
    assert "--cache-type-k" in args
    assert "q8_0" in args
    assert "--cache-type-v" in args
    assert "q4_0" in args


@pytest.mark.asyncio
async def test_load_rejects_quantized_v_without_flash_attn(tmp_path: Path) -> None:
    gguf = tmp_path / "demo.gguf"
    gguf.write_bytes(b"GGUF" * 1024)

    backend = LlamaCppBackend()
    with patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path)
        mock_settings.llama_cpp_server_bin = "/usr/local/bin/llama-server"
        mock_settings.llama_cpp_gpu_layers = 16
        mock_settings.llama_cpp_ctx_size = 8192
        mock_settings.llama_cpp_threads = 8
        mock_settings.llama_cpp_batch_size = 256
        mock_settings.llama_cpp_ubatch_size = 64
        mock_settings.llama_cpp_flash_attn = False  # <-- the gating bit
        mock_settings.llama_cpp_mlock = False
        mock_settings.llama_cpp_embeddings = False
        mock_settings.llama_cpp_startup_timeout_s = 30
        mock_settings.cuda_device_order = "PCI_BUS_ID"

        extra_config = {"llama_cpp": {"cache_type_v": "q4_0"}}
        with pytest.raises(ValueError, match="flash_attn"):
            await backend.load("demo", [0], port=18033, extra_config=extra_config)


def test_load_config_validator_rejects_quantized_v_without_flash_attn() -> None:
    """Pydantic-level validation mirrors the backend gating."""
    from ocabra.schemas.backend_load import LlamaCppLoadConfig

    with pytest.raises(ValueError, match="flash_attn"):
        LlamaCppLoadConfig(cache_type_v="q4_0")

    # f16 V cache is fine without flash_attn.
    cfg = LlamaCppLoadConfig(cache_type_v="f16")
    assert cfg.cache_type_v == "f16"

    # Quantized V cache works when flash_attn is on.
    cfg = LlamaCppLoadConfig(cache_type_v="q4_0", flash_attn=True)
    assert cfg.cache_type_v == "q4_0"


def test_speculative_config_accepts_self_speculative_mtp_without_draft_model_id() -> None:
    """2026-10-27 bug: the schema required draft_model_id and forbade
    spec_type as an unknown field, so qwen3.8:27b's real production config
    (`{"spec_type": "draft-mtp"}`, self-speculative — no external draft
    GGUF, the backend has read spec_type correctly since Sprint 17.4, see
    llama_cpp_backend.py) failed Pydantic validation on every single
    memory-estimate/preset-preview call for this model, silently falling
    back to a 'config inválida' warning + the crude heuristic estimate."""
    from ocabra.schemas.backend_load import LlamaCppLoadConfig, SpeculativeConfig

    cfg = LlamaCppLoadConfig(speculative={"spec_type": "draft-mtp"})
    assert cfg.speculative is not None
    assert cfg.speculative.spec_type == "draft-mtp"
    assert cfg.speculative.draft_model_id is None

    # External draft model (no self-speculation) still works too.
    cfg2 = SpeculativeConfig(draft_model_id="some/draft-model")
    assert cfg2.spec_type is None


def test_speculative_config_rejects_neither_spec_type_nor_draft_model_id() -> None:
    from ocabra.schemas.backend_load import SpeculativeConfig

    with pytest.raises(ValueError, match="spec_type"):
        SpeculativeConfig()


# ---------------------------------------------------------------------------
# Sprint 17.3 — Multi-GPU + MoE CPU offload
# Sprint 17.4 — Speculative decoding + runtime alterno + concurrent slots
# ---------------------------------------------------------------------------


def _patch_settings(mock_settings: MagicMock, models_dir: str) -> None:
    """Apply the baseline llama_cpp settings used by load() tests."""

    mock_settings.models_dir = models_dir
    mock_settings.llama_cpp_server_bin = "/usr/local/bin/llama-server"
    mock_settings.llama_cpp_gpu_layers = 16
    mock_settings.llama_cpp_ctx_size = 8192
    mock_settings.llama_cpp_threads = 8
    mock_settings.llama_cpp_batch_size = 256
    mock_settings.llama_cpp_ubatch_size = 64
    mock_settings.llama_cpp_flash_attn = False
    mock_settings.llama_cpp_mlock = False
    mock_settings.llama_cpp_embeddings = False
    mock_settings.llama_cpp_startup_timeout_s = 30
    mock_settings.cuda_device_order = "PCI_BUS_ID"
    mock_settings.backends_dir = str(Path(models_dir) / "backends")


def test_compose_visible_devices_excludes_disabled() -> None:
    assert _compose_visible_devices([0, 1, 2], [1]) == [0, 2]
    assert _compose_visible_devices([0, 1], None) == [0, 1]
    assert _compose_visible_devices([0, 1], []) == [0, 1]
    # Order is preserved (matters for --main-gpu semantics).
    assert _compose_visible_devices([2, 0, 1], [0]) == [2, 1]


def test_compose_visible_devices_with_gpu_manager_validates() -> None:
    manager = SimpleNamespace(_states={0: object(), 1: object()})
    # Index 5 is unknown to the manager; it must not be silently filtered out
    # of preferred_gpu, but it also should not raise.
    assert _compose_visible_devices([0, 1], [5], gpu_manager=manager) == [0, 1]
    assert _compose_visible_devices([0, 1], [1], gpu_manager=manager) == [0]


@pytest.mark.asyncio
async def test_load_forwards_multi_gpu_and_moe_flags(tmp_path: Path) -> None:
    gguf = tmp_path / "moe.gguf"
    gguf.write_bytes(b"GGUF" * 1024)

    proc = _fake_proc()
    backend = LlamaCppBackend()
    extra = {
        "llama_cpp": {
            "main_gpu": 1,
            "tensor_split": [3, 1],
            "split_mode": "row",
            "n_cpu_moe": 4,
            "override_tensor": "exps=CPU",
        }
    }
    with (
        patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(LlamaCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        _patch_settings(mock_settings, str(tmp_path))
        await backend.load("moe", [0, 1], port=18032, extra_config=extra)

    args = list(create_proc.await_args.args)
    assert "--main-gpu" in args and args[args.index("--main-gpu") + 1] == "1"
    assert "--tensor-split" in args and args[args.index("--tensor-split") + 1] == "3,1"
    assert "--split-mode" in args and args[args.index("--split-mode") + 1] == "row"
    assert "--n-cpu-moe" in args and args[args.index("--n-cpu-moe") + 1] == "4"
    assert "--override-tensor" in args
    assert args[args.index("--override-tensor") + 1] == "exps=CPU"


@pytest.mark.asyncio
async def test_load_disabled_gpus_filters_cuda_visible_devices(tmp_path: Path) -> None:
    gguf = tmp_path / "demo.gguf"
    gguf.write_bytes(b"GGUF" * 1024)

    proc = _fake_proc()
    backend = LlamaCppBackend()
    extra = {"llama_cpp": {"disabled_gpus": [0]}}
    with (
        patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(LlamaCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        _patch_settings(mock_settings, str(tmp_path))
        info = await backend.load("demo", [0, 1], port=18033, extra_config=extra)

    env = create_proc.await_args.kwargs["env"]
    # GPU 0 was disabled; only GPU 1 should be visible to the worker.
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert info.gpu_indices == [1]


@pytest.mark.asyncio
async def test_load_evenly_strategy_autocomputes_tensor_split(tmp_path: Path) -> None:
    gguf = tmp_path / "demo.gguf"
    gguf.write_bytes(b"GGUF" * 1024)

    proc = _fake_proc()
    # Mock GPU manager: GPU 0 = 12 GB, GPU 1 = 24 GB.
    manager = MagicMock()
    manager._states = {0: object(), 1: object()}
    manager.get_state = AsyncMock(
        side_effect=lambda idx: SimpleNamespace(total_vram_mb={0: 12 * 1024, 1: 24 * 1024}[idx])
    )

    backend = LlamaCppBackend(gpu_manager=manager)
    extra = {"llama_cpp": {"split_strategy": "evenly"}}
    with (
        patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(LlamaCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        _patch_settings(mock_settings, str(tmp_path))
        await backend.load("demo", [0, 1], port=18034, extra_config=extra)

    args = list(create_proc.await_args.args)
    # Smallest GPU is normalised to 1 → ratios become 1,2.
    assert "--tensor-split" in args
    csv = args[args.index("--tensor-split") + 1]
    assert csv == "1,2"
    manager.get_state.assert_awaited()


def test_compute_evenly_tensor_split_returns_none_without_manager() -> None:
    backend = LlamaCppBackend()
    # No event loop required because we exercise the helper directly through
    # the public surface used by load(): asyncio.run is fine for one call.
    import asyncio

    result = asyncio.run(backend._compute_evenly_tensor_split([0, 1]))
    assert result is None


# --- Sprint 17.4 — Speculative decoding + runtime alterno ---


@pytest.mark.asyncio
async def test_load_passes_speculative_flags(tmp_path: Path) -> None:
    """Speculative draft + draft_n/min/p_min should reach llama-server cmd."""
    target = tmp_path / "target.gguf"
    target.write_bytes(b"GGUF" * 1024)
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"GGUF" * 256)

    proc = _fake_proc()
    backend = LlamaCppBackend()
    extra_config = {
        "llama_cpp": {
            "speculative": {
                "draft_model_id": str(draft),
                "draft_n": 8,
                "draft_min": 2,
                "draft_p_min": 0.4,
            },
            "parallel_slots": 4,
            "cont_batching": True,
        }
    }
    with (
        patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(LlamaCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        _patch_settings(mock_settings, str(tmp_path))
        await backend.load("target", [1], port=18045, extra_config=extra_config)

    args = list(create_proc.await_args.args)
    assert "--model-draft" in args
    assert str(draft) in args
    assert "--spec-draft-n-max" in args and args[args.index("--spec-draft-n-max") + 1] == "8"
    assert "--spec-draft-n-min" in args and args[args.index("--spec-draft-n-min") + 1] == "2"
    assert "--spec-draft-p-min" in args
    assert "--parallel" in args and args[args.index("--parallel") + 1] == "4"
    assert "--cont-batching" in args


@pytest.mark.asyncio
async def test_load_passes_self_contained_mtp_spec_type(tmp_path: Path) -> None:
    """spec_type alone (no draft_model_id) self-speculates from a nextn block
    baked into the main GGUF — no --model-draft should be added."""
    target = tmp_path / "target.gguf"
    target.write_bytes(b"GGUF" * 1024)

    proc = _fake_proc()
    backend = LlamaCppBackend()
    extra_config = {
        "llama_cpp": {
            "speculative": {
                "spec_type": "draft-mtp",
                "draft_n": 4,
            },
        }
    }
    with (
        patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(LlamaCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        _patch_settings(mock_settings, str(tmp_path))
        await backend.load("target", [1], port=18046, extra_config=extra_config)

    args = list(create_proc.await_args.args)
    assert "--model-draft" not in args
    assert "--spec-type" in args and args[args.index("--spec-type") + 1] == "draft-mtp"
    assert "--spec-draft-n-max" in args and args[args.index("--spec-draft-n-max") + 1] == "4"


@pytest.mark.asyncio
async def test_load_passes_mmproj_flag(tmp_path: Path) -> None:
    """2026-10-27: vision support (gemma4:26b's clip/gemma4v projector) was
    entirely unimplemented — extra_config had nowhere to put a --mmproj
    path. Validated live against the real gemma4:26b GGUFs (external
    draft-mtp model + mmproj together) before adding this."""
    target = tmp_path / "target.gguf"
    target.write_bytes(b"GGUF" * 1024)
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"GGUF" * 256)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"GGUF" * 512)

    proc = _fake_proc()
    backend = LlamaCppBackend()
    extra_config = {
        "llama_cpp": {
            "speculative": {
                "spec_type": "draft-mtp",
                "draft_model_id": str(draft),
            },
            "mmproj_model_id": str(mmproj),
        }
    }
    with (
        patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(LlamaCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        _patch_settings(mock_settings, str(tmp_path))
        await backend.load("target", [1], port=18047, extra_config=extra_config)

    args = list(create_proc.await_args.args)
    assert "--mmproj" in args and args[args.index("--mmproj") + 1] == str(mmproj)
    assert "--model-draft" in args and args[args.index("--model-draft") + 1] == str(draft)
    assert "--spec-type" in args and args[args.index("--spec-type") + 1] == "draft-mtp"


@pytest.mark.asyncio
async def test_get_capabilities_reports_vision_when_mmproj_configured() -> None:
    backend = LlamaCppBackend()
    backend._model_configs["gemma4-26b"] = {"mmproj_model_id": "/data/mmproj.gguf"}

    caps = await backend.get_capabilities("gemma4-26b")

    assert caps.vision is True


def _write_minimal_gguf(path: Path, context_length: int) -> None:
    import struct

    kv = {
        "general.architecture": "testarch",
        "testarch.block_count": 1,
        "testarch.attention.head_count": 1,
        "testarch.embedding_length": 1,
        "testarch.context_length": context_length,
    }
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", 0)
    out += struct.pack("<Q", len(kv))
    for key, value in kv.items():
        key_bytes = key.encode("utf-8")
        out += struct.pack("<Q", len(key_bytes)) + key_bytes
        if isinstance(value, str):
            out += struct.pack("<I", 8)
            val_bytes = value.encode("utf-8")
            out += struct.pack("<Q", len(val_bytes)) + val_bytes
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", int(value))
    path.write_bytes(bytes(out))


@pytest.mark.asyncio
async def test_get_capabilities_reports_native_context_from_gguf(tmp_path: Path) -> None:
    """native_context_length is the GGUF's architectural ceiling, independent
    of whatever --ctx-size preset happens to be configured right now — it
    must NOT collapse to the (much smaller) currently-configured ctx_size."""
    model_file = tmp_path / "model.gguf"
    _write_minimal_gguf(model_file, context_length=262144)
    backend = LlamaCppBackend()
    backend._model_configs["gemma4-26b"] = {
        "ctx_size": 32768,
        "model_file": str(model_file),
    }

    caps = await backend.get_capabilities("gemma4-26b")

    assert caps.context_length == 32768
    assert caps.native_context_length == 262144


@pytest.mark.asyncio
async def test_get_capabilities_native_context_falls_back_to_zero_when_unresolvable() -> None:
    backend = LlamaCppBackend()
    backend._model_configs["never-loaded"] = {"ctx_size": 8192}

    caps = await backend.get_capabilities("never-loaded")

    assert caps.native_context_length == 0


@pytest.mark.asyncio
async def test_load_passes_chat_template_override(tmp_path: Path) -> None:
    args = await _capture_load_cmd(
        tmp_path,
        {"llama_cpp": {"chat_template": "llama3"}},
    )

    assert "--chat-template" in args
    assert args[args.index("--chat-template") + 1] == "llama3"


def test_get_binary_path_default_uses_resolve_server_bin(tmp_path: Path) -> None:
    backend = LlamaCppBackend()
    with patch.object(backend, "_resolve_server_bin", return_value="/opt/cuda/llama-server"):
        assert backend._get_binary_path(None) == "/opt/cuda/llama-server"
        assert backend._get_binary_path("cuda") == "/opt/cuda/llama-server"


def test_get_binary_path_alternate_runtime(tmp_path: Path) -> None:
    backend = LlamaCppBackend()
    bin_path = tmp_path / "backends" / "llama_cpp_cpu" / "bin" / "llama-server"
    bin_path.parent.mkdir(parents=True)
    bin_path.write_text("#!/bin/sh\n")

    with patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings:
        mock_settings.backends_dir = str(tmp_path / "backends")
        with patch(
            "ocabra.backends.llama_cpp_backend.read_backend_metadata",
            return_value=None,
        ):
            assert backend._get_binary_path("cpu") == str(bin_path)


def test_get_binary_path_missing_runtime_raises(tmp_path: Path) -> None:
    backend = LlamaCppBackend()
    with patch("ocabra.backends.llama_cpp_backend.settings") as mock_settings:
        mock_settings.backends_dir = str(tmp_path / "backends")
        with patch(
            "ocabra.backends.llama_cpp_backend.read_backend_metadata",
            return_value=None,
        ):
            with pytest.raises(FileNotFoundError, match="rocm"):
                backend._get_binary_path("rocm")
