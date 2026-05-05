from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends.llama_cpp_backend import LlamaCppBackend


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
