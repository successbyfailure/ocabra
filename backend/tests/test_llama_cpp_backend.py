from __future__ import annotations

from pathlib import Path
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
        mock_settings.models_dir = str(tmp_path)
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

        info = await backend.load("demo", [1], port=18031, extra_config={})

    assert info.backend_type == "llama_cpp"
    assert info.port == 18031
    assert info.pid == 5151
    args = create_proc.await_args.args
    assert "llama_cpp_worker.py" in str(args[1])
    assert "--gpu-layers" in args
    assert "--flash-attn" in args
    assert info.vram_used_mb > 0


@pytest.mark.asyncio
async def test_capabilities_use_embedding_flag_and_context() -> None:
    backend = LlamaCppBackend()
    backend._model_configs["demo-embed"] = {"embedding": True, "ctx_size": 16384}

    caps = await backend.get_capabilities("demo-embed")

    assert caps.embeddings is True
    assert caps.chat is False
    assert caps.context_length == 16384
