from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends.bitnet_backend import BitnetBackend


def _fake_proc(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.pid = 4242
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    return proc


@pytest.mark.asyncio
async def test_load_requires_port(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    backend = BitnetBackend()
    with patch("ocabra.backends.bitnet_backend.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path)
        with pytest.raises(ValueError, match="requires 'port'"):
            await backend.load("model", [0])


@pytest.mark.asyncio
async def test_load_success(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    proc = _fake_proc(returncode=None)
    backend = BitnetBackend()
    with (
        patch("ocabra.backends.bitnet_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch.object(BitnetBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        mock_settings.models_dir = str(tmp_path)
        mock_settings.bitnet_server_bin = "/usr/local/bin/bitnet-server"
        mock_settings.bitnet_gpu_layers = 0
        mock_settings.bitnet_ctx_size = 4096
        mock_settings.bitnet_threads = None
        mock_settings.bitnet_batch_size = 512
        mock_settings.bitnet_ubatch_size = 128
        mock_settings.bitnet_parallel = 1
        mock_settings.bitnet_flash_attn = False
        mock_settings.bitnet_mlock = True
        mock_settings.bitnet_startup_timeout_s = 30
        mock_settings.cuda_device_order = "PCI_BUS_ID"

        info = await backend.load("model", [0], port=18021, extra_config={})

    assert info.backend_type == "bitnet"
    assert info.port == 18021
    assert info.pid == 4242
    assert info.vram_used_mb == 0


@pytest.mark.asyncio
async def test_get_vram_estimate_from_gpu_layers(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    backend = BitnetBackend()
    with patch("ocabra.backends.bitnet_backend.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path)
        mock_settings.bitnet_gpu_layers = 16
        estimate = await backend.get_vram_estimate_mb("model")
    assert estimate == 200


@pytest.mark.asyncio
async def test_capabilities_use_context_from_settings() -> None:
    backend = BitnetBackend()
    with patch("ocabra.backends.bitnet_backend.settings") as mock_settings:
        mock_settings.bitnet_ctx_size = 8192
        caps = await backend.get_capabilities("any-model")
    assert caps.chat is True
    assert caps.completion is True
    assert caps.streaming is True
    assert caps.context_length == 8192
