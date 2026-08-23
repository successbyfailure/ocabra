from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends.bitnet_backend import BitnetBackend
from ocabra.core.model_manager import ModelState
from ocabra.core.model_manager_helpers import (
    estimate_bitnet_vram_from_config,
    resolve_bitnet_gpu_layers,
)


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
        mock_settings.bitnet_cache_type_k = None
        mock_settings.bitnet_cache_type_v = None
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


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Bonsai-27B-Q1_0.gguf", True),
        ("Ternary-Bonsai-27B-Q1_0.gguf", True),
        ("ggml-model-i2_s.gguf", False),
        ("falcon3-1b-instruct-1.58bit-gguf-i2_s.gguf", False),
    ],
)
def test_is_prismml_model(filename: str, expected: bool) -> None:
    assert BitnetBackend._is_prismml_model(Path(filename)) is expected


def test_select_server_bin_prefers_prismml_for_bonsai() -> None:
    backend = BitnetBackend()
    with (
        patch.object(
            BitnetBackend, "_resolve_prismml_bin", return_value="/bin/prismml/bonsai-server"
        ),
        patch.object(BitnetBackend, "_resolve_microsoft_bin", return_value="/bin/bitnet-server"),
    ):
        assert (
            backend._select_server_bin(Path("Bonsai-27B-Q1_0.gguf")) == "/bin/prismml/bonsai-server"
        )
        assert backend._select_server_bin(Path("ggml-model-i2_s.gguf")) == "/bin/bitnet-server"


def test_select_server_bin_raises_when_prismml_missing() -> None:
    backend = BitnetBackend()
    with patch.object(BitnetBackend, "_resolve_prismml_bin", return_value=None):
        with pytest.raises(FileNotFoundError, match="PrismML llama.cpp fork"):
            backend._select_server_bin(Path("Bonsai-27B-Q1_0.gguf"))


def test_build_options_defaults_gpu_layers_for_prismml() -> None:
    backend = BitnetBackend()
    with patch("ocabra.backends.bitnet_backend.settings") as mock_settings:
        mock_settings.bitnet_gpu_layers = 0
        mock_settings.bitnet_ctx_size = 4096
        mock_settings.bitnet_threads = None
        mock_settings.bitnet_batch_size = 512
        mock_settings.bitnet_ubatch_size = 128
        mock_settings.bitnet_parallel = 1
        mock_settings.bitnet_flash_attn = False
        mock_settings.bitnet_mlock = True
        mock_settings.bitnet_cache_type_k = None
        mock_settings.bitnet_cache_type_v = None

        micro = backend._build_options({}, is_prismml=False)
        bonsai = backend._build_options({}, is_prismml=True)
        # Explicit override still wins over the GPU-first Bonsai default.
        overridden = backend._build_options({"gpu_layers": 10}, is_prismml=True)

    assert micro["gpu_layers"] == 0
    assert bonsai["gpu_layers"] == 99
    assert overridden["gpu_layers"] == 10


def test_scheduler_defaults_prismml_to_gpu_and_uses_gguf_size(tmp_path: Path) -> None:
    gguf = tmp_path / "Bonsai-27B-Q1_0.gguf"
    with gguf.open("wb") as handle:
        handle.truncate(600 * 1024 * 1024)
    state = ModelState(
        model_id="bitnet/prism-ml/Bonsai-27B-gguf",
        backend_model_id="prism-ml/Bonsai-27B-gguf",
        display_name="Bonsai 27B",
        backend_type="bitnet",
        extra_config={"model_path": str(gguf)},
    )

    assert resolve_bitnet_gpu_layers(state, 0) == 99
    assert estimate_bitnet_vram_from_config(
        state,
        default_gpu_layers=0,
        models_dir=tmp_path,
    ) >= 648


def test_scheduler_respects_explicit_prismml_cpu_override() -> None:
    state = ModelState(
        model_id="bitnet/prism-ml/Bonsai-27B-gguf",
        display_name="Bonsai 27B",
        backend_type="bitnet",
        extra_config={"gpu_layers": 0},
    )

    assert resolve_bitnet_gpu_layers(state, 0) == 0


@pytest.mark.asyncio
async def test_quantised_kv_cache_forces_flash_attn(tmp_path: Path) -> None:
    gguf = tmp_path / "Bonsai-27B-Q1_0.gguf"
    gguf.write_bytes(b"GGUF")

    proc = _fake_proc(returncode=None)
    sub = AsyncMock(return_value=proc)
    backend = BitnetBackend()
    with (
        patch("ocabra.backends.bitnet_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=sub),
        patch.object(BitnetBackend, "_wait_for_startup", new=AsyncMock()),
        patch.object(
            BitnetBackend, "_resolve_prismml_bin", return_value="/bin/prismml/bonsai-server"
        ),
    ):
        mock_settings.models_dir = str(tmp_path)
        mock_settings.bitnet_gpu_layers = 0
        mock_settings.bitnet_ctx_size = 4096
        mock_settings.bitnet_threads = None
        mock_settings.bitnet_batch_size = 512
        mock_settings.bitnet_ubatch_size = 128
        mock_settings.bitnet_parallel = 1
        mock_settings.bitnet_flash_attn = False
        mock_settings.bitnet_mlock = False
        mock_settings.bitnet_cache_type_k = None
        mock_settings.bitnet_cache_type_v = None
        mock_settings.bitnet_startup_timeout_s = 30
        mock_settings.cuda_device_order = "PCI_BUS_ID"

        await backend.load(
            "Bonsai-27B-Q1_0", [0], port=18022, extra_config={"cache_type_k": "q8_0"}
        )

    cmd = list(sub.call_args.args)
    assert "--cache-type-k" in cmd
    assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"
    # A non-f16 KV cache requires flash-attention even though it was disabled.
    # The PrismML fork needs the explicit value form `--flash-attn on`.
    assert "--flash-attn" in cmd
    assert cmd[cmd.index("--flash-attn") + 1] == "on"
