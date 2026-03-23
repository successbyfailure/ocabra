from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends.tensorrt_llm_backend import TensorRTLLMBackend


def _fake_proc(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.pid = 7373
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    return proc


@pytest.mark.asyncio
async def test_backend_disabled_by_feature_flag() -> None:
    with patch("ocabra.backends.tensorrt_llm_backend.settings") as mock_settings:
        mock_settings.tensorrt_llm_enabled = False
        mock_settings.tensorrt_llm_launch_mode = "binary"
        mock_settings.tensorrt_llm_serve_bin = "/usr/local/bin/trtllm-serve"
        backend = TensorRTLLMBackend()

    assert backend.is_enabled() is False
    with pytest.raises(RuntimeError, match="feature flag"):
        await backend.load("demo", [0], port=18051)


@pytest.mark.asyncio
async def test_load_success_when_enabled_binary_mode(tmp_path: Path) -> None:
    engine_root = tmp_path / "engines"
    engine_dir = engine_root / "meta-llama--demo"
    engine_dir.mkdir(parents=True)
    (engine_dir / "rank0.engine").write_bytes(b"engine")
    tokenizer_dir = tmp_path / "models" / "meta-llama" / "demo"
    tokenizer_dir.mkdir(parents=True)

    proc = _fake_proc()
    with patch("ocabra.backends.tensorrt_llm_backend.settings") as mock_settings:
        mock_settings.tensorrt_llm_enabled = True
        mock_settings.tensorrt_llm_launch_mode = "binary"
        mock_settings.tensorrt_llm_python_bin = "/usr/bin/python3"
        mock_settings.tensorrt_llm_serve_bin = str(tmp_path / "trtllm-serve")
        await asyncio.to_thread(
            Path(mock_settings.tensorrt_llm_serve_bin).write_text,
            "#!/bin/sh\n",
            encoding="utf-8",
        )
        mock_settings.tensorrt_llm_serve_module = "tensorrt_llm.commands.serve"
        mock_settings.tensorrt_llm_engines_dir = str(engine_root)
        mock_settings.tensorrt_llm_backend = "tensorrt"
        mock_settings.tensorrt_llm_tokenizer_path = ""
        mock_settings.tensorrt_llm_max_batch_size = 16
        mock_settings.tensorrt_llm_context_length = 4096
        mock_settings.tensorrt_llm_trust_remote_code = False
        mock_settings.tensorrt_llm_startup_timeout_s = 120
        mock_settings.models_dir = str(tmp_path / "models")
        mock_settings.cuda_device_order = "PCI_BUS_ID"

        backend = TensorRTLLMBackend()
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
            patch.object(TensorRTLLMBackend, "_wait_for_startup", new=AsyncMock()),
        ):
            info = await backend.load("meta-llama/demo", [1], port=18052)

    assert info.backend_type == "tensorrt_llm"
    assert info.port == 18052
    args = create_proc.await_args.args
    assert "tensorrt_llm_worker.py" in str(args[1])
    assert "--launch-mode" in args
    assert args[args.index("--launch-mode") + 1] == "binary"
    assert "--engine-dir" in args
    assert str(engine_dir) in args


@pytest.mark.asyncio
async def test_load_success_when_enabled_module_mode(tmp_path: Path) -> None:
    engine_root = tmp_path / "engines"
    engine_dir = engine_root / "demo"
    engine_dir.mkdir(parents=True)
    (engine_dir / "rank0.engine").write_bytes(b"engine")

    proc = _fake_proc()
    with patch("ocabra.backends.tensorrt_llm_backend.settings") as mock_settings:
        mock_settings.tensorrt_llm_enabled = True
        mock_settings.tensorrt_llm_launch_mode = "module"
        mock_settings.tensorrt_llm_python_bin = "/usr/bin/python3"
        mock_settings.tensorrt_llm_serve_bin = "/usr/local/bin/trtllm-serve"
        mock_settings.tensorrt_llm_serve_module = "tensorrt_llm.commands.serve"
        mock_settings.tensorrt_llm_engines_dir = str(engine_root)
        mock_settings.tensorrt_llm_backend = "tensorrt"
        mock_settings.tensorrt_llm_tokenizer_path = ""
        mock_settings.tensorrt_llm_max_batch_size = None
        mock_settings.tensorrt_llm_context_length = None
        mock_settings.tensorrt_llm_trust_remote_code = False
        mock_settings.tensorrt_llm_startup_timeout_s = 120
        mock_settings.models_dir = str(tmp_path / "models")
        mock_settings.cuda_device_order = "PCI_BUS_ID"

        with patch.object(TensorRTLLMBackend, "_module_available", return_value=True):
            backend = TensorRTLLMBackend()
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
            patch.object(TensorRTLLMBackend, "_wait_for_startup", new=AsyncMock()),
        ):
            info = await backend.load("demo", [0], port=18061)

    assert info.backend_type == "tensorrt_llm"
    args = create_proc.await_args.args
    assert "--launch-mode" in args
    assert args[args.index("--launch-mode") + 1] == "module"
    assert "--python-bin" in args
    assert args[args.index("--python-bin") + 1] == "/usr/bin/python3"
    assert "--serve-module" in args
    assert args[args.index("--serve-module") + 1] == "tensorrt_llm.commands.serve"


def test_detect_disabled_reason_module_missing_module() -> None:
    with patch("ocabra.backends.tensorrt_llm_backend.settings") as mock_settings:
        mock_settings.tensorrt_llm_enabled = True
        mock_settings.tensorrt_llm_launch_mode = "module"
        mock_settings.tensorrt_llm_python_bin = "/usr/bin/python3"
        mock_settings.tensorrt_llm_serve_module = "tensorrt_llm.commands.serve"
        mock_settings.tensorrt_llm_serve_bin = "/usr/local/bin/trtllm-serve"
        with patch("ocabra.backends.tensorrt_llm_backend.shutil.which", return_value="/usr/bin/python3"):
            with patch.object(TensorRTLLMBackend, "_module_available", return_value=False):
                backend = TensorRTLLMBackend()

    assert backend.is_enabled() is False
    assert backend.disabled_reason == "serve module not available: tensorrt_llm.commands.serve"
