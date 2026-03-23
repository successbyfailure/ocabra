from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends.sglang_backend import SGLangBackend


def _fake_proc(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.pid = 6262
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    return proc


@pytest.mark.asyncio
async def test_load_success_uses_local_model_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "meta-llama" / "Meta-Llama-3-8B-Instruct"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))

    proc = _fake_proc()
    backend = SGLangBackend()
    with (
        patch("ocabra.backends.sglang_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch.object(SGLangBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        mock_settings.models_dir = str(tmp_path)
        mock_settings.hf_cache_dir = str(tmp_path / "hf_cache")
        mock_settings.cuda_device_order = "PCI_BUS_ID"
        mock_settings.hf_token = ""
        mock_settings.sglang_server_module = "sglang.launch_server"
        mock_settings.sglang_python_bin = "/opt/sglang-venv/bin/python"
        mock_settings.sglang_tensor_parallel_size = None
        mock_settings.sglang_context_length = 32768
        mock_settings.sglang_mem_fraction_static = 0.8
        mock_settings.sglang_trust_remote_code = True
        mock_settings.sglang_disable_radix_cache = False
        mock_settings.sglang_startup_timeout_s = 120

        info = await backend.load("meta-llama/Meta-Llama-3-8B-Instruct", [0, 1], port=18041)

    assert info.backend_type == "sglang"
    assert info.port == 18041
    args = create_proc.await_args.args
    assert "sglang_worker.py" in str(args[1])
    assert "--tp" in args
    assert str(model_dir) in args


@pytest.mark.asyncio
async def test_capabilities_from_local_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "mistral" / "demo"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 8192,
            }
        )
    )
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))

    backend = SGLangBackend()
    with patch("ocabra.backends.sglang_backend.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path)
        mock_settings.hf_cache_dir = str(tmp_path / "hf_cache")
        caps = await backend.get_capabilities("mistral/demo")

    assert caps.chat is True
    assert caps.tools is True
    assert caps.context_length == 8192
