"""ASY-04: ``get_vram_estimate_mb`` must not walk the model tree on the event loop.

Each backend summed weight sizes with ``rglob`` + ``stat`` inside the
coroutine; over a large HF snapshot that blocks every other request. The walk
now runs in a worker thread (``ocabra.backends._fs.total_file_size``).
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

_real_rglob = Path.rglob


def _recording_rglob(calls: list[int]):
    def rglob(self, pattern, *args, **kwargs):
        calls.append(threading.get_ident())
        return _real_rglob(self, pattern, *args, **kwargs)

    return rglob


def _weights(root: Path, name: str = "model.safetensors", size: int = 4096) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(b"x" * size)
    return root


async def _estimate_off_loop(backend, model_id: str, **kwargs) -> int:
    calls: list[int] = []
    loop_thread = threading.get_ident()
    with patch.object(Path, "rglob", _recording_rglob(calls)):
        result = await backend.get_vram_estimate_mb(model_id, **kwargs)
    assert calls, "the estimate no longer walks the tree; update this test"
    assert loop_thread not in calls, "rglob ran on the event loop thread"
    return result


@pytest.mark.asyncio
async def test_vllm(tmp_path: Path) -> None:
    from ocabra.backends.vllm_backend import VLLMBackend

    model_dir = _weights(tmp_path / "m")
    backend = VLLMBackend()
    with patch.object(backend, "_resolve_local_model_dir", return_value=model_dir):
        await _estimate_off_loop(backend, "org/m", extra_config={"vllm": {}})


@pytest.mark.asyncio
async def test_sglang(tmp_path: Path) -> None:
    from ocabra.backends.sglang_backend import SGLangBackend

    model_dir = _weights(tmp_path / "m", name="pytorch_model.bin")
    backend = SGLangBackend()
    with patch.object(backend, "_resolve_local_model_dir", return_value=model_dir):
        await _estimate_off_loop(backend, "org/m")


@pytest.mark.asyncio
async def test_mageflow(tmp_path: Path) -> None:
    from ocabra.backends.mageflow_backend import MageFlowBackend

    _weights(tmp_path / "org" / "m", size=2 * 1024 * 1024)
    with patch("ocabra.backends.mageflow_backend.settings.models_dir", str(tmp_path)):
        assert await _estimate_off_loop(MageFlowBackend(), "org/m") == 3  # ceil(2 MiB * 1.1)


@pytest.mark.asyncio
async def test_whisper(tmp_path: Path) -> None:
    from ocabra.backends.whisper_backend import WhisperBackend

    _weights(tmp_path / "acme" / "custom-asr", size=1024 * 1024 * 1024)
    with patch("ocabra.backends.whisper_backend.settings.models_dir", str(tmp_path)):
        estimate = await _estimate_off_loop(WhisperBackend(), "acme/custom-asr")
    assert estimate == int(1024 * 1.4)


@pytest.mark.asyncio
async def test_tensorrt_llm(tmp_path: Path) -> None:
    from ocabra.backends.tensorrt_llm_backend import TensorRTLLMBackend

    engine_dir = _weights(tmp_path / "eng" / "engine", name="rank0.engine")
    backend = TensorRTLLMBackend()
    with (
        patch.object(backend, "is_enabled", return_value=True),
        patch.object(backend, "_resolve_engine_dir", return_value=engine_dir.parent),
    ):
        await _estimate_off_loop(backend, "org/m")
