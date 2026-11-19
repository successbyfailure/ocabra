"""BUG-42 / BUG-46: worker startup must stop as soon as the worker dies.

whisper, tts and acestep used to poll ``/health`` until their startup timeout
(minutes) even when the worker process had already exited, and the error said
nothing about why. They now fail at once with the exit code and the log tail.
tts also leaked the log descriptor when unloading a worker that had died.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.backends.acestep_backend import AceStepBackend
from ocabra.backends.tts_backend import TTSBackend
from ocabra.backends.whisper_backend import WhisperBackend


class _DeadProcess:
    """A worker that crashed right after being spawned."""

    pid = 4242
    returncode = 1

    def terminate(self) -> None:  # pragma: no cover - never alive
        raise AssertionError("terminate() on a dead process")

    async def wait(self) -> int:
        return 1


def _log(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("Traceback ...\nImportError: libcudart.so.12: cannot open shared object\n")
    return path


async def _assert_fast_failure(coro) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        await asyncio.wait_for(coro, timeout=3)
    message = str(exc_info.value)
    assert "exit code 1" in message
    assert "libcudart.so.12" in message


@pytest.mark.asyncio
async def test_whisper_load_fails_fast_when_worker_exits(tmp_path: Path) -> None:
    backend = WhisperBackend()
    log_path = _log(tmp_path, "whisper.log")
    with (
        patch(
            "ocabra.backends.whisper_backend.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_DeadProcess()),
        ),
        patch("ocabra.backends.whisper_backend._worker_log_path", return_value=str(log_path)),
        patch("ocabra.backends.whisper_backend.settings.whisper_startup_timeout_s", 600),
        patch.object(backend, "health_check", new=AsyncMock(return_value=False)),
    ):
        await _assert_fast_failure(backend.load("openai/whisper-small", [0], port=18100))
    assert "openai/whisper-small" not in backend._workers


@pytest.mark.asyncio
async def test_tts_load_fails_fast_when_worker_exits(tmp_path: Path) -> None:
    backend = TTSBackend()
    log_path = _log(tmp_path, "tts.log")
    worker_py = tmp_path / "tts_worker.py"
    worker_py.write_text("")
    with (
        patch(
            "ocabra.backends.tts_backend.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_DeadProcess()),
        ),
        patch("ocabra.backends.tts_backend._worker_log_path", return_value=str(log_path)),
        patch("ocabra.backends.tts_backend.WORKER_PATH", worker_py),
        patch.object(backend, "health_check", new=AsyncMock(return_value=False)),
        # 60 retries x 0.5 s = 30 s on the old code.
    ):
        await _assert_fast_failure(backend.load("hexgrad/Kokoro-82M", [0], port=18101))


@pytest.mark.asyncio
async def test_acestep_load_fails_fast_when_worker_exits(tmp_path: Path) -> None:
    backend = AceStepBackend()
    log_path = _log(tmp_path, "acestep.log")
    with (
        patch(
            "ocabra.backends.acestep_backend.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_DeadProcess()),
        ),
        patch("ocabra.backends.acestep_backend._worker_log_path", return_value=str(log_path)),
        patch("ocabra.backends.acestep_backend.settings.acestep_external_api_url", ""),
        patch("ocabra.backends.acestep_backend.settings.acestep_startup_timeout_s", 600),
        patch.object(
            backend, "_resolve_project_paths", return_value=(tmp_path, "/usr/bin/python3")
        ),
        patch.object(backend, "health_check", new=AsyncMock(return_value=False)),
    ):
        await _assert_fast_failure(backend.load("ace-step/ACE-Step-v1-3.5B", [0], port=18102))


@pytest.mark.asyncio
async def test_tts_unload_closes_log_of_dead_worker(tmp_path: Path) -> None:
    backend = TTSBackend()
    log_path = _log(tmp_path, "tts.log")
    worker_py = tmp_path / "tts_worker.py"
    worker_py.write_text("")
    process = _DeadProcess()
    process.returncode = None  # alive during load
    with (
        patch(
            "ocabra.backends.tts_backend.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ),
        patch("ocabra.backends.tts_backend._worker_log_path", return_value=str(log_path)),
        patch("ocabra.backends.tts_backend.WORKER_PATH", worker_py),
        patch.object(backend, "_wait_until_healthy", new=AsyncMock(return_value=True)),
    ):
        await backend.load("hexgrad/Kokoro-82M", [0], port=18103)
    log_file = backend._workers["hexgrad/Kokoro-82M"].log_file
    process.returncode = -9  # the worker dies (OOM killer) while serving
    await backend.unload("hexgrad/Kokoro-82M")
    assert log_file.closed
