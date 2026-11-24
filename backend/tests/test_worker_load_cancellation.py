"""BUG-02: a cancelled load must not leave its worker running.

diffusers, upscaler, flashvsr and mageflow registered the process only after
the health wait and cleaned up with ``except Exception``, which
``CancelledError`` skips. Cancelling a load (the UI's «Cancelar carga», or a
client that gives up) left the worker loading up to 20 GB onto the GPU with
nobody tracking it, and ``unload()`` found nothing to kill.
"""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import AsyncMock, patch

import pytest

BACKENDS = [
    ("ocabra.backends.diffusers_backend", "DiffusersBackend"),
    ("ocabra.backends.upscaler_backend", "UpscalerBackend"),
    ("ocabra.backends.flashvsr_backend", "FlashVSRBackend"),
    ("ocabra.backends.mageflow_backend", "MageFlowBackend"),
]


class _Process:
    pid = 4242

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.terminate()

    async def wait(self) -> int:
        return self.returncode or 0


def _backend(module_name: str, class_name: str, tmp_path, monkeypatch):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:  # tolerate a renamed class, find the one defined there
        cls = next(
            v
            for v in vars(module).values()
            if isinstance(v, type) and v.__module__ == module_name and hasattr(v, "load")
        )
    monkeypatch.setattr(module.settings, "models_dir", str(tmp_path))
    (tmp_path / "some-model").mkdir()
    script = tmp_path / "worker.py"
    script.write_text("")
    backend = cls()
    if hasattr(backend, "_worker_script_path"):
        monkeypatch.setattr(backend, "_worker_script_path", lambda: script)
    if hasattr(backend, "_resolve_python_bin"):
        monkeypatch.setattr(backend, "_resolve_python_bin", lambda: "python")
    return backend


async def _hang(**_kwargs) -> None:
    await asyncio.sleep(3600)


async def _stop(process, **_kwargs) -> None:
    """Stand-in for ``_process.stop_process`` that marks the fake terminated
    instead of signalling a real process group (REF-02)."""
    process.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(("module_name", "class_name"), BACKENDS)
async def test_cancelling_the_load_kills_the_worker(module_name, class_name, tmp_path, monkeypatch):
    backend = _backend(module_name, class_name, tmp_path, monkeypatch)
    process = _Process()

    with (
        patch(f"{module_name}.wait_until_healthy", new=_hang),
        patch(f"{module_name}.stop_process", new=_stop),
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
    ):
        task = asyncio.create_task(backend.load("some-model", [0], port=18555))
        for _ in range(50):
            await asyncio.sleep(0)
            if backend._processes:
                break
        assert backend._processes, "the worker is registered while it loads"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert process.terminated
    assert "some-model" not in backend._processes


@pytest.mark.asyncio
@pytest.mark.parametrize(("module_name", "class_name"), BACKENDS)
async def test_unload_during_the_load_finds_the_worker(module_name, class_name, tmp_path, monkeypatch):
    """The other half: the model manager's cancel path calls unload(), which
    used to find an empty registry and return."""
    backend = _backend(module_name, class_name, tmp_path, monkeypatch)
    process = _Process()

    with (
        patch(f"{module_name}.wait_until_healthy", new=_hang),
        patch(f"{module_name}.stop_process", new=_stop),
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
    ):
        task = asyncio.create_task(backend.load("some-model", [0], port=18556))
        for _ in range(50):
            await asyncio.sleep(0)
            if backend._processes:
                break
        await backend.unload("some-model")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert process.terminated
