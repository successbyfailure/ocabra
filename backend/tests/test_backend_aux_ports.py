"""BUG-44: ports a backend opens by itself come from the managed worker range.

vLLM's memory-profile probe and whisper.cpp's diarization sidecar used to bind
an ephemeral port (``bind(("127.0.0.1", 0))``) outside
``settings.worker_port_range_*``: invisible to the WorkerPool and to the
docker-agent's range checks. With an allocator attached they take it from the
pool and hand it back when the process stops.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ocabra.backends.vllm_backend import VLLMBackend
from ocabra.backends.whispercpp_backend import WhisperCppBackend
from ocabra.config import settings
from ocabra.core.worker_pool import WorkerPool


def _attach(backend) -> WorkerPool:
    pool = WorkerPool()
    backend.attach_port_allocator(pool.assign_port, pool.release_port)
    return pool


def _in_range(port: int) -> bool:
    return settings.worker_port_range_start <= port < settings.worker_port_range_end


@pytest.mark.asyncio
async def test_vllm_memory_probe_uses_and_returns_a_pool_port() -> None:
    backend = VLLMBackend()
    pool = _attach(backend)
    seen: list[int] = []

    async def fake_probe(model_id, *, port, gpu_index, extra_config):
        seen.append(port)
        assert port in pool._used_ports
        return {"status": "ok"}

    with patch.object(backend, "_run_memory_profile_probe", side_effect=fake_probe):
        await backend.estimate_memory_profile("org/model", gpu_index=0)

    assert len(seen) == 1 and _in_range(seen[0])
    assert seen[0] not in pool._used_ports


@pytest.mark.asyncio
async def test_vllm_memory_probe_returns_the_port_on_failure() -> None:
    backend = VLLMBackend()
    pool = _attach(backend)
    with (
        patch.object(
            backend, "_run_memory_profile_probe", new=AsyncMock(side_effect=RuntimeError("x"))
        ),
        pytest.raises(RuntimeError),
    ):
        await backend.estimate_memory_profile("org/model", gpu_index=0)
    assert pool._used_ports == set()


def _fake_process() -> MagicMock:
    proc = MagicMock()
    proc.returncode = None
    proc.pid = 4242
    proc.wait = AsyncMock(return_value=0)
    return proc


@pytest.mark.asyncio
async def test_whispercpp_diarization_sidecar_port_comes_from_pool(tmp_path: Path) -> None:
    backend = WhisperCppBackend()
    pool = _attach(backend)
    python_bin = tmp_path / "python3"
    python_bin.write_bytes(b"#!/bin/sh")
    healthy = httpx.Response(200, request=httpx.Request("GET", "http://x"))

    with (
        patch.object(backend, "_resolve_diarization_python_bin", return_value=str(python_bin)),
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=_fake_process())),
        patch("httpx.AsyncClient.get", new=AsyncMock(return_value=healthy)),
        patch("ocabra.backends.whispercpp_backend.open", create=True),
    ):
        sidecar = await backend._ensure_diarization_worker("m", {})

    assert _in_range(sidecar.port)
    assert sidecar.port in pool._used_ports

    with patch(
        "ocabra.backends.whispercpp_backend.terminate_process_group", new=AsyncMock()
    ):
        await backend._kill_diarization_worker("m")
    assert sidecar.port not in pool._used_ports


@pytest.mark.asyncio
async def test_whispercpp_dead_sidecar_port_is_released_on_relaunch(tmp_path: Path) -> None:
    backend = WhisperCppBackend()
    pool = _attach(backend)
    python_bin = tmp_path / "python3"
    python_bin.write_bytes(b"#!/bin/sh")
    healthy = httpx.Response(200, request=httpx.Request("GET", "http://x"))
    first_proc = _fake_process()

    with (
        patch.object(backend, "_resolve_diarization_python_bin", return_value=str(python_bin)),
        patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=[first_proc, _fake_process()]),
        ),
        patch("httpx.AsyncClient.get", new=AsyncMock(return_value=healthy)),
        patch("ocabra.backends.whispercpp_backend.open", create=True),
    ):
        first = await backend._ensure_diarization_worker("m", {})
        first_proc.returncode = 1  # the sidecar crashes
        second = await backend._ensure_diarization_worker("m", {})

    assert second is not first
    assert pool._used_ports == {second.port}


def test_registering_a_backend_gives_it_the_managed_port_range():
    """BUG-44: the allocator existed but nothing attached it."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ocabra.core.worker_pool import WorkerPool

    pool = WorkerPool()
    backend = SimpleNamespace(attach_port_allocator=MagicMock())
    pool.register_backend("vllm", backend)
    assign, release = backend.attach_port_allocator.call_args.args
    assert assign == pool.assign_port and release == pool.release_port
    pool.register_backend("plain", SimpleNamespace())  # no hook: nothing breaks
