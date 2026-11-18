"""BUG-08: parar un worker tiene que parar también los procesos que lanzó.

voxtral (``vllm serve``), vibeasr (``asr_stream_server``) y whisper.cpp (el
sidecar de pyannote, y el ffmpeg de ``--convert``) se paraban con
``terminate()``/``kill()`` sobre el worker solo: los nietos quedaban huérfanos
y, en el caso de vllm, con la memoria de la GPU cogida para siempre.

Estos tests lanzan procesos de verdad (``sh`` + ``sleep``) que imitan a un
worker con un hijo y comprueban que el hijo no sobrevive.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends._process_group import terminate_process_group
from ocabra.backends.vibeasr_backend import VibeAsrBackend, _VibeAsrWorker
from ocabra.backends.voxtral_backend import VoxtralBackend, _VoxtralWorker
from ocabra.backends.whispercpp_backend import (
    WhisperCppBackend,
    _DiarizationSidecar,
    _WhisperCppWorker,
)

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")


def _alive(pid: int) -> bool:
    """True si ``pid`` existe y no es un zombi."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # El estado va justo después del ")" que cierra el nombre.
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


async def _wait_dead(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return not _alive(pid)


def _kill_quietly(pid: int | None) -> None:
    if pid:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def _spawn_worker_with_child(
    tmp_path: Path, **kwargs: Any
) -> tuple[asyncio.subprocess.Process, int]:
    """Un "worker" (sh) que deja un hijo (sleep) corriendo; devuelve ambos pids.

    Se lanza igual que lo lanza el backend en cuestión: los tests capturan los
    kwargs reales de ``create_subprocess_exec`` del ``load()`` y los pasan aquí.
    """
    pid_file = tmp_path / "child.pid"
    script = f"sleep 300 & echo $! > {pid_file}; wait"
    proc = await asyncio.create_subprocess_exec("sh", "-c", script, **kwargs)
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.02)
    child_pid = int(pid_file.read_text().strip())
    assert _alive(child_pid)
    return proc, child_pid


async def _spawn_kwargs_from_load(backend_load, module: str) -> dict[str, Any]:
    """kwargs con los que ``load()`` lanza el worker (sin lanzar nada)."""
    fake = MagicMock()
    fake.returncode = None
    fake.pid = 4242
    spawn = AsyncMock(return_value=fake)
    with patch(f"{module}.asyncio.create_subprocess_exec", new=spawn):
        try:
            await backend_load()
        except Exception:  # noqa: BLE001 — solo interesan los kwargs del spawn
            pass
    assert spawn.await_count >= 1, "load() no llegó a lanzar el worker"
    kwargs = dict(spawn.await_args.kwargs)
    for key in ("stdout", "stderr", "env"):
        kwargs.pop(key, None)
    return kwargs


# --- El helper ----------------------------------------------------------------


async def test_el_helper_mata_al_nieto_por_las_buenas(tmp_path: Path) -> None:
    proc, child = await _spawn_worker_with_child(tmp_path, start_new_session=True)
    try:
        await terminate_process_group(proc, grace_s=5)
        assert await _wait_dead(child)
        assert proc.returncode is not None
    finally:
        _kill_quietly(child)


async def test_el_helper_mata_al_nieto_con_sigkill(tmp_path: Path) -> None:
    proc, child = await _spawn_worker_with_child(tmp_path, start_new_session=True)
    try:
        await terminate_process_group(proc, grace_s=0, graceful=False)
        assert await _wait_dead(child)
    finally:
        _kill_quietly(child)


async def test_el_helper_barre_nietos_aunque_el_worker_ya_haya_muerto(tmp_path: Path) -> None:
    """El worker murió por su cuenta (OOM, SIGKILL): su hijo sigue en el grupo."""
    proc, child = await _spawn_worker_with_child(tmp_path, start_new_session=True)
    try:
        os.kill(proc.pid, signal.SIGKILL)
        await proc.wait()
        assert _alive(child)
        await terminate_process_group(proc, grace_s=5)
        assert await _wait_dead(child)
    finally:
        _kill_quietly(child)


# --- Cada backend --------------------------------------------------------------


async def test_voxtral_unload_no_deja_huerfano_a_vllm(tmp_path: Path) -> None:
    backend = VoxtralBackend()
    with (
        patch("ocabra.backends.voxtral_backend.settings") as mock_settings,
        patch("ocabra.backends.voxtral_backend.open", create=True),
        patch("ocabra.backends.voxtral_backend.venv_nvidia_ld_library_path", return_value=None),
        patch.object(VoxtralBackend, "_wait_until_healthy", new=AsyncMock(return_value=True)),
    ):
        mock_settings.backends_dir = str(tmp_path)
        mock_settings.voxtral_python_bin = None
        mock_settings.voxtral_startup_timeout_s = 1
        mock_settings.hf_token = None
        kwargs = await _spawn_kwargs_from_load(
            lambda: backend.load("m", [0], port=18001), "ocabra.backends.voxtral_backend"
        )
    backend._workers.clear()

    proc, child = await _spawn_worker_with_child(tmp_path, **kwargs)
    try:
        backend._workers["m"] = _VoxtralWorker(process=proc, info=MagicMock(), log_file=None)
        await backend.unload("m")
        assert await _wait_dead(child), "vllm serve sobrevivió al unload del worker"
    finally:
        _kill_quietly(child)


async def test_vibeasr_unload_no_deja_huerfano_al_stream_server(tmp_path: Path) -> None:
    backend = VibeAsrBackend()
    with (
        patch("ocabra.backends.vibeasr_backend.settings") as mock_settings,
        patch("ocabra.backends.vibeasr_backend.open", create=True),
        patch("ocabra.backends.vibeasr_backend.venv_nvidia_ld_library_path", return_value=None),
        patch.object(VibeAsrBackend, "_resolve_stream_bin", return_value="/bin/true"),
        patch.object(VibeAsrBackend, "_resolve_models", return_value=("vae", "lm")),
        patch.object(VibeAsrBackend, "_resolve_python_bin", return_value=sys.executable),
        patch.object(VibeAsrBackend, "_wait_until_healthy", new=AsyncMock(return_value=True)),
    ):
        mock_settings.vibeasr_threads = 2
        mock_settings.vibeasr_startup_timeout_s = 1
        mock_settings.backends_dir = str(tmp_path)
        kwargs = await _spawn_kwargs_from_load(
            lambda: backend.load("m", [], port=18002), "ocabra.backends.vibeasr_backend"
        )
    backend._workers.clear()

    proc, child = await _spawn_worker_with_child(tmp_path, **kwargs)
    try:
        backend._workers["m"] = _VibeAsrWorker(process=proc, info=MagicMock(), log_file=None)
        await backend.unload("m")
        assert await _wait_dead(child), "asr_stream_server sobrevivió al unload"
    finally:
        _kill_quietly(child)


async def _whispercpp_load_kwargs(tmp_path: Path) -> dict[str, Any]:
    model = tmp_path / "ggml-m.bin"
    model.write_bytes(b"x")
    backend = WhisperCppBackend()
    with (
        patch("ocabra.backends.whispercpp_backend.settings") as mock_settings,
        patch("ocabra.backends.whispercpp_backend.open", create=True),
        patch("ocabra.backends.whispercpp_backend.read_backend_metadata", return_value=None),
        patch.object(WhisperCppBackend, "_wait_for_startup", new=AsyncMock()),
    ):
        mock_settings.backends_dir = str(tmp_path)
        mock_settings.whispercpp_threads = 2
        mock_settings.whispercpp_server_bin = "/bin/true"
        return await _spawn_kwargs_from_load(
            lambda: backend.load(
                "m", [], port=18003, extra_config={"whisper_cpp": {"model_path": str(model)}}
            ),
            "ocabra.backends.whispercpp_backend",
        )


@pytest.mark.parametrize("graceful", [True, False])
async def test_whispercpp_parar_el_server_para_tambien_a_sus_hijos(
    tmp_path: Path, graceful: bool
) -> None:
    kwargs = await _whispercpp_load_kwargs(tmp_path)
    backend = WhisperCppBackend()
    proc, child = await _spawn_worker_with_child(tmp_path, **kwargs)
    log_file = MagicMock()
    try:
        backend._processes["m"] = _WhisperCppWorker(process=proc, port=1, log_file=log_file)
        await backend._kill_process("m", graceful=graceful)
        assert await _wait_dead(child)
        log_file.close.assert_called_once()
    finally:
        _kill_quietly(child)


async def test_whispercpp_el_sidecar_de_diarizacion_va_en_su_propio_grupo(
    tmp_path: Path,
) -> None:
    backend = WhisperCppBackend()
    python_bin = tmp_path / "python3"
    python_bin.write_bytes(b"#!/bin/sh")
    fake = MagicMock()
    fake.returncode = None
    fake.pid = 4242
    spawn = AsyncMock(return_value=fake)
    import httpx

    healthy = httpx.Response(200, request=httpx.Request("GET", "http://x"))
    with (
        patch.object(backend, "_resolve_diarization_python_bin", return_value=str(python_bin)),
        patch("ocabra.backends.whispercpp_backend.asyncio.create_subprocess_exec", new=spawn),
        patch("httpx.AsyncClient.get", new=AsyncMock(return_value=healthy)),
        patch("ocabra.backends.whispercpp_backend.open", create=True),
    ):
        await backend._ensure_diarization_worker("m", {})
    assert spawn.await_args.kwargs.get("start_new_session") is True

    backend._diarization_workers.clear()
    proc, child = await _spawn_worker_with_child(tmp_path, start_new_session=True)
    try:
        backend._diarization_workers["m"] = _DiarizationSidecar(
            process=proc, port=1, log_file=MagicMock()
        )
        await backend._kill_diarization_worker("m")
        assert await _wait_dead(child)
    finally:
        _kill_quietly(child)


# --- Lado worker: el nieto muere aunque el worker muera de un SIGKILL ---------


_WORKERS_DIR = Path(__file__).resolve().parents[1] / "workers"


async def test_parent_death_preexec_mata_al_hijo_si_el_worker_recibe_sigkill(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        f"""
        import os, signal, subprocess, sys, time
        sys.path.insert(0, {str(_WORKERS_DIR)!r})
        from process_guard import parent_death_preexec
        child = subprocess.Popen(["sleep", "300"], preexec_fn=parent_death_preexec())
        print(child.pid, flush=True)
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGKILL)
        """
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE
    )
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    child = int(line.strip())
    try:
        await proc.wait()
        assert proc.returncode == -signal.SIGKILL
        assert await _wait_dead(child), "el hijo sobrevivió al SIGKILL del worker"
    finally:
        _kill_quietly(child)


def test_los_workers_lanzan_a_sus_hijos_atados_a_su_vida() -> None:
    """voxtral (vllm serve) y vibeasr (asr_stream_server, ffmpeg) usan el preexec."""
    for name in ("voxtral_worker.py", "vibeasr_worker.py"):
        source = (_WORKERS_DIR / name).read_text()
        assert "preexec_fn=parent_death_preexec()" in source, name


async def test_voxtral_worker_lanza_vllm_con_parent_death(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("uvicorn")
    from workers import voxtral_worker

    spawn = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(voxtral_worker.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(voxtral_worker, "open", lambda *a, **k: MagicMock(), raising=False)
    await voxtral_worker._start_vllm_omni("m", 18999, [])
    preexec = spawn.await_args.kwargs.get("preexec_fn")
    assert callable(preexec)
    # Y no lo saca de nuestro grupo: el backend para el grupo entero.
    assert not spawn.await_args.kwargs.get("start_new_session")
