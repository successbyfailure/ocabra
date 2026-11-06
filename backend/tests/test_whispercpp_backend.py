from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ocabra.backends.base import ModalityType
from ocabra.backends.whispercpp_backend import (
    WhisperCppBackend,
    _extract_audio_file,
    _is_diarize_requested,
)


def test_supported_modality_is_audio_transcription() -> None:
    assert WhisperCppBackend.supported_modalities() == {ModalityType.AUDIO_TRANSCRIPTION}


@pytest.mark.asyncio
async def test_capabilities_and_zero_vram() -> None:
    backend = WhisperCppBackend()
    caps = await backend.get_capabilities("any")
    assert caps.audio_transcription is True
    # CPU-only backend reserves no VRAM.
    assert await backend.get_vram_estimate_mb("any") == 0


def test_install_spec_native_build() -> None:
    spec = WhisperCppBackend().install_spec
    assert spec.git_repo == "https://github.com/ggerganov/whisper.cpp"
    assert spec.extra_bins["server"] == "bin/whisper-server"
    assert spec.include_core_runtime is False


def test_resolve_model_file_explicit_config_wins(tmp_path: Path) -> None:
    model_file = tmp_path / "ggml-custom.bin"
    model_file.write_bytes(b"GGML")
    backend = WhisperCppBackend()
    resolved = backend._resolve_model_file(
        "whatever", {"whisper_cpp": {"model_path": str(model_file)}}
    )
    assert resolved == model_file


def test_resolve_model_file_finds_in_models_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "whisper_cpp" / "models"
    models_dir.mkdir(parents=True)
    model_file = models_dir / "ggml-large-v3-turbo-q8_0.bin"
    model_file.write_bytes(b"GGML")

    backend = WhisperCppBackend()
    with patch("ocabra.backends.whispercpp_backend.settings") as mock_settings:
        mock_settings.backends_dir = str(tmp_path)
        resolved = backend._resolve_model_file("large-v3-turbo-q8_0", {})

    assert resolved == model_file


def test_resolve_model_file_raises_when_missing(tmp_path: Path) -> None:
    models_dir = tmp_path / "whisper_cpp" / "models"
    models_dir.mkdir(parents=True)
    backend = WhisperCppBackend()
    with patch("ocabra.backends.whispercpp_backend.settings") as mock_settings:
        mock_settings.backends_dir = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            backend._resolve_model_file("missing-model", {})


@pytest.mark.parametrize(
    ("payload", "expected_name", "expected_ct"),
    [
        (b"abc", "audio.wav", "application/octet-stream"),
        (("clip.mp3", b"abc"), "clip.mp3", "application/octet-stream"),
        (("clip.wav", b"abc", "audio/wav"), "clip.wav", "audio/wav"),
        (
            {"filename": "x.flac", "content": b"abc", "content_type": "audio/flac"},
            "x.flac",
            "audio/flac",
        ),
    ],
)
def test_extract_audio_file(payload: object, expected_name: str, expected_ct: str) -> None:
    name, data, content_type = _extract_audio_file({"file": payload})
    assert name == expected_name
    assert data == b"abc"
    assert content_type == expected_ct


def test_extract_audio_file_missing_raises() -> None:
    with pytest.raises(ValueError, match="Missing 'file'"):
        _extract_audio_file({})


@pytest.mark.asyncio
async def test_load_requires_port(tmp_path: Path) -> None:
    backend = WhisperCppBackend()
    with patch("ocabra.backends.whispercpp_backend.settings") as mock_settings:
        mock_settings.backends_dir = str(tmp_path)
        with pytest.raises(ValueError, match="requires 'port'"):
            await backend.load("m", [])


@pytest.mark.asyncio
async def test_health_check_false_when_not_loaded() -> None:
    backend = WhisperCppBackend()
    assert await backend.health_check("missing") is False


# ---------------------------------------------------------------------------
# Diarization sidecar — 2026-11-05: whisper.cpp has no in-process pyannote
# equivalent (whisper-server is a native binary), so diarize=true forwards to
# workers/diarization_worker.py, a lazily-launched sidecar reusing the
# "whisper" backend's own venv (pyannote.audio already lives there).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("False", False),
        ("no", False),
        ("true", True),
        ("1", True),
        ("yes", True),
        (True, True),
    ],
)
def test_is_diarize_requested(raw: object, expected: bool) -> None:
    assert _is_diarize_requested(raw) is expected


def test_resolve_diarization_python_bin_uses_whisper_backend_venv(tmp_path: Path) -> None:
    backend = WhisperCppBackend()
    python_bin = tmp_path / "python3"
    python_bin.write_bytes(b"#!/bin/sh")
    with patch(
        "ocabra.backends.whispercpp_backend.read_backend_metadata",
        return_value={"python_bin": str(python_bin)},
    ):
        assert backend._resolve_diarization_python_bin() == str(python_bin)


def test_resolve_diarization_python_bin_raises_when_whisper_not_installed() -> None:
    backend = WhisperCppBackend()
    with patch(
        "ocabra.backends.whispercpp_backend.read_backend_metadata",
        return_value=None,
    ):
        with pytest.raises(FileNotFoundError, match="whisper"):
            backend._resolve_diarization_python_bin()


def _fake_process(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.pid = 4242
    proc.wait = AsyncMock(return_value=0)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    return proc


@pytest.mark.asyncio
async def test_ensure_diarization_worker_launches_and_caches(tmp_path: Path) -> None:
    backend = WhisperCppBackend()
    python_bin = tmp_path / "python3"
    python_bin.write_bytes(b"#!/bin/sh")
    proc = _fake_process()
    healthy = httpx.Response(200, request=httpx.Request("GET", "http://x"))

    with (
        patch.object(
            backend, "_resolve_diarization_python_bin", return_value=str(python_bin)
        ),
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as create_proc,
        patch("httpx.AsyncClient.get", new=AsyncMock(return_value=healthy)),
        patch("ocabra.backends.whispercpp_backend.open", create=True),
    ):
        sidecar = await backend._ensure_diarization_worker("m", {})

    assert sidecar.process is proc
    create_proc.assert_awaited_once()
    # A second call must reuse the cached, still-alive sidecar instead of
    # launching another one.
    with patch("asyncio.create_subprocess_exec", new=AsyncMock()) as create_proc_again:
        again = await backend._ensure_diarization_worker("m", {})
    create_proc_again.assert_not_awaited()
    assert again is sidecar


@pytest.mark.asyncio
async def test_ensure_diarization_worker_concurrent_calls_launch_once(tmp_path: Path) -> None:
    """2026-11-05: the equivalent lazy-load in whisper_worker.py has no lock,
    so two concurrent diarize=true requests arriving before the first pyannote
    load finishes build two pipelines concurrently. Guard against the same
    race here with the per-model_id asyncio.Lock in _ensure_diarization_worker."""
    backend = WhisperCppBackend()
    python_bin = tmp_path / "python3"
    python_bin.write_bytes(b"#!/bin/sh")
    proc = _fake_process()
    healthy = httpx.Response(200, request=httpx.Request("GET", "http://x"))

    async def _slow_create_subprocess_exec(*_args, **_kwargs):
        await asyncio.sleep(0.01)
        return proc

    with (
        patch.object(
            backend, "_resolve_diarization_python_bin", return_value=str(python_bin)
        ),
        patch(
            "asyncio.create_subprocess_exec", side_effect=_slow_create_subprocess_exec
        ) as create_proc,
        patch("httpx.AsyncClient.get", new=AsyncMock(return_value=healthy)),
        patch("ocabra.backends.whispercpp_backend.open", create=True),
    ):
        results = await asyncio.gather(
            backend._ensure_diarization_worker("m", {}),
            backend._ensure_diarization_worker("m", {}),
        )

    assert create_proc.call_count == 1
    assert results[0] is results[1]


@pytest.mark.asyncio
async def test_forward_request_diarize_merges_sidecar_response(tmp_path: Path) -> None:
    """diarize=true must: (1) force verbose_json against whisper-server so
    there are segments to hand off, (2) post audio + those segments to the
    diarization sidecar, (3) return the sidecar's merged response instead of
    whisper-server's own (non-diarized) one."""
    backend = WhisperCppBackend()
    proc = _fake_process()
    backend._processes["m"] = type(
        "Entry", (), {"process": proc, "port": 18123, "log_file": None}
    )()

    transcribe_response = httpx.Response(
        200,
        json={
            "text": "hola",
            "language": "es",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hola"}],
        },
        request=httpx.Request("POST", "http://x"),
    )
    diarized_response = httpx.Response(
        200,
        json={
            "text": "[SPEAKER_00] hola",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "hola", "speaker": "SPEAKER_00"}
            ],
            "speakers": ["SPEAKER_00"],
            "diarization": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        },
        request=httpx.Request("POST", "http://x"),
    )

    post_calls: list[str] = []

    async def _fake_post(self, url, **kwargs):
        post_calls.append(url)
        if "/transcribe" in url:
            return transcribe_response
        return diarized_response

    fake_sidecar = type("Sidecar", (), {"port": 19999})()

    with (
        patch("httpx.AsyncClient.post", new=_fake_post),
        patch.object(
            backend, "_ensure_diarization_worker", new=AsyncMock(return_value=fake_sidecar)
        ) as ensure_sidecar,
    ):
        result = await backend.forward_request(
            "m", "/transcribe", {"file": b"RIFF....", "diarize": "true"}
        )

    assert result == json.loads(diarized_response.text)
    ensure_sidecar.assert_awaited_once()
    assert any("/transcribe" in url for url in post_calls)
    assert any(":19999" in url for url in post_calls)


@pytest.mark.asyncio
async def test_forward_request_without_diarize_skips_sidecar() -> None:
    backend = WhisperCppBackend()
    proc = _fake_process()
    backend._processes["m"] = type(
        "Entry", (), {"process": proc, "port": 18123, "log_file": None}
    )()

    transcribe_response = httpx.Response(
        200,
        json={"text": "hola", "segments": []},
        request=httpx.Request("POST", "http://x"),
    )

    with (
        patch("httpx.AsyncClient.post", new=AsyncMock(return_value=transcribe_response)),
        patch.object(backend, "_ensure_diarization_worker", new=AsyncMock()) as ensure_sidecar,
    ):
        result = await backend.forward_request("m", "/transcribe", {"file": b"RIFF...."})

    assert result == {"text": "hola", "segments": []}
    ensure_sidecar.assert_not_awaited()
