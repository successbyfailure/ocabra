from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ocabra.backends.base import ModalityType
from ocabra.backends.whispercpp_backend import WhisperCppBackend, _extract_audio_file


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


def test_resolve_server_bin_custom_override(tmp_path: Path) -> None:
    custom = tmp_path / "whisper-server"
    custom.write_bytes(b"binary")
    backend = WhisperCppBackend()
    with patch("ocabra.backends.whispercpp_backend.settings") as mock_settings:
        mock_settings.whispercpp_custom_server_bin = str(custom)
        assert backend._resolve_server_bin() == str(custom)


def test_resolve_server_bin_custom_override_missing_raises(tmp_path: Path) -> None:
    backend = WhisperCppBackend()
    with patch("ocabra.backends.whispercpp_backend.settings") as mock_settings:
        mock_settings.whispercpp_custom_server_bin = str(tmp_path / "missing")
        with pytest.raises(FileNotFoundError):
            backend._resolve_server_bin()


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
