from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ocabra.backends.base import ModalityType
from ocabra.backends.vibeasr_backend import VibeAsrBackend, _extract_audio_file


def test_supported_modality_is_audio_transcription() -> None:
    assert VibeAsrBackend.supported_modalities() == {ModalityType.AUDIO_TRANSCRIPTION}


@pytest.mark.asyncio
async def test_capabilities_and_zero_vram() -> None:
    backend = VibeAsrBackend()
    caps = await backend.get_capabilities("any")
    assert caps.audio_transcription is True
    # CPU-first backend reserves no VRAM.
    assert await backend.get_vram_estimate_mb("any") == 0


def test_install_spec_native_build() -> None:
    spec = VibeAsrBackend().install_spec
    assert spec.git_repo == "https://github.com/microsoft/VibeASR.cpp.git"
    assert spec.extra_bins["stream_server"] == "bin/asr_stream_server"
    assert "ffmpeg" in spec.apt_packages


def test_resolve_models_finds_vae_and_lm(tmp_path: Path) -> None:
    model_dir = tmp_path / "microsoft--VibeVoice-ASR-BitNet"
    model_dir.mkdir()
    vae = model_dir / "vibeasr-vae-encoder-i8_s.gguf"
    lm = model_dir / "vibeasr-lm-i2_s-embed-q6_k.gguf"
    vae.write_bytes(b"GGUF")
    lm.write_bytes(b"GGUF")

    backend = VibeAsrBackend()
    with patch("ocabra.backends.vibeasr_backend.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path)
        resolved_vae, resolved_lm = backend._resolve_models("microsoft--VibeVoice-ASR-BitNet", {})

    assert resolved_vae == str(vae)
    assert resolved_lm == str(lm)


def test_resolve_models_explicit_config_wins(tmp_path: Path) -> None:
    backend = VibeAsrBackend()
    cfg = {"vae_model": "/models/vae.gguf", "lm_model": "/models/lm.gguf"}
    assert backend._resolve_models("whatever", cfg) == ("/models/vae.gguf", "/models/lm.gguf")


def test_resolve_models_raises_when_missing(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    backend = VibeAsrBackend()
    with patch("ocabra.backends.vibeasr_backend.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path)
        with pytest.raises(FileNotFoundError, match="VAE .* and LM"):
            backend._resolve_models("empty", {})


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
    backend = VibeAsrBackend()
    with patch("ocabra.backends.vibeasr_backend.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path)
        with pytest.raises(ValueError, match="requires 'port'"):
            await backend.load("m", [])
