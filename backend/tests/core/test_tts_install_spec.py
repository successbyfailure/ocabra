"""Tests for the tts backend ``install_spec`` and python-bin resolution.

Bloque 15 Fase 2 — segundo backend migrado al contrato
:class:`BackendInstallSpec`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ocabra.backends.base import BackendInstallSpec
from ocabra.backends.tts_backend import TTSBackend
from ocabra.config import settings


# ---------------------------------------------------------------------------
# install_spec
# ---------------------------------------------------------------------------


def test_install_spec_shape() -> None:
    spec = TTSBackend().install_spec

    assert isinstance(spec, BackendInstallSpec)
    assert spec.oci_image == "ghcr.io/ocabra/backend-tts"
    assert spec.oci_tags.get("cuda12"), "expected a cuda12 OCI tag"
    assert spec.pip_packages, "expected non-empty pip_packages"
    assert spec.display_name, "expected a non-empty display_name"
    assert "TTS" in spec.tags
    assert spec.estimated_size_mb > 0


def test_install_spec_pip_packages_cover_engines() -> None:
    """The spec must carry every TTS engine the worker switches between."""

    packages = " ".join(TTSBackend().install_spec.pip_packages).lower()

    for required in ("torch", "torchaudio", "transformers", "qwen-tts", "kokoro", "soundfile"):
        assert required in packages, f"missing required dep '{required}' in install_spec"


def test_install_spec_requests_cuda_torch_index() -> None:
    """Torch must come from the CUDA 12.4 wheel index."""

    urls = " ".join(TTSBackend().install_spec.pip_extra_index_urls)
    assert "download.pytorch.org/whl/cu" in urls


def test_install_spec_opts_in_to_core_runtime() -> None:
    """Worker is a FastAPI app so the venv needs the oCabra core runtime."""

    assert TTSBackend().install_spec.include_core_runtime is True


# ---------------------------------------------------------------------------
# _resolve_python_bin
# ---------------------------------------------------------------------------


def test_resolve_python_bin_defaults_to_sys_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    assert TTSBackend()._resolve_python_bin() == sys.executable


def test_resolve_python_bin_honours_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    tts_dir = tmp_path / "tts"
    tts_dir.mkdir(parents=True)
    (tts_dir / "metadata.json").write_text(
        json.dumps({"backend_type": "tts", "python_bin": "/fake/bin/python"}),
        encoding="utf-8",
    )

    assert TTSBackend()._resolve_python_bin() == "/fake/bin/python"


def test_resolve_python_bin_handles_corrupt_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    tts_dir = tmp_path / "tts"
    tts_dir.mkdir(parents=True)
    (tts_dir / "metadata.json").write_text("not valid json", encoding="utf-8")

    assert TTSBackend()._resolve_python_bin() == sys.executable
