"""Tests for the whisper backend ``install_spec`` and python-bin resolution.

Bloque 15 Fase 2 — piloto de migración del backend ``whisper`` al contrato
:class:`BackendInstallSpec`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ocabra.backends.base import BackendInstallSpec
from ocabra.backends.whisper_backend import WhisperBackend
from ocabra.config import settings


# ---------------------------------------------------------------------------
# install_spec
# ---------------------------------------------------------------------------


def test_install_spec_shape() -> None:
    backend = WhisperBackend()
    spec = backend.install_spec

    assert isinstance(spec, BackendInstallSpec)
    assert spec.oci_image == "ghcr.io/ocabra/backend-whisper"
    assert spec.oci_tags.get("cuda12"), "expected a cuda12 OCI tag"
    assert spec.pip_packages, "expected non-empty pip_packages"
    assert spec.display_name, "expected a non-empty display_name"
    assert "STT" in spec.tags
    assert spec.estimated_size_mb > 0


def test_install_spec_pip_packages_include_core_audio_deps() -> None:
    """The spec must carry the packages the worker actually imports."""

    backend = WhisperBackend()
    packages = " ".join(backend.install_spec.pip_packages).lower()

    for required in (
        "faster-whisper",
        "soundfile",
        "transformers",
        "pyannote.audio",
        "nemo_toolkit",
        "torch",
    ):
        assert required in packages, f"missing required dep '{required}' in install_spec"


def test_install_spec_requests_cuda_torch_index() -> None:
    """Torch must come from the CUDA 12.4 wheel index, not the CPU-only default."""

    spec = WhisperBackend().install_spec
    urls = " ".join(spec.pip_extra_index_urls)
    assert "download.pytorch.org/whl/cu" in urls, (
        "expected a pytorch CUDA wheel index so torch installs with GPU support"
    )


def test_install_spec_opts_in_to_core_runtime() -> None:
    """Worker runs a FastAPI app so the venv needs the oCabra core runtime."""

    assert WhisperBackend().install_spec.include_core_runtime is True


# ---------------------------------------------------------------------------
# _resolve_python_bin
# ---------------------------------------------------------------------------


def test_resolve_python_bin_defaults_to_sys_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no ``metadata.json`` exists, we fall back to ``sys.executable``."""

    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    backend = WhisperBackend()

    assert backend._resolve_python_bin() == sys.executable


def test_resolve_python_bin_honours_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``metadata.json`` specifies a ``python_bin`` we return it verbatim."""

    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir(parents=True)
    (whisper_dir / "metadata.json").write_text(
        json.dumps({"backend_type": "whisper", "python_bin": "/fake/bin/python"}),
        encoding="utf-8",
    )

    backend = WhisperBackend()
    assert backend._resolve_python_bin() == "/fake/bin/python"


def test_resolve_python_bin_handles_corrupt_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed ``metadata.json`` must not crash the backend."""

    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir(parents=True)
    (whisper_dir / "metadata.json").write_text("not valid json", encoding="utf-8")

    backend = WhisperBackend()
    assert backend._resolve_python_bin() == sys.executable


def test_resolve_python_bin_missing_key_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``python_bin`` is absent/empty we fall back gracefully."""

    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir(parents=True)
    (whisper_dir / "metadata.json").write_text(
        json.dumps({"backend_type": "whisper"}), encoding="utf-8"
    )

    backend = WhisperBackend()
    assert backend._resolve_python_bin() == sys.executable
