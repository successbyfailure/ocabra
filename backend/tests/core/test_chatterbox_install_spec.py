"""Tests for the chatterbox backend ``install_spec`` and python-bin resolution."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from ocabra.backends.base import BackendInstallSpec
from ocabra.backends.chatterbox_backend import ChatterboxBackend
from ocabra.config import settings


def _make_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# install_spec
# ---------------------------------------------------------------------------


def test_install_spec_shape() -> None:
    spec = ChatterboxBackend().install_spec

    assert isinstance(spec, BackendInstallSpec)
    assert spec.oci_image == "ghcr.io/ocabra/backend-chatterbox"
    assert spec.oci_tags.get("cuda12")
    assert spec.pip_packages
    assert spec.display_name
    assert "TTS" in spec.tags


def test_install_spec_pip_packages_cover_chatterbox() -> None:
    packages = " ".join(ChatterboxBackend().install_spec.pip_packages).lower()

    for required in ("torch", "torchaudio", "chatterbox-tts", "soundfile"):
        assert required in packages, f"missing required dep '{required}' in install_spec"


def test_install_spec_requests_cuda_torch_index() -> None:
    urls = " ".join(ChatterboxBackend().install_spec.pip_extra_index_urls)
    assert "download.pytorch.org/whl/cu" in urls


def test_install_spec_opts_in_to_core_runtime() -> None:
    assert ChatterboxBackend().install_spec.include_core_runtime is True


# ---------------------------------------------------------------------------
# _resolve_python_bin
# ---------------------------------------------------------------------------


def test_resolve_python_bin_prefers_metadata_over_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When metadata.json declares a python_bin that exists, use it even if a
    legacy ``settings.chatterbox_python_bin`` would also be valid."""

    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    chatterbox_dir = tmp_path / "chatterbox"
    chatterbox_dir.mkdir(parents=True)
    venv_python = chatterbox_dir / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    _make_executable(venv_python)
    (chatterbox_dir / "metadata.json").write_text(
        json.dumps({"backend_type": "chatterbox", "python_bin": str(venv_python)}),
        encoding="utf-8",
    )

    legacy = tmp_path / "legacy" / "bin" / "python"
    legacy.parent.mkdir(parents=True)
    _make_executable(legacy)
    monkeypatch.setattr(settings, "chatterbox_python_bin", str(legacy))

    assert ChatterboxBackend()._resolve_python_bin() == str(venv_python)


def test_resolve_python_bin_falls_back_to_legacy_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without metadata.json, prefer settings.chatterbox_python_bin (fat image)."""

    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    legacy = tmp_path / "legacy" / "bin" / "python"
    legacy.parent.mkdir(parents=True)
    _make_executable(legacy)
    monkeypatch.setattr(settings, "chatterbox_python_bin", str(legacy))

    assert ChatterboxBackend()._resolve_python_bin() == str(legacy)


def test_resolve_python_bin_handles_corrupt_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    chatterbox_dir = tmp_path / "chatterbox"
    chatterbox_dir.mkdir(parents=True)
    (chatterbox_dir / "metadata.json").write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr(settings, "chatterbox_python_bin", "/nope/bin/python")

    assert ChatterboxBackend()._resolve_python_bin() == sys.executable


def test_resolve_python_bin_final_fallback_is_sys_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "chatterbox_python_bin", "/nonexistent/bin/python")

    assert ChatterboxBackend()._resolve_python_bin() == sys.executable
