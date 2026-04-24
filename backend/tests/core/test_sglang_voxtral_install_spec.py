"""Install_spec tests for sglang and voxtral backends (Bloque 15 Fase 2)."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from ocabra.backends.base import BackendInstallSpec
from ocabra.backends.sglang_backend import SGLangBackend
from ocabra.backends.voxtral_backend import VoxtralBackend
from ocabra.config import settings


def _make_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# sglang
# ---------------------------------------------------------------------------


def test_sglang_install_spec_shape() -> None:
    spec = SGLangBackend().install_spec
    assert isinstance(spec, BackendInstallSpec)
    assert spec.oci_image == "ghcr.io/ocabra/backend-sglang"
    assert any("sglang" in p.lower() for p in spec.pip_packages)
    assert "LLM" in spec.tags


def test_sglang_install_spec_uses_cuda_index() -> None:
    urls = " ".join(SGLangBackend().install_spec.pip_extra_index_urls)
    assert "download.pytorch.org/whl/cu" in urls


def test_sglang_resolve_python_bin_prefers_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    sg_dir = tmp_path / "sglang"
    sg_dir.mkdir(parents=True)
    venv_python = sg_dir / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    _make_executable(venv_python)
    (sg_dir / "metadata.json").write_text(
        json.dumps({"backend_type": "sglang", "python_bin": str(venv_python)}),
        encoding="utf-8",
    )
    legacy = tmp_path / "legacy" / "bin" / "python"
    legacy.parent.mkdir(parents=True)
    _make_executable(legacy)
    monkeypatch.setattr(settings, "sglang_python_bin", str(legacy))

    assert SGLangBackend()._resolve_python_bin() == str(venv_python)


def test_sglang_resolve_python_bin_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    legacy = tmp_path / "legacy" / "bin" / "python"
    legacy.parent.mkdir(parents=True)
    _make_executable(legacy)
    monkeypatch.setattr(settings, "sglang_python_bin", str(legacy))

    assert SGLangBackend()._resolve_python_bin() == str(legacy)


def test_sglang_resolve_python_bin_final_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "sglang_python_bin", "/nonexistent/python")

    assert SGLangBackend()._resolve_python_bin() == sys.executable


# ---------------------------------------------------------------------------
# voxtral
# ---------------------------------------------------------------------------


def test_voxtral_install_spec_shape() -> None:
    spec = VoxtralBackend().install_spec
    assert isinstance(spec, BackendInstallSpec)
    assert spec.oci_image == "ghcr.io/ocabra/backend-voxtral"
    pkgs = " ".join(spec.pip_packages).lower()
    assert "vllm" in pkgs and "vllm-omni" in pkgs
    assert "TTS" in spec.tags


def test_voxtral_resolve_python_bin_prefers_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    vx_dir = tmp_path / "voxtral"
    vx_dir.mkdir(parents=True)
    venv_python = vx_dir / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    _make_executable(venv_python)
    (vx_dir / "metadata.json").write_text(
        json.dumps({"backend_type": "voxtral", "python_bin": str(venv_python)}),
        encoding="utf-8",
    )
    legacy = tmp_path / "legacy" / "bin" / "python"
    legacy.parent.mkdir(parents=True)
    _make_executable(legacy)
    monkeypatch.setattr(settings, "voxtral_python_bin", str(legacy))

    assert VoxtralBackend()._resolve_python_bin() == str(venv_python)


def test_voxtral_resolve_python_bin_final_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "voxtral_python_bin", "/nonexistent/python")

    assert VoxtralBackend()._resolve_python_bin() == sys.executable
