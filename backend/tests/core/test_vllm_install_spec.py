"""Install_spec tests for vllm backend (Bloque 15 Fase 2)."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from ocabra.backends.base import BackendInstallSpec
from ocabra.backends.vllm_backend import VLLMBackend
from ocabra.config import settings


def _make_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def test_install_spec_shape() -> None:
    spec = VLLMBackend().install_spec
    assert isinstance(spec, BackendInstallSpec)
    assert spec.oci_image == "ghcr.io/ocabra/backend-vllm"
    assert any(p.lower().startswith("vllm") for p in spec.pip_packages)
    assert "LLM" in spec.tags
    assert spec.estimated_size_mb >= 5000


def test_install_spec_uses_cuda_index() -> None:
    urls = " ".join(VLLMBackend().install_spec.pip_extra_index_urls)
    assert "download.pytorch.org/whl/cu" in urls


def test_resolve_python_bin_defaults_to_sys_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    assert VLLMBackend()._resolve_python_bin() == sys.executable


def test_resolve_python_bin_honours_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    vllm_dir = tmp_path / "vllm"
    vllm_dir.mkdir(parents=True)
    venv_python = vllm_dir / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    _make_executable(venv_python)
    (vllm_dir / "metadata.json").write_text(
        json.dumps({"backend_type": "vllm", "python_bin": str(venv_python)}),
        encoding="utf-8",
    )

    assert VLLMBackend()._resolve_python_bin() == str(venv_python)
