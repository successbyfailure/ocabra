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


def test_install_spec_declares_no_extra_torch_index() -> None:
    """vLLM ships its own CUDA wheels from PyPI, so the spec deliberately
    declares no extra index. The ``download.pytorch.org/whl/cu124`` entry this
    test used to require was dropped on purpose in 9936b74 (2026-07-26) when
    vLLM was upgraded; the test kept asserting it and had been failing ever
    since. Keeping it pinned would now be actively wrong — the backend runs
    CUDA 13.0, so a cu124 index would resolve mismatched wheels."""
    spec = VLLMBackend().install_spec
    assert spec.pip_extra_index_urls == []
    # The nvcc components are what actually pins the CUDA toolchain.
    assert any(p.startswith("nvidia-cuda-nvcc==") for p in spec.pip_packages)


def test_install_spec_pins_vllm_to_the_installed_version() -> None:
    """The declared pin is the installer's source of truth, so letting it drift
    behind the venv means a reinstall silently downgrades the backend — which
    is exactly what happened: the venv reached 0.32.0 while the spec still said
    0.29.0 (reconciled 2026-11-06). This asserts the pin is exact, not that it
    is any particular number, so a deliberate bump does not break the test."""
    pins = [p for p in VLLMBackend().install_spec.pip_packages if p.startswith("vllm==")]
    assert len(pins) == 1, f"expected exactly one exact vllm pin, got {pins}"


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
