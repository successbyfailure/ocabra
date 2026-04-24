"""Tests for the diffusers backend ``install_spec`` and python-bin resolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ocabra.backends.base import BackendInstallSpec
from ocabra.backends.diffusers_backend import DiffusersBackend
from ocabra.config import settings


def test_install_spec_shape() -> None:
    spec = DiffusersBackend().install_spec

    assert isinstance(spec, BackendInstallSpec)
    assert spec.oci_image == "ghcr.io/ocabra/backend-diffusers"
    assert spec.oci_tags.get("cuda12")
    assert spec.pip_packages
    assert spec.display_name
    assert "Image" in spec.tags
    assert spec.estimated_size_mb > 0


def test_install_spec_pip_packages_cover_diffusers_stack() -> None:
    packages = " ".join(DiffusersBackend().install_spec.pip_packages).lower()

    for required in ("torch", "diffusers", "accelerate", "transformers", "pillow", "safetensors"):
        assert required in packages, f"missing required dep '{required}' in install_spec"


def test_install_spec_requests_cuda_torch_index() -> None:
    urls = " ".join(DiffusersBackend().install_spec.pip_extra_index_urls)
    assert "download.pytorch.org/whl/cu" in urls


def test_install_spec_opts_in_to_core_runtime() -> None:
    assert DiffusersBackend().install_spec.include_core_runtime is True


def test_resolve_python_bin_defaults_to_sys_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    assert DiffusersBackend()._resolve_python_bin() == sys.executable


def test_resolve_python_bin_honours_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    diff_dir = tmp_path / "diffusers"
    diff_dir.mkdir(parents=True)
    (diff_dir / "metadata.json").write_text(
        json.dumps({"backend_type": "diffusers", "python_bin": "/fake/bin/python"}),
        encoding="utf-8",
    )

    assert DiffusersBackend()._resolve_python_bin() == "/fake/bin/python"


def test_resolve_python_bin_handles_corrupt_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    diff_dir = tmp_path / "diffusers"
    diff_dir.mkdir(parents=True)
    (diff_dir / "metadata.json").write_text("not valid json", encoding="utf-8")

    assert DiffusersBackend()._resolve_python_bin() == sys.executable
