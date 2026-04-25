"""Tests for native-build backend install specs (Bloque 15 Fase 2 — Ronda 1).

Covers ``llama_cpp``, ``bitnet`` and ``acestep`` — the three backends migrated
in Ronda 1 of bloque 15. Each test stays small: spec shape + path resolver
priority (modular metadata → legacy settings).
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from ocabra.backends.acestep_backend import AceStepBackend
from ocabra.backends.base import BackendInstallSpec
from ocabra.backends.bitnet_backend import BitnetBackend
from ocabra.backends.llama_cpp_backend import LlamaCppBackend
from ocabra.config import settings


def _make_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# llama_cpp
# ---------------------------------------------------------------------------


def test_llama_cpp_install_spec_shape() -> None:
    spec = LlamaCppBackend().install_spec

    assert isinstance(spec, BackendInstallSpec)
    assert spec.git_repo == "https://github.com/ggml-org/llama.cpp"
    assert spec.post_install_script == "backend/scripts/install_llama_cpp.sh"
    assert spec.extra_bins == {"server": "bin/llama-server"}
    assert spec.include_core_runtime is False
    # Native build needs apt deps.
    for required in ("build-essential", "cmake", "git"):
        assert required in spec.apt_packages


def test_llama_cpp_post_install_script_ships_in_repo() -> None:
    spec = LlamaCppBackend().install_spec
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / spec.post_install_script
    assert script.is_file(), f"missing {script}"
    assert script.stat().st_mode & stat.S_IEXEC


def test_llama_cpp_resolve_server_bin_prefers_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    backend_dir = tmp_path / "llama_cpp"
    backend_dir.mkdir()
    bin_path = backend_dir / "bin" / "llama-server"
    bin_path.parent.mkdir()
    _make_executable(bin_path)
    (backend_dir / "metadata.json").write_text(
        json.dumps({"extra_bins": {"server": str(bin_path)}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "llama_cpp_server_bin", "/legacy/llama-server")

    assert LlamaCppBackend()._resolve_server_bin() == str(bin_path)


def test_llama_cpp_resolve_server_bin_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "llama_cpp_server_bin", "/legacy/llama-server")

    assert LlamaCppBackend()._resolve_server_bin() == "/legacy/llama-server"


# ---------------------------------------------------------------------------
# bitnet
# ---------------------------------------------------------------------------


def test_bitnet_install_spec_shape() -> None:
    spec = BitnetBackend().install_spec

    assert spec.git_repo == "https://github.com/microsoft/BitNet.git"
    assert spec.git_recursive is True, "BitNet has submodules"
    assert spec.post_install_script == "backend/scripts/install_bitnet.sh"
    assert spec.extra_bins == {"server": "bin/bitnet-server"}
    assert spec.include_core_runtime is False
    for required in ("build-essential", "cmake", "git"):
        assert required in spec.apt_packages


def test_bitnet_post_install_script_ships_in_repo() -> None:
    spec = BitnetBackend().install_spec
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / spec.post_install_script
    assert script.is_file(), f"missing {script}"
    assert script.stat().st_mode & stat.S_IEXEC


def test_bitnet_resolve_server_bin_prefers_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    backend_dir = tmp_path / "bitnet"
    backend_dir.mkdir()
    bin_path = backend_dir / "bin" / "bitnet-server"
    bin_path.parent.mkdir()
    _make_executable(bin_path)
    (backend_dir / "metadata.json").write_text(
        json.dumps({"extra_bins": {"server": str(bin_path)}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "bitnet_server_bin", "/legacy/bitnet-server")

    assert BitnetBackend()._resolve_server_bin() == str(bin_path)


def test_bitnet_resolve_server_bin_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "bitnet_server_bin", "/legacy/bitnet-server")

    assert BitnetBackend()._resolve_server_bin() == "/legacy/bitnet-server"


# ---------------------------------------------------------------------------
# acestep
# ---------------------------------------------------------------------------


def test_acestep_install_spec_shape() -> None:
    spec = AceStepBackend().install_spec

    assert spec.git_repo == "https://github.com/ace-step/ACE-Step-1.5"
    assert spec.post_install_script == "backend/scripts/install_acestep.sh"
    # torch must be pre-installed via pip with cu124 index BEFORE uv sync runs.
    assert any("torch" in pkg for pkg in spec.pip_packages)
    assert any("download.pytorch.org" in url for url in spec.pip_extra_index_urls)
    # apt deps for clone + audio.
    for required in ("git", "ffmpeg", "libsndfile1", "curl"):
        assert required in spec.apt_packages


def test_acestep_post_install_script_ships_in_repo() -> None:
    spec = AceStepBackend().install_spec
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / spec.post_install_script
    assert script.is_file(), f"missing {script}"
    assert script.stat().st_mode & stat.S_IEXEC


def test_acestep_resolve_project_paths_prefers_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    backend_dir = tmp_path / "acestep"
    backend_dir.mkdir()
    src_dir = backend_dir / "src"
    src_dir.mkdir()
    venv_python = backend_dir / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    _make_executable(venv_python)
    (backend_dir / "metadata.json").write_text(
        json.dumps(
            {"src_dir": str(src_dir), "python_bin": str(venv_python)}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "acestep_project_dir", "/legacy/acestep")

    project_dir, python_bin = AceStepBackend()._resolve_project_paths()
    assert project_dir == src_dir
    assert python_bin == str(venv_python)


def test_acestep_resolve_project_paths_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without metadata, fall back to legacy settings.acestep_project_dir."""

    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    legacy_project = tmp_path / "legacy_project"
    legacy_project.mkdir()
    legacy_venv = legacy_project / ".venv" / "bin" / "python"
    legacy_venv.parent.mkdir(parents=True)
    _make_executable(legacy_venv)
    monkeypatch.setattr(settings, "acestep_project_dir", str(legacy_project))

    project_dir, python_bin = AceStepBackend()._resolve_project_paths()
    assert project_dir == legacy_project
    assert python_bin == str(legacy_venv)
