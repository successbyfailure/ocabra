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


# ``_resolve_server_bin`` se partio en dos al integrar Bonsai/PrismML (908d96b):
# ``_resolve_microsoft_bin`` mantiene la prioridad metadata -> ajuste legacy, y
# ``_select_server_bin(model_file)`` elige entre el binario de Microsoft y el
# fork PrismML segun el GGUF.


def test_bitnet_resolve_microsoft_bin_prefers_metadata(
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

    assert BitnetBackend()._resolve_microsoft_bin() == str(bin_path)


def test_bitnet_resolve_microsoft_bin_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El ajuste heredado vale, pero solo si el binario esta en disco."""
    legacy = tmp_path / "legacy-bitnet-server"
    legacy.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "bitnet_server_bin", str(legacy))

    assert BitnetBackend()._resolve_microsoft_bin() == str(legacy)


def test_bitnet_resolve_microsoft_bin_no_inventa_una_ruta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Devolvia la ruta configurada existiera o no, y se ejecutaba nada.

    Este nodo retiro ese binario el 2026-11-09 (su llama.cpp era de enero y
    no lo usaba ningun modelo), asi que la ausencia tiene que notarse.
    """
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "bitnet_server_bin", "/legacy/no-existe")

    assert BitnetBackend()._resolve_microsoft_bin() is None


def test_bitnet_select_server_bin_uses_microsoft_for_i2s_gguf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "legacy-bitnet-server"
    legacy.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "bitnet_server_bin", str(legacy))

    modelo = tmp_path / "BitNet-b1.58-2B-4T-i2_s.gguf"
    assert BitnetBackend()._select_server_bin(modelo) == str(legacy)


def test_bitnet_select_server_bin_uses_prismml_for_bonsai_gguf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "bitnet_server_bin", "/legacy/bitnet-server")
    # A diferencia del binario de Microsoft, que se devuelve tal cual venga del
    # ajuste, el de PrismML solo se acepta si existe de verdad: si no, la
    # resolucion devuelve None para dar un error explicito.
    bonsai_bin = tmp_path / "bonsai-server"
    _make_executable(bonsai_bin)
    monkeypatch.setattr(settings, "bitnet_prismml_server_bin", str(bonsai_bin))

    modelo = tmp_path / "Bonsai-27B-Q1_0_g128.gguf"
    assert BitnetBackend()._select_server_bin(modelo) == str(bonsai_bin)


def test_bitnet_select_server_bin_explains_missing_prismml_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un GGUF de Bonsai sin el fork instalado tiene que decir que falta el
    fork, no caer al binario de Microsoft (que no sabe leer Q1_0)."""
    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    monkeypatch.setattr(settings, "bitnet_server_bin", "/legacy/bitnet-server")
    monkeypatch.setattr(settings, "bitnet_prismml_server_bin", "")

    modelo = tmp_path / "Bonsai-27B-Q1_0_g128.gguf"
    with pytest.raises(FileNotFoundError, match="PrismML"):
        BitnetBackend()._select_server_bin(modelo)


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
        json.dumps({"src_dir": str(src_dir), "python_bin": str(venv_python)}),
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
