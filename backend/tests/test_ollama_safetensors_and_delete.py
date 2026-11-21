"""inkling-2-mini y ravenx-chaos-agent (2026-11-21).

1. Un tag de Ollama en el formato nuevo (un blob por tensor, ``format:
   safetensors``, motor MLX) no trae GGUF: se registró en llamacpp_mks y no
   podía cargar nunca («sin model_path del GGUF»). Solo lo sirve Ollama.
2. Borrar un modelo GGUF descargado de HF dejaba la carpeta (16 GB): el
   nombre se deducía del id público y no del artefacto.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocabra.api.internal import models as models_api
from ocabra.core import artifacts
from ocabra.core.compatibility import backends_for_artifact


def _manifest(root: Path, name: str, tag: str, layers: list[dict]) -> None:
    path = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"layers": layers}))


@pytest.fixture
def ollama_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(artifacts, "ollama_root", lambda: str(tmp_path))
    _manifest(
        tmp_path,
        "inkling-2-mini",
        "26b-a5b",
        [{"mediaType": "application/vnd.ollama.image.tensor", "digest": "sha256:t", "name": "w"}],
    )
    _manifest(
        tmp_path,
        "gemma4",
        "26b",
        [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:g"}],
    )
    return tmp_path


def test_a_tag_without_gguf_is_only_served_by_ollama(ollama_store: Path) -> None:
    assert artifacts.ollama_tag_has_gguf("inkling-2-mini:26b-a5b") is False
    assert artifacts.ollama_tag_has_gguf("gemma4:26b") is True
    assert artifacts.ollama_tag_has_gguf("missing:1b") is None

    tag = {"kind": "ollama_tag", "source": "ollama", "ref": "inkling-2-mini:26b-a5b"}
    assert [e["backend"] for e in backends_for_artifact(tag)] == ["ollama"]
    gguf = {"kind": "ollama_tag", "source": "ollama", "ref": "gemma4:26b"}
    assert "llamacpp_mks" in [e["backend"] for e in backends_for_artifact(gguf)]
    # Not installed yet (searching before the pull): nothing to rule out.
    unknown = {"kind": "ollama_tag", "source": "ollama", "ref": "missing:1b"}
    assert "llamacpp_mks" in [e["backend"] for e in backends_for_artifact(unknown)]


def _hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(models_api.settings, "models_dir", str(tmp_path))
    hf = tmp_path / "huggingface"
    (hf / "org--Chaos-GGUF--Chaos-Q4_K_M").mkdir(parents=True)
    (hf / "org--Chaos-GGUF--Chaos-Q4_K_M" / "Chaos-Q4_K_M.gguf").write_bytes(b"x")
    return hf


def test_the_download_folder_comes_from_the_artifact(tmp_path, monkeypatch) -> None:
    hf = _hf(tmp_path, monkeypatch)
    folder = hf / "org--Chaos-GGUF--Chaos-Q4_K_M"
    assert models_api._unshared_download_dir(str(folder / "Chaos-Q4_K_M.gguf"), []) == folder
    assert models_api._unshared_download_dir(str(folder), []) == folder


def test_a_folder_another_model_uses_is_kept(tmp_path, monkeypatch) -> None:
    hf = _hf(tmp_path, monkeypatch)
    folder = hf / "org--Chaos-GGUF--Chaos-Q4_K_M"
    shared = [str(folder / "Chaos-Q4_K_M.gguf")]
    assert models_api._unshared_download_dir(str(folder), shared) is None


def test_never_outside_the_download_root(tmp_path, monkeypatch) -> None:
    hf = _hf(tmp_path, monkeypatch)
    assert models_api._unshared_download_dir(str(hf), []) is None
    assert models_api._unshared_download_dir(str(tmp_path), []) is None
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert models_api._unshared_download_dir(str(other), []) is None
