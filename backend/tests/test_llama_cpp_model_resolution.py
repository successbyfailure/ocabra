"""BUG-43: llama.cpp must load exactly the GGUF it was given, never a guess."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ocabra.backends.llama_cpp_backend import LlamaCppBackend
from tests.test_llama_cpp_backend import _capture_load_cmd


def _gguf(path: Path, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _resolve(models_dir: Path, model_id: str, extra_config: dict | None = None) -> Path:
    with patch("ocabra.backends.llama_cpp_backend.settings.models_dir", str(models_dir)):
        return LlamaCppBackend()._resolve_model_file(model_id, extra_config or {})


def test_missing_explicit_model_path_fails_instead_of_searching(tmp_path: Path) -> None:
    # The artifact pinned a blob that has since been deleted; a file with the
    # same name exists elsewhere (another model's download).
    _gguf(tmp_path / "other-model" / "demo.gguf")
    gone = tmp_path / "blobs" / "demo.gguf"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve(tmp_path, "demo", {"llama_cpp": {"model_path": str(gone)}})


def test_explicit_model_path_wins(tmp_path: Path) -> None:
    pinned = _gguf(tmp_path / "pinned" / "x.gguf")
    _gguf(tmp_path / "demo.gguf")
    assert _resolve(tmp_path, "demo", {"llama_cpp": {"model_path": str(pinned)}}) == pinned


def test_ambiguous_stem_search_fails_loudly(tmp_path: Path) -> None:
    _gguf(tmp_path / "a" / "model-Q4_K_M.gguf", mtime=1_000)
    _gguf(tmp_path / "b" / "model-Q4_K_M.gguf", mtime=2_000)
    with pytest.raises(FileNotFoundError, match="ambiguous"):
        _resolve(tmp_path, "model-Q4_K_M")


def test_unique_stem_search_still_resolves(tmp_path: Path) -> None:
    only = _gguf(tmp_path / "deep" / "dir" / "model-Q4_K_M.gguf")
    assert _resolve(tmp_path, "model-Q4_K_M") == only


def test_dotted_tag_does_not_match_a_shorter_stem(tmp_path: Path) -> None:
    # Path("qwen3.8:27b").stem == "qwen3": the old search served qwen3.gguf.
    _gguf(tmp_path / "somewhere" / "qwen3.gguf")
    with pytest.raises(FileNotFoundError, match="not found"):
        _resolve(tmp_path, "qwen3.8:27b")


@pytest.mark.asyncio
async def test_absolute_lora_path_is_passed_through(tmp_path: Path) -> None:
    """The LoRA adapters are not resolved through the model lookup: an
    absolute path under the HF tree (as in qwen3.8:27b-pi) reaches the worker
    untouched, next to a pinned base model."""
    base = _gguf(tmp_path / "blobs" / "sha256-base")
    lora = _gguf(tmp_path / "huggingface" / "acme--pi-lora" / "pi-lora-f16.gguf")
    argv = await _capture_load_cmd(
        tmp_path,
        {"llama_cpp": {"model_path": str(base), "lora": str(lora)}},
    )
    assert argv[argv.index("--model-path") + 1] == str(base)
    assert argv[argv.index("--lora") + 1] == str(lora)
