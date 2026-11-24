"""Characterization tests for ocabra.core.model_files (REF-05 step 1, DUP-01)."""

from __future__ import annotations

from pathlib import Path

from ocabra.core.model_files import find_local_hf_dir, sum_hf_weight_files_mb


def _make_model_dir(root: Path, rel: str) -> Path:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    return d


def test_find_direct_models_dir_layout(tmp_path: Path) -> None:
    d = _make_model_dir(tmp_path, "Org/Model")
    assert find_local_hf_dir("Org/Model", tmp_path) == d


def test_find_flat_huggingface_layout(tmp_path: Path) -> None:
    d = _make_model_dir(tmp_path, "huggingface/Org--Model")
    assert find_local_hf_dir("Org/Model", tmp_path) == d


def test_find_snapshot_layout(tmp_path: Path) -> None:
    d = _make_model_dir(tmp_path, "huggingface/models--Org--Model/snapshots/abc")
    assert find_local_hf_dir("Org/Model", tmp_path) == d


def test_strips_worker_key_suffix(tmp_path: Path) -> None:
    d = _make_model_dir(tmp_path, "huggingface/Org--Model")
    assert find_local_hf_dir("Org/Model::0123456789ab", tmp_path) == d


def test_hf_cache_layout_prefers_ref_main(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    root = cache / "hub" / "models--Org--Model"
    snap = root / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("deadbeef\n")
    assert find_local_hf_dir("Org/Model", tmp_path / "models", cache) == snap


def test_missing_returns_none(tmp_path: Path) -> None:
    assert find_local_hf_dir("Org/Nope", tmp_path) is None
    assert find_local_hf_dir("", tmp_path) is None


def test_sum_weight_files_mb(tmp_path: Path) -> None:
    (tmp_path / "a.safetensors").write_bytes(b"x" * (1024 * 1024))
    (tmp_path / "b.bin").write_bytes(b"y" * (2 * 1024 * 1024))
    (tmp_path / "ignore.txt").write_bytes(b"z" * 1024)
    assert sum_hf_weight_files_mb(tmp_path) == 3.0
