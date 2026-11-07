"""ocabra.core.artifacts — resolution of deployment artifacts to files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocabra.core import artifacts


def _ollama_store(root: Path, tag_path: str, layers: dict[str, str]) -> None:
    media = {v: k for k, v in artifacts.OLLAMA_LAYER_MEDIA.items()}
    manifest = root / "manifests" / tag_path
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {"mediaType": media[kind], "digest": digest.replace("-", ":", 1)}
                    for kind, digest in layers.items()
                ]
            }
        )
    )


def test_ollama_gguf_resolves_through_the_current_manifest(tmp_path: Path) -> None:
    """The blob is looked up on every resolution: after an ``ollama pull`` that
    updates the tag, the deployment follows the new blob instead of the old path."""
    _ollama_store(
        tmp_path,
        "registry.ollama.ai/library/gemma4/26b",
        {"model": "sha256-aaa", "projector": "sha256-ppp"},
    )
    art = {
        "kind": "gguf",
        "source": "ollama",
        "ref": "gemma4:26b",
        "mmproj": {"source": "ollama", "layer": "projector"},
    }

    extra = artifacts.materialize("llamacpp_mks", art, {"llama_cpp": {"ctx_size": 8192}}, tmp_path)
    assert extra["llama_cpp"] == {
        "ctx_size": 8192,
        "model_path": str(tmp_path / "blobs" / "sha256-aaa"),
        "mmproj_model_id": str(tmp_path / "blobs" / "sha256-ppp"),
    }

    (tmp_path / "manifests").rename(tmp_path / "old")
    _ollama_store(
        tmp_path,
        "registry.ollama.ai/library/gemma4/26b",
        {"model": "sha256-bbb", "projector": "sha256-ppp"},
    )
    extra = artifacts.materialize("llamacpp_mks", art, extra, tmp_path)
    assert extra["llama_cpp"]["model_path"] == str(tmp_path / "blobs" / "sha256-bbb")


def test_missing_tag_does_not_raise_from_materialize(tmp_path: Path) -> None:
    art = {"kind": "gguf", "source": "ollama", "ref": "gone:latest"}
    assert artifacts.materialize("llamacpp_mks", art, {"a": 1}, tmp_path) == {"a": 1}
    with pytest.raises(artifacts.ArtifactResolutionError):
        artifacts.resolve_paths(art, tmp_path)


def test_strip_derived_removes_paths_but_keeps_options() -> None:
    extra = {
        "model_path": "/x",
        "llama_cpp": {
            "model_path": "/y",
            "mmproj_model_id": "/p",
            "ctx_size": 1,
            "speculative": {"draft_model_id": "/d", "draft_n": 4},
        },
    }
    assert artifacts.strip_derived("llamacpp_mks", extra) == {
        "llama_cpp": {"ctx_size": 1, "speculative": {"draft_n": 4}}
    }


def test_ui_round_trip_of_paths_is_ignored_but_a_real_edit_updates_the_artifact() -> None:
    art = {
        "kind": "gguf",
        "source": "ollama",
        "ref": "gemma4:26b",
        "draft": {"source": "ollama", "layer": "draft"},
    }
    current = {
        "llama_cpp": {"model_path": "/blobs/a", "speculative": {"draft_model_id": "/blobs/d"}}
    }

    assert artifacts.apply_path_edits("llamacpp_mks", art, current, current) == art

    edit = {
        "llama_cpp": {
            "model_path": "/blobs/a",
            "speculative": {"draft_model_id": "/models/other.gguf"},
        }
    }
    updated = artifacts.apply_path_edits("llamacpp_mks", art, edit, current)
    assert updated["draft"] == {"source": "local", "path": "/models/other.gguf"}
    assert updated["source"] == "ollama" and updated["ref"] == "gemma4:26b"


def test_backend_ref_prefers_the_artifact() -> None:
    assert (
        artifacts.backend_ref({"ref": "BAAI/bge-reranker-v2-m3"}, "bge-reranker-v2-m3")
        == "BAAI/bge-reranker-v2-m3"
    )
    assert artifacts.backend_ref(None, "x") == "x"


@pytest.mark.parametrize(
    ("tag", "path"),
    [
        ("gemma4:26b", "registry.ollama.ai/library/gemma4/26b"),
        ("gemma4", "registry.ollama.ai/library/gemma4/latest"),
        ("user/m:t", "registry.ollama.ai/user/m/t"),
        ("hf.co/huihui-ai/X-GGUF:UD-Q5_K_XL", "hf.co/huihui-ai/X-GGUF/UD-Q5_K_XL"),
    ],
)
def test_manifest_path_roundtrip(tag: str, path: str) -> None:
    assert artifacts.ollama_manifest_relpath(tag) == path
    expected = tag if ":" in tag.rsplit("/", 1)[-1] else f"{tag}:latest"
    assert artifacts.ollama_tag_from_relpath(path) == expected
