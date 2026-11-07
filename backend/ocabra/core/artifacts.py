"""Model artifacts: what file(s) a deployment serves and where they are on disk.

A deployment (``model_configs`` row) carries an ``artifact`` descriptor instead
of hard-coding paths in ``extra_config``. The descriptor is the source of
truth; the concrete paths the backends read (``model_path``,
``mmproj_model_id``, ``speculative.draft_model_id``) are **derived** from it in
memory every time the state is built or a load starts, and never persisted.

Why: llama.cpp deployments of Ollama models used to pin
``/data/ollama_models/blobs/sha256-…`` by absolute path. An ``ollama pull``
that updates the tag, or a prune, changes the blob and left the deployment
pointing at a file that no longer exists. ``{"source": "ollama", "ref": tag}``
is resolved through the tag's manifest on each load instead.

Descriptor shape (all keys optional except ``kind``)::

    {"kind": "gguf" | "ollama_tag" | "hf_repo" | "ctranslate2" | "ggml" | ...,
     "source": "ollama" | "hf" | "local",
     "ref": "<backend reference: Ollama tag, HF repo, voice name, ...>",
     "path": "/abs/path",            # source hf/local single-file artifacts
     "repo": "org/name", "file": "x.gguf",
     "mmproj": {"source": "ollama", "layer": "projector"} | {"source": "local", "path": ...},
     "draft":  {"source": "ollama", "layer": "draft"} | {"source": "ollama", "ref": tag, "layer": "model"}
               | {"source": "local", "path": ...}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

LLAMA_CPP_FAMILY = frozenset({"llama_cpp", "llamacpp_mks"})
LLAMA_CPP_SECTION = "llama_cpp"

OLLAMA_REGISTRY_HOST = "registry.ollama.ai"
OLLAMA_LAYER_MEDIA = {
    "application/vnd.ollama.image.model": "model",
    "application/vnd.ollama.image.projector": "projector",
    "application/vnd.ollama.image.draft": "draft",
}
DEFAULT_OLLAMA_ROOT = "/data/ollama_models"

# Artifact kind each backend serves when it is not a single GGUF file.
KIND_BY_BACKEND = {
    "ollama": "ollama_tag",
    "whisper": "ctranslate2",
    "whisper_cpp": "ggml",
    "piper": "onnx_voice",
    "diffusers": "diffusers_tree",
    "mage": "diffusers_tree",
    "vllm": "hf_repo",
    "sglang": "hf_repo",
    "tensorrt_llm": "trt_engine",
    "tts": "hf_repo",
    "chatterbox": "hf_repo",
    "voxtral": "hf_repo",
    "upscaler": "esrgan",
    "flashvsr": "flashvsr",
    "acestep": "acestep",
    "vibeasr": "gguf_pair",
}

# Keys of extra_config that are derived from the artifact (never persisted).
ARTIFACT_KEYS = ("model_path", "mmproj_model_id", "mmproj_path")


def ollama_root() -> str:
    """Ollama model store as this process sees it."""
    from ocabra.config import settings

    return getattr(settings, "ollama_shared_models_dir", "") or DEFAULT_OLLAMA_ROOT


def config_section_for(backend_type: str) -> str:
    """extra_config section a backend reads its options from."""
    return LLAMA_CPP_SECTION if backend_type in LLAMA_CPP_FAMILY else backend_type


# ── Ollama manifests ─────────────────────────────────────────────────────────


def ollama_manifest_relpath(tag_ref: str) -> str:
    """``gemma4:26b`` → ``registry.ollama.ai/library/gemma4/26b``;
    ``hf.co/org/repo:Q5`` → ``hf.co/org/repo/Q5`` (explicit registry host)."""
    head, _, last = tag_ref.rpartition("/")
    name_last, _, tag = last.partition(":")
    name = f"{head}/{name_last}" if head else name_last
    tag = tag or "latest"
    if "/" not in name:
        return f"{OLLAMA_REGISTRY_HOST}/library/{name}/{tag}"
    if "." in name.split("/", 1)[0]:
        return f"{name}/{tag}"
    return f"{OLLAMA_REGISTRY_HOST}/{name}/{tag}"


def ollama_tag_from_relpath(relpath: str) -> str:
    parts = relpath.split("/")
    if parts and parts[0] == OLLAMA_REGISTRY_HOST:
        parts = parts[1:]
    *name_parts, tag = parts
    if name_parts and name_parts[0] == "library":
        name_parts = name_parts[1:]
    return f"{'/'.join(name_parts)}:{tag}"


def _layers_from_manifest(data: dict) -> dict[str, str]:
    layers: dict[str, str] = {}
    for layer in data.get("layers") or []:
        kind = OLLAMA_LAYER_MEDIA.get(layer.get("mediaType"))
        digest = str(layer.get("digest") or "").replace(":", "-")
        if kind and digest:
            layers[kind] = digest
    return layers


def read_ollama_manifest(
    tag_ref: str, root: str | Path = DEFAULT_OLLAMA_ROOT
) -> dict[str, str] | None:
    """Blob digests (``model``/``projector``/``draft``) of one installed tag, or None."""
    path = Path(root) / "manifests" / ollama_manifest_relpath(tag_ref)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    layers = _layers_from_manifest(data)
    return layers if "model" in layers else None


def scan_ollama_manifests(root: str | Path = DEFAULT_OLLAMA_ROOT) -> dict[str, dict[str, str]]:
    """Map every installed Ollama tag to its blob digests."""
    out: dict[str, dict[str, str]] = {}
    manifests = Path(root) / "manifests"
    if not manifests.is_dir():
        return out
    for path in manifests.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        layers = _layers_from_manifest(data)
        if "model" in layers:
            out[ollama_tag_from_relpath(str(path.relative_to(manifests)))] = layers
    return out


# ── resolution ───────────────────────────────────────────────────────────────


class ArtifactResolutionError(RuntimeError):
    """The artifact cannot be mapped to files on disk (tag removed, etc.)."""


def resolve_paths(
    artifact: dict | None, ollama_root: str | Path = DEFAULT_OLLAMA_ROOT
) -> dict[str, str]:
    """Concrete file paths of an artifact: ``model`` and optionally ``mmproj``/``draft``.

    Returns ``{}`` for artifacts without files the backend reads by path
    (Ollama tags, HF repos resolved by the backend itself, voices, ...).
    """
    if not isinstance(artifact, dict):
        return {}
    blobs = Path(ollama_root) / "blobs"
    out: dict[str, str] = {}
    layers: dict[str, str] | None = None
    if artifact.get("kind") == "gguf" and artifact.get("source") == "ollama":
        ref = str(artifact.get("ref") or "")
        layers = read_ollama_manifest(ref, ollama_root)
        if layers is None:
            raise ArtifactResolutionError(f"Ollama tag {ref!r} is not installed (no manifest)")
        out["model"] = str(blobs / layers["model"])
    elif artifact.get("path"):
        out["model"] = str(artifact["path"])

    for label in ("mmproj", "draft"):
        sub = artifact.get(label)
        if not isinstance(sub, dict):
            continue
        if sub.get("source") == "ollama":
            sub_layers = layers
            if sub.get("ref"):
                sub_layers = read_ollama_manifest(str(sub["ref"]), ollama_root)
            layer = sub.get("layer") or ("projector" if label == "mmproj" else "draft")
            digest = (sub_layers or {}).get(layer)
            if not digest:
                raise ArtifactResolutionError(
                    f"{label}: layer {layer!r} not found for {sub.get('ref') or artifact.get('ref')!r}"
                )
            out[label] = str(blobs / digest)
        elif sub.get("path"):
            out[label] = str(sub["path"])
    return out


def materialize(
    backend_type: str,
    artifact: dict | None,
    extra_config: dict | None,
    ollama_root: str | Path = DEFAULT_OLLAMA_ROOT,
) -> dict:
    """Return a copy of ``extra_config`` with the artifact's paths written where
    the backend reads them. Never raises: on a resolution failure the error is
    logged and the paths are left out, so the load fails with the backend's
    own "file not found" instead of a startup crash."""
    extra = dict(extra_config or {})
    if not isinstance(artifact, dict):
        return extra
    try:
        paths = resolve_paths(artifact, ollama_root)
    except ArtifactResolutionError as exc:
        logger.warning(
            "artifact_resolution_failed", backend=backend_type, artifact=artifact, error=str(exc)
        )
        return extra
    if not paths:
        return extra
    section_key = config_section_for(backend_type)
    section = dict(extra.get(section_key) or {}) if isinstance(extra.get(section_key), dict) else {}
    if "model" in paths:
        section["model_path"] = paths["model"]
    if "mmproj" in paths:
        section["mmproj_model_id"] = paths["mmproj"]
    if "draft" in paths:
        spec = (
            dict(section.get("speculative") or {})
            if isinstance(section.get("speculative"), dict)
            else {}
        )
        spec["draft_model_id"] = paths["draft"]
        section["speculative"] = spec
    extra[section_key] = section
    return extra


def strip_derived(backend_type: str, extra_config: dict | None) -> dict | None:
    """Remove artifact-derived keys before persisting extra_config.

    The UI round-trips the extra_config it received (paths included); without
    this a PATCH would re-pin the paths that the artifact is meant to own.
    """
    if not isinstance(extra_config, dict):
        return extra_config
    extra = dict(extra_config)
    for key in ARTIFACT_KEYS:
        extra.pop(key, None)
    section_key = config_section_for(backend_type)
    section = extra.get(section_key)
    if isinstance(section, dict):
        section = dict(section)
        for key in ARTIFACT_KEYS:
            section.pop(key, None)
        spec = section.get("speculative")
        if isinstance(spec, dict) and "draft_model_id" in spec:
            spec = {k: v for k, v in spec.items() if k != "draft_model_id"}
            if spec:
                section["speculative"] = spec
            else:
                section.pop("speculative")
        extra[section_key] = section
    return extra


def backend_ref(artifact: dict | None, fallback: str) -> str:
    """Identifier the backend receives as ``model_id`` (Ollama tag, HF repo, voice...)."""
    if isinstance(artifact, dict) and artifact.get("ref"):
        return str(artifact["ref"])
    return fallback


def artifact_from_legacy_extra(
    backend_type: str, backend_model_id: str, extra_config: dict | None
) -> dict:
    """Artifact descriptor for a deployment registered the old way (path in extra_config)."""
    extra = extra_config if isinstance(extra_config, dict) else {}
    section = extra.get(config_section_for(backend_type))
    section = section if isinstance(section, dict) else {}

    def opt(key: str) -> Any:
        return section.get(key, extra.get(key))

    if backend_type == "ollama":
        return {"kind": "ollama_tag", "source": "ollama", "ref": backend_model_id}
    gguf = backend_type in LLAMA_CPP_FAMILY or backend_type == "bitnet"
    art: dict[str, Any] = {
        "kind": "gguf" if gguf else KIND_BY_BACKEND.get(backend_type, "unknown"),
        "source": "local",
        "ref": backend_model_id,
    }
    if opt("model_path"):
        art["path"] = str(opt("model_path"))
    mmproj = opt("mmproj_model_id") or opt("mmproj_path")
    if mmproj:
        art["mmproj"] = {"source": "local", "path": str(mmproj)}
    spec = opt("speculative")
    if isinstance(spec, dict) and spec.get("draft_model_id"):
        art["draft"] = {"source": "local", "path": str(spec["draft_model_id"])}
    return art


def apply_path_edits(
    backend_type: str,
    artifact: dict | None,
    incoming_extra: dict | None,
    current_extra: dict | None,
) -> dict | None:
    """Fold user edits of derived paths back into the artifact.

    The UI still edits ``model_path`` / ``mmproj_model_id`` /
    ``speculative.draft_model_id`` as plain fields. A value equal to what the
    artifact currently resolves to is a round-trip and is ignored; a different
    one is a real edit and repoints that part of the artifact to a local file.
    Returns the (possibly) updated artifact.
    """
    if not isinstance(artifact, dict) or not isinstance(incoming_extra, dict):
        return artifact
    section_key = config_section_for(backend_type)

    def read(extra: dict | None, key: str) -> Any:
        extra = extra if isinstance(extra, dict) else {}
        sec = extra.get(section_key) if isinstance(extra.get(section_key), dict) else {}
        if key == "draft_model_id":
            spec = sec.get("speculative") if isinstance(sec.get("speculative"), dict) else {}
            return spec.get(key)
        return sec.get(key, extra.get(key))

    updated = dict(artifact)
    for key, label in (
        ("model_path", None),
        ("mmproj_model_id", "mmproj"),
        ("draft_model_id", "draft"),
    ):
        new_value = read(incoming_extra, key)
        if not new_value or new_value == read(current_extra, key):
            continue
        if label is None:
            updated.update(source="local", path=str(new_value))
            updated.pop("other_tags", None)
        else:
            updated[label] = {"source": "local", "path": str(new_value)}
    return updated
