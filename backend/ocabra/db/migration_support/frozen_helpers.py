"""Frozen copies of the live helpers migration 0034 uses (INF-12).

:mod:`.model_unification` decides what migration 0034 does, and it used to
import these from ``ocabra.core.artifacts`` and ``ocabra.core.config_keys``:
an edit to live code there silently changed what 0034 does on a database that
has not run it yet. These are verbatim copies as of 2026-11-19, i.e. what 0034 did
on a fresh database until now. Do not edit them to follow live code; that is
the point.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

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
    "strata": "strata_config",
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


def config_section_for(backend_type: str) -> str:
    """extra_config section a backend reads its options from."""
    return LLAMA_CPP_SECTION if backend_type in LLAMA_CPP_FAMILY else backend_type


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
        if backend_type == "strata":
            section.pop("config_path", None)
        spec = section.get("speculative")
        if isinstance(spec, dict) and "draft_model_id" in spec:
            spec = {k: v for k, v in spec.items() if k != "draft_model_id"}
            if spec:
                section["speculative"] = spec
            else:
                section.pop("speculative")
        extra[section_key] = section
    return extra


_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")

# Dicts whose keys belong to someone else's schema (HF config, CLI flags,
# environment, chat-template kwargs...) or to internal bookkeeping: their
# keys are left exactly as they are.
FOREIGN_KEYS = frozenset(
    {
        "hf_overrides",
        "extra_args",
        "env",
        "chat_template_kwargs",
        "limit_mm_per_prompt",
        "speculative_config",
        "override_generation_config",
        "registry_metadata",
        "runtime_capabilities",
        "_runtime",
    }
)


def camel_to_snake(key: str) -> str:
    return _CAMEL_RE.sub("_", key).lower()


def normalize_case(
    value: Any, depth: int = 3, prefer: Literal["snake", "camel"] = "snake"
) -> tuple[Any, list[str]]:
    """camelCase → snake_case on config keys. Returns (normalised, dropped).

    On a clash between both spellings, ``prefer`` decides: ``"snake"`` for
    stored data (it is the value the readers apply today), ``"camel"`` for an
    incoming edit from the UI (it is what the user just typed). Recurses into
    the backend sections and their own sub-dicts (``speculative``) up to
    ``depth``, never into :data:`FOREIGN_KEYS`.
    """
    if not isinstance(value, dict) or depth <= 0:
        return value, []
    out: dict[str, Any] = {}
    dropped: list[str] = []
    for key, val in value.items():
        snake = camel_to_snake(key)
        is_camel = snake != key
        twins = [k for k in value if k != key and camel_to_snake(k) == snake]
        if twins and (prefer == "camel") != is_camel:
            # this spelling loses against its twin
            twin = twins[0]
            if value[twin] != val:
                dropped.append(f"{key}={val!r} (gana {twin}={value[twin]!r})")
            continue
        if snake in FOREIGN_KEYS:
            out[snake] = val
            continue
        sub, sub_dropped = normalize_case(val, depth - 1, prefer)
        out[snake] = sub
        dropped.extend(f"{key}.{d}" for d in sub_dropped)
    return out, dropped
