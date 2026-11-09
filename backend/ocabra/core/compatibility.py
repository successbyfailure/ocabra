"""Which backends can serve an artifact (model/backends unification).

One rule set, used by:

- adding a deployment to an existing model (``derive_artifact``): which other
  backends can serve the very same file, and with which artifact descriptor;
- search / download (``backends_for_files``): which backends a repo or an
  Ollama tag would be servable by, so the UI can show backend icons.

The rules mirror the verified matrix in
docs/tasks/model-backends-unification-plan.md: GGUF is the only format really
shared between backends (llama.cpp upstream, the llama.cpp-mks fork and Ollama
when the file comes from Ollama; BitNet only for ternary/1-bit quants);
safetensors LLM/encoder repos go to vLLM/SGLang; everything else (STT, TTS,
image, music, upscaling) is one format → one backend.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Preference order inside a family: first = suggested default. llamacpp_mks is
# the production llama.cpp (decision 2026-11-07); Ollama is a fallback.
GGUF_BACKENDS = ("llamacpp_mks", "llama_cpp", "ollama")
HF_LLM_BACKENDS = ("vllm", "sglang")

# Marcadores de *cuantizacion*, no de familia. "bonsai" estaba aqui y mandaba
# al backend ternario cualquier cosa con ese nombre: bonsai-2:27b es un qwen35
# con projector de vision y file_type 41 que la build moderna de llama.cpp
# carga sin problema, mientras el bitnet-server se abortaba con
# GGML_ASSERT(info->type < GGML_TYPE_COUNT) por no conocer el tipo
# (2026-11-09). El ternario de verdad se sigue reconociendo por su quant:
# Bonsai-27B-Q1_0.gguf lleva q1_0.
_TERNARY_RE = re.compile(r"(bitnet|i2_s|q1_0|1\.58bit|tq[12]_0)", re.IGNORECASE)


def is_ternary_gguf(name: str) -> bool:
    return bool(_TERNARY_RE.search(name or ""))


def _entry(backend: str, status: str, reason: str, installed: set[str] | None) -> dict[str, Any]:
    return {
        "backend": backend,
        "status": status,  # compatible | probable
        "installed": True if installed is None else backend in installed,
        "reason": reason,
    }


def backends_for_artifact(
    artifact: dict | None, installed: set[str] | None = None
) -> list[dict[str, Any]]:
    """Backends that can serve the file described by an artifact, preferred first."""
    if not isinstance(artifact, dict):
        return []
    kind = artifact.get("kind")
    source = artifact.get("source")
    name = str(artifact.get("file") or artifact.get("path") or artifact.get("ref") or "")
    if kind in {"gguf", "ollama_tag"}:
        if is_ternary_gguf(name):
            return [_entry("bitnet", "compatible", "GGUF ternario/1-bit", installed)]
        out = [
            _entry("llamacpp_mks", "compatible", "GGUF", installed),
            _entry("llama_cpp", "compatible", "GGUF", installed),
        ]
        if source == "ollama":
            out.append(_entry("ollama", "compatible", "tag de Ollama", installed))
        return out
    if kind == "hf_repo":
        return [
            _entry("vllm", "compatible", "safetensors", installed),
            _entry("sglang", "probable", "safetensors (sin probar en este nodo)", installed),
        ]
    return []


def derive_artifact(artifact: dict, target_backend: str) -> dict:
    """Artifact for a new deployment of the same file on ``target_backend``.

    Raises ValueError when the backend cannot serve that file.
    """
    allowed = {e["backend"] for e in backends_for_artifact(artifact)}
    if target_backend not in allowed:
        raise ValueError(
            f"Backend '{target_backend}' cannot serve this artifact "
            f"({artifact.get('kind')}); compatible: {', '.join(sorted(allowed)) or 'none'}."
        )
    kind = artifact.get("kind")
    if target_backend == "ollama":
        return {"kind": "ollama_tag", "source": "ollama", "ref": artifact["ref"]}
    if kind == "ollama_tag":
        # Same blob, read through the tag's manifest on every load. The
        # projector/draft layers are picked up when the tag ships them.
        return {
            "kind": "gguf",
            "source": "ollama",
            "ref": artifact["ref"],
            "mmproj": {"source": "ollama", "layer": "projector", "optional": True},
            "draft": {"source": "ollama", "layer": "draft", "optional": True},
        }
    return dict(artifact)


# ── search / download ────────────────────────────────────────────────────────


def backends_for_files(
    files: Iterable[str],
    *,
    pipeline_tag: str | None = None,
    library_name: str | None = None,
    tags: Iterable[str] = (),
    suggested: str | None = None,
    installed: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Backend icons for a HF repo (or one of its variants) from its file list.

    ``suggested`` is the single backend the registry heuristics picked
    (``_infer_backend``); it is kept for the non-shared formats (TTS, STT,
    diffusers…) where it is the only answer.
    """
    names = [str(f) for f in files]
    lowered = [n.lower() for n in names]
    tagset = {str(t).lower() for t in tags}
    ggufs = [n for n in names if n.lower().endswith(".gguf")]
    has_safetensors = any(n.endswith((".safetensors", ".bin")) for n in lowered)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(backend: str, status: str, reason: str) -> None:
        if backend not in seen:
            seen.add(backend)
            out.append(_entry(backend, status, reason, installed))

    if ggufs:
        model_ggufs = [g for g in ggufs if "mmproj" not in g.lower()]
        if model_ggufs and all(is_ternary_gguf(g) for g in model_ggufs):
            add("bitnet", "compatible", "GGUF ternario/1-bit")
        else:
            add("llamacpp_mks", "compatible", "GGUF")
            add("llama_cpp", "compatible", "GGUF")
    is_llm_like = (pipeline_tag or "") in {
        "text-generation",
        "image-text-to-text",
        "feature-extraction",
        "sentence-similarity",
        "text-classification",
        "text-ranking",
        "",
    } and (library_name or "transformers") in {"transformers", "sentence-transformers", ""}
    if (
        has_safetensors
        and is_llm_like
        and suggested in {None, "vllm", "sglang", "llama_cpp", "llamacpp_mks"}
    ):
        add("vllm", "compatible", "safetensors")
        add("sglang", "probable", "safetensors (sin probar en este nodo)")
    if suggested and suggested not in seen and suggested not in {"llama_cpp", "llamacpp_mks"}:
        add(suggested, "compatible", "formato propio del backend")
    if "gguf" in tagset and not ggufs:
        add("llamacpp_mks", "probable", "etiquetado gguf")
    return out


def backends_for_ollama_tag(
    installed: set[str] | None = None, name: str = ""
) -> list[dict[str, Any]]:
    """An Ollama tag is a GGUF: Ollama itself plus both llama.cpp builds."""
    if is_ternary_gguf(name):
        return [_entry("bitnet", "probable", "GGUF ternario/1-bit", installed)]
    return [
        _entry("llamacpp_mks", "compatible", "GGUF del tag", installed),
        _entry("llama_cpp", "compatible", "GGUF del tag", installed),
        _entry("ollama", "compatible", "tag de Ollama", installed),
    ]
