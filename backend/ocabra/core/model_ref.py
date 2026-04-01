from __future__ import annotations

KNOWN_BACKEND_TYPES = {
    "vllm",
    "acestep",
    "llama_cpp",
    "sglang",
    "tensorrt_llm",
    "bitnet",
    "diffusers",
    "whisper",
    "tts",
    "ollama",
}


def _split_canonical_model_ref(model_ref: str) -> tuple[str, str] | None:
    value = str(model_ref or "").strip()
    if not value or "/" not in value:
        return None

    backend, backend_model_id = value.split("/", 1)
    backend = backend.strip().lower()
    backend_model_id = backend_model_id.strip()
    if backend not in KNOWN_BACKEND_TYPES:
        return None
    if not backend_model_id:
        return None
    return backend, backend_model_id


def build_model_ref(backend_type: str, backend_model_id: str) -> str:
    backend = str(backend_type or "").strip().lower()
    model = str(backend_model_id or "").strip()
    if backend not in KNOWN_BACKEND_TYPES:
        raise ValueError(f"Unknown backend type '{backend_type}'")
    if not model:
        raise ValueError("backend model id must not be empty")
    return f"{backend}/{model}"


def parse_model_ref(model_ref: str) -> tuple[str, str]:
    value = str(model_ref or "").strip()
    if not value:
        raise ValueError("model id must not be empty")

    parsed = _split_canonical_model_ref(value)
    if parsed is None:
        raise ValueError(
            f"Invalid model id '{model_ref}'. Expected canonical format 'backend/model'."
        )
    return parsed


def normalize_model_ref(backend_type: str, model_ref: str) -> tuple[str, str]:
    """
    Normalize model identifiers into canonical form `backend/model`.

    Supports legacy persisted IDs:
    - plain names without prefix (e.g. `devstral-small-2:24b`)
    - repo-style names for non-canonical prefixes (e.g. `openai/whisper-medium`)

    Returns:
      (canonical_model_id, backend_model_id)
    """
    backend = str(backend_type or "").strip().lower()
    if backend not in KNOWN_BACKEND_TYPES:
        raise ValueError(f"Unknown backend type '{backend_type}'")

    raw = str(model_ref or "").strip()
    if not raw:
        raise ValueError("model id must not be empty")

    parsed = _split_canonical_model_ref(raw)
    if parsed is not None:
        parsed_backend, parsed_backend_model_id = parsed
        if parsed_backend == backend:
            backend_model_id = parsed_backend_model_id
        else:
            # Keep full identifier as backend-native payload to avoid data loss.
            backend_model_id = raw
    else:
        backend_model_id = raw

    canonical_model_id = build_model_ref(backend, backend_model_id)
    return canonical_model_id, backend_model_id
