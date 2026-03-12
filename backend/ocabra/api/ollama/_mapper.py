"""
Helpers to map model names between Ollama and oCabra internal IDs.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

_DEFAULT_OLLAMA_TO_INTERNAL: dict[str, str] = {
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
}

_SIZE_RE = re.compile(r"(?P<size>\d+(?:\.\d+)?)\s*[Bb]")


class OllamaNameMapper:
    """Bidirectional mapper between Ollama names and internal model IDs."""

    def __init__(self, extra_map: Mapping[str, str] | None = None) -> None:
        self._ollama_to_internal: dict[str, str] = {
            _normalize_ollama_name(key): val for key, val in _DEFAULT_OLLAMA_TO_INTERNAL.items()
        }
        if extra_map:
            self._ollama_to_internal.update({
                _normalize_ollama_name(key): val for key, val in extra_map.items()
            })
        self._internal_to_ollama: dict[str, str] = {
            internal: ollama for ollama, internal in self._ollama_to_internal.items()
        }

    def to_internal(self, ollama_name: str) -> str:
        """Convert an Ollama model name to an internal model ID."""
        normalized = _normalize_ollama_name(ollama_name)
        if normalized in self._ollama_to_internal:
            return self._ollama_to_internal[normalized]

        # If the caller already passed an internal model id, keep it untouched.
        if "/" in ollama_name:
            return ollama_name

        return ollama_name

    def to_ollama(self, model_id: str) -> str:
        """Convert an internal model ID to an Ollama model name."""
        if model_id in self._internal_to_ollama:
            return self._internal_to_ollama[model_id]

        heuristic = _infer_ollama_name(model_id)
        return heuristic or model_id


def _normalize_ollama_name(value: str) -> str:
    return value.strip().lower()


def _infer_ollama_name(model_id: str) -> str | None:
    # Example: meta-llama/Llama-3.2-3B-Instruct -> llama3.2:3b
    short = model_id.split("/")[-1]
    short_lower = short.lower()

    if short_lower.startswith("llama-"):
        version = short_lower.replace("llama-", "", 1)
        version = version.split("-", 1)[0]
        size_match = _SIZE_RE.search(short)
        if size_match:
            size = _normalize_size_tag(size_match.group("size"))
            return f"llama{version}:{size}b"

    size_match = _SIZE_RE.search(short)
    if size_match:
        size = _normalize_size_tag(size_match.group("size"))
        base = short_lower.replace("-instruct", "")
        base = base.replace("_", "-")
        return f"{base}:{size}b"

    return None


def _normalize_size_tag(value: str) -> str:
    try:
        numeric = float(value)
    except ValueError:
        return value.lower()
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).rstrip("0").rstrip(".")


async def resolve_model(model_manager, requested_name: str) -> tuple[str, object | None]:
    """
    Resolve a requested Ollama model name to an internal model id.

    Native Ollama model ids win over compatibility aliases so preinstalled
    Ollama models remain addressable even when a mapper rule exists.
    """
    exact_name = requested_name.strip()
    if exact_name:
        exact_state = await model_manager.get_state(exact_name)
        if exact_state is not None:
            return exact_name, exact_state

    mapper = OllamaNameMapper()
    model_id = mapper.to_internal(requested_name)
    if model_id == exact_name:
        return model_id, None

    mapped_state = await model_manager.get_state(model_id)
    return model_id, mapped_state
