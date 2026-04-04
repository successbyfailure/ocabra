"""
Helpers to map model names between Ollama and oCabra internal IDs.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ocabra.api.openai._deps import resolve_model as resolve_openai_model
from ocabra.core.model_ref import build_model_ref, parse_model_ref

if TYPE_CHECKING:
    from ocabra.api._deps_auth import UserContext

_DEFAULT_OLLAMA_TO_INTERNAL: dict[str, str] = {
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
}

_SIZE_RE = re.compile(r"(?P<size>\d+(?:\.\d+)?)\s*[Bb]")


class OllamaNameMapper:
    """Bidirectional mapper between Ollama names and canonical model refs."""

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
        """Convert an Ollama model name to canonical oCabra model ref."""
        value = ollama_name.strip()
        if not value:
            raise ValueError("model name must not be empty")

        try:
            # Already canonical model ref.
            parse_model_ref(value)
            return value
        except ValueError:
            pass

        normalized = _normalize_ollama_name(value)
        mapped_internal = self._ollama_to_internal.get(normalized)
        if mapped_internal:
            return build_model_ref("vllm", mapped_internal)

        return build_model_ref("ollama", value)

    def to_ollama(self, model_id: str) -> str:
        """Convert a canonical oCabra model ref to an Ollama model name."""
        backend_type, backend_model_id = parse_model_ref(model_id)
        if backend_type == "ollama":
            return backend_model_id

        if backend_model_id in self._internal_to_ollama:
            return self._internal_to_ollama[backend_model_id]

        heuristic = _infer_ollama_name(backend_model_id)
        return heuristic or backend_model_id


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


async def resolve_model(
    model_manager,
    requested_name: str,
    user: UserContext | None = None,
) -> tuple[str, object | None]:
    """Resolve a requested Ollama model name to a canonical oCabra model ref.

    Native Ollama canonical refs win over compatibility aliases.

    If *user* is provided and the resolved model is not in the user's accessible
    model set, the model is treated as not found (returns ``(id, None)``) to
    avoid leaking existence to unauthorised callers.

    Args:
        model_manager: The application :class:`ModelManager`.
        requested_name: Ollama model name or canonical oCabra model ref.
        user: Optional resolved :class:`UserContext`; used to filter model access.

    Returns:
        Tuple of ``(resolved_model_id, ModelState | None)``.
    """
    exact_name = requested_name.strip()
    if exact_name:
        try:
            parsed_backend, _ = parse_model_ref(exact_name)
        except ValueError:
            parsed_backend = None

        if parsed_backend == "ollama":
            exact_state = await model_manager.get_state(exact_name)
            if exact_state is not None:
                if user is not None and not user.is_admin and exact_name not in user.accessible_model_ids:
                    return exact_name, None
                return exact_name, exact_state

        if parsed_backend != "ollama":
            ollama_canonical = build_model_ref("ollama", exact_name)
            native_state = await model_manager.get_state(ollama_canonical)
            if native_state is not None:
                if user is not None and not user.is_admin and ollama_canonical not in user.accessible_model_ids:
                    return ollama_canonical, None
                return ollama_canonical, native_state

    resolved_model_id, resolved_state = await resolve_openai_model(model_manager, exact_name, user=user)
    if resolved_state is not None:
        return resolved_model_id, resolved_state

    mapper = OllamaNameMapper()
    model_id = mapper.to_internal(requested_name)
    mapped_state = await model_manager.get_state(model_id)
    if mapped_state is not None and user is not None:
        if not user.is_admin and model_id not in user.accessible_model_ids:
            return model_id, None
    return model_id, mapped_state
