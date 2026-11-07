"""snake_case normalisation of model ``extra_config`` keys.

Persisted ``extra_config`` is snake_case since migration 0034. Before it, the
UI wrote camelCase (``ctxSize``) while every option reader checks snake_case
first (``ctx_size``), so once both spellings existed an edit from the UI was
silently ignored (qwen3.8:27b-abliterated showed ctx 229376 and ran 40960).
Normalising at every write path keeps a single spelling per key.
"""

from __future__ import annotations

import re
from typing import Any, Literal

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
