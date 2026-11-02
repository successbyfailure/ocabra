"""Shared resolution of profile output-token limits."""

from __future__ import annotations

from typing import Any


def resolve_max_output_tokens(
    profile: Any,
    *,
    context_length: int = 0,
    backend_limit: int = 0,
) -> int | None:
    """Return the explicit enforceable output cap, bounded by context.

    A total context window is not an output limit. Profile policy wins over a
    backend-declared cap; zero or missing values mean that no independent cap
    is known and callers must only enforce prompt + output <= context.
    """

    policy = getattr(profile, "runtime_policy", None)
    policy = policy if isinstance(policy, dict) else {}
    raw_limit = policy.get("max_output_tokens")
    if raw_limit is None:
        raw_limit = backend_limit
    try:
        limit = int(raw_limit or 0)
    except (TypeError, ValueError):
        return None
    if limit <= 0:
        return None
    if context_length > 0:
        return min(limit, context_length)
    return limit
