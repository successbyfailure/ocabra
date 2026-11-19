"""Pure planning primitives for adaptive profile runtimes.

This module deliberately has no database or backend dependencies.  It resolves
``MAX``, selects the smallest suitable preset, and creates a stable identity for
the runner that the lifecycle layer will eventually load.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

SelectionReason = Literal["loaded", "default", "smallest_sufficient"]


class ContextCapacityError(ValueError):
    """Raised when no enabled preset can serve a request token budget."""


@dataclass(frozen=True)
class ResolvedRuntimePreset:
    """A persisted preset after its context and placement have been resolved."""

    preset_key: str
    effective_context: int
    parallelism: int
    kv_cache_dtype: str
    placement_policy: str = "auto"
    gpu_indices: tuple[int, ...] = ()
    enabled: bool = True
    is_default: bool = False
    priority: int = 100
    runtime_key: str | None = None


@dataclass(frozen=True)
class RuntimeSelection:
    preset: ResolvedRuntimePreset
    reason: SelectionReason
    reused_loaded_runtime: bool


def build_runtime_key(
    *,
    model_id: str,
    backend_type: str,
    effective_context: int,
    parallelism: int,
    kv_cache_dtype: str,
    placement_policy: str,
    gpu_indices: Iterable[int] = (),
    load_flags: Mapping[str, Any] | None = None,
    pool_id: str | None = None,
    placement: Mapping[str, Any] | None = None,
) -> str:
    """Create a stable runner identity, independent of the preset's UI name.

    ``placement`` (the preset's own GPU/CPU placement) only enters the identity
    when set, so presets without one keep the key they always had.
    """

    if effective_context <= 0:
        raise ValueError("effective_context must be positive")
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    canonical = {
        "backend": backend_type,
        "context": effective_context,
        "gpus": sorted(set(gpu_indices)),
        "kv": kv_cache_dtype,
        "load_flags": load_flags or {},
        "model": model_id,
        "parallelism": parallelism,
        "placement": placement_policy,
        "pool": pool_id,
    }
    if placement:
        canonical["preset_placement"] = dict(placement)
    try:
        payload = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except TypeError as exc:
        raise ValueError("load_flags must be JSON serializable") from exc
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"runtime:{digest}"


def select_runtime_preset(
    presets: Iterable[ResolvedRuntimePreset],
    *,
    required_tokens: int,
    loaded_runtime_keys: Iterable[str] = (),
    selection: Literal["prefer_loaded", "prefer_default"] = "prefer_loaded",
) -> RuntimeSelection:
    """Choose a preset for ``prompt + reserved output`` tokens.

    ``prefer_loaded`` implements the normal hot-runner path.  A larger loaded
    context is intentionally valid for a smaller request. ``prefer_default`` is
    the explicit opt-out and chooses the default first when it covers the job.
    """

    if required_tokens <= 0:
        raise ValueError("required_tokens must be positive")
    if selection not in {"prefer_loaded", "prefer_default"}:
        raise ValueError(f"unsupported selection policy: {selection}")

    candidates = [
        preset
        for preset in presets
        if preset.enabled and preset.effective_context >= required_tokens
    ]
    if not candidates:
        raise ContextCapacityError(
            f"no enabled runtime preset covers {required_tokens} tokens"
        )

    def order_key(preset: ResolvedRuntimePreset) -> tuple[int, int, str]:
        return (preset.effective_context, preset.priority, preset.preset_key)

    loaded = set(loaded_runtime_keys)
    loaded_candidates = [
        preset
        for preset in candidates
        if preset.runtime_key is not None and preset.runtime_key in loaded
    ]
    default = next((preset for preset in candidates if preset.is_default), None)

    if selection == "prefer_default" and default is not None:
        chosen = default
        reason: SelectionReason = "default"
    elif loaded_candidates:
        chosen = min(loaded_candidates, key=order_key)
        reason = "loaded"
    elif default is not None:
        chosen = default
        reason = "default"
    else:
        chosen = min(candidates, key=order_key)
        reason = "smallest_sufficient"

    return RuntimeSelection(
        preset=chosen,
        reason=reason,
        reused_loaded_runtime=(
            chosen.runtime_key is not None and chosen.runtime_key in loaded
        ),
    )
