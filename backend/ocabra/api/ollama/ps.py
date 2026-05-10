"""
GET /api/ps — list currently loaded models in Ollama-compatible format.

Mirrors Ollama's ``/api/ps`` endpoint: returns only profiles whose backing
model is in the LOADED state. Each entry includes the standard Ollama fields
(``name``, ``model``, ``size``, ``digest``, ``details``, ``expires_at``,
``size_vram``) plus an oCabra-specific ``expected_load_seconds`` for clients
that want to estimate cold-start cost when a model is *not* in this list.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request

from ocabra.api._deps_auth import UserContext
from ocabra.config import settings

from ._shared import get_ollama_user

router = APIRouter()


@router.get("/ps", summary="List running models")
async def list_running(
    request: Request,
    user: UserContext = Depends(get_ollama_user),
) -> dict:
    """List all profiles whose model is currently LOADED.

    Returns the Ollama ``/api/ps`` schema:
      ``{"models": [{"name", "model", "size", "digest", "details",
                     "expires_at", "size_vram", ...}]}``.
    """
    model_manager = request.app.state.model_manager
    profile_registry = getattr(request.app.state, "profile_registry", None)

    models: list[dict[str, Any]] = []

    if profile_registry is not None:
        enabled_profiles = await profile_registry.list_enabled()
    else:
        enabled_profiles = []

    seen_base_models: set[str] = set()

    for profile in enabled_profiles:
        if (
            not user.is_admin
            and profile.profile_id not in user.accessible_model_ids
            and profile.base_model_id not in user.accessible_model_ids
        ):
            continue

        base_state = await model_manager.get_state(profile.base_model_id)
        if base_state is None or base_state.status.value != "loaded":
            continue

        seen_base_models.add(profile.base_model_id)
        backend_model_id = base_state.backend_model_id or profile.base_model_id
        size_bytes = _estimate_size_bytes(backend_model_id, base_state.vram_used_mb)

        models.append(
            {
                "name": profile.profile_id,
                "model": profile.profile_id,
                "size": size_bytes,
                "size_vram": int(max(0, base_state.vram_used_mb)) * 1024 * 1024,
                "digest": "sha256:"
                + hashlib.sha256(profile.profile_id.encode("utf-8")).hexdigest(),
                "details": {
                    "parent_model": profile.base_model_id,
                    "format": _infer_format(backend_model_id),
                    "family": profile.category,
                    "families": [profile.category],
                    "parameter_size": "unknown",
                    "quantization_level": "F16",
                },
                "expires_at": _compute_expires_at(base_state),
                "expected_load_seconds": await model_manager.get_expected_load_seconds(
                    profile.base_model_id
                ),
            }
        )

    # Fallback: include any LOADED model without an enabled profile (legacy or
    # admin-managed models). Admins see them all; regular users only see ones
    # they have access to.
    all_states = await model_manager.list_states()
    for state in all_states:
        if state.model_id in seen_base_models:
            continue
        if state.status.value != "loaded":
            continue
        if not user.is_admin and state.model_id not in user.accessible_model_ids:
            continue

        backend_model_id = state.backend_model_id or state.model_id
        size_bytes = _estimate_size_bytes(backend_model_id, state.vram_used_mb)
        models.append(
            {
                "name": state.model_id,
                "model": state.model_id,
                "size": size_bytes,
                "size_vram": int(max(0, state.vram_used_mb)) * 1024 * 1024,
                "digest": "sha256:"
                + hashlib.sha256(state.model_id.encode("utf-8")).hexdigest(),
                "details": {
                    "parent_model": "",
                    "format": _infer_format(backend_model_id),
                    "family": state.backend_type,
                    "families": [state.backend_type],
                    "parameter_size": "unknown",
                    "quantization_level": "F16",
                },
                "expires_at": _compute_expires_at(state),
                "expected_load_seconds": await model_manager.get_expected_load_seconds(
                    state.model_id
                ),
            }
        )

    return {"models": models}


def _estimate_size_bytes(model_id: str, vram_used_mb: int) -> int:
    model_dir = Path(settings.models_dir) / model_id
    if model_dir.exists():
        size = sum(
            path.stat().st_size
            for path in model_dir.rglob("*")
            if path.is_file() and path.suffix in {".safetensors", ".bin", ".gguf"}
        )
        if size > 0:
            return int(size)
    return int(max(0, vram_used_mb)) * 1024 * 1024


def _infer_format(model_id: str) -> str:
    lower = model_id.lower()
    if lower.endswith(".gguf"):
        return "gguf"
    return "safetensors"


def _compute_expires_at(state) -> str:
    """Best-effort idle-eviction timestamp: ``last_request_at + idle_timeout``.

    Falls back to ``loaded_at`` when there has been no traffic yet, and to
    ``now`` if neither is available.
    """
    timeout_s = max(0, int(getattr(settings, "idle_timeout_seconds", 0) or 0))
    base = state.last_request_at or state.loaded_at or datetime.now(UTC)
    expires = base + timedelta(seconds=timeout_s) if timeout_s > 0 else base
    return expires.astimezone(UTC).isoformat().replace("+00:00", "Z")
