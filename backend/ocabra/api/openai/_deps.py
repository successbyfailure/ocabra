"""
Shared dependencies for OpenAI API endpoints.

Provides model/profile resolution, capability checks, and request forwarding
helpers used by all ``/v1/*`` endpoint modules.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from fastapi import Depends, HTTPException, Request

from ocabra.api._deps_auth import UserContext, get_current_user

if TYPE_CHECKING:
    from ocabra.core.model_manager import ModelManager, ModelState
    from ocabra.core.profile_registry import ProfileRegistry
    from ocabra.db.model_config import ModelProfile

logger = structlog.get_logger(__name__)


def _ensure_load_timeout_s() -> int:
    from ocabra.config import settings

    return max(60, int(settings.model_load_wait_timeout_s))


def _openai_error(
    message: str,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
    status_code: int = 400,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


async def get_openai_user(
    user: UserContext = Depends(get_current_user),
) -> UserContext:
    """Resolve auth for OpenAI-compatible endpoints.

    Delegates to ``get_current_user`` which already handles:
    - Bearer API key resolution.
    - Cookie JWT resolution.
    - Anonymous access when ``require_api_key_openai=False``.
    - HTTP 401 when ``require_api_key_openai=True`` and no credentials provided.

    Returns:
        Resolved :class:`UserContext` for the caller.

    Raises:
        HTTPException 401: When authentication is required but missing or invalid.
    """
    return user


async def resolve_model(
    model_manager: ModelManager,
    model_id: str,
    user: UserContext | None = None,
) -> tuple[str, ModelState | None]:
    """Resolve a model id by canonical id or backend_model_id alias.

    Resolution order:
    1) Exact canonical match (model_id)
    2) First state whose backend_model_id equals requested value

    If *user* is provided and the resolved model is not in the user's accessible
    model set, the model is treated as not found (404) to avoid leaking existence.

    Args:
        model_manager: The application :class:`ModelManager`.
        model_id: Requested model identifier (canonical or alias).
        user: Optional resolved :class:`UserContext`; used to filter model access.

    Returns:
        Tuple of ``(resolved_model_id, ModelState | None)``.
    """
    requested = str(model_id or "").strip()
    if not requested:
        return "", None

    exact = await model_manager.get_state(requested)
    if exact is not None:
        resolved_id = requested
        resolved_state = exact
    else:
        states = await model_manager.list_states()
        resolved_id = requested
        resolved_state = None
        for state in states:
            if state.backend_model_id == requested:
                resolved_id = state.model_id
                resolved_state = state
                break

    if resolved_state is not None and user is not None:
        if not user.is_admin and resolved_id not in user.accessible_model_ids:
            return resolved_id, None

    return resolved_id, resolved_state


# ── Worker key helpers ───────────────────────────────────────────


def compute_worker_key(base_model_id: str, load_overrides: dict | None) -> str:
    """Derive a worker key from a base model id and optional load overrides.

    When *load_overrides* is empty or ``None`` the key equals *base_model_id*
    (shared worker). Otherwise a short hash is appended so that different
    override combinations get separate workers.
    """
    if not load_overrides:
        return base_model_id
    # Deterministic JSON → hash
    canonical = json.dumps(load_overrides, sort_keys=True, separators=(",", ":"))
    short_hash = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"{base_model_id}::{short_hash}"


# ── Profile resolution ───────────────────────────────────────────


def get_profile_registry(request: Request) -> ProfileRegistry:
    """Return the :class:`ProfileRegistry` stored on ``app.state``."""
    registry = getattr(request.app.state, "profile_registry", None)
    if registry is None:
        raise _openai_error(
            "Profile registry not available.",
            "server_error",
            code="service_unavailable",
            status_code=503,
        )
    return registry


async def resolve_profile(
    profile_id: str,
    model_manager: ModelManager,
    profile_registry: ProfileRegistry,
    *,
    user: UserContext | None = None,
) -> tuple[ModelProfile, ModelState]:
    """Resolve a *profile_id* to its ``(ModelProfile, ModelState)`` pair.

    Resolution order:

    1. Exact match in :class:`ProfileRegistry` by *profile_id*. The profile
       must be enabled.
    2. **Legacy fallback** (when ``settings.legacy_model_id_fallback`` is
       ``True``): if *profile_id* contains ``/`` it looks like a canonical
       ``model_id``; find the model and its default profile, log a
       deprecation warning.
    3. If nothing matches, raise HTTP 404.

    Access control: when *user* is provided and the resolved profile id is
    not in the user's ``accessible_model_ids`` set the profile is treated as
    not found (404) to avoid leaking existence.
    """
    from ocabra.config import settings

    requested = str(profile_id or "").strip()
    if not requested:
        raise _openai_error(
            "The 'model' field is required.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    # 1. Direct profile lookup
    profile = await profile_registry.get(requested)
    if profile is not None:
        if not profile.enabled:
            raise _openai_error(
                f"The model '{requested}' is not available.",
                "invalid_request_error",
                param="model",
                code="model_not_found",
                status_code=404,
            )
        # Access control on profile_id
        if user is not None and not user.is_admin and requested not in user.accessible_model_ids:
            raise _openai_error(
                f"The model '{requested}' does not exist.",
                "invalid_request_error",
                param="model",
                code="model_not_found",
                status_code=404,
            )
        worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
        state = await _ensure_worker_loaded(
            model_manager,
            profile.base_model_id,
            worker_key,
            profile.load_overrides,
        )
        return profile, state

    # 2. Legacy fallback: canonical model_id with '/'
    if "/" in requested and settings.legacy_model_id_fallback:
        logger.warning(
            "legacy_model_id_fallback",
            requested=requested,
            hint="Clients should migrate to profile_id. "
            "Set LEGACY_MODEL_ID_FALLBACK=false to disable.",
        )
        # Find the model
        model_state = await model_manager.get_state(requested)
        if model_state is not None:
            # Access control on model_id
            if (
                user is not None
                and not user.is_admin
                and requested not in user.accessible_model_ids
            ):
                raise _openai_error(
                    f"The model '{requested}' does not exist.",
                    "invalid_request_error",
                    param="model",
                    code="model_not_found",
                    status_code=404,
                )
            # Find default profile for this model
            profiles = await profile_registry.list_by_model(requested)
            default_profile = next((p for p in profiles if p.is_default and p.enabled), None)
            if default_profile is None:
                # Try any enabled profile
                default_profile = next((p for p in profiles if p.enabled), None)
            if default_profile is not None:
                worker_key = compute_worker_key(
                    default_profile.base_model_id,
                    default_profile.load_overrides,
                )
                state = await _ensure_worker_loaded(
                    model_manager,
                    default_profile.base_model_id,
                    worker_key,
                    default_profile.load_overrides,
                )
                return default_profile, state

    # 3. Nothing matched → 404
    raise _openai_error(
        f"The model '{requested}' does not exist.",
        "invalid_request_error",
        param="model",
        code="model_not_found",
        status_code=404,
    )


async def _ensure_worker_loaded(
    model_manager: ModelManager,
    base_model_id: str,
    worker_key: str,
    load_overrides: dict | None,
) -> ModelState:
    """Ensure the worker identified by *worker_key* is loaded.

    When *worker_key* differs from *base_model_id* (non-empty overrides),
    we check if a virtual model entry already exists in ModelManager; if not,
    we create one by cloning the base model's state and applying
    ``load_overrides`` as extra config.

    Returns the :class:`ModelState` of the loaded worker.
    """
    # If worker_key == base_model_id, just use the normal path
    if worker_key == base_model_id:
        state = await model_manager.get_state(base_model_id)
        if state is None:
            raise _openai_error(
                f"Base model '{base_model_id}' is not configured.",
                "invalid_request_error",
                param="model",
                code="model_not_found",
                status_code=404,
            )
        # Delegate to the existing ensure_loaded for actual loading
        return await _do_ensure_loaded(model_manager, base_model_id)

    # Dedicated worker: check if a state already exists for this key
    state = await model_manager.get_state(worker_key)
    if state is not None:
        return await _do_ensure_loaded(model_manager, worker_key)

    # Clone from base model
    base_state = await model_manager.get_state(base_model_id)
    if base_state is None:
        raise _openai_error(
            f"Base model '{base_model_id}' is not configured.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    merged_extra = {**base_state.extra_config, **(load_overrides or {})}
    try:
        await model_manager.add_model(
            model_id=worker_key,
            backend_type=base_state.backend_type,
            display_name=f"{base_state.display_name} (override)",
            load_policy=base_state.load_policy.value,
            auto_reload=base_state.auto_reload,
            preferred_gpu=base_state.preferred_gpu,
            extra_config=merged_extra,
        )
    except Exception:
        # May already exist from concurrent request
        pass

    return await _do_ensure_loaded(model_manager, worker_key)


async def _do_ensure_loaded(
    model_manager: ModelManager,
    model_id: str,
) -> ModelState:
    """Core loading logic extracted from ``ensure_loaded``.

    Triggers on-demand loading, waits for LOADING state, and touches
    ``last_request_at``.
    """
    from ocabra.core.model_manager import ModelStatus

    state = await model_manager.get_state(model_id)
    if state is None:
        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    async def _touch(resolved_id: str, request_at: datetime) -> None:
        state.last_request_at = request_at
        touch = getattr(model_manager, "touch_last_request_at", None)
        if touch is None:
            return
        result = touch(resolved_id, request_at)
        if inspect.isawaitable(result):
            await result

    if state.status == ModelStatus.LOADED:
        await _touch(model_id, datetime.now(UTC))
        return state

    if state.status in (ModelStatus.CONFIGURED, ModelStatus.UNLOADED, ModelStatus.ERROR):
        try:
            await model_manager.load(model_id)
        except Exception as exc:
            from ocabra.core.scheduler import InsufficientVRAMError

            if isinstance(exc, InsufficientVRAMError):
                raise _openai_error(
                    str(exc),
                    "invalid_request_error",
                    code="insufficient_vram",
                    status_code=409,
                ) from exc
            raise _openai_error(
                f"Failed to load model '{model_id}': {exc}",
                "server_error",
                code="model_load_failed",
                status_code=503,
            ) from exc

        state = await model_manager.get_state(model_id)
        if state and state.status == ModelStatus.LOADED:
            await _touch(model_id, datetime.now(UTC))
            return state

    if state and state.status == ModelStatus.LOADING:
        for _ in range(_ensure_load_timeout_s()):
            await asyncio.sleep(1)
            state = await model_manager.get_state(model_id)
            if state and state.status == ModelStatus.LOADED:
                await _touch(model_id, datetime.now(UTC))
                return state
        raise _openai_error(
            f"Model '{model_id}' did not finish loading in time.",
            "server_error",
            code="model_load_timeout",
            status_code=503,
        )

    detail_suffix = ""
    if state and state.error_message:
        detail_suffix = f" detail: {state.error_message}"

    raise _openai_error(
        (
            f"Model '{model_id}' is not available "
            f"(status: {state.status.value if state else 'unknown'}).{detail_suffix}"
        ),
        "server_error",
        code="model_unavailable",
        status_code=503,
    )


def merge_profile_defaults(profile: ModelProfile, body: dict) -> dict:
    """Merge a profile's ``request_defaults`` and ``assets`` into a request body.

    The profile's ``request_defaults`` serve as a base; the client body
    overrides anything explicitly set. Asset injection (e.g. ``voice_ref``)
    is applied *after* the merge and cannot be overridden by the client.
    """
    defaults = profile.request_defaults or {}
    merged: dict[str, Any] = {**defaults, **body}

    # Asset injection — controlled paths that clients cannot override
    assets = profile.assets or {}
    voice_ref_info = assets.get("voice_ref")
    if isinstance(voice_ref_info, dict) and voice_ref_info.get("path"):
        merged["voice_ref"] = voice_ref_info["path"]
    elif isinstance(voice_ref_info, str) and voice_ref_info:
        merged["voice_ref"] = voice_ref_info

    return merged


async def ensure_loaded(
    model_manager: ModelManager,
    model_id: str,
    user: UserContext | None = None,
) -> ModelState:
    """
    Ensure a model is LOADED before forwarding a request.
    Triggers on-demand loading if CONFIGURED, UNLOADED, or ERROR.
    Waits up to settings.model_load_wait_timeout_s if already LOADING.
    On success, updates the model's last-request timestamp and persists it.

    If *user* is provided, model access is filtered by the user's accessible model
    set (mirrors the filtering done in /v1/models).
    """
    from ocabra.core.model_manager import ModelStatus

    async def _touch_last_request_at(resolved_id: str, request_at: datetime) -> None:
        state.last_request_at = request_at
        touch = getattr(model_manager, "touch_last_request_at", None)
        if touch is None:
            return
        result = touch(resolved_id, request_at)
        if inspect.isawaitable(result):
            await result

    resolved_model_id, state = await resolve_model(model_manager, model_id, user=user)
    if state is None:
        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    if state.status == ModelStatus.LOADED:
        request_at = datetime.now(UTC)
        await _touch_last_request_at(resolved_model_id, request_at)
        return state

    if state.status in (ModelStatus.CONFIGURED, ModelStatus.UNLOADED, ModelStatus.ERROR):
        try:
            await model_manager.load(resolved_model_id)
        except Exception as exc:
            # Surface resource/scheduler failures as clear client-visible errors.
            from ocabra.core.scheduler import InsufficientVRAMError

            if isinstance(exc, InsufficientVRAMError):
                raise _openai_error(
                    str(exc),
                    "invalid_request_error",
                    code="insufficient_vram",
                    status_code=409,
                ) from exc
            raise _openai_error(
                f"Failed to load model '{resolved_model_id}': {exc}",
                "server_error",
                code="model_load_failed",
                status_code=503,
            ) from exc

        state = await model_manager.get_state(resolved_model_id)
        if state and state.status == ModelStatus.LOADED:
            request_at = datetime.now(UTC)
            await _touch_last_request_at(resolved_model_id, request_at)
            return state

    if state and state.status == ModelStatus.LOADING:
        for _ in range(_ensure_load_timeout_s()):
            await asyncio.sleep(1)
            state = await model_manager.get_state(resolved_model_id)
            if state and state.status == ModelStatus.LOADED:
                request_at = datetime.now(UTC)
                await _touch_last_request_at(resolved_model_id, request_at)
                return state
        raise _openai_error(
            f"Model '{resolved_model_id}' did not finish loading in time.",
            "server_error",
            code="model_load_timeout",
            status_code=503,
        )

    detail_suffix = ""
    if state and state.error_message:
        detail_suffix = f" detail: {state.error_message}"

    raise _openai_error(
        (
            f"Model '{resolved_model_id}' is not available "
            f"(status: {state.status.value if state else 'unknown'}).{detail_suffix}"
        ),
        "server_error",
        code="model_unavailable",
        status_code=503,
    )


def check_capability(state: ModelState, capability: str, endpoint: str) -> None:
    """Raise 400 if the model lacks the required capability."""
    if not getattr(state.capabilities, capability, False):
        raise _openai_error(
            f"The model '{state.model_id}' does not support {endpoint}.",
            "invalid_request_error",
            param="model",
            code="model_not_capable",
        )


def to_backend_body(state: ModelState, body: dict) -> dict:
    """Copy request payload and normalize model field to backend-native model id."""
    payload = dict(body)
    payload["model"] = state.backend_model_id
    if payload.get("stream") is True:
        stream_options = payload.get("stream_options")
        if isinstance(stream_options, dict):
            payload["stream_options"] = {**stream_options, "include_usage": True}
        else:
            payload["stream_options"] = {"include_usage": True}
    return payload


def raise_upstream_http_error(exc: httpx.HTTPStatusError) -> None:
    """
    Translate worker HTTP errors into OpenAI-compatible API errors.

    Preserves upstream status code and error payload when possible.
    """
    status_code = exc.response.status_code
    body_text = exc.response.text

    try:
        parsed = json.loads(body_text) if body_text else None
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        if "detail" in parsed:
            raise HTTPException(status_code=status_code, detail=parsed["detail"])
        if "error" in parsed:
            raise HTTPException(status_code=status_code, detail=parsed)

    message = body_text.strip() if body_text else str(exc)
    raise _openai_error(
        message,
        "invalid_request_error" if 400 <= status_code < 500 else "server_error",
        status_code=status_code,
    )
