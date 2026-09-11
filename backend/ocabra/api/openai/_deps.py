"""
Shared dependencies for OpenAI API endpoints.

Provides model/profile resolution, capability checks, and request forwarding
helpers used by all ``/v1/*`` endpoint modules.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from fastapi import Depends, HTTPException, Request

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.core.model_manager_helpers import compute_worker_key

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


async def _wait_for_unloading_transition(
    model_manager: ModelManager,
    model_id: str,
    state: ModelState,
) -> ModelState:
    """Wait for an in-progress unload before attempting a new load.

    An unload is a transient state, not a terminal model failure.  Treating it
    as unavailable makes requests race with idle/manual eviction and produces a
    needless 503 even though the model can be loaded again immediately after.
    """
    from ocabra.core.model_manager import ModelStatus

    if state.status != ModelStatus.UNLOADING:
        return state

    deadline = asyncio.get_running_loop().time() + _ensure_load_timeout_s()
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.1)
        current = await model_manager.get_state(model_id)
        if current is None:
            break
        if current.status != ModelStatus.UNLOADING:
            return current

    exc = _openai_error(
        f"Model '{model_id}' did not finish unloading in time.",
        "server_error",
        code="model_transition_timeout",
        status_code=503,
    )
    exc.headers = {"Retry-After": "2"}
    raise exc


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
    request_body: dict | None = None,
    router_resolver: object | None = None,
    request_state: object | None = None,
) -> tuple[ModelProfile, ModelState]:
    """Resolve a *profile_id* to its ``(ModelProfile, ModelState)`` pair.

    Resolution order:

    1. Exact match in :class:`ProfileRegistry` by *profile_id*. The profile
       must be enabled.
    2. Canonical ``model_id`` fallback: find the model and its default enabled
       profile (or the first enabled profile). Canonical ids are stable public
       identifiers even when legacy alias fallback is disabled.
    3. If nothing matches, raise HTTP 404.

    Access control: when *user* is provided and the resolved profile id is
    not in the user's ``accessible_model_ids`` set the profile is treated as
    not found (404) to avoid leaking existence.
    """
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
        # Access control: check both profile_id and base_model_id since
        # group_models stores canonical model_ids (e.g. "ollama/qwen3:8b")
        # while clients send profile_ids (e.g. "qwen3:8b").
        if (
            user is not None
            and not user.is_admin
            and requested not in user.accessible_model_ids
            and profile.base_model_id not in user.accessible_model_ids
        ):
            raise _openai_error(
                f"The model '{requested}' does not exist.",
                "invalid_request_error",
                param="model",
                code="model_not_found",
                status_code=404,
            )
        # Bloque 20 — Router profiles: if this profile declares routing_targets
        # delegate to the RouterResolver, which returns the best target under
        # current conditions (loaded / loadable / session veto / estimator).
        # The router's own attribution flows via ``via_router_profile_id`` on
        # the eventual request_stat row so ``/stats/routing`` can aggregate it.
        if router_resolver is not None and router_resolver.is_router(profile):
            resolved = await router_resolver.pick(profile, request_body=request_body)
            target = await profile_registry.get(resolved.target_profile_id)
            if target is not None:
                worker_key = compute_worker_key(target.base_model_id, target.load_overrides)
                state = await _ensure_worker_loaded(
                    model_manager,
                    target.base_model_id,
                    worker_key,
                    target.load_overrides,
                )
                # Router attribution — StatsMiddleware (BaseHTTPMiddleware)
                # doesn't propagate contextvars set by the handler back to
                # itself after ``call_next`` (async-boundary in Starlette),
                # so we hang the value on ``request.state``: shared object,
                # accessible from both sides of the boundary, no reliance on
                # asyncio.copy_context semantics.
                if request_state is not None:
                    try:
                        request_state.via_router_profile_id = profile.profile_id
                        # Also record the resolved target profile_id so the
                        # stats collector can group routing decisions by the
                        # profile that actually served the request (vs the
                        # target's canonical model_id).
                        request_state.resolved_model_id = target.profile_id
                    except Exception:  # noqa: BLE001 — never break resolution
                        pass
                return target, state
            # Router failed to resolve any target (shouldn't happen — pick
            # always returns something), fall through to plain path so the
            # request still gets served with the router's own base_model_id.

        worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
        state = await _ensure_worker_loaded(
            model_manager,
            profile.base_model_id,
            worker_key,
            profile.load_overrides,
        )
        return profile, state

    # 2. Canonical model id fallback.  This is not a legacy alias: internal
    # APIs and persisted configuration advertise this stable identifier.
    model_state = await model_manager.get_state(requested)
    if model_state is not None:
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
        profiles = await profile_registry.list_by_model(requested)
        selected = next((p for p in profiles if p.is_default and p.enabled), None)
        if selected is None:
            selected = next((p for p in profiles if p.enabled), None)
        if selected is None:
            raise _openai_error(
                f"The model '{requested}' has no enabled profile.",
                "invalid_request_error",
                param="model",
                code="profile_not_configured",
                status_code=404,
            )
        worker_key = compute_worker_key(selected.base_model_id, selected.load_overrides)
        state = await _ensure_worker_loaded(
            model_manager,
            selected.base_model_id,
            worker_key,
            selected.load_overrides,
        )
        return selected, state

    # 3. Nothing matched → 404
    logger.warning(
        "resolve_profile_not_found",
        requested=requested,
        has_slash="/" in requested,
    )
    raise _openai_error(
        f"The model '{requested}' does not exist.",
        "invalid_request_error",
        param="model",
        code="model_not_found",
        status_code=404,
    )


# OpenAPI: shared response metadata for streaming endpoints. Surfaces the
# X-Ocabra-* headers (only set when the request asks for stream=true) and a
# short description of the SSE comment events the server interleaves.
STREAMING_LOAD_RESPONSE_DOC: dict = {
    200: {
        "description": (
            "Successful response. When ``stream=true`` the body is "
            "``text/event-stream``; oCabra pre-flushes the response headers "
            "below before triggering any model load, and interleaves two "
            "named SSE events carrying load progress (same convention as "
            "``ocabra.tool_started`` / ``ocabra.tool_result``):\n\n"
            '``event: ocabra.model_loading\\ndata: {"model_id": ..., '
            '"worker_key": ..., "status": ..., '
            '"expected_wait_seconds": ...}\\n\\n`` on stream open, and '
            '``event: ocabra.model_ready\\ndata: {"model_id": ..., '
            '"load_duration_ms": ..., "was_cold_start": ...}\\n\\n`` '
            "once the model is ready. Clients that don't recognise these "
            "named events ignore them; the rest of the stream is plain "
            "OpenAI-format ``data: {...}`` chunks."
        ),
        "headers": {
            "X-Ocabra-Model-Status": {
                "description": (
                    "Worker status when the stream was opened: "
                    "``configured``, ``loading``, ``loaded``, etc. "
                    "Streaming responses only."
                ),
                "schema": {"type": "string"},
            },
            "X-Ocabra-Model-Id": {
                "description": "Canonical model id resolved from the request.",
                "schema": {"type": "string"},
            },
            "X-Ocabra-Expected-Wait-Seconds": {
                "description": (
                    "Median load time (seconds) over the last 5 successful "
                    "loads of this model. Only emitted when the worker is "
                    "not yet ``loaded`` and historical samples exist."
                ),
                "schema": {"type": "integer"},
            },
        },
    },
}


SSE_KEEPALIVE_INTERVAL_S = 10.0


async def keepalive_until_done(task: asyncio.Task, interval: float = SSE_KEEPALIVE_INTERVAL_S):
    """Yield ``b": keepalive\\n\\n"`` every *interval* seconds until *task* finishes.

    Mirrors the pattern from :mod:`ocabra.agents.executor`: shields the task
    so the wait_for timeout doesn't cancel it, emits a comment-line keepalive
    on each timeout to keep proxies (NPM, Cloudflare, ...) from closing the
    connection during long model loads, and exits when the task completes —
    leaving the caller to inspect ``task.result()`` / ``task.exception()``.
    """
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except TimeoutError:
            yield b": keepalive\n\n"
        except Exception:
            # The caller owns error serialization.  Do not let a failed load
            # escape from the keepalive iterator after the SSE response has
            # already started; leave it to inspect task.exception().
            return


def sse_ocabra_event(event: str, **fields: object) -> bytes:
    """Encode an oCabra load-status hint as an SSE *comment* frame.

    Previously emitted a named ``event: ocabra.<event>`` frame with a JSON
    ``data:`` payload. Strict OpenAI clients don't filter by event name — they
    parse every ``data:`` line as a ``ChatCompletionChunk`` and choke on the
    non-chunk JSON. A comment line (starting with ``:``) is ignored by every
    spec-compliant SSE parser (the OpenAI SDK included), so it can't be
    mis-parsed, while still forcing a header flush and keeping the connection
    alive during cold starts. oCabra-aware clients may read the JSON after the
    ``:``; status is also available out-of-band via the ``X-Ocabra-*`` headers,
    ``GET /v1/models/{id}`` (``ocabra`` extension) and
    ``GET /ocabra/models/{id}/status``.
    """
    payload = {"event": f"ocabra.{event}", **fields}
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    return f": {encoded}\n\n".encode()


async def build_model_status_headers(
    model_manager: ModelManager,
    worker_key: str,
    base_model_id: str,
) -> dict[str, str]:
    """Return diagnostic headers describing the worker's current load state.

    Always includes ``X-Ocabra-Model-Status`` and ``X-Ocabra-Model-Id``. When
    the worker is not yet loaded, also adds
    ``X-Ocabra-Expected-Wait-Seconds`` based on historical load times. Headers
    are intended to be flushed *before* the load begins (streaming endpoints)
    so the client learns the wait time up front.
    """
    state = await model_manager.get_state(worker_key)
    if state is None and worker_key != base_model_id:
        state = await model_manager.get_state(base_model_id)

    if state is None:
        return {}

    headers: dict[str, str] = {
        "X-Ocabra-Model-Status": state.status.value,
        "X-Ocabra-Model-Id": state.model_id,
    }
    if state.status.value != "loaded":
        # Try the worker key first (covers per-profile workers); fall back to
        # the base model so we still surface a hint when this is the first
        # load of an override variant.
        expected = await model_manager.get_expected_load_seconds(worker_key)
        if expected is None and worker_key != base_model_id:
            expected = await model_manager.get_expected_load_seconds(base_model_id)
        if expected is not None:
            headers["X-Ocabra-Expected-Wait-Seconds"] = str(expected)
    return headers


async def lookup_profile(
    profile_id: str,
    profile_registry: ProfileRegistry,
    *,
    user: UserContext | None = None,
) -> ModelProfile:
    """Resolve *profile_id* to its :class:`ModelProfile` **without** loading.

    Mirrors :func:`resolve_profile` for lookup/access-control but does not
    call :func:`_ensure_worker_loaded`. Use this when you need the profile
    metadata before deciding whether/when to trigger the load (e.g. to
    pre-flush response headers in a streaming endpoint).
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
        if (
            user is not None
            and not user.is_admin
            and requested not in user.accessible_model_ids
            and profile.base_model_id not in user.accessible_model_ids
        ):
            raise _openai_error(
                f"The model '{requested}' does not exist.",
                "invalid_request_error",
                param="model",
                code="model_not_found",
                status_code=404,
            )
        return profile

    if "/" in requested and settings.legacy_model_id_fallback:
        # Legacy path: callers must use resolve_profile to get the loaded state
        # for legacy model ids, since we'd otherwise need to query the model
        # manager here. Surface a 404 — streaming endpoints should fall back
        # to resolve_profile in that case.
        pass

    raise _openai_error(
        f"The model '{requested}' does not exist.",
        "invalid_request_error",
        param="model",
        code="model_not_found",
        status_code=404,
    )


async def ensure_worker_loaded(
    model_manager: ModelManager,
    base_model_id: str,
    worker_key: str,
    load_overrides: dict | None,
) -> ModelState:
    """Public wrapper around :func:`_ensure_worker_loaded`.

    Exposed for streaming endpoints that pre-flush headers and trigger the
    actual load from inside the SSE generator.
    """
    return await _ensure_worker_loaded(model_manager, base_model_id, worker_key, load_overrides)


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
        logger.warning("ensure_loaded_state_missing", model_id=model_id)
        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    state = await _wait_for_unloading_transition(model_manager, model_id, state)

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
            if state is None or state.status != ModelStatus.LOADING:
                break
        if state and state.status == ModelStatus.LOADING:
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


def get_federation_manager(request: Request):
    """Return the FederationManager from app state, or None if disabled."""
    return getattr(request.app.state, "federation_manager", None)


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

    state = await _wait_for_unloading_transition(
        model_manager,
        resolved_model_id,
        state,
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


def _thinking_disable_intent(body: dict) -> bool | None:
    """Infer whether the client wants reasoning/thinking disabled.

    Returns True (disable), False (force enable) or None (unspecified) by
    inspecting the various spellings clients use: Ollama's native ``think``,
    vLLM's ``chat_template_kwargs.enable_thinking`` and the OpenAI-style
    ``reasoning`` / ``reasoning_effort`` fields.
    """
    if isinstance(body.get("think"), bool):
        return not body["think"]

    cct = body.get("chat_template_kwargs")
    if isinstance(cct, dict) and isinstance(cct.get("enable_thinking"), bool):
        return not cct["enable_thinking"]

    reasoning = body.get("reasoning")
    if isinstance(reasoning, bool):
        return not reasoning
    if isinstance(reasoning, str) and reasoning.strip().lower() in {"none", "off", "false"}:
        return True

    effort = body.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip().lower() in {"none", "off", "false"}:
        return True

    return None


def _apply_thinking_controls(state: ModelState, payload: dict) -> None:
    """Map the client's thinking-disable intent onto backend-specific knobs.

    Backends disagree on how (and whether) reasoning can be turned off:

    - **vLLM / sglang** honour it through the chat template, so we set
      ``chat_template_kwargs.enable_thinking = false``. This genuinely
      suppresses the ``<think>`` block for models that support the flag
      (e.g. Qwen3) and is silently ignored by templates that don't.
    - **Ollama** cannot disable reasoning through its OpenAI-compatible
      endpoint for current reasoning models: a string ``reasoning`` 400s
      (it expects an ``openai.Reasoning`` object) and ``reasoning_effort``
      only relocates the chain-of-thought inline. So we inject nothing and
      just drop the vLLM-only ``chat_template_kwargs`` to avoid confusing the
      upstream. Disabling reasoning there needs a prompt-level switch (e.g.
      Qwen's ``/no_think``) or an upstream Ollama fix (ollama/ollama#15288).
    """
    backend_type = getattr(state, "backend_type", "")
    if backend_type == "ollama":
        payload.pop("chat_template_kwargs", None)
        return
    if backend_type in {"vllm", "sglang"} and _thinking_disable_intent(payload) is True:
        cct = payload.get("chat_template_kwargs")
        cct = dict(cct) if isinstance(cct, dict) else {}
        cct.setdefault("enable_thinking", False)
        payload["chat_template_kwargs"] = cct


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
    _apply_thinking_controls(state, payload)
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


def stream_error_details(exc: Exception) -> tuple[str, str]:
    """Return a client-safe message and stable code for an in-band SSE error.

    Streaming response headers have already been sent when worker forwarding
    fails, so the upstream HTTP status cannot be returned directly. Preserve
    the worker's JSON error message in the SSE envelope and encode its status
    class in a stable oCabra code for request statistics.
    """
    from ocabra.core.worker_pool import InferenceTimeoutError

    if isinstance(exc, InferenceTimeoutError):
        return str(exc), "generation_timeout"

    if not isinstance(exc, httpx.HTTPStatusError):
        return str(exc), "stream_error"

    status_code = exc.response.status_code
    body_text = exc.response.text
    message = ""
    try:
        parsed = json.loads(body_text) if body_text else None
    except (TypeError, ValueError):
        parsed = None

    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        error = parsed.get("error")
        if isinstance(detail, str):
            message = detail
        elif isinstance(detail, dict):
            message = str(detail.get("message") or "")
        if not message and isinstance(error, str):
            message = error
        elif not message and isinstance(error, dict):
            message = str(error.get("message") or "")

    if not message:
        message = body_text.strip() if body_text else str(exc)

    if status_code == 429:
        code = "upstream_rate_limited"
    elif 400 <= status_code < 500:
        code = "upstream_invalid_request"
    else:
        code = "upstream_server_error"

    logger.warning(
        "worker_stream_http_error",
        upstream_status=status_code,
        upstream_code=code,
        error_message=message[:1000],
        worker_url=str(exc.request.url),
    )
    return message, code
