"""
Shared dependencies for OpenAI API endpoints.

Provides model/profile resolution, capability checks, and request forwarding
helpers used by all ``/v1/*`` endpoint modules.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from fastapi import Depends, HTTPException, Request

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.core.model_manager_helpers import compute_worker_key  # noqa: F401
from ocabra.core.model_ref import is_llama_cpp_family
from ocabra.core.runtime_coordinator import (
    OutputTokenLimitError,
    RuntimePlan,
    plan_profile_runtime,
)
from ocabra.core.runtime_presets import ContextCapacityError

if TYPE_CHECKING:
    from ocabra.core.model_manager import ModelManager, ModelState
    from ocabra.core.profile_registry import ProfileRegistry
    from ocabra.db.model_config import ModelProfile

logger = structlog.get_logger(__name__)


def _ensure_load_timeout_s() -> int:
    from ocabra.config import settings

    return max(60, int(settings.model_load_wait_timeout_s))


async def _wait_for_service_gpu_and_retry_load(
    model_manager: ModelManager,
    model_id: str,
    state: ModelState | None,
) -> ModelState | None:
    """After a load fails with InsufficientVRAMError, check whether the cause is
    an external service (Hunyuan, TRELLIS.2, ...) holding a temporary VRAM
    reservation on this model's preferred GPU (see ServiceManager.reserve_gpu_vram)
    — as opposed to genuine, indefinite capacity exhaustion. If so, poll until the
    reservation clears (or we time out) and retry the load, instead of failing
    the request immediately. Plain models competing for VRAM with each other are
    unaffected: this only activates when a service reservation is the reason.

    Returns the freshly loaded state on success, or None if the caller should
    fall through to the normal InsufficientVRAMError → 409 response.
    """
    from ocabra.core.model_manager import ModelStatus
    from ocabra.core.scheduler import InsufficientVRAMError

    gpu_manager = model_manager.gpu_manager
    target_gpu = state.preferred_gpu if state else None
    if gpu_manager is None or target_gpu is None:
        return None
    if not gpu_manager.has_service_lock(target_gpu):
        return None

    blockers = gpu_manager.service_locks(target_gpu)
    model_manager.mark_service_gpu_wait(model_id, ", ".join(blockers) or "?")
    try:
        for _ in range(_ensure_load_timeout_s()):
            await asyncio.sleep(1)
            if gpu_manager.has_service_lock(target_gpu):
                continue
            try:
                await model_manager.load(model_id)
            except InsufficientVRAMError:
                # Reservation cleared but something else (a genuinely concurrent
                # load) grabbed the VRAM first — keep waiting for the timeout.
                continue
            new_state = await model_manager.get_state(model_id)
            if new_state and new_state.status == ModelStatus.LOADED:
                return new_state
        return None
    finally:
        model_manager.clear_service_gpu_wait(model_id)


def _openai_error(
    message: str,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
    status_code: int = 400,
    headers: dict[str, str] | None = None,
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
        headers=headers,
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

    raise _openai_error(
        f"Model '{model_id}' did not finish unloading in time.",
        "server_error",
        code="model_transition_timeout",
        status_code=503,
        headers={"Retry-After": "2"},
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
        if not user.can_access_model(resolved_id):
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
            headers={"Retry-After": "5"},
        )
    return registry


async def ocabra_response_headers(
    request: object,
    model_manager: ModelManager,
    worker_key: str,
    base_model_id: str,
) -> dict[str, str]:
    """Compact wrapper for post-load endpoints.

    Reads ``pre_load_status`` and ``load_started_wall`` off ``request.state``
    (populated by :func:`_snapshot_pre_load`) and calls
    :func:`build_model_status_headers` with the cold-start + router fields
    already filled in. Endpoints get a one-liner for the whole header set.
    """
    st = getattr(request, "state", None)
    pre_status = getattr(st, "pre_load_status", None) if st is not None else None
    load_started = getattr(st, "load_started_wall", None) if st is not None else None
    via_router = getattr(st, "via_router_profile_id", None) if st is not None else None
    router_target = getattr(st, "resolved_model_id", None) if st is not None else None
    elapsed_ms: int | None = None
    if load_started is not None:
        elapsed_ms = int(max(0.0, time.monotonic() - load_started) * 1000)
    return await build_model_status_headers(
        model_manager,
        worker_key,
        base_model_id,
        pre_status=pre_status,
        elapsed_ms=elapsed_ms,
        via_router_profile_id=via_router,
        router_target_profile_id=router_target,
    )


async def _snapshot_pre_load(
    model_manager: ModelManager,
    worker_key: str,
    request_state: object | None,
) -> None:
    """Stamp ``pre_load_status`` + ``load_started_wall`` on ``request.state``.

    Called right before ``_ensure_worker_loaded`` so downstream endpoints can
    tell whether the request paid for a cold start and how long the load
    took. No-op when ``request_state`` isn't provided.
    """
    if request_state is None:
        return
    try:
        if hasattr(request_state, "pre_load_status"):
            return  # first snapshot wins (e.g. router fallback path)
        state = await model_manager.get_state(worker_key)
        request_state.pre_load_status = state.status.value if state is not None else "unknown"
        request_state.load_started_wall = time.monotonic()
    except Exception:  # noqa: BLE001 — never block resolution on telemetry
        pass


async def resolve_profile_runtime_plan(
    profile: ModelProfile,
    model_manager: ModelManager,
    request_body: dict | None,
    *,
    request_state: object | None = None,
) -> RuntimePlan:
    """Select a profile runtime preset and expose its attribution on request state."""

    try:
        plan = await plan_profile_runtime(profile, model_manager, request_body)
    except OutputTokenLimitError as exc:
        raise _openai_error(
            str(exc),
            "invalid_request_error",
            param=exc.parameter,
            code="max_tokens_exceeded",
            status_code=400,
        ) from exc
    except ContextCapacityError as exc:
        raise _openai_error(
            str(exc),
            "invalid_request_error",
            param="model",
            code="context_length_exceeded",
            status_code=400,
        ) from exc
    if request_state is not None:
        try:
            request_state.runtime_worker_key = plan.worker_key
            request_state.runtime_preset_key = plan.preset_key
            request_state.runtime_key = plan.runtime_key
            request_state.runtime_effective_context = plan.effective_context
            request_state.runtime_required_tokens = plan.required_tokens
            request_state.runtime_selection_reason = plan.selection_reason
            request_id = getattr(request_state, "inflight_request_id", None)
            old_worker = getattr(request_state, "inflight_model_id", None)
            if request_id and old_worker and old_worker != plan.worker_key:
                model_manager.rebind_request(request_id, old_worker, plan.worker_key)
                request_state.inflight_model_id = plan.worker_key
        except Exception:  # noqa: BLE001 — telemetry must not break inference
            pass
    return plan


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

    # 1. Hidden compatibility alias, then direct profile lookup. Aliases may
    # pin a historical ``-ctx*`` id to one canonical runtime preset.
    alias_resolver = getattr(profile_registry, "resolve_alias", None)
    alias_target = await alias_resolver(requested) if alias_resolver is not None else None
    profile = alias_target[0] if alias_target is not None else None
    if alias_target is not None and alias_target[1]:
        extension = dict(request_body.get("ocabra") or {}) if isinstance(request_body, dict) else {}
        extension["context_preset"] = alias_target[1]
        request_body = {**(request_body or {}), "ocabra": extension}
    if profile is None:
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
        if user is not None and not user.can_access_model(
            profile.profile_id, profile.base_model_id
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
            from ocabra.core.router_resolver import NoRoutableTargetError

            try:
                resolved = await router_resolver.pick(profile, request_body=request_body)
            except NoRoutableTargetError as exc:
                raise _openai_error(
                    str(exc),
                    "server_error",
                    param="model",
                    code="router_no_target",
                    status_code=503,
                ) from exc
            target = await profile_registry.get(resolved.target_profile_id)
            if target is not None and target.base_model_id is not None:
                runtime_plan = await resolve_profile_runtime_plan(
                    target,
                    model_manager,
                    request_body,
                    request_state=request_state,
                )
                worker_key = runtime_plan.worker_key
                await _snapshot_pre_load(model_manager, worker_key, request_state)
                state = await _ensure_worker_loaded(
                    model_manager,
                    target.base_model_id,
                    worker_key,
                    runtime_plan.load_overrides,
                    preferred_gpu=runtime_plan.preferred_gpu,
                    load_policy=runtime_plan.load_policy,
                    required_tokens=runtime_plan.required_tokens,
                )
                # Router attribution — StatsMiddleware (BaseHTTPMiddleware)
                # doesn't propagate contextvars set by the handler back to
                # itself after ``call_next`` (async-boundary in Starlette),
                # so we hang the value on ``request.state``: shared object,
                # accessible from both sides of the boundary, no reliance on
                # asyncio.copy_context semantics.
                #
                # Skip attribution when the router resolved to itself (the
                # "no_immediate_winner" fallback in RouterResolver.pick).
                # Otherwise ``/stats/routing`` shows self-loops (router → same
                # router) that carry no useful information — the request was
                # served by the router's own base_model_id, no different from
                # a plain profile call.
                if request_state is not None and target.profile_id != profile.profile_id:
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
            raise _openai_error(
                f"Router '{requested}' resolved an invalid target.",
                "server_error",
                param="model",
                code="router_invalid_target",
                status_code=503,
            )

        if profile.base_model_id is None:
            raise _openai_error(
                f"Router '{requested}' is unavailable while routing is disabled.",
                "server_error",
                param="model",
                code="routing_disabled",
                status_code=503,
            )

        runtime_plan = await resolve_profile_runtime_plan(
            profile,
            model_manager,
            request_body,
            request_state=request_state,
        )
        worker_key = runtime_plan.worker_key
        await _snapshot_pre_load(model_manager, worker_key, request_state)
        state = await _ensure_worker_loaded(
            model_manager,
            profile.base_model_id,
            worker_key,
            runtime_plan.load_overrides,
            preferred_gpu=runtime_plan.preferred_gpu,
            load_policy=runtime_plan.load_policy,
            required_tokens=runtime_plan.required_tokens,
        )
        return profile, state

    # 2. Canonical model id fallback.  This is not a legacy alias: internal
    # APIs and persisted configuration advertise this stable identifier.
    model_state = await model_manager.get_state(requested)
    if model_state is not None:
        if user is not None and not user.can_access_model(requested):
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
        runtime_plan = await resolve_profile_runtime_plan(
            selected,
            model_manager,
            request_body,
            request_state=request_state,
        )
        worker_key = runtime_plan.worker_key
        await _snapshot_pre_load(model_manager, worker_key, request_state)
        state = await _ensure_worker_loaded(
            model_manager,
            selected.base_model_id,
            worker_key,
            runtime_plan.load_overrides,
            preferred_gpu=runtime_plan.preferred_gpu,
            load_policy=runtime_plan.load_policy,
            required_tokens=runtime_plan.required_tokens,
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


# OpenAPI: canonical X-Ocabra-* header set emitted by every endpoint that
# resolves a model (chat, completions, embeddings, pooling/score/rerank/
# classify, audio/transcriptions, audio/speech, audio/generate,
# images/generations, images/edits). Streaming endpoints flush these before
# the load blocks; non-streaming attach them to the final response.
OCABRA_MODEL_HEADERS: dict = {
    "X-Ocabra-Model-Status": {
        "description": (
            "Worker status at emit time (``configured``, ``loading``, "
            "``loaded``, ``unloaded``, ``error``). Always present."
        ),
        "schema": {"type": "string"},
    },
    "X-Ocabra-Model-Id": {
        "description": ("Canonical model id resolved from the request (post-router)."),
        "schema": {"type": "string"},
    },
    "X-Ocabra-Expected-Wait-Seconds": {
        "description": (
            "Median load time (seconds) over the last 5 successful loads of "
            "this model. Emitted only when the worker isn't ``loaded`` yet "
            "and historical samples exist."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Load-Queue-Depth": {
        "description": (
            "Total loads currently queued (waiting + active) behind the "
            "cross-backend load gate. Emitted only under pressure."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Load-Active": {
        "description": (
            "Loads actually running right now. With the current mutex this "
            "is at most 1; kept explicit so clients can distinguish "
            "queued-behind vs. loading-now."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Was-Cold-Start": {
        "description": (
            "``1`` when the request paid for a model load. Omitted when the "
            "model was already warm. Requires the endpoint to have "
            "captured the pre-load state (all resource endpoints do)."
        ),
        "schema": {"type": "string", "enum": ["1"]},
    },
    "X-Ocabra-Load-Duration-Ms": {
        "description": (
            "Milliseconds the request spent waiting for the model to be "
            "ready. Only emitted alongside ``X-Ocabra-Was-Cold-Start``."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Router": {
        "description": (
            "When the requested model was a router profile, this is the "
            "``profile_id`` the client asked for. See "
            "``X-Ocabra-Router-Target`` for what actually served."
        ),
        "schema": {"type": "string"},
    },
    "X-Ocabra-Router-Target": {
        "description": (
            "``profile_id`` of the profile that actually served the request "
            "after the router picked. Present only alongside "
            "``X-Ocabra-Router``."
        ),
        "schema": {"type": "string"},
    },
    "X-Ocabra-Slot-Queue-Saturated": {
        "description": (
            "``1`` (non-streaming only) when the request arrived to find "
            "every one of the worker's fair-share concurrency slots "
            "already occupied (see ``FairSlotGate``) and had to wait its "
            "turn — covers queueing on an *already-loaded* model, distinct "
            "from ``X-Ocabra-Expected-Wait-Seconds``/``model_loading`` "
            "which cover a cold model load. No queue *position* is given: "
            "the gate alternates by least-recently-served user, not strict "
            "FIFO, so a numeric position would be unstable. Streaming "
            "clients get the equivalent ``ocabra.slot_queued`` SSE comment "
            "instead (see the streaming response description above)."
        ),
        "schema": {"type": "string", "enum": ["1"]},
    },
    "X-Ocabra-Slot-In-Flight": {
        "description": (
            "Requests in flight on that worker at arrival time. Present "
            "only alongside ``X-Ocabra-Slot-Queue-Saturated``."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Slot-Count": {
        "description": (
            "Configured concurrency slots for that worker. Present only "
            "alongside ``X-Ocabra-Slot-Queue-Saturated``."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-KV-Queue-Saturated": {
        "description": (
            "``1`` when the request had to wait because its conservative KV "
            "reservation did not fit in the unified pool remainder. Distinct "
            "from physical slot saturation."
        ),
        "schema": {"type": "string", "enum": ["1"]},
    },
    "X-Ocabra-KV-Capacity-Tokens": {
        "description": (
            "Logical unified-pool capacity used by oCabra admission. Present "
            "alongside ``X-Ocabra-KV-Queue-Saturated``."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-KV-Reserved-Tokens": {
        "description": (
            "Logical KV tokens already reserved when the request arrived. "
            "This is an oCabra estimate, not observed llama.cpp cell usage."
        ),
        "schema": {"type": "integer"},
    },
}


#: Shared by every 503/429 ``responses`` entry below — ``Retry-After`` is
#: only present on genuinely transient failures (worker still loading,
#: registry not ready yet, circuit breaker, at-capacity). Permanent/config
#: failures (bad router target, federation disabled, a model stuck in
#: ``ModelStatus.ERROR``) return the same status codes without this header.
RETRY_AFTER_HEADER_DOC: dict = {
    "Retry-After": {
        "description": (
            "Seconds to wait before retrying. Present only when the "
            "failure is transient (model/worker loading, registry warming "
            "up post-restart, circuit breaker cooldown, at-capacity). "
            "Absent for permanent/config failures — retrying won't help."
        ),
        "schema": {"type": "integer"},
    },
}

#: Reused across every OpenAI-compatible endpoint that loads a model
#: on-demand — 503 covers "model/worker not available yet" (transient, see
#: ``RETRY_AFTER_HEADER_DOC``), 429 covers the per-worker concurrency cap.
#: 409/504 (2026-10-31) cover failures from the request-dispatch path
#: itself (``raise_inference_dispatch_error``) — previously undocumented
#: here even though the codes (``insufficient_vram``, ``generation_timeout``,
#: ``worker_degraded``) already existed and were used by the streaming
#: in-band error path (``stream_error_details``).
LOAD_FAILURE_RESPONSES_DOC: dict = {
    409: {
        "description": (
            "Not enough free VRAM to load the model even after waiting "
            "for scheduler eviction (``insufficient_vram``). Permanent "
            "given the server's current state — no ``Retry-After``."
        ),
    },
    429: {
        "description": (
            "The resolved worker is already at its concurrency cap "
            "(``model_at_capacity``) — admission control rejected this "
            "request rather than queue it indefinitely."
        ),
        "headers": RETRY_AFTER_HEADER_DOC,
    },
    503: {
        "description": (
            "Model/worker not available — still loading, failed to load "
            "(``model_load_failed``), an internal registry isn't ready yet "
            "post-restart, or the per-worker circuit breaker is open after "
            "repeated failures (``worker_degraded``). See ``Retry-After`` "
            "below for whether retrying makes sense."
        ),
        "headers": RETRY_AFTER_HEADER_DOC,
    },
    504: {
        "description": (
            "The worker accepted the request but didn't respond within "
            "oCabra's inference timeout (``generation_timeout``) — "
            "transient, see ``Retry-After``."
        ),
        "headers": RETRY_AFTER_HEADER_DOC,
    },
}


STREAMING_LOAD_RESPONSE_DOC: dict = {
    200: {
        "description": (
            "Successful response. When ``stream=true`` the body is "
            "``text/event-stream``; oCabra pre-flushes the ``X-Ocabra-*`` "
            "headers below **before** triggering any model load, and "
            "interleaves two oCabra-aware SSE **comment** frames carrying "
            "load progress:\n\n"
            '``: {"event":"ocabra.model_loading","model_id":...,'
            '"worker_key":...,"status":...,'
            '"expected_wait_seconds":...}\\n\\n`` on stream open, and\n\n'
            '``: {"event":"ocabra.model_ready","model_id":...,'
            '"worker_key":...,"load_duration_ms":...,'
            '"was_cold_start":...}\\n\\n`` once the model is ready.\n\n'
            "A third, unrelated to cold-start, is emitted when the model "
            "is already loaded but every fair-share concurrency slot is "
            "occupied (see ``X-Ocabra-Slot-Queue-Saturated`` below for the "
            "non-streaming equivalent and why there's no queue position):\n\n"
            '``: {"event":"ocabra.slot_queued","model_id":...,'
            '"in_flight":...,"slots":...}\\n\\n``\n\n'
            "The leading ``:`` makes these SSE **comments**, ignored by "
            "strict OpenAI SDK parsers. oCabra-aware clients may parse the "
            "JSON after the colon. If the load fails after headers have "
            'been flushed, the error travels as ``data: {"error": ...}`` '
            "in the same stream (not as an HTTP status)."
        ),
        "headers": OCABRA_MODEL_HEADERS,
    },
    **LOAD_FAILURE_RESPONSES_DOC,
}


NON_STREAMING_LOAD_RESPONSE_DOC: dict = {
    200: {
        "description": (
            "Successful response. Non-streaming endpoints attach the "
            "``X-Ocabra-*`` headers below to the final response — clients "
            "then learn retrospectively whether the request paid for a cold "
            "start (``X-Ocabra-Was-Cold-Start`` + ``-Load-Duration-Ms``) "
            "and whether the server is under load (``-Load-Queue-Depth`` / "
            "``-Load-Active``)."
        ),
        "headers": OCABRA_MODEL_HEADERS,
    },
    **LOAD_FAILURE_RESPONSES_DOC,
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
    *,
    pre_status: str | None = None,
    elapsed_ms: int | None = None,
    via_router_profile_id: str | None = None,
    router_target_profile_id: str | None = None,
) -> dict[str, str]:
    """Return the canonical ``X-Ocabra-*`` header set for any request.

    Base fields (always emitted when the state is known):
      - ``X-Ocabra-Model-Status`` — worker state at header-flush time.
      - ``X-Ocabra-Model-Id`` — canonical model id after router resolution.

    Pressure hints (emitted only when the worker isn't yet ``loaded``):
      - ``X-Ocabra-Expected-Wait-Seconds`` — median of the last 5 loads.
      - ``X-Ocabra-Load-Queue-Depth`` / ``-Load-Active`` — how many other
        loads are pending/running behind the backend gate.

    Cold-start flags (post-load; require ``pre_status`` from *before* the
    load call):
      - ``X-Ocabra-Was-Cold-Start`` — ``"1"`` when the client paid for a
        load. Omitted when the model was already warm.
      - ``X-Ocabra-Load-Duration-Ms`` — how long the request had to wait
        for the model to be ready. Requires ``elapsed_ms``.

    Router attribution (chat/completions routers):
      - ``X-Ocabra-Router`` — the router profile the client asked for.
      - ``X-Ocabra-Router-Target`` — the target profile that actually ran.

    Streaming endpoints call this **before** the load with no cold-start /
    router kwargs; non-streaming ones call it **after** and pass
    ``pre_status``/``elapsed_ms`` so the response fully describes what
    happened.
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
    # Cross-request load pressure. Clients render this as "queued behind N
    # cold starts" so the wait doesn't feel silent.
    queue_waiters = int(getattr(model_manager, "_load_queue_waiters", 0) or 0)
    in_progress = len(getattr(model_manager, "_load_in_progress", set()) or ())
    if queue_waiters or in_progress:
        headers["X-Ocabra-Load-Queue-Depth"] = str(queue_waiters + in_progress)
        headers["X-Ocabra-Load-Active"] = str(in_progress)

    # Cold-start attribution — set only when the caller captured the state
    # before triggering the load. ``pre_status="loaded"`` is treated as
    # "warm request" and both fields are omitted.
    if pre_status is not None and pre_status != "loaded":
        headers["X-Ocabra-Was-Cold-Start"] = "1"
        if elapsed_ms is not None and elapsed_ms >= 0:
            headers["X-Ocabra-Load-Duration-Ms"] = str(int(elapsed_ms))

    # Router attribution — same shape as chat/completions streaming today.
    if via_router_profile_id:
        headers["X-Ocabra-Router"] = str(via_router_profile_id)
        if router_target_profile_id:
            headers["X-Ocabra-Router-Target"] = str(router_target_profile_id)
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
        if user is not None and not user.can_access_model(requested, profile.base_model_id):
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
    *,
    preferred_gpu: int | None = None,
    load_policy: str | None = None,
    required_tokens: int | None = None,
) -> ModelState:
    """Public wrapper around :func:`_ensure_worker_loaded`.

    Exposed for streaming endpoints that pre-flush headers and trigger the
    actual load from inside the SSE generator.
    """
    return await _ensure_worker_loaded(
        model_manager,
        base_model_id,
        worker_key,
        load_overrides,
        preferred_gpu=preferred_gpu,
        load_policy=load_policy,
        required_tokens=required_tokens,
    )


async def _ensure_worker_loaded(
    model_manager: ModelManager,
    base_model_id: str,
    worker_key: str,
    load_overrides: dict | None,
    *,
    preferred_gpu: int | None = None,
    load_policy: str | None = None,
    required_tokens: int | None = None,
) -> ModelState:
    """Ensure the worker identified by *worker_key* is loaded.

    When *worker_key* differs from *base_model_id* (non-empty overrides),
    we check if a virtual model entry already exists in ModelManager; if not,
    we create one by cloning the base model's state and applying
    ``load_overrides`` as extra config.

    Returns the :class:`ModelState` of the loaded worker.
    """

    async def _mark_runtime(state: ModelState) -> None:
        marker = getattr(model_manager, "mark_runtime_request", None)
        if marker is None:
            return
        result = marker(state.model_id, required_tokens=required_tokens)
        if inspect.isawaitable(result):
            await result

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
        state = await _do_ensure_loaded(model_manager, base_model_id)
        await _mark_runtime(state)
        return state

    # Clone from the current base configuration. Derived states are retained in
    # memory after unload, while the base configuration remains mutable. Always
    # resolve the base first so an unloaded derived worker cannot restart with a
    # stale option such as vision_mode or mmproj_model_id.
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

    # Dedicated worker: reuse an existing transient state, but refresh it when
    # no process is running. Mutating a loaded/loading state would make its
    # metadata disagree with the arguments of the live backend process.
    state = await model_manager.get_state(worker_key)
    if state is not None:
        from ocabra.core.model_manager import LoadPolicy, ModelStatus

        if state.status not in (ModelStatus.LOADED, ModelStatus.LOADING):
            state.extra_config = merged_extra
            state.preferred_gpu = (
                preferred_gpu if preferred_gpu is not None else base_state.preferred_gpu
            )
            state.load_policy = LoadPolicy(load_policy or base_state.load_policy.value)
            state.pin_permanent = base_state.pin_permanent
        state = await _do_ensure_loaded(model_manager, worker_key)
        await _mark_runtime(state)
        return state

    try:
        # Derived workers are in-memory only: the profile that owns the
        # overrides is already persisted in ``model_profiles`` and knows how
        # to reconstruct this state on demand, so a second ``model_configs``
        # row would just show up as a phantom "(override)" entry in the
        # model list. Restart drops these — that's fine, they get recreated
        # by the next request that resolves the profile.
        await model_manager.add_model(
            model_id=worker_key,
            backend_type=base_state.backend_type,
            display_name=f"{base_state.display_name} (override)",
            load_policy=load_policy or base_state.load_policy.value,
            pin_permanent=base_state.pin_permanent,
            preferred_gpu=(
                preferred_gpu if preferred_gpu is not None else base_state.preferred_gpu
            ),
            extra_config=merged_extra,
            persist=False,
            backend_model_id_override=base_state.backend_model_id,
        )
    except Exception:
        # May already exist from concurrent request
        pass

    state = await _do_ensure_loaded(model_manager, worker_key)
    await _mark_runtime(state)
    return state


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
        if touch is not None:
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
                retried_state = await _wait_for_service_gpu_and_retry_load(
                    model_manager, model_id, state
                )
                if retried_state is not None:
                    await _touch(model_id, datetime.now(UTC))
                    return retried_state
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
                headers={"Retry-After": "10"},
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
                headers={"Retry-After": "10"},
            )

    detail_suffix = ""
    if state and state.error_message:
        detail_suffix = f" detail: {state.error_message}"

    # A model stuck in ERROR needs a human to look at it (bad weights, repeated
    # crash, etc.) — retrying on a timer won't help, so skip the hint there.
    # Any other non-terminal status (CONFIGURED, UNLOADING, ...) is a normal
    # transitional state a short wait can resolve.
    retry_headers = None if state and state.status == ModelStatus.ERROR else {"Retry-After": "10"}
    raise _openai_error(
        (
            f"Model '{model_id}' is not available "
            f"(status: {state.status.value if state else 'unknown'}).{detail_suffix}"
        ),
        "server_error",
        code="model_unavailable",
        status_code=503,
        headers=retry_headers,
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
                retried_state = await _wait_for_service_gpu_and_retry_load(
                    model_manager, resolved_model_id, state
                )
                if retried_state is not None:
                    request_at = datetime.now(UTC)
                    await _touch_last_request_at(resolved_model_id, request_at)
                    return retried_state
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
                headers={"Retry-After": "10"},
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
            headers={"Retry-After": "10"},
        )

    detail_suffix = ""
    if state and state.error_message:
        detail_suffix = f" detail: {state.error_message}"

    # See the twin ensure_worker_loaded() site above for why ERROR skips the hint.
    retry_headers = None if state and state.status == ModelStatus.ERROR else {"Retry-After": "10"}
    raise _openai_error(
        (
            f"Model '{resolved_model_id}' is not available "
            f"(status: {state.status.value if state else 'unknown'}).{detail_suffix}"
        ),
        "server_error",
        code="model_unavailable",
        status_code=503,
        headers=retry_headers,
    )


def is_batch_origin_request(request: Request) -> bool:
    """True when this call was dispatched by the batch processor
    (``ocabra.core.batch_processor``), not a real client.

    The batch processor calls back into this same app over
    ``httpx.ASGITransport`` carrying the trusted ``X-Gateway-Token`` (the
    same mechanism ``get_current_user`` already uses to impersonate the
    batch owner, see ``_deps_auth.py``) plus a dedicated origin header.
    Only trust the origin header when the gateway token actually matches —
    an external caller can send any header it likes, but can't forge
    ``settings.gateway_service_token``.
    """
    from ocabra.config import settings

    gw_token = request.headers.get("X-Gateway-Token", "")
    if (
        not gw_token
        or not settings.gateway_service_token
        or gw_token != settings.gateway_service_token
    ):
        return False
    return request.headers.get("X-Internal-Request-Origin", "") == "batch"


def fairness_user_key(user: UserContext) -> str:
    """Identity key for the per-worker fair-share admission gate
    (``ocabra.core.fair_slot_gate``). Prefers the stable user_id; falls
    back to username, then a shared "anonymous" bucket when auth is
    disabled (``REQUIRE_API_KEY_*=false``) — every anonymous caller then
    fair-shares as a single "user", which is strictly better than no
    fairness at all but can't tell two anonymous callers apart.
    """
    return user.user_id or user.username or "anonymous"


def fairness_slots(state: ModelState) -> int:
    """How many requests this model's worker can run concurrently, for
    sizing the fair-share gate.

    Thin wrapper over ``resolve_worker_slots``, which is also what admission
    (``ModelManager.effective_inflight_cap``) and the dashboard card use. They
    were three separate implementations and two of them disagreed: admission
    assumed a single slot whenever the runtime-preset coordinator had not
    written ``_runtime.parallelism``, so a llama.cpp model with
    ``parallel_slots=4`` ran on a 4-slot gate while being admitted as if it had
    one. Unknown backends stay conservatively one-at-a-time.
    """
    from ocabra.core.model_manager_helpers import resolve_worker_slots

    return resolve_worker_slots(state) or 1


def fairness_kv_budget(
    state: ModelState,
    *,
    required_tokens: int | None,
    effective_context: int | None = None,
) -> tuple[int | None, int]:
    """Return ``(pool_capacity, request_reservation)`` for fair admission.

    KV weighting is deliberately opt-in at runtime: only llama.cpp workers
    that explicitly enabled the unified cache publish a capacity. Every other
    backend receives ``(None, 0)`` and keeps the historical slot-only gate.
    """
    if not is_llama_cpp_family(state.backend_type) or required_tokens is None:
        return None, 0

    from ocabra.core.model_manager_helpers import resolve_llama_cpp_option

    enabled = resolve_llama_cpp_option(state, "kv_unified", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None, 0

    runtime = state.extra_config.get("_runtime") if isinstance(state.extra_config, dict) else None
    raw_capacity = effective_context
    if raw_capacity is None and isinstance(runtime, dict):
        raw_capacity = runtime.get("effective_context")
    if raw_capacity is None:
        raw_capacity = resolve_llama_cpp_option(state, "ctx_size", None)
    if raw_capacity is None and isinstance(state.extra_config, dict):
        use_case = state.extra_config.get("use_case")
        if isinstance(use_case, dict):
            raw_capacity = use_case.get("context")
    try:
        capacity = int(raw_capacity or 0)
        # Legacy runtime presets express context per slot. Adaptive unified-KV
        # presets instead expose the aggregate physical pool so one request can
        # use all of it; those must not be multiplied by parallelism.
        if (
            isinstance(runtime, dict)
            and runtime.get("preset_key")
            and not runtime.get("context_is_pool")
        ):
            capacity *= max(1, int(runtime.get("parallelism") or 1))
        reservation = max(0, int(required_tokens))
    except (TypeError, ValueError):
        return None, 0
    if capacity <= 0:
        return None, 0
    return capacity, reservation


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


_REASONING_EFFORT_LEVELS = {"low", "medium", "high"}
_REASONING_EFFORT_OFF = {"off", "none", "disabled", "no", "false"}
_REASONING_EFFORT_DEFAULT = {"", "default"}
# 2026-11-05: a caller that never sets reasoning_effort/reasoning_budget/
# max_tokens can leave a reasoning model thinking indefinitely — confirmed
# live against gemma4:26b, a request with no cap at all generated 14k+
# tokens non-stop until oCabra's 1500s dispatch timeout killed it, three
# retries in a row. "medium" gives the model's own chat template a bounded
# hint instead of running with no guidance whatsoever. Hardcoded last-resort
# fallback if settings/per-model config ever holds something invalid.
DEFAULT_REASONING_EFFORT = "medium"


def _resolve_default_reasoning_effort(state: ModelState) -> str:
    """Pick the effort level to use when a caller doesn't set one.

    Precedence: the model's own ``extra_config.default_reasoning_effort``
    (set via the model edit UI) over the global ``GET/PATCH /ocabra/config``
    ``defaultReasoningEffort`` setting. Falls back to
    :data:`DEFAULT_REASONING_EFFORT` if neither resolves to a valid level.
    """
    from ocabra.config import settings

    extra_config = getattr(state, "extra_config", None)
    per_model = None
    if isinstance(extra_config, dict):
        per_model = extra_config.get("default_reasoning_effort") or extra_config.get(
            "defaultReasoningEffort"
        )
    candidate = str(per_model if per_model else settings.default_reasoning_effort).strip().lower()
    if candidate in _REASONING_EFFORT_LEVELS or candidate in _REASONING_EFFORT_OFF:
        return candidate
    return DEFAULT_REASONING_EFFORT


def _reasoning_budget_for_level(level: str) -> int | None:
    """Tope duro de tokens de pensamiento para un nivel de ``reasoning_effort``.

    ``None`` = ese nivel no lleva tope (``high`` por defecto: quien pide
    esfuerzo alto quiere justamente que piense sin prisa).
    """
    from ocabra.config import settings

    por_nivel = {
        "low": settings.reasoning_budget_low_tokens,
        "medium": settings.reasoning_budget_medium_tokens,
        "high": settings.reasoning_budget_high_tokens,
    }
    try:
        budget = int(por_nivel.get(level) or 0)
    except (TypeError, ValueError):
        return None
    return budget if budget > 0 else None


def _apply_reasoning_effort(state: ModelState, payload: dict) -> None:
    """Translate the OpenAI-style ``reasoning_effort`` field into whatever the
    target backend understands. The field is already in the ChatCompletionRequest
    schema; here we fan it out:

    - **vLLM**: forwarded as ``extra_body.chat_template_kwargs.reasoning_effort``
      (vLLM 0.9+ wires that through to the chat template; harmless on older
      versions because unknown kwargs are ignored).
    - **Ollama ≥0.5**: translated to the native ``think`` field
      (``True`` for low/medium/high — Ollama has a single on/off switch —
      and ``False`` for off/none/disabled).
    - **llama.cpp (native)**: the requested level is forwarded as
      ``reasoning_effort`` and paired with the per-request Jinja override
      ``chat_template_kwargs.enable_thinking``. The boolean is required when
      the worker was launched with ``--reasoning off``: a level alone does not
      override that launch default.
    - **Everything else**: silently dropped — no-op for backends with no
      native reasoning control.

    When the caller omits the field (or sends ``default``/``""``) and the
    model advertises ``capabilities.reasoning``, the effective default from
    :func:`_resolve_default_reasoning_effort` is substituted instead of
    leaving reasoning completely unbounded. A non-reasoning model (or an
    explicit ``off``/``low``/``high``) is untouched.

    The input ``reasoning_effort`` key is always popped from the forwarded
    payload so unknown backends don't 400 on it.
    """
    raw = payload.pop("reasoning_effort", None)
    value = "" if raw is None else str(raw).strip().lower()
    if value in _REASONING_EFFORT_DEFAULT:
        if not getattr(state.capabilities, "reasoning", False):
            return
        value = _resolve_default_reasoning_effort(state)
    if value in _REASONING_EFFORT_OFF:
        enabled = False
    elif value in _REASONING_EFFORT_LEVELS:
        enabled = True
    else:
        # Unknown value — leave the payload alone and let the backend 400.
        payload["reasoning_effort"] = raw
        return

    backend = getattr(state, "backend_type", "") or ""
    if backend == "vllm":
        extra = payload.get("extra_body")
        if not isinstance(extra, dict):
            extra = {}
        template_kwargs = extra.get("chat_template_kwargs")
        if not isinstance(template_kwargs, dict):
            template_kwargs = {}
        # Preserve the full "low/medium/high/off" granularity for vLLM — some
        # chat templates branch on the exact level, not just on/off.
        template_kwargs.setdefault("reasoning_effort", value if enabled else "off")
        extra["chat_template_kwargs"] = template_kwargs
        payload["extra_body"] = extra
    elif backend == "ollama":
        # Ollama 0.5+ only has a boolean switch; carry it through. When the
        # payload goes to ``/api/chat`` the ``think`` key sits at the top
        # level; when it goes to the OpenAI-compat ``/v1/chat/completions``
        # Ollama passes unknown top-level keys to the chat template so the
        # same key still works.
        payload.setdefault("think", enabled)
    elif is_llama_cpp_family(backend):
        template_kwargs = payload.get("chat_template_kwargs")
        template_kwargs = dict(template_kwargs) if isinstance(template_kwargs, dict) else {}
        template_kwargs["enable_thinking"] = enabled
        payload["chat_template_kwargs"] = template_kwargs
        payload["reasoning_effort"] = value if enabled else "none"
        if enabled:
            # Chaos starts with reasoning_format=none so ordinary requests do
            # not expose empty think tags. An explicit reasoning request wants
            # the standard split field consumed by OpenAI-compatible clients.
            payload.setdefault("reasoning_format", "deepseek")
            # El nivel, por si solo, no limita nada: ni la plantilla de gemma4
            # menciona ``reasoning_effort`` (verificado leyendo la plantilla que
            # publica el propio llama-server en /props), asi que "low",
            # "medium" y "high" daban resultados identicos. Se traduce a
            # ``reasoning_budget_tokens``, que es el unico tope duro: al
            # agotarse, llama.cpp corta el pensamiento y fuerza la respuesta.
            # Sin esto, una peticion sin max_tokens podia pensar hasta agotar
            # la salida (incidente 2026-11-07: 132.263 tokens en 26 minutos).
            # Un presupuesto explicito del cliente manda sobre el derivado.
            if "reasoning_budget_tokens" not in payload and "reasoning_budget" not in payload:
                derivado = _reasoning_budget_for_level(value)
                if derivado is not None:
                    payload["reasoning_budget_tokens"] = derivado


def _apply_reasoning_budget(state: ModelState, payload: dict) -> None:
    """Translate oCabra's ``reasoning_budget`` extension into llama-server's
    native per-request field — a hard token cap on thinking — when
    exhausted, llama.cpp cuts reasoning short and forces the final answer,
    unlike ``reasoning_effort`` which is only a soft hint the chat template
    may ignore).

    2026-10-29: added after a production incident where gemma4 (llama_cpp,
    no budget cap at all — ``--reasoning-budget`` defaults to -1,
    unrestricted) reasoned for 28k+ tokens on an uncapped request and burned
    through the entire context preset before writing an answer, leaving
    ``content`` empty. No native equivalent on other backends — silently
    dropped there, same pattern as ``reasoning_effort``.

    The wire key is ``reasoning_budget_tokens``, NOT ``reasoning_budget`` —
    confirmed by reading the strings in ``libllama-server-impl.so`` (the
    CLI flag ``--reasoning-budget`` doesn't map 1:1 to its own JSON key; the
    binary groups it with ``thinking_budget_tokens``,
    ``reasoning_budget_start_tag`` etc., all suffixed ``_tokens``/``_tag``).
    An initial implementation forwarded the CLI-flag-shaped name
    (``reasoning_budget``) directly, which the server silently ignored — a
    request with budget=200 still reasoned for thousands of tokens. Fixed
    and verified live the same day: budget=200 against qwen3.8:27b produced
    674 chars of reasoning and a clean ``finish_reason: "stop"`` (vs. ~16.6k
    chars / several minutes unconstrained).
    """
    raw = payload.pop("reasoning_budget", None)
    if raw is None:
        return
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        return
    if is_llama_cpp_family(getattr(state, "backend_type", "")):
        payload.setdefault("reasoning_budget_tokens", budget)


def _normalize_tool_call_entry(tool_call: object) -> object:
    """Backfill the OpenAI-required ``type``/stringified-``arguments`` shape
    on a ``tool_calls`` entry before forwarding it to a strict backend.

    Mirrors ``ocabra.api.ollama.chat._normalize_tool_call_for_backend`` —
    same fix, same root cause (llama-server's OpenAI-compatible endpoint
    rejects outright with "Missing tool call type" if a historical
    ``tool_calls`` entry omits ``type`` or has ``arguments`` as a parsed
    object instead of a JSON string), applied here defensively for
    OpenAI-compatible callers too: found live 2026-10-30 via an external
    agent hitting /v1/chat/completions directly against
    ``llama_cpp/gemma4-26b`` with exactly this shape.
    """
    if not isinstance(tool_call, dict):
        return tool_call
    out = dict(tool_call)
    out.setdefault("type", "function")
    function = out.get("function")
    if isinstance(function, dict):
        function = dict(function)
        if isinstance(function.get("arguments"), dict):
            function["arguments"] = json.dumps(function["arguments"])
        out["function"] = function
    return out


def _normalize_messages_for_backend(messages: object) -> object:
    if not isinstance(messages, list):
        return messages
    out = []
    for raw in messages:
        if not isinstance(raw, dict) or not isinstance(raw.get("tool_calls"), list):
            out.append(raw)
            continue
        msg = dict(raw)
        msg["tool_calls"] = [_normalize_tool_call_entry(tc) for tc in msg["tool_calls"]]
        out.append(msg)
    return out


def to_backend_body(state: ModelState, body: dict) -> dict:
    """Copy request payload and normalize model field to backend-native model id."""
    payload = dict(body)
    # oCabra request extensions are consumed by the coordinator and must not
    # leak into strict OpenAI-compatible upstreams.
    payload.pop("ocabra", None)
    payload["model"] = state.backend_model_id
    if "messages" in payload:
        payload["messages"] = _normalize_messages_for_backend(payload["messages"])
    if payload.get("stream") is True:
        stream_options = payload.get("stream_options")
        if isinstance(stream_options, dict):
            payload["stream_options"] = {**stream_options, "include_usage": True}
        else:
            payload["stream_options"] = {"include_usage": True}
    _apply_reasoning_effort(state, payload)
    _apply_reasoning_budget(state, payload)
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


def raise_inference_dispatch_error(exc: Exception) -> None:
    """Translate ``InferenceTimeoutError``/``WorkerDegradedError`` from
    ``WorkerPool.forward_request`` into the same stable codes the streaming
    path already produces via ``stream_error_details`` — neither is an
    ``httpx.HTTPStatusError``, so a bare ``except httpx.HTTPStatusError``
    around the non-streaming ``forward_request`` call lets both escape
    uncaught into main.py's generic ``Exception`` handler: a 500 with
    ``code`` set to the raw Python exception class name, no ``Retry-After``,
    and no match to any documented status (confirmed 2026-10-31 — both are
    transient and should look like it).
    """
    from ocabra.core.fair_slot_gate import KVCapacityError
    from ocabra.core.worker_pool import InferenceTimeoutError, WorkerDegradedError

    if isinstance(exc, KVCapacityError):
        raise _openai_error(
            str(exc),
            "invalid_request_error",
            code="context_length_exceeded",
            status_code=400,
        ) from exc

    if isinstance(exc, InferenceTimeoutError):
        raise _openai_error(
            str(exc),
            "server_error",
            code="generation_timeout",
            status_code=504,
            headers={"Retry-After": "5"},
        ) from exc
    if isinstance(exc, WorkerDegradedError):
        from ocabra.config import settings

        raise _openai_error(
            str(exc),
            "server_error",
            code="worker_degraded",
            status_code=503,
            headers={"Retry-After": str(settings.worker_circuit_breaker_cooldown_s)},
        ) from exc
    raise exc


def stream_error_details(exc: Exception) -> tuple[str, str]:
    """Return a client-safe message and stable code for an in-band SSE error.

    Streaming response headers have already been sent when worker forwarding
    fails, so the upstream HTTP status cannot be returned directly. Preserve
    the worker's JSON error message in the SSE envelope and encode its status
    class in a stable oCabra code for request statistics.
    """
    from ocabra.core.fair_slot_gate import KVCapacityError
    from ocabra.core.worker_pool import InferenceTimeoutError, WorkerDegradedError

    if isinstance(exc, KVCapacityError):
        return str(exc), "context_length_exceeded"

    if isinstance(exc, InferenceTimeoutError):
        return str(exc), "generation_timeout"

    if isinstance(exc, WorkerDegradedError):
        return str(exc), "worker_degraded"

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
