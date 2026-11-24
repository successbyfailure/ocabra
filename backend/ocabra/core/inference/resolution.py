"""Model/profile/router resolution for the request path.

Resolves the id a client sent (canonical id, backend alias, profile id or
router profile) into the concrete ``(ModelProfile, ModelState)`` pair that
will serve the request, choosing the runtime preset and triggering the load
along the way. Moved out of ``api`` (REF-09) so ``core`` callers — video
jobs, the realtime session — no longer import from ``api``; the ``api`` layer
re-exports these via ``_deps``.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import structlog

from ocabra.core.inference.errors import (
    _openai_error,
    engine_suffixed_error,
    router_pick_error,
)
from ocabra.core.inference.loading import (
    _do_ensure_loaded,
    _ensure_worker_loaded,
    _snapshot_pre_load,
)
from ocabra.core.runtime_coordinator import (
    OutputTokenLimitError,
    RuntimePlan,
    plan_profile_runtime,
)
from ocabra.core.runtime_presets import ContextCapacityError

if TYPE_CHECKING:
    from ocabra.api._deps_auth import UserContext
    from ocabra.core.model_manager import ModelManager, ModelState
    from ocabra.core.profile_registry import ProfileRegistry
    from ocabra.db.model_config import ModelProfile

logger = structlog.get_logger(__name__)


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
        # A model key or a backend reference can match several deployments
        # (``gemma4:26b`` is the Ollama tag of ``ollama/gemma4:26b`` *and* the
        # backend reference of ``llamacpp_mks/gemma4:26b``). Taking the first
        # one made the answer depend on dict order; prefer the deployment that
        # serves the model's public name, then a model-key match.
        states = await model_manager.list_states()
        resolved_id = requested
        resolved_state = None
        candidates = [
            s
            for s in states
            if s.backend_model_id == requested or getattr(s, "model_key", None) == requested
        ]
        if candidates:
            is_default = getattr(model_manager, "is_default_deployment", None)

            def _rank(s) -> tuple[int, int]:
                default_rank = 0 if (is_default is None or is_default(s)) else 1
                key_rank = 0 if getattr(s, "model_key", None) == requested else 1
                return (default_rank, key_rank)

            resolved_state = min(candidates, key=_rank)
            resolved_id = resolved_state.model_id

    if resolved_state is not None and user is not None:
        if not user.can_access_model(resolved_id):
            return resolved_id, None

    return resolved_id, resolved_state


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
    # The chosen worker had its context shrunk in place: grow it back before a
    # request that needs more than it holds now (~0.15-0.5 s), evicting what
    # took the room it gave up if the GPUs no longer have it.
    live_effective_context = plan.live_effective_context
    if (
        live_effective_context is not None
        and plan.required_tokens is not None
        and plan.required_tokens > live_effective_context
    ):
        try:
            await model_manager.grow_context_back(plan.worker_key)
        except Exception as exc:  # noqa: BLE001 - any failure leaves it small
            logger.warning(
                "live_context_regrow_failed",
                worker_key=plan.worker_key,
                required_tokens=plan.required_tokens,
                live_effective_context=live_effective_context,
                error=str(exc),
            )
            # The room went to an on-demand vision projector, which llama-server
            # cannot unload. A text request that needs that room gets the
            # worker reloaded without it (the load below brings it back at its
            # full context) instead of a 503 that the client retries forever:
            # 2026-11-16, ornith-1.5:35b answered 11 consecutive 503s after one
            # image shrank it from 229k to 143k.
            if await model_manager.release_vision_by_reload(plan.worker_key, request_body):
                live_effective_context = None
            else:
                raise _openai_error(
                    "The model is running with a reduced context and could not grow back "
                    f"to fit {plan.required_tokens} tokens right now; retry shortly.",
                    "server_error",
                    code="context_temporarily_reduced",
                    status_code=503,
                    headers={"Retry-After": "5"},
                ) from exc
        live_effective_context = None
    effective_context = plan.effective_context
    if live_effective_context is not None and effective_context is not None:
        effective_context = min(effective_context, live_effective_context)

    if request_state is not None:
        try:
            request_state.runtime_worker_key = plan.worker_key
            request_state.runtime_preset_key = plan.preset_key
            request_state.runtime_key = plan.runtime_key
            request_state.runtime_effective_context = effective_context
            request_state.runtime_required_tokens = plan.required_tokens
            request_state.runtime_kv_reservation_tokens = plan.kv_reservation_tokens
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
    required_capability: str | tuple[str, ...] | None = None,
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

    # 0. Names that pin the engine (``gemma4:26b-ollama``) are rejected: the
    # engine serving a model is a deployment decision that can change, and a
    # client holding such a name pins itself to the losing side of it. The
    # model name is the normal way in; ``<backend>/<modelo>`` is the way to
    # ask for one engine on purpose.
    suffixed = getattr(profile_registry, "engine_suffixed_name", None)
    alternativa = await suffixed(requested) if suffixed is not None else None
    if alternativa is not None:
        raise engine_suffixed_error(requested, *alternativa)

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
                resolved = await router_resolver.pick(
                    profile,
                    request_body=request_body,
                    required_capability=required_capability,
                    exclude_targets=getattr(request_state, "router_excluded_targets", None),
                )
            except NoRoutableTargetError as exc:
                raise router_pick_error(exc) from exc
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
                note_router_resolution(profile, target)
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


# (target profile id, router request_defaults) of the router that resolved this
# request. Resolution swaps the router for its target profile, so without this
# the router's own defaults were never applied to anything.
_ROUTER_DEFAULTS: ContextVar[tuple[str, dict] | None] = ContextVar(
    "ocabra_router_request_defaults", default=None
)


def note_router_resolution(router: ModelProfile, target: ModelProfile) -> None:
    """Remember the router's ``request_defaults`` for the target it chose."""
    defaults = router.request_defaults if isinstance(router.request_defaults, dict) else None
    if defaults and target.profile_id != router.profile_id:
        _ROUTER_DEFAULTS.set((target.profile_id, dict(defaults)))


def merge_profile_defaults(profile: ModelProfile, body: dict) -> dict:
    """Merge a profile's ``request_defaults`` and ``assets`` into a request body.

    Precedence, lowest first: the profile's ``request_defaults``, then those of
    the router the client named (when this profile is the target it chose),
    then the client body. Asset injection (e.g. ``voice_ref``) is applied
    *after* the merge and cannot be overridden by the client.
    """
    defaults = profile.request_defaults or {}
    routed = _ROUTER_DEFAULTS.get()
    router_defaults = routed[1] if routed and routed[0] == profile.profile_id else {}
    merged: dict[str, Any] = {**defaults, **router_defaults, **body}

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
    Ensure a model is LOADED before forwarding a request (legacy names).

    Resolves *model_id* the legacy way (filtered by *user*'s accessible models,
    as ``/v1/models`` does) and loads it with :func:`_do_ensure_loaded`.

    This used to be a second copy of that loader whose wait loop did not stop
    when the load failed: a load that failed in 2 s kept the request waiting
    the whole ``model_load_wait_timeout_s`` and then answered "did not finish
    loading in time" (BUG-16). Only the Ollama layer calls it.
    """
    resolved_model_id, state = await resolve_model(model_manager, model_id, user=user)
    if state is None:
        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )
    return await _do_ensure_loaded(model_manager, resolved_model_id)
