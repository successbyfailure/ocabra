"""Router resolver — picks the best target for a router profile.

Bloque 20 — Etapa 6.

A router profile has a ``routing_targets`` list (ordered ``profile_id``
strings). When a client asks for the router, the resolver walks that list
and returns the first candidate that satisfies the current constraints:

    1. Already LOADED, not saturated, not held by a live Realtime session.
    2. UNLOADED but loadable without disrupting a busy neighbour.
    3. The estimator says it's still cheaper than the next candidate.

Every decision produces a ``reason`` string that lands both in the
audit-trail sent to the client (via the attribution banner in Playground)
and in ``request_stats.via_router_profile_id`` for the Stats/Routing view.

Never blocks a load: the resolver returns a ResolvedTarget and the
caller runs its normal ``_do_ensure_loaded`` path on that target. If no
target passes the checks the resolver still returns something (the first
enabled non-loop candidate) so the caller experiences the same failure
mode as a plain profile — nothing new to handle upstream.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from ocabra.config import settings


# Contextvar-based attribution so the stats collector can pick up the
# router's profile_id without threading a scoped parameter through every
# handler. Set inside ``resolve_profile`` when the router picks a target;
# read at the far end of the middleware once the response is fully out.
current_router_profile_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ocabra_current_router_profile_id",
    default=None,
)

if TYPE_CHECKING:
    from ocabra.core.duration_estimator import DurationEstimator
    from ocabra.core.model_manager import ModelManager, ModelState
    from ocabra.core.profile_registry import ModelProfile, ProfileRegistry
    from ocabra.core.session_registry import SessionRegistry

logger = structlog.get_logger(__name__)


@dataclass
class CandidateAudit:
    """One step of the resolution walk — for the audit trail."""

    target_profile_id: str
    outcome: str    # "loaded" | "loadable" | "session_veto" | "estimator_slow" |
                    # "load_would_disrupt" | "not_found" | "disabled" | "loop"


@dataclass
class ResolvedTarget:
    """Router's final decision.

    ``base_model_id`` is the canonical model_id the caller will actually
    load through ``_do_ensure_loaded``. ``target_profile_id`` is the
    profile that wraps that model — that's what the router logically
    chose, and what shows up in the client-facing attribution banner.
    """

    target_profile_id: str
    base_model_id: str
    load_overrides: dict | None
    reason: str
    considered: list[CandidateAudit] = field(default_factory=list)


class RoutingLoopError(RuntimeError):
    """Raised when a router points back at itself (directly or through a chain).

    Configuration error; the resolver logs the cycle and 500s at the caller
    so the operator notices and fixes the setup instead of silently getting
    weird behaviour.
    """


class RouterResolver:
    """Stateless router — one instance per app, injected everywhere."""

    def __init__(
        self,
        *,
        profile_registry: "ProfileRegistry",
        model_manager: "ModelManager",
        duration_estimator: "DurationEstimator | None" = None,
        session_registry: "SessionRegistry | None" = None,
    ) -> None:
        self._profile_registry = profile_registry
        self._model_manager = model_manager
        self._duration_estimator = duration_estimator
        self._session_registry = session_registry

    # ── Public API ────────────────────────────────────────────

    def is_router(self, profile: "ModelProfile") -> bool:
        """A profile is a router when it declares ``routing_targets``.

        The feature flag ``routing_enabled`` short-circuits this — flipping
        it off makes the whole system behave as if no router existed
        (routers fall back to their own base_model_id).
        """
        if not settings.routing_enabled:
            return False
        targets = getattr(profile, "routing_targets", None)
        return isinstance(targets, list) and len(targets) > 0

    async def pick(
        self,
        router: "ModelProfile",
        *,
        request_body: dict | None = None,
    ) -> ResolvedTarget:
        """Walk the router's ``routing_targets`` and pick the best one now.

        ``request_body`` is used to feed the estimator (input_tokens,
        work_size_meta) so cost comparisons between candidates account for
        actual request size. Missing/malformed body → the estimator falls
        back to the raw percentile with its own confidence.
        """
        from ocabra.core.model_manager import ModelStatus

        audit: list[CandidateAudit] = []
        # Loop prevention: routers can point at other routers, but a cycle
        # (A→B→A) means someone typed a bad config — surface as a hard
        # error, not silent recursion.
        visited: set[str] = {router.profile_id}

        target_ids = list(router.routing_targets or [])
        # Fall back to the router's own base_model_id at the very end so a
        # router whose targets all fail still behaves like a plain profile.
        # We wrap it as a virtual candidate: the caller looks it up via the
        # profile itself (no separate profile lookup needed).
        last_resort: ResolvedTarget | None = None

        input_tokens = None
        max_tokens = None
        work_size_meta = None
        if isinstance(request_body, dict):
            input_tokens = request_body.get("input_tokens")
            max_tokens = request_body.get("max_tokens")
            work_size_meta = request_body.get("work_size_meta")

        for target_id in target_ids:
            if target_id in visited:
                audit.append(CandidateAudit(target_id, "loop"))
                logger.warning(
                    "router_loop_detected",
                    router=router.profile_id,
                    target=target_id,
                    chain=list(visited),
                )
                continue
            visited.add(target_id)

            target = await self._profile_registry.get(target_id)
            if target is None:
                audit.append(CandidateAudit(target_id, "not_found"))
                continue
            if not target.enabled:
                audit.append(CandidateAudit(target_id, "disabled"))
                continue

            # Nested router? Resolve it recursively — the visited set
            # carries the cycle guard across levels.
            if self.is_router(target):
                try:
                    nested = await self.pick(target, request_body=request_body)
                except RoutingLoopError:
                    audit.append(CandidateAudit(target_id, "loop"))
                    continue
                if nested.reason != "no_immediate_winner":
                    audit.append(CandidateAudit(target_id, f"nested:{nested.reason}"))
                    return ResolvedTarget(
                        target_profile_id=nested.target_profile_id,
                        base_model_id=nested.base_model_id,
                        load_overrides=nested.load_overrides,
                        reason=nested.reason,
                        considered=audit + nested.considered,
                    )

            state = await self._model_manager.get_state(target.base_model_id)
            worker_key = target.base_model_id  # profile_id-based worker_keys land here in Etapa 6+

            # 1) Loaded, unsaturated, unreserved → hard win.
            if state is not None and state.status == ModelStatus.LOADED:
                if self._session_registry is not None and self._session_registry.has_active_session_holding(
                    worker_key
                ):
                    audit.append(CandidateAudit(target_id, "session_veto"))
                    continue
                if self._model_manager.is_busy(worker_key):
                    # Busy is soft: only skip when the estimator thinks the
                    # remaining time is longer than a cold-start fallback
                    # elsewhere. Otherwise wait for the neighbour and reuse
                    # the hot cache.
                    if self._should_skip_for_busy(worker_key, target, request_body):
                        audit.append(CandidateAudit(target_id, "load_would_disrupt"))
                        continue
                audit.append(CandidateAudit(target_id, "loaded"))
                return ResolvedTarget(
                    target_profile_id=target.profile_id,
                    base_model_id=target.base_model_id,
                    load_overrides=target.load_overrides,
                    reason="primary_loaded",
                    considered=audit,
                )

            # 2) Reserved by session even though unloaded — shouldn't happen
            # in practice (unload releases the registry entry) but guard
            # anyway so we don't accidentally hit a stale hold.
            if self._session_registry is not None and self._session_registry.has_active_session_holding(
                worker_key
            ):
                audit.append(CandidateAudit(target_id, "session_veto"))
                continue

            # 3) Unloaded / configured / error but loadable without evicting
            # a busy neighbour → accept, let the ensure-loaded path handle
            # the actual load.
            audit.append(CandidateAudit(target_id, "loadable"))
            candidate = ResolvedTarget(
                target_profile_id=target.profile_id,
                base_model_id=target.base_model_id,
                load_overrides=target.load_overrides,
                reason="loadable_no_disruption",
                considered=list(audit),
            )
            # Only keep the FIRST loadable candidate as the winner — the
            # ordering of ``routing_targets`` expresses the operator's
            # preference among functionally equivalent fallbacks.
            if last_resort is None:
                last_resort = candidate
                return candidate

        # Nothing survived — fall back to the router's own base_model_id.
        # This is what a plain profile would do, so upstream behaviour
        # matches the pre-router world.
        if last_resort is not None:
            return last_resort
        return ResolvedTarget(
            target_profile_id=router.profile_id,
            base_model_id=router.base_model_id,
            load_overrides=router.load_overrides,
            reason="no_immediate_winner",
            considered=audit,
        )

    # ── Internals ────────────────────────────────────────────

    def _should_skip_for_busy(
        self,
        worker_key: str,
        target: "ModelProfile",
        request_body: dict | None,
    ) -> bool:
        """Estimator-based tie-break for busy candidates.

        If the estimator has low confidence, we conservatively wait (the
        cache is hot). If it says "I know this neighbour will take longer
        than a cold-start fallback", we skip.
        """
        if self._duration_estimator is None:
            return False
        remaining = self._duration_estimator.in_flight_remaining(
            worker_key, model_id=target.base_model_id
        )
        if remaining is None:
            return False
        # Fall back only if the remaining time is more than a full minute —
        # smaller windows are cheaper to wait out than to pay a cold start
        # on another candidate.
        return remaining > 60.0


__all__ = [
    "CandidateAudit",
    "ResolvedTarget",
    "RouterResolver",
    "RoutingLoopError",
]
