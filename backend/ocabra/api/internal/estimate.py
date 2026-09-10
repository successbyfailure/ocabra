"""POST /ocabra/estimate — expose the DurationEstimator to clients.

Bloque 20 — Etapa 1.

Clients (Playground first, third-party integrations later) call this before
they send a real inference request to know roughly how long it will take.
Returns the estimate exactly as the internal consumers see it, so any
mismatch between "what the UI shows" and "what the router decided" is a bug
in the UI, not in the estimator.
"""

from __future__ import annotations

from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.core.duration_estimator import DurationEstimator

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["estimate"])


class EstimateRequest(BaseModel):
    model: str = Field(..., description="profile_id, base model_id, or backend/model form.")
    # The body of a would-be inference request. Fields we understand today:
    # - ``max_tokens`` for chat/completions/embeddings.
    # - anything else is currently ignored; family-specific estimators
    #   (Etapa 5) will consult ``audio_seconds``, ``frames``, ``steps``, ...
    body: dict = Field(default_factory=dict)


class EstimateResponse(BaseModel):
    expected_seconds: float
    p50_seconds: float
    p95_seconds: float
    confidence: float
    source: Literal["regression", "percentile", "family_default"]
    sample_count: int
    cold_start_seconds: float


@router.post(
    "/estimate",
    response_model=EstimateResponse,
    summary="Estimate how long an inference request will take",
    description=(
        "Returns the DurationEstimator's prediction for the given model + body. "
        "Consumers should look at ``confidence`` before trusting ``expected_seconds`` "
        "for irreversible decisions — below the ``router_confidence_floor`` the number "
        "is a rough hint and ``p95_seconds`` is a safer bound."
    ),
    responses={
        404: {"description": "Model or profile not found"},
        503: {"description": "Estimator not initialised yet"},
    },
)
async def estimate_request(
    body: EstimateRequest,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> EstimateResponse:
    estimator: DurationEstimator | None = getattr(
        request.app.state, "duration_estimator", None
    )
    if estimator is None:
        raise HTTPException(status_code=503, detail="Duration estimator not ready yet")

    # Resolve the model_id + currently_loaded state. We deliberately allow
    # both a profile_id and a canonical model_id (backend/name) so clients
    # don't have to know which they hold.
    profile_registry = getattr(request.app.state, "profile_registry", None)
    model_manager = request.app.state.model_manager

    # Two IDs matter here:
    #   * ``stats_key`` — the ``model_id`` recorded in request_stats. For
    #     profile-based traffic that's the profile_id (e.g. "gemma4:26b").
    #     For direct legacy calls it's the canonical form ("ollama/…"). The
    #     estimator's calibration table is keyed by whatever request_stats
    #     stored, so this is what we look up.
    #   * ``base_model_id`` — the canonical model behind the profile. Used
    #     to consult model_manager for the currently_loaded state, since the
    #     model_manager only knows canonical ids.
    stats_key: str | None = None
    base_model_id: str | None = None
    backend_type: str | None = None

    if profile_registry is not None:
        profile = await profile_registry.get(body.model)
        if profile is not None:
            stats_key = profile.profile_id
            base_model_id = profile.base_model_id

    if stats_key is None:
        # Legacy path: canonical model_id.
        state = await model_manager.get_state(body.model)
        if state is not None:
            stats_key = state.model_id
            base_model_id = state.model_id
            backend_type = state.backend_type

    if stats_key is None or base_model_id is None:
        raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found")

    state = await model_manager.get_state(base_model_id)
    currently_loaded = bool(
        state is not None and getattr(state.status, "value", "") == "loaded"
    )
    if backend_type is None and state is not None:
        backend_type = state.backend_type

    # Try to pull the chat-shape signals out of the body: ``max_tokens`` is
    # taken as-is; ``input_tokens`` is estimated from ``messages`` via a
    # 4-char-per-token heuristic if not provided explicitly. Cheap and good
    # enough — clients that care about precision can pass ``input_tokens``
    # directly.
    input_tokens = body.body.get("input_tokens")
    if input_tokens is None:
        messages = body.body.get("messages")
        if isinstance(messages, list):
            char_count = 0
            for msg in messages:
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, str):
                    char_count += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            char_count += len(part["text"])
            input_tokens = max(1, char_count // 4)
    max_tokens = body.body.get("max_tokens")

    result = await estimator.estimate(
        stats_key,
        backend_type=backend_type,
        currently_loaded=currently_loaded,
        input_tokens=int(input_tokens) if isinstance(input_tokens, (int, float)) else None,
        max_tokens=int(max_tokens) if isinstance(max_tokens, (int, float)) else None,
    )
    return EstimateResponse(**result.as_dict())
