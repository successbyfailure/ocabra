"""Internal REST endpoints for the OpenAI TTS voice-name compatibility shim.

Independent of the router/``routing_targets`` system on purpose — see
``ocabra/core/openai_tts_compat.py`` and docs/CONTRACTS.md.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.core.openai_tts_compat import OpenAITTSCompatRegistry
from ocabra.database import AsyncSessionLocal
from ocabra.schemas.openai_tts_compat import OpenAITTSVoiceMapOut, OpenAITTSVoiceMapUpsert

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["openai-tts-compat"])


def _get_registry(request: Request) -> OpenAITTSCompatRegistry:
    registry = getattr(request.app.state, "openai_tts_compat", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI TTS compat registry not available",
            headers={"Retry-After": "5"},
        )
    return registry


@router.get(
    "/openai-tts-voice-map",
    summary="List OpenAI TTS voice-name compatibility pins",
    description=(
        "Every configured (openai_model, voice_name) -> target_profile_id "
        "pin. Empty when nothing is pinned — clients that only know "
        "OpenAI's official TTS model/voice names (tts-1, tts-1-hd, "
        "gpt-4o-mini-tts + alloy, echo...) get a plain 404 until you add "
        "one."
    ),
)
async def list_entries(
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict]:
    registry = _get_registry(request)
    return [
        OpenAITTSVoiceMapOut.model_validate(e).model_dump(mode="json") for e in registry.list_all()
    ]


@router.put(
    "/openai-tts-voice-map/{openai_model}/{voice_name}",
    summary="Pin an OpenAI (model, voice) pair to one of oCabra's own profiles",
    description=(
        "target_profile_id must be an existing, enabled oCabra profile — "
        "execution or router, no constraint either way. Creates the pin "
        "if it doesn't exist yet, overwrites it otherwise."
    ),
    responses={400: {"description": "target_profile_id missing, not found, or disabled"}},
)
async def upsert_entry(
    openai_model: str,
    voice_name: str,
    body: OpenAITTSVoiceMapUpsert,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        try:
            entry = await registry.upsert(
                session,
                openai_model=openai_model,
                voice_name=voice_name,
                target_profile_id=body.target_profile_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OpenAITTSVoiceMapOut.model_validate(entry).model_dump(mode="json")


@router.delete(
    "/openai-tts-voice-map/{openai_model}/{voice_name}",
    summary="Remove an OpenAI TTS voice-name compatibility pin",
    responses={404: {"description": "Pin not found"}},
)
async def delete_entry(
    openai_model: str,
    voice_name: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        deleted = await registry.delete(session, openai_model=openai_model, voice_name=voice_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Pin not found.")
    return {"ok": True}
