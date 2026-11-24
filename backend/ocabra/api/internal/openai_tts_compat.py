"""Internal REST endpoints for the OpenAI TTS voice-name compatibility shim.

Independent of the router/``routing_targets`` system on purpose — see
``ocabra/core/openai_tts_compat.py`` and docs/CONTRACTS.md.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.api.internal._openai_compat import get_compat_registry
from ocabra.core.openai_tts_compat import OpenAITTSCompatRegistry
from ocabra.database import AsyncSessionLocal
from ocabra.schemas.openai_tts_compat import OpenAITTSVoiceMapOut, OpenAITTSVoiceMapUpsert

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["openai-tts-compat"])


def _get_registry(request: Request) -> OpenAITTSCompatRegistry:
    return get_compat_registry(request, attr="openai_tts_compat", label="TTS")


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
    """List every OpenAI TTS (model, voice) -> profile pin.

    Requires the ``model_manager`` role. Empty when nothing is pinned, in which
    case OpenAI's own names (``tts-1`` + ``alloy``...) get a 404 on
    ``/v1/audio/speech``.

    Returns:
        A list of ``OpenAITTSVoiceMapOut`` dicts.

    Raises:
        HTTPException 503: the compat registry is not initialised yet.
    """
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
    """Create or overwrite the pin of an OpenAI (model, voice) pair.

    Requires the ``model_manager`` role.

    Args:
        openai_model: OpenAI TTS model name (e.g. ``tts-1``).
        voice_name: OpenAI voice name (e.g. ``alloy``).
        body: ``OpenAITTSVoiceMapUpsert`` with ``target_profile_id`` (an existing,
            enabled execution or router profile).

    Returns:
        The stored ``OpenAITTSVoiceMapOut``.

    Raises:
        HTTPException 400: the target profile is missing, unknown or disabled.
        HTTPException 503: the compat registry is not initialised yet.
    """
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
    """Remove the pin of an OpenAI (model, voice) pair.

    Requires the ``model_manager`` role.

    Args:
        openai_model: OpenAI TTS model name.
        voice_name: OpenAI voice name.

    Returns:
        ``{"ok": True}``.

    Raises:
        HTTPException 404: no such pin.
        HTTPException 503: the compat registry is not initialised yet.
    """
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        deleted = await registry.delete(session, openai_model=openai_model, voice_name=voice_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Pin not found.")
    return {"ok": True}
