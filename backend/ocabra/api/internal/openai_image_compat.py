"""Internal REST endpoints for the OpenAI image model-name compatibility shim.

Same design as ``openai_tts_compat``: see ``ocabra/core/openai_image_compat.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.core.openai_image_compat import OpenAIImageCompatRegistry
from ocabra.database import AsyncSessionLocal
from ocabra.schemas.openai_tts_compat import OpenAIImageModelMapOut, OpenAIImageModelMapUpsert

router = APIRouter(tags=["openai-image-compat"])


def _get_registry(request: Request) -> OpenAIImageCompatRegistry:
    registry = getattr(request.app.state, "openai_image_compat", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI image compat registry not available",
            headers={"Retry-After": "5"},
        )
    return registry


@router.get(
    "/openai-image-model-map",
    summary="List OpenAI image model-name compatibility pins",
    description=(
        "Every configured openai_model -> target_profile_id pin. Clients that "
        "only know OpenAI's image model names (gpt-image-1, dall-e-3...) get a "
        "plain 404 for any name without one."
    ),
)
async def list_entries(
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict]:
    """List every pin.

    Returns:
        A list of ``{openai_model, target_profile_id, created_at, updated_at}``.
    """
    registry = _get_registry(request)
    return [
        OpenAIImageModelMapOut.model_validate(e).model_dump(mode="json")
        for e in registry.list_all()
    ]


@router.put(
    "/openai-image-model-map/{openai_model}",
    summary="Pin an OpenAI image model name to one of oCabra's own profiles",
    responses={400: {"description": "target_profile_id missing, not found, or disabled"}},
)
async def upsert_entry(
    openai_model: str,
    body: OpenAIImageModelMapUpsert,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Create or overwrite the pin for ``openai_model``.

    Args:
        openai_model: OpenAI name, e.g. ``gpt-image-1``.
        body: ``target_profile_id`` — an existing, enabled profile or router.

    Returns:
        The stored pin.
    """
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        try:
            entry = await registry.upsert(
                session, openai_model=openai_model, target_profile_id=body.target_profile_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OpenAIImageModelMapOut.model_validate(entry).model_dump(mode="json")


@router.delete(
    "/openai-image-model-map/{openai_model}",
    summary="Remove an OpenAI image model-name compatibility pin",
    responses={404: {"description": "Pin not found"}},
)
async def delete_entry(
    openai_model: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Remove the pin for ``openai_model`` (that name goes back to 404).

    Returns:
        ``{"ok": true}``.
    """
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        deleted = await registry.delete(session, openai_model=openai_model)
    if not deleted:
        raise HTTPException(status_code=404, detail="Pin not found.")
    return {"ok": True}
