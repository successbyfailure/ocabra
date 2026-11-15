"""OpenAI image model-name compatibility shim.

A client that only knows OpenAI's official image model names
(``gpt-image-1``, ``gpt-image-1-mini``, ``dall-e-3``, ``dall-e-2``) cannot send
any oCabra-specific ``model``. This registry maps each of those names to one
of oCabra's own profiles (execution or router), consulted by
``api/openai/images.py`` before the normal profile resolution. Same design as
``openai_tts_compat`` (keyed by model name alone; images have no voice).
"""

from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ocabra.db.model_config import ModelProfile, OpenAIImageModelMap

logger = structlog.get_logger(__name__)

# What OpenAI does when ``model`` is omitted on /v1/images/*.
OPENAI_DEFAULT_IMAGE_MODEL = "dall-e-2"


class OpenAIImageCompatRegistry:
    """In-memory cache of ``openai_image_model_map`` rows, backed by PostgreSQL."""

    def __init__(self) -> None:
        self._entries: dict[str, OpenAIImageModelMap] = {}

    async def load_all(self, session: AsyncSession) -> None:
        result = await session.execute(select(OpenAIImageModelMap))
        self._entries = {row.openai_model: row for row in result.scalars().all()}
        logger.info("openai_image_compat_loaded", count=len(self._entries))

    def resolve(self, openai_model: str | None) -> str | None:
        """Target profile_id for this OpenAI name, or None without a pin.

        An empty ``model`` is looked up as ``dall-e-2``, OpenAI's own default,
        so a client that omits it still lands somewhere once that pin exists.
        """
        entry = self._entries.get(openai_model or OPENAI_DEFAULT_IMAGE_MODEL)
        return entry.target_profile_id if entry is not None else None

    def list_all(self) -> list[OpenAIImageModelMap]:
        return sorted(self._entries.values(), key=lambda e: e.openai_model)

    async def upsert(
        self, session: AsyncSession, *, openai_model: str, target_profile_id: str
    ) -> OpenAIImageModelMap:
        if not openai_model:
            raise ValueError("openai_model is required.")
        result = await session.execute(
            select(ModelProfile.profile_id, ModelProfile.enabled).where(
                ModelProfile.profile_id == target_profile_id
            )
        )
        row = result.first()
        if row is None:
            raise ValueError(f"target_profile_id '{target_profile_id}' not found.")
        if not row.enabled:
            raise ValueError(f"target_profile_id '{target_profile_id}' is disabled.")

        db_entry = (
            await session.execute(
                select(OpenAIImageModelMap).where(OpenAIImageModelMap.openai_model == openai_model)
            )
        ).scalar_one_or_none()
        if db_entry is None:
            db_entry = OpenAIImageModelMap(
                openai_model=openai_model, target_profile_id=target_profile_id
            )
            session.add(db_entry)
        else:
            db_entry.target_profile_id = target_profile_id
        await session.commit()
        await session.refresh(db_entry)
        self._entries[openai_model] = db_entry
        logger.info(
            "openai_image_compat_upserted",
            openai_model=openai_model,
            target_profile_id=target_profile_id,
        )
        return db_entry

    async def delete(self, session: AsyncSession, *, openai_model: str) -> bool:
        if openai_model not in self._entries:
            return False
        db_entry = (
            await session.execute(
                select(OpenAIImageModelMap).where(OpenAIImageModelMap.openai_model == openai_model)
            )
        ).scalar_one_or_none()
        if db_entry is not None:
            await session.delete(db_entry)
            await session.commit()
        del self._entries[openai_model]
        logger.info("openai_image_compat_deleted", openai_model=openai_model)
        return True
