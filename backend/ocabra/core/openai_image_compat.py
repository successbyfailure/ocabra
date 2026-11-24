"""OpenAI image model-name compatibility shim.

A client that only knows OpenAI's official image model names
(``gpt-image-1``, ``gpt-image-1-mini``, ``dall-e-3``, ``dall-e-2``) cannot send
any oCabra-specific ``model``. This registry maps each of those names to one
of oCabra's own profiles (execution or router), consulted by
``api/openai/images.py`` before the normal profile resolution. Same design as
``openai_tts_compat`` (keyed by model name alone; images have no voice).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from ocabra.core._openai_compat import OpenAICompatRegistry
from ocabra.db.model_config import OpenAIImageModelMap

# What OpenAI does when ``model`` is omitted on /v1/images/*.
OPENAI_DEFAULT_IMAGE_MODEL = "dall-e-2"


class OpenAIImageCompatRegistry(OpenAICompatRegistry[str, OpenAIImageModelMap]):
    """In-memory cache of ``openai_image_model_map`` rows, backed by PostgreSQL."""

    _model = OpenAIImageModelMap
    _log_prefix = "openai_image_compat"

    def _row_key(self, row: OpenAIImageModelMap) -> str:
        return row.openai_model

    def _pk_filter(self, key: str) -> list[ColumnElement[bool]]:
        return [OpenAIImageModelMap.openai_model == key]

    def resolve(self, openai_model: str | None) -> str | None:
        """Target profile_id for this OpenAI name, or None without a pin.

        An empty ``model`` is looked up as ``dall-e-2``, OpenAI's own default,
        so a client that omits it still lands somewhere once that pin exists.
        """
        entry = self._entries.get(openai_model or OPENAI_DEFAULT_IMAGE_MODEL)
        return entry.target_profile_id if entry is not None else None

    async def upsert(
        self, session: AsyncSession, *, openai_model: str, target_profile_id: str
    ) -> OpenAIImageModelMap:
        if not openai_model:
            raise ValueError("openai_model is required.")
        return await self._upsert_key(
            session,
            openai_model,
            target_profile_id,
            lambda: OpenAIImageModelMap(
                openai_model=openai_model, target_profile_id=target_profile_id
            ),
            openai_model=openai_model,
        )

    async def delete(self, session: AsyncSession, *, openai_model: str) -> bool:
        return await self._delete_key(session, openai_model, openai_model=openai_model)
