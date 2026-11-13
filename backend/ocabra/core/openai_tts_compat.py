"""OpenAI TTS voice-name compatibility shim.

Independent of the router/``routing_targets`` system (see
``docs/CONTRACTS.md`` and ``router_resolver.py``): a client that only knows
OpenAI's official TTS model names (``tts-1``, ``tts-1-hd``,
``gpt-4o-mini-tts``) and voice names (``alloy``, ``echo``...) has no way to
send anything oCabra-specific at all — not even a custom ``model``. This
registry lets an operator map each ``(openai_model, voice_name)`` pair
straight to one of oCabra's own profiles (execution or router, no
constraint either way), consulted by ``api/openai/audio.py`` before the
normal profile resolution.
"""

from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ocabra.db.model_config import ModelProfile, OpenAITTSVoiceMap

logger = structlog.get_logger(__name__)


class OpenAITTSCompatRegistry:
    """In-memory cache of ``openai_tts_voice_map`` rows, backed by PostgreSQL."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], OpenAITTSVoiceMap] = {}

    async def load_all(self, session: AsyncSession) -> None:
        result = await session.execute(select(OpenAITTSVoiceMap))
        rows = result.scalars().all()
        self._entries = {(row.openai_model, row.voice_name): row for row in rows}
        logger.info("openai_tts_compat_loaded", count=len(self._entries))

    def resolve(self, openai_model: str, voice_name: str | None) -> str | None:
        """Return the target profile_id for this (model, voice) pair, or
        None when there's no pin — callers fall back to resolving
        ``openai_model`` literally, same as if this registry didn't exist.
        """
        if not voice_name:
            return None
        entry = self._entries.get((openai_model, voice_name))
        return entry.target_profile_id if entry is not None else None

    def list_all(self) -> list[OpenAITTSVoiceMap]:
        return sorted(self._entries.values(), key=lambda e: (e.openai_model, e.voice_name))

    async def upsert(
        self,
        session: AsyncSession,
        *,
        openai_model: str,
        voice_name: str,
        target_profile_id: str,
    ) -> OpenAITTSVoiceMap:
        if not openai_model or not voice_name:
            raise ValueError("openai_model and voice_name are both required.")
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

        existing = self._entries.get((openai_model, voice_name))
        if existing is not None:
            existing.target_profile_id = target_profile_id
            entry = existing
        else:
            entry = OpenAITTSVoiceMap(
                openai_model=openai_model,
                voice_name=voice_name,
                target_profile_id=target_profile_id,
            )
            session.add(entry)
        await session.commit()
        await session.refresh(entry)
        self._entries[(openai_model, voice_name)] = entry
        logger.info(
            "openai_tts_compat_upserted",
            openai_model=openai_model,
            voice_name=voice_name,
            target_profile_id=target_profile_id,
        )
        return entry

    async def delete(self, session: AsyncSession, *, openai_model: str, voice_name: str) -> bool:
        entry = self._entries.get((openai_model, voice_name))
        if entry is None:
            return False
        result = await session.execute(
            select(OpenAITTSVoiceMap).where(
                OpenAITTSVoiceMap.openai_model == openai_model,
                OpenAITTSVoiceMap.voice_name == voice_name,
            )
        )
        db_entry = result.scalar_one_or_none()
        if db_entry is not None:
            await session.delete(db_entry)
            await session.commit()
        del self._entries[(openai_model, voice_name)]
        logger.info("openai_tts_compat_deleted", openai_model=openai_model, voice_name=voice_name)
        return True
