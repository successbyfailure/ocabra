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

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from ocabra.core._openai_compat import OpenAICompatRegistry
from ocabra.db.model_config import OpenAITTSVoiceMap


class OpenAITTSCompatRegistry(OpenAICompatRegistry[tuple[str, str], OpenAITTSVoiceMap]):
    """In-memory cache of ``openai_tts_voice_map`` rows, backed by PostgreSQL."""

    _model = OpenAITTSVoiceMap
    _log_prefix = "openai_tts_compat"

    def _row_key(self, row: OpenAITTSVoiceMap) -> tuple[str, str]:
        return (row.openai_model, row.voice_name)

    def _pk_filter(self, key: tuple[str, str]) -> list[ColumnElement[bool]]:
        openai_model, voice_name = key
        return [
            OpenAITTSVoiceMap.openai_model == openai_model,
            OpenAITTSVoiceMap.voice_name == voice_name,
        ]

    def resolve(self, openai_model: str, voice_name: str | None) -> str | None:
        """Return the target profile_id for this (model, voice) pair, or
        None when there's no pin — callers fall back to resolving
        ``openai_model`` literally, same as if this registry didn't exist.
        """
        if not voice_name:
            return None
        entry = self._entries.get((openai_model, voice_name))
        return entry.target_profile_id if entry is not None else None

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
        return await self._upsert_key(
            session,
            (openai_model, voice_name),
            target_profile_id,
            lambda: OpenAITTSVoiceMap(
                openai_model=openai_model,
                voice_name=voice_name,
                target_profile_id=target_profile_id,
            ),
            openai_model=openai_model,
            voice_name=voice_name,
        )

    async def delete(self, session: AsyncSession, *, openai_model: str, voice_name: str) -> bool:
        return await self._delete_key(
            session,
            (openai_model, voice_name),
            openai_model=openai_model,
            voice_name=voice_name,
        )
