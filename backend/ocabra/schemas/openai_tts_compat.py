"""Schemas for the OpenAI TTS voice-name compatibility shim."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class OpenAITTSVoiceMapUpsert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_profile_id: str = Field(..., min_length=1, max_length=512)


class OpenAITTSVoiceMapOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    openai_model: str
    voice_name: str
    target_profile_id: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
