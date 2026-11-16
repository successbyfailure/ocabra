"""Schemas for the automatic preset setup of a llama.cpp model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AutoSetupRequest(BaseModel):
    """Devices a model may use; ``apply`` writes the planned presets."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    gpus: list[int] = Field(default_factory=list)
    cpu: bool = False
    apply: bool = False
