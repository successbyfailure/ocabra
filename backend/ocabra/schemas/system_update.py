"""Request bodies of /ocabra/system/update (oCabra updating itself)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SelfUpdateApply(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Release to install (``X.Y.Z``); empty = the newest published.
    version: str | None = Field(default=None, pattern=r"^v?\d+\.\d+\.\d+$")
    # Skip the idle check and allow reinstalling the same version.
    force: bool = False


class SelfUpdatePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    policy: Literal["off", "notify", "auto"] | None = None
    idle_minutes: int | None = Field(default=None, ge=1, le=1440, alias="idleMinutes")
