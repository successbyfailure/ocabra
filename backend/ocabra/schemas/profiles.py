"""Pydantic schemas for model profile CRUD endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProfileCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(..., min_length=1, max_length=512)
    display_name: str | None = None
    description: str | None = None
    category: str = "llm"
    load_overrides: dict[str, Any] | None = None
    request_defaults: dict[str, Any] | None = None
    enabled: bool = True
    is_default: bool = False
    # Bloque 20 — Router profiles. Ordered list of target profile_ids the
    # RouterResolver walks when this profile is requested. Non-null makes
    # the profile a router; null keeps it a plain profile.
    routing_targets: list[str] | None = None


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = None
    description: str | None = None
    category: str | None = None
    load_overrides: dict[str, Any] | None = None
    request_defaults: dict[str, Any] | None = None
    enabled: bool | None = None
    is_default: bool | None = None
    routing_targets: list[str] | None = None


class ProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    profile_id: str
    base_model_id: str
    display_name: str | None = None
    description: str | None = None
    category: str
    load_overrides: dict[str, Any] | None = None
    request_defaults: dict[str, Any] | None = None
    assets: dict[str, Any] | None = None
    enabled: bool
    is_default: bool
    routing_targets: list[str] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
