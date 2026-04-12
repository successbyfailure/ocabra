"""Pydantic schemas for federation CRUD endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PeerCreate(BaseModel):
    """Request body for adding a new federation peer."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=255)
    url: str = Field(..., min_length=1, max_length=1024)
    api_key: str = Field(..., min_length=1)
    access_level: str = Field(default="inference", pattern=r"^(inference|full)$")


class PeerUpdate(BaseModel):
    """Request body for updating a federation peer."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=255)
    url: str | None = Field(default=None, min_length=1, max_length=1024)
    api_key: str | None = Field(default=None, min_length=1)
    access_level: str | None = Field(default=None, pattern=r"^(inference|full)$")
    enabled: bool | None = None


class PeerOut(BaseModel):
    """Response schema for a federation peer."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    url: str
    access_level: str
    enabled: bool
    online: bool = False
    last_heartbeat: datetime | None = None
    models: list[dict[str, Any]] = []
    gpus: list[dict[str, Any]] = []
    load: dict[str, Any] = {}
    created_at: datetime | None = None
    updated_at: datetime | None = None


class HeartbeatGpu(BaseModel):
    """GPU info in heartbeat response."""

    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int


class HeartbeatModel(BaseModel):
    """Model info in heartbeat response."""

    model_id: str
    status: str
    profiles: list[str] = []


class HeartbeatLoad(BaseModel):
    """Load info in heartbeat response."""

    active_requests: int = 0
    gpu_utilization_avg_pct: float = 0.0


class HeartbeatResponse(BaseModel):
    """Response for the heartbeat endpoint."""

    node_id: str
    node_name: str
    version: str
    uptime_seconds: float
    gpus: list[HeartbeatGpu]
    models: list[HeartbeatModel]
    load: HeartbeatLoad


class PeerTestResult(BaseModel):
    """Result of a peer connection test."""

    success: bool
    node_id: str | None = None
    node_name: str | None = None
    latency_ms: float | None = None
    error: str | None = None
