"""Pydantic schemas for federation CRUD endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PeerCreate(BaseModel):
    """Request body for adding a new federation peer.

    ``connection_mode="http"`` (default): this node calls ``url`` with
    ``api_key`` — both required. ``connection_mode="tunnel_in"``: the peer
    dials this node; ``url``/``api_key`` are not used and the response carries
    a one-time ``tunnel_token`` for the peer to present. (Normally tunnel peers
    are created by accepting a join request instead.)
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=255)
    url: str = Field(default="", max_length=1024)
    api_key: str = Field(default="")
    access_level: str = Field(default="inference", pattern=r"^(inference|full)$")
    connection_mode: str = Field(default="http", pattern=r"^(http|tunnel_in)$")
    notes: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _http_needs_url_and_key(self) -> PeerCreate:
        if self.connection_mode == "http" and (not self.url.strip() or not self.api_key.strip()):
            raise ValueError("url and api_key are required for connection_mode='http'")
        return self


class PeerUpdate(BaseModel):
    """Request body for updating a federation peer."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=255)
    url: str | None = Field(default=None, min_length=1, max_length=1024)
    api_key: str | None = Field(default=None, min_length=1)
    access_level: str | None = Field(default=None, pattern=r"^(inference|full)$")
    enabled: bool | None = None
    notes: str | None = Field(default=None, max_length=4000)


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
    connection_mode: str = "http"
    mode: str = "manual"
    notes: str | None = None
    remote_node_id: str | None = None
    local_access_level: str | None = Field(
        default=None, description="What this peer may do on us (None = we don't serve it)"
    )
    join_state: str | None = Field(
        default=None, description="Outgoing join request: pending | denied | expired"
    )
    join_verification_code: str | None = None
    dial_state: str | None = Field(
        default=None,
        description="tunnel_out: connecting | connected | backoff | rejected | stopped",
    )
    last_error: str | None = None
    tunnel_connected: bool = False
    tunnel: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Live tunnel info: remote node name/version, connected_at, active streams, "
            "requests sent/served."
        ),
    )


class PeerCreated(PeerOut):
    """Response for peer creation / token rotation. ``tunnel_token`` is shown once."""

    tunnel_token: str | None = None


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
    runtime_preset: str | None = None
    effective_context: int | None = None
    parallelism: int | None = None


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


# ── Phase 5: Remote operations schemas ──────────────────────────


class RemoteLoadRequest(BaseModel):
    """Optional configuration for remote model load.

    All fields are optional; when omitted the peer uses its own defaults.
    """

    model_config = ConfigDict(extra="forbid")

    preferred_gpu: int | None = Field(default=None, description="GPU index on the peer to load onto")
    extra_config: dict[str, Any] | None = Field(default=None, description="Extra backend-specific configuration")


class RemoteDownloadRequest(BaseModel):
    """Request body for triggering a model download on a remote peer.

    Mirrors the local POST /ocabra/downloads request schema.
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., pattern=r"^(huggingface|ollama|bitnet)$", description="Download source")
    model_ref: str = Field(..., min_length=1, description="Model reference (e.g. 'org/model-name')")
    artifact: str | None = Field(default=None, description="Specific artifact/file to download")
    register_config: dict[str, Any] | None = Field(default=None, description="Auto-register config after download")


# ── Join requests / invites ─────────────────────────────────────


class JoinRequestCreate(BaseModel):
    """Body a node sends to ask to join this one (public endpoint)."""

    model_config = ConfigDict(extra="ignore")

    node_id: str = Field(..., min_length=1, max_length=255)
    node_name: str = Field(..., min_length=1, max_length=255)
    version: str | None = Field(default=None, max_length=64)
    gpus: list[dict[str, Any]] | None = None
    message: str | None = Field(default=None, max_length=2000)
    mode: str = Field(default="swarm", pattern=r"^(swarm)$")
    invite_code: str | None = Field(default=None, max_length=200)


class JoinRequestDecision(BaseModel):
    """Optional overrides when accepting a join request."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=255)
    notes: str | None = Field(default=None, max_length=4000)


class JoinOutgoing(BaseModel):
    """Ask another node (``hub_url``) to let this node join its federation."""

    model_config = ConfigDict(extra="forbid")

    hub_url: str = Field(..., min_length=8, max_length=1024)
    message: str | None = Field(default=None, max_length=2000)
    invite_code: str | None = Field(default=None, max_length=200)
    name: str | None = Field(default=None, min_length=1, max_length=255)
    mode: str = Field(default="swarm", pattern=r"^(swarm)$")


class InviteCreate(BaseModel):
    """Create a pre-approved join code."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1, max_length=255)
    mode: str = Field(default="swarm", pattern=r"^(swarm)$")
    max_uses: int = Field(default=1, ge=1, le=100)
    expires_in_hours: float | None = Field(default=24, gt=0, le=24 * 365)
