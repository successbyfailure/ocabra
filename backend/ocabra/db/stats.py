import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ocabra.database import Base


class RequestStat(Base):
    __tablename__ = "request_stats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    backend_type: Mapped[str | None] = mapped_column(String(64), index=True)
    request_kind: Mapped[str | None] = mapped_column(String(64), index=True)
    endpoint_path: Mapped[str | None] = mapped_column(String(256))
    status_code: Mapped[int | None] = mapped_column(Integer)
    gpu_index: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    # Vatios-hora estimados consumidos durante el request
    energy_wh: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    # Authenticated user who made the request; NULL for anonymous callers.
    user_id: Mapped[uuid.UUID | None] = mapped_column(sa.Uuid, nullable=True)
    # Group associated with the API key used for this request; preserved even if group is deleted.
    group_id: Mapped[uuid.UUID | None] = mapped_column(sa.Uuid, nullable=True, index=True)
    # Name/label of the API key used; stored denormalized so it survives key deletion.
    api_key_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # Node ID of the remote peer when the request was proxied via federation.
    remote_node_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # Agent that originated this request (NULL for non-agent traffic).
    # Set on both the root request and every intermediate hop.
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # When this stat is an intermediate tool-loop hop, points at the root request_stat.
    # Root requests have ``parent_request_id = NULL`` and ``agent_id`` populated.
    parent_request_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("request_stats.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )


class GpuStat(Base):
    __tablename__ = "gpu_stats"

    # Serie temporal agregada por minuto
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, primary_key=True
    )
    gpu_index: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
    utilization_pct: Mapped[float | None] = mapped_column(Float)
    vram_used_mb: Mapped[int | None] = mapped_column(Integer)
    power_draw_w: Mapped[float | None] = mapped_column(Float)
    temperature_c: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("ix_gpu_stats_recorded_at_gpu", "recorded_at", "gpu_index"),
    )


class ModelLoadStat(Base):
    __tablename__ = "model_load_stats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    backend_type: Mapped[str | None] = mapped_column(String(64), index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    gpu_count: Mapped[int | None] = mapped_column(Integer)
    gpu_indices: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_model_load_stats_started_at_model", "started_at", "model_id"),
    )


class ServiceGenerationStat(Base):
    """One row per completed (or force-evicted) generation job on an interactive service."""

    __tablename__ = "service_generation_stats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    service_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    service_type: Mapped[str | None] = mapped_column(String(64), index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    gpu_index: Mapped[int | None] = mapped_column(Integer)
    vram_peak_mb: Mapped[int | None] = mapped_column(Integer)
    # True when the generation was interrupted by a forced eviction (grace period expired)
    evicted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_svc_gen_stats_started_at_svc", "started_at", "service_id"),
    )


class ServerStat(Base):
    __tablename__ = "server_stats"

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, primary_key=True
    )
    cpu_power_w: Mapped[float | None] = mapped_column(Float)
    cpu_temp_c: Mapped[float | None] = mapped_column(Float)
    total_gpu_power_w: Mapped[float | None] = mapped_column(Float)
    total_power_w: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("ix_server_stats_recorded_at", "recorded_at"),
    )


class ToolCallStat(Base):
    """One row per MCP tool_call executed inside an agent tool-loop.

    Linked to the root ``request_stats`` row via ``request_stat_id`` when
    available (the root is the entry point created by the StatsMiddleware for
    the HTTP request that invoked the agent).  ``tool_args_redacted`` must
    already have secrets redacted by the AgentExecutor — we never persist raw
    args.  ``result_summary`` is truncated to ``mcp_result_max_bytes`` (text).
    """

    __tablename__ = "tool_call_stats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    request_stat_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("request_stats.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    mcp_server_alias: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(256), nullable=False)
    tool_args_redacted: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ok")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    hop_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index(
            "ix_tool_call_stats_alias_tool",
            "mcp_server_alias",
            "tool_name",
        ),
    )
