import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
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
