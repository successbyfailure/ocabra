import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
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
