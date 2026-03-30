from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ocabra.database import Base


class TrtllmCompileJob(Base):
    """Persistent record of a TensorRT-LLM engine compilation job."""

    __tablename__ = "trtllm_compile_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_model: Mapped[str] = mapped_column(Text, nullable=False)
    engine_name: Mapped[str] = mapped_column(Text, nullable=False)
    gpu_indices: Mapped[list] = mapped_column(JSON, nullable=False)
    dtype: Mapped[str] = mapped_column(String(32), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    # pending | running | done | failed | cancelled
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", index=True
    )
    # convert | build | null
    phase: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress_pct: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Final engine directory path
    engine_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_trtllm_compile_jobs_status", "status"),
    )
