"""Persisted benchmark runs and their individual attempts."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ocabra.database import Base


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    app: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    suite: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[str | None] = mapped_column(Text, nullable=True)
    models: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    suite_args: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued", index=True)
    phase: Mapped[str] = mapped_column(Text, nullable=False, default="queued")
    auto_pause: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    pause_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_model_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    workspace_path: Mapped[str] = mapped_column(Text, nullable=False)
    result: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    contaminated_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_activity_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    attempts: Mapped[list[BenchmarkAttempt]] = relationship(
        "BenchmarkAttempt",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="BenchmarkAttempt.started_at",
    )

    __table_args__ = (Index("ix_benchmark_runs_created_at", "created_at"),)


class BenchmarkAttempt(Base):
    __tablename__ = "benchmark_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("benchmark_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model: Mapped[str] = mapped_column(Text, nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    workspace_path: Mapped[str] = mapped_column(Text, nullable=False)
    log_path: Mapped[str] = mapped_column(Text, nullable=False)
    return_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    environment: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    external_requests: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    run: Mapped[BenchmarkRun] = relationship("BenchmarkRun", back_populates="attempts")

    __table_args__ = (
        Index("ix_benchmark_attempts_run_model", "run_id", "model"),
        Index(
            "uq_benchmark_attempts_run_number",
            "run_id",
            "model",
            "attempt_number",
            unique=True,
        ),
    )
