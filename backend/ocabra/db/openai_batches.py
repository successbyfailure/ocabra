"""SQLAlchemy ORM models for OpenAI-compatible Files and Batches APIs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ocabra.database import Base


class OpenAIFile(Base):
    __tablename__ = "openai_files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="uploaded")
    status_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": f"file-{self.id}",
            "object": "file",
            "bytes": self.bytes,
            "created_at": int(self.created_at.timestamp()),
            "filename": self.filename,
            "purpose": self.purpose,
            "status": self.status,
            "status_details": self.status_details,
        }


class OpenAIBatch(Base):
    __tablename__ = "openai_batches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
    )
    endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    input_file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("openai_files.id", ondelete="RESTRICT"),
        nullable=False,
    )
    completion_window: Mapped[str] = mapped_column(Text, nullable=False, default="24h")
    status: Mapped[str] = mapped_column(Text, nullable=False, default="validating", index=True)
    output_file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("openai_files.id", ondelete="SET NULL"),
        nullable=True,
    )
    error_file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("openai_files.id", ondelete="SET NULL"),
        nullable=True,
    )
    errors: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    request_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    batch_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    in_progress_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finalizing_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelling_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_openai_dict(self) -> dict[str, Any]:
        def ts(value: datetime | None) -> int | None:
            return int(value.timestamp()) if value else None

        return {
            "id": f"batch_{self.id}",
            "object": "batch",
            "endpoint": self.endpoint,
            "errors": self.errors,
            "input_file_id": f"file-{self.input_file_id}",
            "completion_window": self.completion_window,
            "status": self.status,
            "output_file_id": f"file-{self.output_file_id}" if self.output_file_id else None,
            "error_file_id": f"file-{self.error_file_id}" if self.error_file_id else None,
            "created_at": ts(self.created_at),
            "in_progress_at": ts(self.in_progress_at),
            "expires_at": ts(self.expires_at),
            "finalizing_at": ts(self.finalizing_at),
            "completed_at": ts(self.completed_at),
            "failed_at": ts(self.failed_at),
            "expired_at": ts(self.expired_at),
            "cancelling_at": ts(self.cancelling_at),
            "cancelled_at": ts(self.cancelled_at),
            "request_counts": {
                "total": self.request_total,
                "completed": self.request_completed,
                "failed": self.request_failed,
            },
            "metadata": self.batch_metadata,
        }
