"""Trabajos de generación de vídeo (``/v1/videos``).

El prompt no se guarda: vive en memoria mientras el trabajo está en cola o
generándose (privacidad). Lo que queda en la fila es lo que hace falta para
contestar ``GET /v1/videos/{id}`` y servir el MP4 hasta que caduque.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ocabra.database import Base


class VideoJob(Base):
    __tablename__ = "video_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    model: Mapped[str] = mapped_column(Text, nullable=False)
    # queued | in_progress | completed | failed
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def public_id(self) -> str:
        return f"video_{self.id.hex}"

    def to_openai_dict(self) -> dict[str, Any]:
        """Objeto ``video`` de la API de OpenAI."""

        def ts(value: datetime | None) -> int | None:
            return int(value.timestamp()) if value is not None else None

        return {
            "id": self.public_id,
            "object": "video",
            "model": self.model,
            "status": self.status,
            "progress": int(self.progress or 0),
            "created_at": ts(self.created_at),
            "completed_at": ts(self.completed_at),
            "expires_at": ts(self.expires_at),
            "seconds": str(self.seconds),
            "size": f"{self.width}x{self.height}",
            "seed": self.seed,
            "remixed_from_video_id": None,
            "error": self.error,
        }
