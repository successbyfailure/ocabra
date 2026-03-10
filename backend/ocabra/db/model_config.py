import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ocabra.database import Base


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(512))
    backend_type: Mapped[str] = mapped_column(String(64), nullable=False)
    load_policy: Mapped[str] = mapped_column(String(32), nullable=False, default="on_demand")
    auto_reload: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preferred_gpu: Mapped[int | None] = mapped_column(Integer)
    extra_config: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class EvictionSchedule(Base):
    __tablename__ = "eviction_schedules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # NULL = schedule global (aplica a todos los modelos)
    model_id: Mapped[str | None] = mapped_column(String(512), index=True)
    cron_expr: Mapped[str] = mapped_column(String(128), nullable=False)
    # evict_warm | evict_all | reload
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
