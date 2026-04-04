"""
SQLAlchemy model and helpers for persisted server config overrides.

The settings object is seeded from .env at startup, then any rows in
``server_config`` are applied on top.  PATCH /ocabra/config writes here so
changes survive container restarts.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy import DateTime, String, Text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from ocabra.database import Base

# Fields that must NOT be persisted (infra-level, binary paths, secrets).
# These are always sourced from .env only.
_ENV_ONLY_FIELDS: frozenset[str] = frozenset({
    "models_dir",
    "hf_cache_dir",
    "ai_models_root",
    "hf_token",
    "ollama_base_url",
    "database_url",
    "redis_url",
    "api_host",
    "api_port",
    "app_version",
    "gateway_service_token",
    "llama_cpp_server_bin",
    "bitnet_server_bin",
    "sglang_python_bin",
    "sglang_server_module",
    "tensorrt_llm_serve_bin",
    "tensorrt_llm_docker_bin",
    "tensorrt_llm_docker_image",
    "tensorrt_llm_python_bin",
    "tensorrt_llm_serve_module",
    "tensorrt_llm_docker_models_mount_host",
    "tensorrt_llm_docker_models_mount_container",
    "tensorrt_llm_docker_hf_cache_mount_host",
    "tensorrt_llm_docker_hf_cache_mount_container",
    "tensorrt_llm_engines_dir",
    "tensorrt_llm_host_helper_image",
    "worker_port_range_start",
    "worker_port_range_end",
    "frontend_uid",
    "frontend_gid",
})


class ServerConfigOverride(Base):
    __tablename__ = "server_config"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=sa.func.now(),
    )


async def load_overrides(session: AsyncSession) -> dict[str, str]:
    """Return all persisted config overrides as a {key: raw_value} dict."""
    result = await session.execute(sa.select(ServerConfigOverride))
    return {row.key: row.value for row in result.scalars().all()}


async def save_override(session: AsyncSession, key: str, value: object) -> None:
    """Upsert a single config override.  *value* is JSON-encoded for storage."""
    if key in _ENV_ONLY_FIELDS:
        return
    raw = json.dumps(value)
    existing = await session.get(ServerConfigOverride, key)
    if existing is None:
        session.add(ServerConfigOverride(key=key, value=raw, updated_at=datetime.now(UTC)))
    else:
        existing.value = raw
        existing.updated_at = datetime.now(UTC)


def apply_overrides_to_settings(settings_obj: object, overrides: dict[str, str]) -> None:
    """Apply DB overrides onto the settings object (mutates in place)."""
    for key, raw in overrides.items():
        if key in _ENV_ONLY_FIELDS:
            continue
        if not hasattr(settings_obj, key):
            continue
        try:
            value = json.loads(raw)
            setattr(settings_obj, key, value)
        except Exception:
            pass
