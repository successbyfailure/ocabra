import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ocabra.database import Base


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(512))
    backend_type: Mapped[str] = mapped_column(String(64), nullable=False)
    load_policy: Mapped[str] = mapped_column(String(32), nullable=False, default="on_demand")
    # Only meaningful when load_policy == "pin". False (default, "soft pin"):
    # preloaded at startup, immune to idle-timeout and routine LRU eviction,
    # but still evictable as an absolute last resort when nothing else can
    # free enough VRAM (see ModelManager._get_pressure_eviction_candidates).
    # True ("hard pin"): excluded from that last-resort path too — never
    # evicted by anything except an explicit admin unload (2026-10-30).
    pin_permanent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preferred_gpu: Mapped[int | None] = mapped_column(Integer)
    extra_config: Mapped[dict | None] = mapped_column(JSONB)
    # GGUF tokenizer fingerprint (Sprint 17.4) — populated by the local scanner
    # for llama.cpp/bitnet models. Nullable for non-GGUF backends.
    vocab_size: Mapped[int | None] = mapped_column(Integer)
    bos_id: Mapped[int | None] = mapped_column(Integer)
    eos_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    profiles: Mapped[list["ModelProfile"]] = relationship(
        "ModelProfile",
        back_populates="base_model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ModelProfile(Base):
    """A public-facing profile derived from a base model.

    Execution profiles belong to one physical model. Router profiles are
    model-independent public entry points whose ordered targets are profiles.
    """

    __tablename__ = "model_profiles"
    __table_args__ = (
        CheckConstraint(
            "(profile_kind = 'execution' AND base_model_id IS NOT NULL "
            "AND (routing_targets IS NULL OR routing_targets = '[]'::jsonb)) OR "
            "(profile_kind = 'router' AND base_model_id IS NULL "
            "AND jsonb_typeof(routing_targets) = 'array' "
            "AND jsonb_array_length(routing_targets) > 0)",
            name="ck_model_profile_kind_shape",
        ),
    )

    profile_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    profile_kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default="execution"
    )
    base_model_id: Mapped[str | None] = mapped_column(
        String(512),
        ForeignKey("model_configs.model_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    display_name: Mapped[str | None] = mapped_column(String(512))
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="llm")
    load_overrides: Mapped[dict | None] = mapped_column(JSONB)
    request_defaults: Mapped[dict | None] = mapped_column(JSONB)
    assets: Mapped[dict | None] = mapped_column(JSONB)
    # Router profiles delegate to these ordered public profile ids. They do
    # not own a base model: every successful request resolves to a target.
    #
    # ``none_as_null=True``: without it, SQLAlchemy's JSONB binds a Python
    # ``None`` as the JSON value ``null`` rather than SQL ``NULL``.
    # ``ck_model_profile_kind_shape`` checks ``routing_targets IS NULL OR
    # routing_targets = '[]'::jsonb`` — a JSON ``null`` satisfies neither
    # branch, so every execution-profile insert/update that leaves this
    # unset (i.e. every non-router profile) violated the constraint.
    routing_targets: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))
    # Bloque 21 — selection/hysteresis policy shared by the profile's
    # context/runtime presets. Presets themselves live in their own table so
    # they can be addressed and mutated independently from the public profile.
    runtime_policy: Mapped[dict | None] = mapped_column(JSONB)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    base_model: Mapped["ModelConfig | None"] = relationship(
        "ModelConfig", back_populates="profiles"
    )
    runtime_presets: Mapped[list["ProfileRuntimePreset"]] = relationship(
        "ProfileRuntimePreset",
        back_populates="profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
        order_by="ProfileRuntimePreset.priority, ProfileRuntimePreset.context_tokens",
    )
    aliases: Mapped[list["ModelProfileAlias"]] = relationship(
        "ModelProfileAlias",
        back_populates="profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )


class ModelProfileAlias(Base):
    """Hidden compatibility name that resolves to a canonical profile preset."""

    __tablename__ = "model_profile_aliases"

    alias_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey(
            "model_profiles.profile_id", ondelete="CASCADE", onupdate="CASCADE"
        ),
        nullable=False,
        index=True,
    )
    preset_key: Mapped[str | None] = mapped_column(String(64))
    hidden: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    profile: Mapped["ModelProfile"] = relationship(
        "ModelProfile", back_populates="aliases"
    )


class ProfileRuntimePreset(Base):
    """One load-affecting runtime configuration exposed by a public profile."""

    __tablename__ = "profile_runtime_presets"
    __table_args__ = (
        UniqueConstraint("profile_id", "preset_key", name="uq_profile_runtime_preset_key"),
        CheckConstraint(
            "(context_mode = 'fixed' AND context_tokens > 0) OR "
            "(context_mode = 'max' AND context_tokens IS NULL)",
            name="ck_profile_runtime_preset_context",
        ),
        CheckConstraint(
            "parallelism > 0", name="ck_profile_runtime_preset_parallelism"
        ),
        CheckConstraint("priority >= 0", name="ck_profile_runtime_preset_priority"),
        CheckConstraint(
            "NOT is_default OR enabled", name="ck_profile_runtime_preset_default_enabled"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    profile_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("model_profiles.profile_id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )
    preset_key: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    context_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="fixed")
    context_tokens: Mapped[int | None] = mapped_column(Integer)
    parallelism: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    kv_cache_dtype: Mapped[str] = mapped_column(String(16), nullable=False, default="q8")
    placement_policy: Mapped[str] = mapped_column(String(16), nullable=False, default="auto")
    preferred_gpus: Mapped[list[int] | None] = mapped_column(JSONB)
    execution_policy: Mapped[str] = mapped_column(String(16), nullable=False, default="auto")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    profile: Mapped["ModelProfile"] = relationship(
        "ModelProfile", back_populates="runtime_presets"
    )


class EvictionSchedule(Base):
    __tablename__ = "eviction_schedules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # NULL = schedule global (aplica a todos los modelos)
    model_id: Mapped[str | None] = mapped_column(String(512), index=True)
    cron_expr: Mapped[str] = mapped_column(String(128), nullable=False)
    # evict_warm | evict_all | reload
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


GLOBAL_SCHEDULE_START_ACTION = "evict_all"
GLOBAL_SCHEDULE_END_ACTION = "reload"

# Per-model schedules use evict_warm (no point in evicting PIN through a
# per-model rule) + reload. Same UI shape, narrower scope.
MODEL_SCHEDULE_START_ACTION = "evict_warm"
MODEL_SCHEDULE_END_ACTION = "reload"


def _normalize_days(days: Sequence[int]) -> list[int]:
    normalized = sorted({int(day) for day in days if 0 <= int(day) <= 6})
    if not normalized:
        raise ValueError("globalSchedules.days must contain at least one day between 0 and 6")
    return normalized


def _parse_time(value: str) -> tuple[int, int]:
    parts = str(value or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time value: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"Invalid time value: {value!r}")
    return hour, minute


def _format_time(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def _days_to_cron_field(days: Sequence[int]) -> str:
    return ",".join(str(day) for day in _normalize_days(days))


def _shift_days(days: Sequence[int], offset: int) -> list[int]:
    return sorted({(int(day) + offset) % 7 for day in _normalize_days(days)})


def _schedule_payload_to_rows(
    *,
    model_id: str | None,
    schedule_id: str,
    days: Sequence[int],
    start: str,
    end: str,
    enabled: bool,
    start_action: str,
    end_action: str,
    id_label: str,
) -> list[EvictionSchedule]:
    start_hour, start_minute = _parse_time(start)
    end_hour, end_minute = _parse_time(end)
    normalized_days = _normalize_days(days)
    start_total_minutes = start_hour * 60 + start_minute
    end_total_minutes = end_hour * 60 + end_minute
    end_days = (
        normalized_days
        if end_total_minutes > start_total_minutes
        else _shift_days(normalized_days, 1)
    )

    label = str(schedule_id).strip()
    if not label:
        raise ValueError(f"{id_label}.id must be a non-empty string")

    return [
        EvictionSchedule(
            model_id=model_id,
            cron_expr=f"{start_minute} {start_hour} * * {_days_to_cron_field(normalized_days)}",
            action=start_action,
            label=label,
            enabled=bool(enabled),
        ),
        EvictionSchedule(
            model_id=model_id,
            cron_expr=f"{end_minute} {end_hour} * * {_days_to_cron_field(end_days)}",
            action=end_action,
            label=label,
            enabled=bool(enabled),
        ),
    ]


def global_schedule_payload_to_rows(
    schedule_id: str,
    days: Sequence[int],
    start: str,
    end: str,
    enabled: bool,
) -> list[EvictionSchedule]:
    return _schedule_payload_to_rows(
        model_id=None,
        schedule_id=schedule_id,
        days=days,
        start=start,
        end=end,
        enabled=enabled,
        start_action=GLOBAL_SCHEDULE_START_ACTION,
        end_action=GLOBAL_SCHEDULE_END_ACTION,
        id_label="globalSchedules",
    )


def model_schedule_payload_to_rows(
    model_id: str,
    schedule_id: str,
    days: Sequence[int],
    start: str,
    end: str,
    enabled: bool,
) -> list[EvictionSchedule]:
    if not model_id:
        raise ValueError("model_id is required for per-model schedules")
    return _schedule_payload_to_rows(
        model_id=model_id,
        schedule_id=schedule_id,
        days=days,
        start=start,
        end=end,
        enabled=enabled,
        start_action=MODEL_SCHEDULE_START_ACTION,
        end_action=MODEL_SCHEDULE_END_ACTION,
        id_label="schedules",
    )


def _parse_day_field(field: str) -> list[int]:
    field = str(field or "").strip()
    if not field:
        raise ValueError("Empty cron weekday field")
    if field == "*":
        return list(range(7))

    days: set[int] = set()
    for part in field.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_expr, end_expr = token.split("-", 1)
            start = int(start_expr)
            end = int(end_expr)
            if start > end:
                raise ValueError(f"Unsupported wrapped cron weekday range: {field!r}")
            days.update(range(start, end + 1))
            continue
        days.add(int(token))
    return sorted(day % 7 for day in days)


def _cron_to_time_and_days(cron_expr: str) -> tuple[list[int], str]:
    minute, hour, _day, _month, weekday = str(cron_expr or "").split()
    return _parse_day_field(weekday), _format_time(int(hour), int(minute))


def _group_schedule_rows(
    rows: Sequence[EvictionSchedule],
    *,
    start_action: str,
    end_action: str,
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        label = str(row.label or row.id)
        bucket = grouped.setdefault(label, {"enabled": True})
        bucket["enabled"] = bool(bucket["enabled"]) and bool(row.enabled)
        bucket.setdefault("label", label)
        if row.action == start_action:
            days, start = _cron_to_time_and_days(row.cron_expr)
            bucket["days"] = days
            bucket["start"] = start
        elif row.action == end_action:
            _days, end = _cron_to_time_and_days(row.cron_expr)
            bucket["end"] = end

    payloads: list[dict[str, object]] = []
    for label, bucket in grouped.items():
        days = bucket.get("days")
        start = bucket.get("start")
        end = bucket.get("end")
        if not isinstance(days, list) or not isinstance(start, str) or not isinstance(end, str):
            continue
        payloads.append(
            {
                "id": label,
                "days": days,
                "start": start,
                "end": end,
                "enabled": bool(bucket.get("enabled", True)),
            }
        )

    return sorted(payloads, key=lambda item: (str(item["start"]), str(item["id"])))


async def get_global_schedule_rows(session) -> list[EvictionSchedule]:
    import sqlalchemy as sa

    result = await session.execute(
        sa.select(EvictionSchedule).where(EvictionSchedule.model_id.is_(None))
    )
    return list(result.scalars().all())


async def replace_global_schedules(session, payloads: Sequence[dict[str, object]]) -> None:
    existing_rows = await get_global_schedule_rows(session)
    for row in existing_rows:
        await session.delete(row)

    new_rows: list[EvictionSchedule] = []
    for payload in payloads:
        new_rows.extend(
            global_schedule_payload_to_rows(
                schedule_id=str(payload["id"]),
                days=payload["days"],  # type: ignore[arg-type]
                start=str(payload["start"]),
                end=str(payload["end"]),
                enabled=bool(payload.get("enabled", True)),
            )
        )

    session.add_all(new_rows)


def global_schedule_rows_to_payload(rows: Sequence[EvictionSchedule]) -> list[dict[str, object]]:
    return _group_schedule_rows(
        rows,
        start_action=GLOBAL_SCHEDULE_START_ACTION,
        end_action=GLOBAL_SCHEDULE_END_ACTION,
    )


async def get_model_schedule_rows(session, model_id: str) -> list[EvictionSchedule]:
    import sqlalchemy as sa

    result = await session.execute(
        sa.select(EvictionSchedule).where(EvictionSchedule.model_id == model_id)
    )
    return list(result.scalars().all())


async def get_all_model_schedule_rows(session) -> list[EvictionSchedule]:
    """Return every non-global schedule, useful to enrich the model listing."""
    import sqlalchemy as sa

    result = await session.execute(
        sa.select(EvictionSchedule).where(EvictionSchedule.model_id.is_not(None))
    )
    return list(result.scalars().all())


async def replace_model_schedules(
    session, model_id: str, payloads: Sequence[dict[str, object]]
) -> None:
    if not model_id:
        raise ValueError("model_id is required for per-model schedules")
    existing_rows = await get_model_schedule_rows(session, model_id)
    for row in existing_rows:
        await session.delete(row)

    new_rows: list[EvictionSchedule] = []
    for payload in payloads:
        new_rows.extend(
            model_schedule_payload_to_rows(
                model_id=model_id,
                schedule_id=str(payload["id"]),
                days=payload["days"],  # type: ignore[arg-type]
                start=str(payload["start"]),
                end=str(payload["end"]),
                enabled=bool(payload.get("enabled", True)),
            )
        )

    session.add_all(new_rows)


def model_schedule_rows_to_payload(
    rows: Sequence[EvictionSchedule],
) -> list[dict[str, object]]:
    return _group_schedule_rows(
        rows,
        start_action=MODEL_SCHEDULE_START_ACTION,
        end_action=MODEL_SCHEDULE_END_ACTION,
    )
