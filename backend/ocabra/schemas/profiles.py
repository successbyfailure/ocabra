"""Pydantic schemas for model profile CRUD endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RuntimePolicy(BaseModel):
    """How a profile selects and retires context runtimes."""

    model_config = ConfigDict(extra="forbid")

    selection: Literal["prefer_loaded", "prefer_default"] = "prefer_loaded"
    restore_default_when_idle: bool = True
    downgrade_grace_seconds: int = Field(default=600, ge=0, le=86400)
    minimum_residency_seconds: int = Field(default=120, ge=0, le=86400)
    small_requests_extend_large_ttl: bool = False
    remote_warm_threshold_seconds: int = Field(default=10, ge=0, le=3600)
    max_context_tokens: int | None = Field(default=None, gt=0)
    # Overrides settings.runtime_preset_default_output_reservation_tokens for
    # this profile only — see estimate_request_token_budget() /
    # plan_profile_runtime(). Reasoning-heavy profiles (gemma4, qwen3.8) need
    # more than the global default reserves, since reasoning_content counts
    # against the same budget as the final answer.
    default_output_reservation_tokens: int | None = Field(default=None, gt=0)


class RuntimePresetCreate(BaseModel):
    """Create one load-affecting runtime preset for a profile."""

    model_config = ConfigDict(extra="forbid")

    preset_key: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    label: str = Field(..., min_length=1, max_length=128)
    context_mode: Literal["fixed", "max"] = "fixed"
    context_tokens: int | None = Field(default=None, gt=0)
    parallelism: int = Field(default=2, ge=1, le=64)
    kv_cache_dtype: Literal["fp16", "bf16", "fp8", "q8", "q4"] = "q8"
    placement_policy: Literal["auto", "preferred", "forced"] = "auto"
    preferred_gpus: list[int] | None = None
    execution_policy: Literal["auto", "local", "remote"] = "auto"
    load_policy: Literal["warm", "on_demand"] = "on_demand"
    enabled: bool = True
    is_default: bool = False
    priority: int = Field(default=100, ge=0)

    @field_validator("preferred_gpus")
    @classmethod
    def validate_preferred_gpus(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(index < 0 for index in value):
            raise ValueError("preferred_gpus must contain non-negative GPU indices")
        if len(set(value)) != len(value):
            raise ValueError("preferred_gpus must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_context_and_default(self) -> RuntimePresetCreate:
        if self.context_mode == "fixed" and self.context_tokens is None:
            raise ValueError("context_tokens is required when context_mode='fixed'")
        if self.context_mode == "max" and self.context_tokens is not None:
            raise ValueError("context_tokens must be null when context_mode='max'")
        if self.is_default and not self.enabled:
            raise ValueError("the default runtime preset must be enabled")
        if self.placement_policy in ("preferred", "forced") and not self.preferred_gpus:
            raise ValueError(
                "preferred_gpus is required for preferred/forced placement"
            )
        return self


class RuntimePresetUpdate(BaseModel):
    """Patch one runtime preset; cross-field validation happens after merge."""

    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, min_length=1, max_length=128)
    context_mode: Literal["fixed", "max"] | None = None
    context_tokens: int | None = Field(default=None, gt=0)
    parallelism: int | None = Field(default=None, ge=1, le=64)
    kv_cache_dtype: Literal["fp16", "bf16", "fp8", "q8", "q4"] | None = None
    placement_policy: Literal["auto", "preferred", "forced"] | None = None
    preferred_gpus: list[int] | None = None
    execution_policy: Literal["auto", "local", "remote"] | None = None
    load_policy: Literal["warm", "on_demand"] | None = None
    enabled: bool | None = None
    is_default: bool | None = None
    priority: int | None = Field(default=None, ge=0)

    @field_validator("preferred_gpus")
    @classmethod
    def validate_preferred_gpus(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(index < 0 for index in value):
            raise ValueError("preferred_gpus must contain non-negative GPU indices")
        if len(set(value)) != len(value):
            raise ValueError("preferred_gpus must not contain duplicates")
        return value


class RuntimePresetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    profile_id: str
    preset_key: str
    label: str
    context_mode: Literal["fixed", "max"]
    context_tokens: int | None = None
    parallelism: int
    kv_cache_dtype: str
    placement_policy: str
    preferred_gpus: list[int] | None = None
    execution_policy: str
    load_policy: str
    enabled: bool
    is_default: bool
    priority: int
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RuntimePresetResolutionOut(BaseModel):
    preset_key: str
    resolved: bool
    requested_context: int | None = None
    effective_context: int | None = None
    model_max: int | None = None
    hardware_max: int | None = None
    policy_max: int | None = None
    requested_parallelism: int
    effective_parallelism: int
    target_gpus: list[int] = Field(default_factory=list)
    fits: bool
    runtime_key: str | None = None
    warnings: list[str] = Field(default_factory=list)


class RuntimePresetBenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cold_start: bool = False
    prompt: str = "Responde únicamente con una frase breve sobre computación."
    max_tokens: int = Field(default=64, ge=1, le=1024)


class RuntimePresetBenchmarkOut(BaseModel):
    preset_key: str
    effective_context: int
    worker_key: str
    was_cold: bool
    load_ms: int
    generation_ms: int
    output_tokens: int | None = None
    tokens_per_second: float | None = None


class ProfileCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(..., min_length=1, max_length=512)
    display_name: str | None = None
    description: str | None = None
    category: str = "llm"
    load_overrides: dict[str, Any] | None = None
    request_defaults: dict[str, Any] | None = None
    enabled: bool = True
    is_default: bool = False
    runtime_policy: RuntimePolicy | None = None
    # Bloque 20 — Router profiles. Ordered list of target profile_ids the
    # RouterResolver walks when this profile is requested. Non-null makes
    # the profile a router; null keeps it a plain profile.
    routing_targets: list[str] | None = None


class RouterCreate(BaseModel):
    """Create a model-independent routing profile."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(..., min_length=1, max_length=512)
    display_name: str | None = None
    description: str | None = None
    category: str = "llm"
    request_defaults: dict[str, Any] | None = None
    enabled: bool = True
    routing_targets: list[str] = Field(..., min_length=1)


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = None
    description: str | None = None
    category: str | None = None
    load_overrides: dict[str, Any] | None = None
    request_defaults: dict[str, Any] | None = None
    enabled: bool | None = None
    is_default: bool | None = None
    runtime_policy: RuntimePolicy | None = None
    routing_targets: list[str] | None = None


class ProfileRename(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_profile_id: str = Field(..., min_length=1, max_length=512)


class ProfileClone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_profile_id: str = Field(..., min_length=1, max_length=512)
    display_name: str | None = None


class ProfileAliasCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias_id: str = Field(
        ..., min_length=1, max_length=512, pattern=r"^[a-z0-9][a-z0-9.\-:]*$"
    )
    preset_key: str | None = Field(default=None, max_length=64)
    hidden: bool = True


class ProfileAliasOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    alias_id: str
    profile_id: str
    preset_key: str | None = None
    hidden: bool = True
    created_at: datetime | None = None


class ProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    profile_id: str
    profile_kind: Literal["execution", "router"] = "execution"
    base_model_id: str | None = None
    display_name: str | None = None
    description: str | None = None
    category: str
    load_overrides: dict[str, Any] | None = None
    request_defaults: dict[str, Any] | None = None
    assets: dict[str, Any] | None = None
    enabled: bool
    is_default: bool
    routing_targets: list[str] | None = None
    runtime_policy: RuntimePolicy | None = None
    runtime_presets: list[RuntimePresetOut] = Field(default_factory=list)
    aliases: list[ProfileAliasOut] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
