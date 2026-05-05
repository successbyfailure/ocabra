"""Pydantic schemas for backend-specific load configuration.

These schemas describe the per-model overrides that the UI persists under
``ModelState.extra_config[<backend>]`` and that backends consume when launching
their worker process. They intentionally use ``None`` defaults so that callers
can serialise with ``model_dump(exclude_none=True)`` and let the runtime fall
back to the global defaults defined in ``ocabra.config.settings``.

Owner: Bloque 17 of the llama.cpp loader parity plan
(``docs/tasks/llama-cpp-parity-plan.md``).

The schema is **strictly additive** across the four sprints:
    * 17.1 — Tier 1 (foundation + trivial flags).
    * 17.2 — KV-quant fields.
    * 17.3 — Multi-GPU + MoE offload.
    * 17.4 — Speculative decoding, alternate runtime, concurrent slots.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

KvCacheType = Literal["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]
SplitMode = Literal["layer", "row", "none"]
SplitStrategy = Literal["evenly", "favor_main"]
RuntimeType = Literal["cuda", "rocm", "vulkan", "cpu"]


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------


class SpeculativeConfig(BaseModel):
    """Speculative decoding parameters (Sprint 17.4).

    The draft model must be a llama.cpp-compatible GGUF with the same
    tokenizer (``vocab_size``, ``bos_id``, ``eos_id``) as the target model.
    """

    model_config = ConfigDict(extra="forbid")

    draft_model_id: str = Field(
        description="Canonical model_id of the draft (smaller) model.",
    )
    draft_n: int | None = Field(
        default=None,
        description="Maximum draft tokens to predict per step (--draft-max).",
    )
    draft_min: int | None = Field(
        default=None,
        description="Minimum draft tokens before validating (--draft-min).",
    )
    draft_p_min: float | None = Field(
        default=None,
        description="Minimum acceptance probability for drafts (--draft-p-min).",
    )


class LlamaCppLoadConfig(BaseModel):
    """Per-model overrides for the llama.cpp backend.

    Field semantics:

    * Numeric ``None`` => use global default (``settings.llama_cpp_*``) or the
      llama-server built-in default.
    * Boolean ``None`` => use global default. ``False`` is a meaningful value
      (e.g. ``mmap=False`` emits ``--no-mmap``) so callers must distinguish
      "unset" from "explicitly false".
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # --- Sprint 17.1 (Tier 1) ---
    gpu_layers: int | None = Field(default=None, ge=0)
    ctx_size: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    ubatch_size: int | None = Field(default=None, ge=1)
    threads: int | None = Field(default=None, ge=1)
    flash_attn: bool | None = None
    mlock: bool | None = None
    embedding: bool | None = None
    # ``mmap`` is the inverted form of ``--no-mmap``: True (or None) means
    # "use mmap", False means "pass --no-mmap to llama-server".
    mmap: bool | None = None
    seed: int | None = None
    no_kv_offload: bool | None = None
    rope_freq_base: float | None = Field(default=None, gt=0.0)
    rope_freq_scale: float | None = Field(default=None, gt=0.0)

    # --- Sprint 17.2 (KV-quant) ---
    cache_type_k: KvCacheType | None = None
    cache_type_v: KvCacheType | None = None  # requires flash_attn=True if != f16

    # --- Sprint 17.3 (Multi-GPU + MoE CPU offload) ---
    main_gpu: int | None = Field(default=None, ge=0)
    tensor_split: list[float] | None = None
    split_mode: SplitMode | None = None
    disabled_gpus: list[int] | None = None
    split_strategy: SplitStrategy | None = None
    n_cpu_moe: int | None = Field(default=None, ge=0)
    override_tensor: str | None = None

    # --- Sprint 17.4 (Speculative decoding + runtime + concurrent slots) ---
    speculative: SpeculativeConfig | None = None
    runtime: RuntimeType | None = None
    parallel_slots: int | None = Field(default=None, ge=1)
    cont_batching: bool | None = None
    keep_alive_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_cache_type_v_requires_flash_attn(self) -> LlamaCppLoadConfig:
        """When ``cache_type_v`` is quantized, ``flash_attn`` must be enabled."""
        if self.cache_type_v is not None and self.cache_type_v != "f16" and not self.flash_attn:
            raise ValueError(
                "cache_type_v != 'f16' requires flash_attn=True (llama.cpp only "
                "supports quantized V cache when flash attention is enabled)."
            )
        return self

    @field_validator("tensor_split")
    @classmethod
    def _validate_tensor_split(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("tensor_split must contain at least one ratio")
        if any(ratio < 0 for ratio in value):
            raise ValueError("tensor_split ratios must be non-negative")
        if all(ratio == 0 for ratio in value):
            raise ValueError("tensor_split must contain at least one non-zero ratio")
        return value

    @field_validator("disabled_gpus")
    @classmethod
    def _validate_disabled_gpus(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(idx < 0 for idx in value):
            raise ValueError("disabled_gpus indices must be non-negative")
        seen: set[int] = set()
        unique: list[int] = []
        for idx in value:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
        return unique
