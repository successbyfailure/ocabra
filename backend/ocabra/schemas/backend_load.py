"""Pydantic schemas for backend-specific load configuration.

These schemas describe the per-model overrides that the UI persists under
``ModelState.extra_config[<backend>]`` and that backends consume when launching
their worker process. They intentionally use ``None`` defaults so that callers
can serialise with ``model_dump(exclude_none=True)`` and let the runtime fall
back to the global defaults defined in ``ocabra.config.settings``.

Owner: Bloque 17 of the llama.cpp loader parity plan
(``docs/tasks/llama-cpp-parity-plan.md``). Each Sprint (17.1-17.4) only ADDS
fields to :class:`LlamaCppLoadConfig`; never rename or remove existing ones.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

KvCacheType = Literal["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------


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
    # Pre-existing fields (kept here so the schema is the single source of
    # truth for the UI and tests):
    gpu_layers: int | None = Field(default=None, ge=0)
    ctx_size: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    ubatch_size: int | None = Field(default=None, ge=1)
    threads: int | None = Field(default=None, ge=1)
    flash_attn: bool | None = None
    mlock: bool | None = None
    embedding: bool | None = None
    # New in Sprint 17.1.
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

    @model_validator(mode="after")
    def _validate_cache_type_v_requires_flash_attn(self) -> LlamaCppLoadConfig:
        """When ``cache_type_v`` is quantized, ``flash_attn`` must be enabled.

        llama.cpp (mirroring LM Studio's gating) only supports a quantized V
        cache when flash attention is active. We surface a friendly error at
        load-config validation time rather than letting llama-server crash on
        startup.
        """
        if self.cache_type_v is not None and self.cache_type_v != "f16" and not self.flash_attn:
            raise ValueError(
                "cache_type_v != 'f16' requires flash_attn=True (llama.cpp only "
                "supports quantized V cache when flash attention is enabled)."
            )
        return self
