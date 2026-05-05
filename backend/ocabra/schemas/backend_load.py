"""Pydantic schemas for backend-specific load configuration.

These schemas describe the per-model overrides that the UI persists under
``ModelState.extra_config[<backend>]`` and that backends consume when launching
their worker process. They intentionally use ``None`` defaults so that callers
can serialise with ``model_dump(exclude_none=True)`` and let the runtime fall
back to the global defaults defined in ``ocabra.config.settings``.

Owner: Sprint 17.1 of the llama.cpp loader parity plan
(``docs/tasks/llama-cpp-parity-plan.md``). Sprints 17.2-17.4 will only ADD
fields to :class:`LlamaCppLoadConfig`; never rename or remove existing ones.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
