"""VRAM planning: model footprint as a function of context and parallelism.

The core insight is that for an autoregressive transformer the KV cache — not the
weights — is what makes VRAM scale with context length and concurrency, and it is
*exactly* computable from the model's architecture metadata:

    KV_bytes_per_token = L · H_kv · (key_length + value_length) · b_kv

where L = layers, H_kv = KV heads (GQA), key/value_length = per-head K/V dim, and
b_kv = bytes per KV element (2 fp16, 1 fp8, ~0.5 for 4-bit KV). Total KV VRAM is
that per-token cost times the number of KV tokens the backend reserves.

This module exposes one primitive (``ModelArch.kv_bytes_per_token``) plus arch
extractors for the three metadata sources (HF ``config.json``, GGUF header, Ollama
``/api/show``) and a thin per-backend translator layer, because each backend
reserves KV differently:

    * Ollama    — num_ctx is per-slot, reserved up front: KV tokens = num_ctx · NUM_PARALLEL
    * llama.cpp — --ctx-size is the TOTAL, split across --parallel: KV tokens = ctx_size
                  (and only offloaded layers hold KV on the GPU)
    * vLLM      — paged KV; the footprint is a FIXED fraction (gpu_memory_utilization)
                  grabbed up front, so the useful question is inverted: what max_model_len
                  / concurrency fits in that pool.

Validated to the MB against measured Ollama footprints (see tests).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

_MB = 1024 * 1024

# CUDA context + compute buffers that don't scale meaningfully with context.
DEFAULT_OVERHEAD_MB = 400


@dataclass(frozen=True)
class ModelArch:
    """Architecture parameters needed to size the KV cache."""

    layers: int  # L (block_count / num_hidden_layers)
    n_kv_heads: int  # H_kv (num_key_value_heads / attention.head_count_kv)
    key_length: int  # per-head K dim (attention.key_length / head_dim)
    value_length: int  # per-head V dim (usually == key_length)
    hidden_size: int = 0
    context_length: int = 0  # native max context the model was trained/config'd for
    weight_dtype_bytes: float = 2.0  # bytes/param, for weight estimates when no file size
    # Exact long-context KV elements for hybrid/sliding-window architectures
    # that cannot be represented by one homogeneous layer/head tuple.
    kv_elements_per_token_override: int = 0
    # Fixed per-sequence VRAM (MiB) for a hybrid SSM/attention model's
    # recurrent-state cache ("rs cache": llama.cpp's llama_memory_recurrent
    # tensors, cache_r_l*/cache_s_l*, one pair per Mamba-style layer). This is
    # completely independent of context length — it is a per-sequence state,
    # not a per-token one — but it DOES scale with the number of concurrent
    # sequences a backend reserves slots for (--parallel / NUM_PARALLEL).
    # 0 for architectures without recurrent layers.
    rs_cache_mb_per_slot: float = 0.0
    # Fixed, ONE-TIME VRAM (MiB) folding together everything a load needs
    # that doesn't scale with context AND isn't derivable from GGUF metadata
    # for this architecture — dominated by a multimodal projector's resident
    # weights (mmproj: loaded once regardless of --parallel, not modeled
    # anywhere else in this module) plus a small residual for local/
    # sliding-window-attention layers' bounded KV cache (saturates at
    # sliding_window tokens — see arch_from_gguf's gemma4 branch for the
    # full calibration). Unlike rs_cache_mb_per_slot this is NOT multiplied
    # by slots or by (1 + rollback_snapshots) — the dominant component
    # (mmproj) is a one-time load, not a per-sequence cost, and scaling the
    # whole constant by slots overcounted it badly (confirmed live: a
    # 4-slot preset's estimated budget collapsed to a few thousand tokens
    # before this fix). The minor per-sequence portion this approximates
    # away is small enough not to matter in practice.
    fixed_load_overhead_mb: float = 0.0

    def kv_bytes_per_token(self, kv_dtype_bytes: float = 2.0) -> int:
        """KV cache bytes for one token across all layers (K and V)."""
        if self.kv_elements_per_token_override > 0:
            return int(self.kv_elements_per_token_override * kv_dtype_bytes)
        return int(
            self.layers * self.n_kv_heads * (self.key_length + self.value_length) * kv_dtype_bytes
        )


def kv_vram_mb(arch: ModelArch, context_tokens: int, kv_dtype_bytes: float = 2.0) -> float:
    """VRAM (MiB) the KV cache occupies for ``context_tokens`` total KV tokens."""
    return arch.kv_bytes_per_token(kv_dtype_bytes) * max(0, context_tokens) / _MB


#: llama.cpp --cache-type-k/--cache-type-v values (and the friendlier aliases
#: the UI/scheduler also accept) mapped to effective bytes/element. Quantized
#: average costs amortize the block header over the block size (q8_0 = 34B/32
#: elems ≈ 1.0, q5_* ≈ 0.65, q4_* ≈ 0.5) — the single source of truth for this
#: conversion; callers must not keep their own copy.
KV_CACHE_TYPE_BYTES: dict[str, float] = {
    "f16": 2.0, "fp16": 2.0, "bf16": 2.0,
    "q8_0": 1.0, "q8": 1.0,
    "q5_1": 0.65, "q5_0": 0.65,
    "q4_1": 0.5, "q4_0": 0.5, "q4": 0.5,
    "iq4_nl": 0.5,
}


def kv_dtype_bytes_for_cache_type(cache_type: str | None) -> float:
    """Bytes/element for a ``--cache-type-k``/``-v`` value. Unknown/``None``
    falls back to 2.0 (f16), llama.cpp's own built-in default."""
    if not cache_type:
        return 2.0
    return KV_CACHE_TYPE_BYTES.get(str(cache_type).lower(), 2.0)


def rs_cache_vram_mb(arch: ModelArch, slots: int = 1, rollback_snapshots: int = 0) -> float:
    """VRAM (MiB) for the hybrid SSM recurrent-state cache ("rs cache") across
    ``slots`` concurrent sequences. 0 for non-hybrid architectures.

    ``rollback_snapshots`` mirrors llama.cpp's ``n_rs_seq`` (partial-rollback
    snapshots the recurrent memory widens its tensors by, to support rewinding
    a rejected speculative draft — self-speculative MTP, EAGLE3, dflash,
    dspark). Each extra snapshot adds one more full copy of the per-slot state,
    so the real cost is ``(1 + rollback_snapshots)`` times larger than the
    plain base case. Defaults to 0 (no speculative rollback) — callers that
    know the model loads with one of those ``--spec-type`` values MUST pass
    the draft length here, or this under-counts the real reservation.
    """
    if arch.rs_cache_mb_per_slot <= 0:
        return 0.0
    return arch.rs_cache_mb_per_slot * max(1, slots) * (1 + max(0, rollback_snapshots))


def fixed_overhead_vram_mb(arch: ModelArch) -> float:
    """VRAM (MiB) for ``arch.fixed_load_overhead_mb``. 0 when the
    architecture has no calibrated value. A ONE-TIME cost — not multiplied
    by slots or by a rollback factor (see the field's docstring).
    """
    return max(0.0, arch.fixed_load_overhead_mb)


#: Empirically measured, NOT derived from source — self-speculative MTP's
#: draft-context sizing (its own small KV cache + compute buffer) and the
#: target context's compute/graph-reserve buffer growth beyond the flat
#: DEFAULT_OVERHEAD_MB weren't traced to their llama.cpp allocation formula.
#: Measured live against the real qwen3.8:27b worker (q8_0 KV, --spec-type
#: draft-mtp, ubatch=128, no --spec-draft-n-max override) by diffing
#: nvidia-smi before/after a clean load plus llama.cpp's own verbose
#: ("-v") per-buffer log lines ("KV buffer size"/"compute buffer size",
#: printed twice — once for the target context, once for the draft one):
#:   1 slot / 32768  ctx: target_compute 161.27 + draft_kv 128.00 + draft_compute 32.50 =  321.77 MiB
#:   2 slots/131072 ctx: target_compute 553.27 + draft_kv 512.00 + draft_compute 33.00 = 1098.27 MiB
#:   3 slots/ 98304 ctx: target_compute 417.53 + draft_kv 384.00 + draft_compute 32.76 =  834.29 MiB
#: NOT monotonic in slots (2 slots costs MORE than 3) — llama.cpp's internal
#: graph/batch sizing doesn't scale as a clean function of slot count, so a
#: curve fit (the original slots^2 guess, calibrated off only the first two
#: points) was overfitting noise and ended up MORE conservative than reality
#: warranted, clamping 3-slot presets below what actually fits. This is a
#: lookup table of real measurements instead of a formula. Verified against
#: all three real loads: using the exact measured value at each point yields
#: a consistent ~775-780 MiB safety margin over the real nvidia-smi total —
#: tight-but-safe, not padded. Add a new entry (same nvidia-smi diff + "-v"
#: log method) rather than trusting extrapolation if a slot count outside
#: this table is ever actually used.
_MTP_DRAFT_OVERHEAD_MEASURED_MB: dict[int, float] = {1: 321.77, 2: 1098.27, 3: 834.29}
#: For slot counts not in the table above: the highest per-slot rate seen
#: among the measured points (2-slot: 1098.27/2), used as a conservative
#: (not necessarily accurate) upper bound — safer than assuming the 3-slot
#: dip continues indefinitely, which is unverified.
_MTP_DRAFT_OVERHEAD_FALLBACK_MB_PER_SLOT = max(
    v / k for k, v in _MTP_DRAFT_OVERHEAD_MEASURED_MB.items()
)


def mtp_draft_overhead_mb(slots: int) -> float:
    """Extra VRAM (MiB) for self-speculative MTP/EAGLE3/dflash/dspark loads,
    beyond KV + rs_cache: the draft model's own KV cache + compute buffer,
    plus the target context's compute buffer (doesn't scale cleanly with
    ``slots`` — see the calibration note on
    ``_MTP_DRAFT_OVERHEAD_MEASURED_MB``). 0 for ``slots <= 0``; callers
    should only add this when the load actually uses one of those
    ``--spec-type`` values (same condition as ``rs_rollback_snapshots > 0``
    in ``rs_cache_vram_mb``) — it is meaningless for a plain/non-speculative
    load.
    """
    if slots <= 0:
        return 0.0
    measured = _MTP_DRAFT_OVERHEAD_MEASURED_MB.get(slots)
    if measured is not None:
        return measured
    return _MTP_DRAFT_OVERHEAD_FALLBACK_MB_PER_SLOT * slots


def estimate_total_vram_mb(
    weights_mb: float,
    arch: ModelArch,
    kv_tokens: int,
    *,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
    slots: int = 1,
    rs_rollback_snapshots: int = 0,
) -> int:
    """weights + KV(kv_tokens) + rs_cache(slots) + fixed_overhead (one-time) + overhead, in MiB."""
    rs_mb = rs_cache_vram_mb(arch, slots=slots, rollback_snapshots=rs_rollback_snapshots)
    fixed_mb = fixed_overhead_vram_mb(arch)
    return int(
        weights_mb + kv_vram_mb(arch, kv_tokens, kv_dtype_bytes) + rs_mb + fixed_mb + overhead_mb
    )


#: Fraction of (free - weights - overhead) actually usable for KV cache after
#: runtime buffers the model-level estimator doesn't track (CUDA graphs,
#: Flash-Attention workspace, Ollama runner scratch, chunked prefill tokens).
#: 0.90 means we assume 10% of the "logical" KV budget is burnt on buffers.
#: Set from measurements: without this margin, gemma4:26b was reporting fits
#: for 262k on a 24 GB 3090 and Ollama rejected the load with HTTP 500.
DEFAULT_KV_SAFETY_FACTOR = 0.90


def max_context_tokens(
    free_mb: float,
    weights_mb: float,
    arch: ModelArch,
    *,
    slots: int = 1,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
    safety_factor: float = DEFAULT_KV_SAFETY_FACTOR,
    rs_rollback_snapshots: int = 0,
) -> int:
    """Largest per-slot context that fits in ``free_mb`` given weights + overhead.

    ``slots`` divides the KV budget (Ollama: NUM_PARALLEL; llama.cpp: --parallel).
    ``safety_factor`` discounts the raw KV budget to leave headroom for runtime
    buffers the estimator doesn't model (CUDA graphs, Flash-Attention workspace,
    scratch allocations). Default 0.90. For hybrid SSM/attention architectures,
    the fixed rs_cache(slots) cost is deducted from the budget up front (it
    does not scale with context, so it must not eat into the KV-per-token math).
    """
    rs_mb = rs_cache_vram_mb(arch, slots=slots, rollback_snapshots=rs_rollback_snapshots)
    fixed_mb = fixed_overhead_vram_mb(arch)
    budget_mb = free_mb - weights_mb - overhead_mb - rs_mb - fixed_mb
    if budget_mb <= 0:
        return 0
    budget_mb *= max(0.0, min(1.0, safety_factor))
    per_token_mb = arch.kv_bytes_per_token(kv_dtype_bytes) / _MB
    if per_token_mb <= 0:
        return 0
    total_tokens = budget_mb / per_token_mb
    return int(total_tokens / max(1, slots))


# ---------------------------------------------------------------------------
# Architecture extraction — HF config.json
# ---------------------------------------------------------------------------

def arch_from_hf_config(cfg: dict) -> ModelArch | None:
    try:
        layers = int(cfg["num_hidden_layers"])
        n_heads = int(cfg.get("num_attention_heads", 0) or 0)
        n_kv = int(cfg.get("num_key_value_heads", n_heads) or n_heads)
        hidden = int(cfg.get("hidden_size", 0) or 0)
        head_dim = int(cfg.get("head_dim") or (hidden // n_heads if n_heads else 0))
        ctx = int(cfg.get("max_position_embeddings", 0) or 0)
        dtype = str(cfg.get("torch_dtype", "float16")).lower()
        if "float32" in dtype:
            wb = 4.0
        elif "float8" in dtype or "fp8" in dtype:
            wb = 1.0
        else:
            wb = 2.0
        if layers <= 0 or n_kv <= 0 or head_dim <= 0:
            return None
        return ModelArch(layers, n_kv, head_dim, head_dim, hidden, ctx, wb)
    except (KeyError, TypeError, ValueError):
        return None


def _hybrid_rs_cache_mb_per_slot(
    recurrent_layers: int,
    ssm_d_conv: int,
    ssm_d_inner: int,
    ssm_d_state: int,
    ssm_n_group: int,
) -> float:
    """VRAM (MiB) for ONE sequence's recurrent-state cache across all of a
    hybrid model's SSM/Mamba-style layers.

    Mirrors llama.cpp's ``llama_hparams::n_embd_r()``/``n_embd_s()``
    (src/llama-hparams.cpp) for the plain Mamba2-style case (the one qwen35 /
    qwen3next / nemotron-h use — not the RWKV/LFM2/KDA/linear-attention
    variants, which size their state differently):

        n_embd_r = (ssm_d_conv - 1) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state)  # conv history
        n_embd_s = ssm_d_state * ssm_d_inner                                         # SSM state

    llama_memory_recurrent allocates two tensors per recurrent layer sized
    ``n_embd_r`` / ``n_embd_s`` elements, always as F32 regardless of the
    configured KV cache dtype (``recurrent_type_r``/``recurrent_type_s`` are
    hardcoded ``GGML_TYPE_F32`` at every ``llama_memory_hybrid*`` construction
    site in llama-model.cpp) — hence the fixed 4 bytes/element below.

    This is the base case: one row per sequence (llama.cpp's ``mem_size`` ==
    ``n_seq_max`` when there is no speculative rollback, i.e. ``n_rs_seq=0``).
    Self-speculative decoding (``--spec-type draft-mtp``/``draft-eagle3``/
    ``draft-dflash``/``draft-dspark``) widens the tensors to
    ``mem_size * (1 + n_rs_seq)`` rows — see ``rs_cache_vram_mb``'s
    ``rollback_snapshots`` parameter, which the caller must supply separately.
    """
    if recurrent_layers <= 0 or ssm_d_inner <= 0 or ssm_d_state <= 0:
        return 0.0
    n_embd_r = max(0, ssm_d_conv - 1) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state)
    n_embd_s = ssm_d_state * ssm_d_inner
    bytes_per_slot = recurrent_layers * (n_embd_r + n_embd_s) * 4  # F32
    return bytes_per_slot / _MB


# ---------------------------------------------------------------------------
# Architecture extraction — Ollama /api/show model_info
# ---------------------------------------------------------------------------

def arch_from_ollama_model_info(model_info: dict, arch_family: str) -> ModelArch | None:
    """``model_info`` is the ``/api/show`` block; keys are namespaced by family
    (e.g. ``qwen3.block_count``). ``arch_family`` comes from ``details.family``.
    """
    prefix = arch_family
    if not prefix:
        return None

    def g(suffix: str):
        return model_info.get(f"{prefix}.{suffix}")

    try:
        layers = int(g("block_count"))
        raw_n_kv = g("attention.head_count_kv")
        n_kv_values = (
            [int(value) for value in raw_n_kv]
            if isinstance(raw_n_kv, list)
            else [int(raw_n_kv)] * layers
        )
        if len(n_kv_values) != layers:
            return None
        n_kv = max(n_kv_values)
        key_len = int(g("attention.key_length"))
        val_len = int(g("attention.value_length") or key_len)
        hidden = int(g("embedding_length") or 0)
        ctx = int(g("context_length") or 0)
        if layers <= 0 or n_kv <= 0 or key_len <= 0:
            return None
        kv_elements_override = 0
        rs_cache_mb_per_slot = 0.0
        sliding_pattern = g("attention.sliding_window_pattern")
        if isinstance(sliding_pattern, list) and len(sliding_pattern) == layers:
            # Local sliding-window layers retain a fixed-size cache. Only the
            # global-attention layers grow with long context.
            kv_elements_override = sum(
                heads * (key_len + val_len)
                for heads, sliding in zip(n_kv_values, sliding_pattern, strict=True)
                if not sliding
            )
        else:
            full_attention_interval = int(g("full_attention_interval") or 0)
            if full_attention_interval > 1:
                # Hybrid SSM/attention models allocate long KV only on their
                # periodic full-attention blocks. The remaining (recurrent)
                # layers hold a fixed-size state instead ("rs cache").
                attention_layers = (
                    layers + full_attention_interval - 1
                ) // full_attention_interval
                kv_elements_override = (
                    attention_layers * n_kv * (key_len + val_len)
                )
                recurrent_layers = layers - attention_layers
                rs_cache_mb_per_slot = _hybrid_rs_cache_mb_per_slot(
                    recurrent_layers,
                    int(g("ssm.conv_kernel") or 0),
                    int(g("ssm.inner_size") or 0),
                    int(g("ssm.state_size") or 0),
                    int(g("ssm.group_count") or 0),
                )
        return ModelArch(
            layers,
            n_kv,
            key_len,
            val_len,
            hidden,
            ctx,
            2.0,
            kv_elements_override,
            rs_cache_mb_per_slot,
        )
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Architecture extraction — GGUF header (llama.cpp files)
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"
# scalar type id -> struct format
_GGUF_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}
_GGUF_SCALAR_SIZE = {t: struct.calcsize(f) for t, f in _GGUF_SCALAR_FMT.items()}
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9


def _gguf_read(f, fmt: str):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _gguf_read_string(f) -> str:
    n = _gguf_read(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


#: Small per-layer arrays (e.g. attention.sliding_window_pattern, one bool
#: per block_count) are worth capturing for hybrid-architecture KV sizing;
#: huge arrays (tokenizer vocab, merges — tens of thousands of entries) are
#: not, both for parse speed and because nothing reads them. This caps what
#: gets materialized into a Python list rather than seek-skipped.
_GGUF_MAX_ARRAY_CAPTURE = 4096


def _gguf_consume_value(f, vtype: int):
    """Read a scalar/string value, or a small scalar array (see
    ``_GGUF_MAX_ARRAY_CAPTURE``); seek past large/string arrays (never
    needed). Returns the scalar/string/list, or None for skipped arrays."""
    if vtype == _GGUF_TYPE_STRING:
        return _gguf_read_string(f)
    if vtype == _GGUF_TYPE_ARRAY:
        elem_type = _gguf_read(f, "<I")
        count = _gguf_read(f, "<Q")
        if elem_type in _GGUF_SCALAR_SIZE:
            if count <= _GGUF_MAX_ARRAY_CAPTURE:
                fmt = _GGUF_SCALAR_FMT[elem_type]
                return [_gguf_read(f, fmt) for _ in range(count)]
            f.seek(_GGUF_SCALAR_SIZE[elem_type] * count, 1)
        elif elem_type == _GGUF_TYPE_STRING:
            for _ in range(count):
                f.seek(_gguf_read(f, "<Q"), 1)
        else:
            raise ValueError(f"unsupported nested GGUF array type {elem_type}")
        return None
    fmt = _GGUF_SCALAR_FMT.get(vtype)
    if fmt is None:
        raise ValueError(f"unknown GGUF value type {vtype}")
    return _gguf_read(f, fmt)


def read_gguf_metadata(path: str | Path) -> dict:
    """Parse the GGUF metadata key/value section. Arrays (tokenizer vocab, etc.)
    are seek-skipped, so this reads only the small header regardless of file size.
    """
    out: dict = {}
    with open(path, "rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        _version = _gguf_read(f, "<I")
        _tensor_count = _gguf_read(f, "<Q")
        kv_count = _gguf_read(f, "<Q")
        for _ in range(kv_count):
            key = _gguf_read_string(f)
            vtype = _gguf_read(f, "<I")
            value = _gguf_consume_value(f, vtype)
            if value is not None:
                out[key] = value
    return out


def arch_from_gguf(path: str | Path) -> ModelArch | None:
    try:
        md = read_gguf_metadata(path)
    except (OSError, ValueError, struct.error):
        return None
    arch = md.get("general.architecture")
    if not arch:
        return None

    def g(suffix: str):
        return md.get(f"{arch}.{suffix}")

    try:
        layers = int(g("block_count"))
        n_heads = int(g("attention.head_count") or 0)
        # head_count_kv is either a scalar, a per-layer array (e.g. gemma4:
        # global-attention layers use fewer KV heads than local/SWA ones —
        # see the sliding_pattern branch below, which uses the per-layer
        # values when both arrays line up), or absent entirely for
        # architectures without grouped-query attention (falls back to
        # head_count: full MHA, n_kv == n_heads).
        raw_n_kv = g("attention.head_count_kv")
        n_kv_values = (
            [int(v) for v in raw_n_kv]
            if isinstance(raw_n_kv, list)
            else None
        )
        n_kv = max(n_kv_values) if n_kv_values else int(raw_n_kv or n_heads or 0)
        hidden = int(g("embedding_length") or 0)
        key_len = g("attention.key_length")
        key_len = int(key_len) if key_len is not None else (hidden // n_heads if n_heads else 0)
        val_len = g("attention.value_length")
        val_len = int(val_len) if val_len is not None else key_len
        ctx = int(g("context_length") or 0)
        if layers <= 0 or n_kv <= 0 or key_len <= 0:
            return None

        kv_elements_override = 0
        rs_cache_mb_per_slot = 0.0
        sliding_pattern = g("attention.sliding_window_pattern")
        if isinstance(sliding_pattern, list) and len(sliding_pattern) == layers:
            # Local sliding-window-attention layers (e.g. gemma4: SWA layers
            # interleaved with periodic global-attention ones) retain only a
            # bounded (sliding_window-sized) KV cache that doesn't grow with
            # context — dropped from the linear-in-context term entirely
            # (same simplification as arch_from_ollama_model_info(), the
            # only place this was modeled before: direct GGUF parsing never
            # captured array-typed metadata at all until now). Only the
            # global-attention layers scale with context_tokens. Uses the
            # per-layer head_count_kv values when present (global and local
            # layers can use different KV head counts, e.g. gemma4) rather
            # than the single max()-derived n_kv.
            per_layer_n_kv = (
                n_kv_values if n_kv_values and len(n_kv_values) == layers else [n_kv] * layers
            )
            kv_elements_override = sum(
                heads * (key_len + val_len)
                for heads, sliding in zip(per_layer_n_kv, sliding_pattern, strict=True)
                if not sliding
            )
        else:
            full_attention_interval = int(g("full_attention_interval") or 0)
            if full_attention_interval > 1:
                # Hybrid SSM/attention models (e.g. qwen35: Mamba-style layers
                # interleaved with periodic full-attention blocks) allocate long
                # KV only on their full-attention blocks; the rest are recurrent
                # layers with a fixed-size state ("rs cache", modeled below) —
                # mirrors the same branch in arch_from_ollama_model_info().
                attention_layers = (layers + full_attention_interval - 1) // full_attention_interval
                kv_elements_override = attention_layers * n_kv * (key_len + val_len)
                recurrent_layers = layers - attention_layers
                rs_cache_mb_per_slot = _hybrid_rs_cache_mb_per_slot(
                    recurrent_layers,
                    int(g("ssm.conv_kernel") or 0),
                    int(g("ssm.inner_size") or 0),
                    int(g("ssm.state_size") or 0),
                    int(g("ssm.group_count") or 0),
                )

        fixed_load_overhead_mb = 0.0
        if arch == "gemma4":
            # The sliding_pattern branch above now correctly derives the
            # pure-KV linear term from gemma4's real per-layer metadata (5
            # global-attention layers out of 30, n_kv=2, 512+512 dims ->
            # 10240 elements/token — confirmed present as actual arrays in
            # the raw GGUF once _gguf_consume_value started capturing small
            # arrays; previously invisible, not actually absent as first
            # assumed). But 2 real loads of llama_cpp/gemma4-26b (1 slot,
            # --spec-type draft-mtp, q8_0 KV, with mmproj — 2026-10-27)
            # measure notably more than that pure-KV math predicts:
            #   ctx=32768  -> 19009 MiB real (pure-KV predicts ~17325 MiB)
            #   ctx=131072 -> 21225 MiB real (pure-KV predicts ~18285 MiB)
            # i.e. a real residual that itself scales with context — almost
            # certainly compute/graph buffers growing with sequence length
            # beyond raw KV storage, not modeled anywhere in this module.
            # Solving residual = C + S*ctx for the two unknowns (both
            # measurements used 1 slot, so C is directly the one-time
            # constant — no slots scaling to back out): S ≈ 13396
            # elements/token extra, C ≈ 1266 MiB — the latter bundles
            # mmproj's resident weights (~1140 MiB, a ONE-TIME load
            # regardless of --parallel, not modeled anywhere else here)
            # plus the local/SWA layers' small bounded cache (~100 MiB,
            # saturates at the 1024-token sliding_window, and in principle
            # scales with slots, but is too small relative to mmproj to
            # bother splitting out) plus residual. Added on top of (not
            # replacing) the derived base so this stays correct if a future
            # Ollama re-export changes the per-layer arrays. Reproduces
            # both real measurements to within ~1 MiB. The two added
            # constants are NOT derived from source — re-measure if this
            # model's real VRAM footprint changes.
            kv_elements_override += 13396
            fixed_load_overhead_mb = 1266.0

        return ModelArch(
            layers, n_kv, key_len, val_len, hidden, ctx, 2.0,
            kv_elements_override, rs_cache_mb_per_slot, fixed_load_overhead_mb,
        )
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-backend translators
# ---------------------------------------------------------------------------

def plan_ollama_vram_mb(
    arch: ModelArch,
    weights_mb: float,
    num_ctx: int,
    num_parallel: int = 1,
    *,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
) -> int:
    """Ollama reserves KV for ``num_ctx`` per slot × ``num_parallel`` slots up front."""
    kv_tokens = max(1, num_ctx) * max(1, num_parallel)
    return estimate_total_vram_mb(
        weights_mb,
        arch,
        kv_tokens,
        kv_dtype_bytes=kv_dtype_bytes,
        overhead_mb=overhead_mb,
        slots=max(1, num_parallel),
    )


def plan_llama_cpp_vram_mb(
    arch: ModelArch,
    file_size_mb: float,
    ctx_size: int,
    *,
    gpu_layers: int | None = None,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
    parallel_slots: int = 1,
    rs_rollback_snapshots: int = 0,
) -> int:
    """llama.cpp: --ctx-size is the TOTAL KV budget (shared across --parallel), and
    only the ``gpu_layers`` offloaded layers hold weights + KV on the GPU.

    ``parallel_slots`` is llama.cpp's ``--parallel`` (number of concurrent
    sequences); it does not change the KV token budget (``ctx_size`` is
    already the total), but it DOES scale the hybrid rs_cache term, which is
    per-sequence rather than per-token. ``rs_rollback_snapshots`` must be set
    by the caller when the load uses self-speculative decoding
    (``--spec-type draft-mtp``/``draft-eagle3``/``draft-dflash``/
    ``draft-dspark``), matching llama.cpp's ``n_rs_seq`` (defaults to the
    draft length, 3, when ``--spec-draft-n-max`` isn't overridden) — omitting
    it under-counts the real reservation for those loads.
    """
    total_layers = max(1, arch.layers)
    offloaded = total_layers if gpu_layers is None else max(0, min(gpu_layers, total_layers))
    if offloaded <= 0:
        return 0
    frac = offloaded / total_layers
    weights_mb = file_size_mb * frac
    kv_arch = arch if offloaded == total_layers else ModelArch(
        offloaded, arch.n_kv_heads, arch.key_length, arch.value_length,
        arch.hidden_size, arch.context_length, arch.weight_dtype_bytes,
        int(arch.kv_elements_per_token_override * frac) if arch.kv_elements_per_token_override else 0,
        arch.rs_cache_mb_per_slot * frac,
        arch.fixed_load_overhead_mb * frac,
    )
    # rs_rollback_snapshots > 0 is the same signal rs_cache_vram_mb() uses for
    # "this load self-speculates" — reuse it here instead of adding yet
    # another spec-type parameter.
    draft_overhead_mb = (
        mtp_draft_overhead_mb(parallel_slots) if rs_rollback_snapshots > 0 else 0.0
    )
    return estimate_total_vram_mb(
        weights_mb,
        kv_arch,
        max(1, ctx_size),
        kv_dtype_bytes=kv_dtype_bytes,
        overhead_mb=overhead_mb + draft_overhead_mb,
        slots=max(1, parallel_slots),
        rs_rollback_snapshots=rs_rollback_snapshots,
    )


def vllm_footprint_mb(total_vram_mb: float, gpu_memory_utilization: float = 0.85) -> int:
    """vLLM grabs this fraction of the GPU up front regardless of context — the KV
    pool is carved from it, so the footprint for placement is essentially fixed."""
    return int(total_vram_mb * gpu_memory_utilization)


def vllm_max_model_len(
    arch: ModelArch,
    weights_mb: float,
    total_vram_mb: float,
    *,
    gpu_memory_utilization: float = 0.85,
    concurrency: int = 1,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
) -> int:
    """Max per-sequence context vLLM can serve for ``concurrency`` simultaneous
    sequences without preemption, given its fixed KV pool."""
    pool_mb = total_vram_mb * gpu_memory_utilization - weights_mb - overhead_mb
    if pool_mb <= 0:
        return 0
    per_token_mb = arch.kv_bytes_per_token(kv_dtype_bytes) / _MB
    if per_token_mb <= 0:
        return 0
    pool_tokens = pool_mb / per_token_mb
    return int(pool_tokens / max(1, concurrency))


# ---------------------------------------------------------------------------
# Capacity report (used by the /capacity endpoint and use-case planning)
# ---------------------------------------------------------------------------

# Backends whose KV footprint is a fixed pool carved from the GPU up front.
_POOLED_BACKENDS = {"vllm", "sglang"}


def _cap(value: int, native: int) -> int:
    return min(value, native) if native else value


def capacity_rows(
    backend_type: str,
    arch: ModelArch,
    weights_mb: float,
    *,
    gpu_total_mb: float,
    slots: tuple[int, ...] = (1, 2, 4),
    gpu_memory_utilization: float = 0.85,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
) -> list[dict]:
    """Per-slot (or per-concurrency, for pooled backends) max context table.

    Computed against the GPU's *total* VRAM — the planning question is "the most
    context this model can serve if it owns this GPU" (oCabra evicts to make room
    on load), not what happens to be free this instant. vLLM/SGLang additionally
    apply gpu_memory_utilization to their fixed pool. Capped at native context.
    """
    native = arch.context_length or 0
    rows: list[dict] = []
    for s in slots:
        if backend_type in _POOLED_BACKENDS:
            ctx = vllm_max_model_len(
                arch, weights_mb, gpu_total_mb,
                gpu_memory_utilization=gpu_memory_utilization,
                concurrency=s, kv_dtype_bytes=kv_dtype_bytes, overhead_mb=overhead_mb,
            )
        else:
            ctx = max_context_tokens(
                gpu_total_mb, weights_mb, arch,
                slots=s, kv_dtype_bytes=kv_dtype_bytes, overhead_mb=overhead_mb,
            )
        rows.append({"slots": s, "max_context": ctx, "max_context_capped": _cap(ctx, native)})
    return rows


def plan_use_case(
    backend_type: str,
    arch: ModelArch,
    weights_mb: float,
    *,
    gpu_total_mb: float,
    requested_context: int | str | None,
    slots: int = 1,
    gpu_memory_utilization: float = 0.85,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
    rs_rollback_snapshots: int = 0,
) -> dict:
    """Resolve a use case (target context + parallelism) to an effective context.

    ``requested_context`` may be an int, or None/"max"/"auto"/0 to mean "the most
    that fits". Returns the effective context (clamped to what fits and to the
    model's native context), the fitting maximum, and any clamp warnings.

    ``rs_rollback_snapshots`` must be set (see ``rs_cache_vram_mb``) when the
    load self-speculates (``--spec-type draft-mtp``/``draft-eagle3``/
    ``draft-dflash``/``draft-dspark``) — it both widens the hybrid rs_cache
    deduction and adds the draft model's own KV+compute overhead
    (``mtp_draft_overhead_mb``). Omitting it for such a load makes this
    function report more context fits than the real process can hold.
    """
    native = arch.context_length or 0
    if backend_type in _POOLED_BACKENDS:
        max_ctx = vllm_max_model_len(
            arch, weights_mb, gpu_total_mb,
            gpu_memory_utilization=gpu_memory_utilization,
            concurrency=slots, kv_dtype_bytes=kv_dtype_bytes, overhead_mb=overhead_mb,
        )
    else:
        draft_overhead_mb = (
            mtp_draft_overhead_mb(slots) if rs_rollback_snapshots > 0 else 0.0
        )
        max_ctx = max_context_tokens(
            gpu_total_mb, weights_mb, arch,
            slots=slots, kv_dtype_bytes=kv_dtype_bytes,
            overhead_mb=overhead_mb + draft_overhead_mb,
            rs_rollback_snapshots=rs_rollback_snapshots,
        )
    max_ctx = _cap(max_ctx, native)

    warnings: list[str] = []
    auto = requested_context in (None, 0, "max", "auto")
    if auto:
        effective = max_ctx
    else:
        effective = int(requested_context)
        if native and effective > native:
            warnings.append(f"contexto {effective} > nativo {native}; capado a {native}")
            effective = native
        if effective > max_ctx:
            warnings.append(
                f"contexto {effective} no cabe con {slots} slots "
                f"(máx {max_ctx}); reducido a {max_ctx}"
            )
            effective = max_ctx

    return {
        "effective_context": effective,
        "max_context": max_ctx,
        "slots": slots,
        "auto": auto,
        "warnings": warnings,
    }


def vram_curve(
    arch: ModelArch,
    weights_mb: float,
    *,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
    points: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144),
) -> list[dict]:
    """Total VRAM (weights + KV + overhead) at a series of single-slot contexts,
    truncated at the model's native context."""
    native = arch.context_length or points[-1]
    out: list[dict] = []
    seen: set[int] = set()
    for ctx in points:
        c = min(ctx, native)
        if c in seen:
            continue
        seen.add(c)
        out.append({
            "context": c,
            "vram_mb": estimate_total_vram_mb(
                weights_mb, arch, c, kv_dtype_bytes=kv_dtype_bytes, overhead_mb=overhead_mb
            ),
        })
    return out
