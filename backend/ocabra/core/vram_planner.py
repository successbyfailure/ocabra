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

    def kv_bytes_per_token(self, kv_dtype_bytes: float = 2.0) -> int:
        """KV cache bytes for one token across all layers (K and V)."""
        return int(
            self.layers * self.n_kv_heads * (self.key_length + self.value_length) * kv_dtype_bytes
        )


def kv_vram_mb(arch: ModelArch, context_tokens: int, kv_dtype_bytes: float = 2.0) -> float:
    """VRAM (MiB) the KV cache occupies for ``context_tokens`` total KV tokens."""
    return arch.kv_bytes_per_token(kv_dtype_bytes) * max(0, context_tokens) / _MB


def estimate_total_vram_mb(
    weights_mb: float,
    arch: ModelArch,
    kv_tokens: int,
    *,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
) -> int:
    """weights + KV(kv_tokens) + overhead, in MiB."""
    return int(weights_mb + kv_vram_mb(arch, kv_tokens, kv_dtype_bytes) + overhead_mb)


def max_context_tokens(
    free_mb: float,
    weights_mb: float,
    arch: ModelArch,
    *,
    slots: int = 1,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
) -> int:
    """Largest per-slot context that fits in ``free_mb`` given weights + overhead.

    ``slots`` divides the KV budget (Ollama: NUM_PARALLEL; llama.cpp: --parallel).
    """
    budget_mb = free_mb - weights_mb - overhead_mb
    if budget_mb <= 0:
        return 0
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
        n_kv = int(g("attention.head_count_kv"))
        key_len = int(g("attention.key_length"))
        val_len = int(g("attention.value_length") or key_len)
        hidden = int(g("embedding_length") or 0)
        ctx = int(g("context_length") or 0)
        if layers <= 0 or n_kv <= 0 or key_len <= 0:
            return None
        return ModelArch(layers, n_kv, key_len, val_len, hidden, ctx, 2.0)
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


def _gguf_consume_value(f, vtype: int):
    """Read a scalar/string value; seek past arrays (never needed). Returns the
    scalar/string, or None for arrays."""
    if vtype == _GGUF_TYPE_STRING:
        return _gguf_read_string(f)
    if vtype == _GGUF_TYPE_ARRAY:
        elem_type = _gguf_read(f, "<I")
        count = _gguf_read(f, "<Q")
        if elem_type in _GGUF_SCALAR_SIZE:
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
        n_kv = int(g("attention.head_count_kv"))
        hidden = int(g("embedding_length") or 0)
        n_heads = int(g("attention.head_count") or 0)
        key_len = g("attention.key_length")
        key_len = int(key_len) if key_len is not None else (hidden // n_heads if n_heads else 0)
        val_len = g("attention.value_length")
        val_len = int(val_len) if val_len is not None else key_len
        ctx = int(g("context_length") or 0)
        if layers <= 0 or n_kv <= 0 or key_len <= 0:
            return None
        return ModelArch(layers, n_kv, key_len, val_len, hidden, ctx, 2.0)
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
        weights_mb, arch, kv_tokens, kv_dtype_bytes=kv_dtype_bytes, overhead_mb=overhead_mb
    )


def plan_llama_cpp_vram_mb(
    arch: ModelArch,
    file_size_mb: float,
    ctx_size: int,
    *,
    gpu_layers: int | None = None,
    kv_dtype_bytes: float = 2.0,
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
) -> int:
    """llama.cpp: --ctx-size is the TOTAL KV budget (shared across --parallel), and
    only the ``gpu_layers`` offloaded layers hold weights + KV on the GPU.
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
    )
    return estimate_total_vram_mb(
        weights_mb, kv_arch, max(1, ctx_size), kv_dtype_bytes=kv_dtype_bytes, overhead_mb=overhead_mb
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
