"""Deterministic VRAM estimator for llama.cpp / GGUF models.

Parses the GGUF header (magic + version + tensor_count + metadata KV pairs)
without loading the model weights, then applies the standard llama.cpp KV cache
formula to produce a per-component byte breakdown suitable for the UI.

Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

from ocabra.schemas.backend_load import KvCacheType, LlamaCppLoadConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"

# GGUF metadata value type IDs (see spec).
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

# Bytes per element for each KV cache type. Values for quantized formats are
# average effective per-element costs (block sizes amortized): q8_0 = 34 bytes
# / 32 elems ≈ 1.0625, q5_1 ≈ 0.6875, q5_0 ≈ 0.625, q4_1 ≈ 0.5625, q4_0 = 0.5.
_KV_BYTES_PER_ELEMENT: dict[KvCacheType, float] = {
    "f16": 2.0,
    "q8_0": 1.0625,
    "q5_1": 0.6875,
    "q5_0": 0.625,
    "q4_1": 0.5625,
    "q4_0": 0.5,
}

# Fallbacks when GGUF metadata is incomplete. Roughly tuned for a 7B-class
# Llama-style model so the estimate is in the right order of magnitude even
# when the file is malformed.
_DEFAULT_N_LAYERS = 32
_DEFAULT_N_KV_HEADS = 8
_DEFAULT_HEAD_DIM = 128
_DEFAULT_EMBEDDING_LENGTH = 4096
_DEFAULT_BATCH_SIZE = 512
_DEFAULT_CTX_SIZE = 4096


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_vram(gguf_path: str, config: LlamaCppLoadConfig) -> dict[str, int]:
    """Estimate per-component VRAM usage (in bytes) for a GGUF + load config.

    Args:
        gguf_path: Absolute path to the ``.gguf`` model file.
        config: Per-model load overrides (only ``ctx_size``, ``batch_size``,
            ``cache_type_k`` and ``cache_type_v`` are consulted).

    Returns:
        ``{"model_bytes", "kv_bytes", "compute_buffer_bytes", "total_bytes"}``
        — all integer byte counts. ``total`` is the sum of the three.

    Raises:
        FileNotFoundError: If ``gguf_path`` does not exist.
    """
    path = Path(gguf_path)
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    model_bytes = path.stat().st_size

    metadata = _parse_gguf_metadata(path)
    arch = _extract_architecture(metadata)

    n_layers = _metadata_int(metadata, f"{arch}.block_count", _DEFAULT_N_LAYERS)
    embedding_length = _metadata_int(
        metadata, f"{arch}.embedding_length", _DEFAULT_EMBEDDING_LENGTH
    )
    head_count = _metadata_int(metadata, f"{arch}.attention.head_count", 0)
    n_kv_heads = _metadata_int(
        metadata, f"{arch}.attention.head_count_kv", head_count or _DEFAULT_N_KV_HEADS
    )
    if head_count > 0 and embedding_length > 0:
        head_dim = max(1, embedding_length // head_count)
    else:
        head_dim = _DEFAULT_HEAD_DIM

    n_ctx = int(config.ctx_size or _DEFAULT_CTX_SIZE)
    batch_size = int(config.batch_size or _DEFAULT_BATCH_SIZE)
    cache_k = config.cache_type_k or "f16"
    cache_v = config.cache_type_v or "f16"

    bytes_k = _KV_BYTES_PER_ELEMENT.get(cache_k, _KV_BYTES_PER_ELEMENT["f16"])
    bytes_v = _KV_BYTES_PER_ELEMENT.get(cache_v, _KV_BYTES_PER_ELEMENT["f16"])

    # Standard formula: per layer we keep one K and one V tensor, each shaped
    # (n_kv_heads * head_dim) * n_ctx. The "× 2" in the spec is split here
    # between K and V to support different quantizations per side.
    kv_elements_per_side = n_layers * n_kv_heads * head_dim * n_ctx
    kv_bytes = int(kv_elements_per_side * bytes_k + kv_elements_per_side * bytes_v)

    # Coarse compute buffer approximation (good enough for the UI sidebar).
    compute_buffer_bytes = int(batch_size * embedding_length * 4)

    total_bytes = model_bytes + kv_bytes + compute_buffer_bytes
    return {
        "model_bytes": int(model_bytes),
        "kv_bytes": int(kv_bytes),
        "compute_buffer_bytes": int(compute_buffer_bytes),
        "total_bytes": int(total_bytes),
    }


# ---------------------------------------------------------------------------
# GGUF parsing helpers
# ---------------------------------------------------------------------------


def _parse_gguf_metadata(path: Path) -> dict[str, Any]:
    """Read only the GGUF header + KV metadata, returning a flat dict.

    We deliberately stop after the metadata section and never touch tensor
    descriptors or weights — that keeps the parser O(metadata_count) and safe
    on multi-GB files.
    """
    metadata: dict[str, Any] = {}
    with path.open("rb") as fh:
        magic = fh.read(4)
        if magic != _GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic!r}): {path}")
        version = _read_u32(fh)
        # tensor_count and metadata_kv_count are uint64 since GGUF v2.
        _tensor_count = _read_u64(fh)
        kv_count = _read_u64(fh)
        metadata["__version__"] = version
        for _ in range(kv_count):
            try:
                key = _read_string(fh)
                value = _read_value(fh)
            except (struct.error, OSError):
                # Truncated / malformed metadata — stop early but keep what we
                # have. The estimator falls back to defaults for missing keys.
                break
            metadata[key] = value
    return metadata


def _extract_architecture(metadata: dict[str, Any]) -> str:
    """Return the model architecture, e.g. ``"llama"`` or ``"qwen2"``."""
    arch = metadata.get("general.architecture")
    if isinstance(arch, str) and arch:
        return arch
    return "llama"


def _metadata_int(metadata: dict[str, Any], key: str, default: int) -> int:
    """Coerce a metadata value to a positive int, falling back on default."""
    value = metadata.get(key)
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
        return default
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return default


def _read_u32(fh) -> int:
    return struct.unpack("<I", fh.read(4))[0]


def _read_u64(fh) -> int:
    return struct.unpack("<Q", fh.read(8))[0]


def _read_string(fh) -> str:
    length = _read_u64(fh)
    raw = fh.read(length)
    return raw.decode("utf-8", errors="replace")


def _read_value(fh) -> Any:
    type_id = _read_u32(fh)
    return _read_typed_value(fh, type_id)


def _read_typed_value(fh, type_id: int) -> Any:  # noqa: PLR0911 — explicit dispatch
    if type_id == _GGUF_TYPE_UINT8:
        return struct.unpack("<B", fh.read(1))[0]
    if type_id == _GGUF_TYPE_INT8:
        return struct.unpack("<b", fh.read(1))[0]
    if type_id == _GGUF_TYPE_UINT16:
        return struct.unpack("<H", fh.read(2))[0]
    if type_id == _GGUF_TYPE_INT16:
        return struct.unpack("<h", fh.read(2))[0]
    if type_id == _GGUF_TYPE_UINT32:
        return struct.unpack("<I", fh.read(4))[0]
    if type_id == _GGUF_TYPE_INT32:
        return struct.unpack("<i", fh.read(4))[0]
    if type_id == _GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", fh.read(4))[0]
    if type_id == _GGUF_TYPE_BOOL:
        return bool(struct.unpack("<B", fh.read(1))[0])
    if type_id == _GGUF_TYPE_STRING:
        return _read_string(fh)
    if type_id == _GGUF_TYPE_UINT64:
        return _read_u64(fh)
    if type_id == _GGUF_TYPE_INT64:
        return struct.unpack("<q", fh.read(8))[0]
    if type_id == _GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", fh.read(8))[0]
    if type_id == _GGUF_TYPE_ARRAY:
        elem_type = _read_u32(fh)
        length = _read_u64(fh)
        # Cap array size to avoid pathological metadata; we don't actually
        # consume array values for any of the keys we care about.
        if length > 1_000_000:
            raise ValueError(f"GGUF array too large: {length}")
        return [_read_typed_value(fh, elem_type) for _ in range(length)]
    raise ValueError(f"Unknown GGUF metadata type id: {type_id}")


# ---------------------------------------------------------------------------
# Sync convenience for non-async callers
# ---------------------------------------------------------------------------


def estimate_vram_safe(gguf_path: str, config: LlamaCppLoadConfig) -> dict[str, int] | None:
    """Best-effort wrapper that returns ``None`` if estimation fails.

    Useful from FastAPI handlers where we'd rather degrade to the legacy probe
    than 500 the request.
    """
    try:
        if not os.path.isfile(gguf_path):
            return None
        return estimate_vram(gguf_path, config)
    except Exception:  # noqa: BLE001 — caller prefers None over crashing
        return None
