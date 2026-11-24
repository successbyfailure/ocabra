"""Single GGUF header parser with a per-file cache (DUP-03).

Six places parsed the GGUF header by hand (five in ``registry.local_scanner``,
one in ``vram_planner``), each re-reading the file. This is the only parser now:
it reads the key/value metadata and the tensor table once, header-only
(tokenizer vocab arrays are seek-skipped so the read is bounded regardless of
file size), and memoizes the result by ``(path, mtime, size)`` so repeated
lookups — the model list, the scheduler's arch read, speculative probing — cost
one parse.

Outputs are a superset of what each old parser extracted, so every caller is
reproduced exactly:

    * ``metadata``        — scalars, strings and small scalar arrays
    * ``array_lengths``   — element count of every array key (vocab size, ...)
    * ``tensor_names``    — names in the tensor table (speculative-type probing)
    * ``expert_mb_by_layer`` — MiB of MoE expert tensors per layer (--n-cpu-moe)
"""

from __future__ import annotations

import functools
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

_GGUF_MAGIC = b"GGUF"
# scalar type id -> struct format
_GGUF_SCALAR_FMT = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<B",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_GGUF_SCALAR_SIZE = {t: struct.calcsize(f) for t, f in _GGUF_SCALAR_FMT.items()}
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9

#: Small per-layer arrays (e.g. attention.sliding_window_pattern, one bool
#: per block_count) are worth capturing for hybrid-architecture KV sizing;
#: huge arrays (tokenizer vocab, merges — tens of thousands of entries) are
#: not, both for parse speed and because nothing reads them. This caps what
#: gets materialized into a Python list rather than seek-skipped.
_GGUF_MAX_ARRAY_CAPTURE = 4096

# Expert tensors as llama.cpp itself matches them for --n-cpu-moe/--cpu-moe
# (LLM_FFN_EXPS_REGEX, ``\.ffn_(up|down|gate|gate_up)_(ch|)exps``). The shared
# expert (``ffn_*_shexp``) is not one of them and stays on the GPU.
_EXPERT_TENSOR = re.compile(r"^blk\.(\d+)\.ffn_(?:up|down|gate|gate_up)_(?:ch|)exps")
# Tables only ever read by row (get_rows): the token embeddings and, in
# per-layer-embedding models (qwen4exp, gemma3n), one row per token per layer.
# llama.cpp keeps them in host RAM, but a token touches a few KB of them, so
# they do not need to be resident: the mapping pages in what is read.
_LOOKUP_TENSOR = re.compile(r"^(?:per_layer_)?token_embd\.weight$")


@dataclass(frozen=True)
class GgufHeader:
    """Everything the header parsers used to extract, read once per file."""

    metadata: dict
    array_lengths: dict[str, int]
    tensor_names: tuple[str, ...]
    expert_mb_by_layer: dict[int, float]
    #: MiB of row-lookup tables (see ``_LOOKUP_TENSOR``) in this file.
    lookup_mb: float = 0.0


def _gguf_read(f, fmt: str):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _gguf_read_string(f) -> str:
    n = _gguf_read(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def _gguf_consume_value(f, vtype: int) -> tuple[object, int | None]:
    """Read a scalar/string value, or a small scalar array (see
    ``_GGUF_MAX_ARRAY_CAPTURE``); seek past large/string arrays (never
    materialized). Returns ``(value, array_length)`` where ``value`` is the
    scalar/string/list or ``None`` for a skipped array, and ``array_length`` is
    the element count for arrays or ``None`` for scalars/strings."""
    if vtype == _GGUF_TYPE_STRING:
        return _gguf_read_string(f), None
    if vtype == _GGUF_TYPE_ARRAY:
        elem_type = _gguf_read(f, "<I")
        count = _gguf_read(f, "<Q")
        if elem_type in _GGUF_SCALAR_SIZE:
            if count <= _GGUF_MAX_ARRAY_CAPTURE:
                fmt = _GGUF_SCALAR_FMT[elem_type]
                return [_gguf_read(f, fmt) for _ in range(count)], count
            f.seek(_GGUF_SCALAR_SIZE[elem_type] * count, 1)
        elif elem_type == _GGUF_TYPE_STRING:
            for _ in range(count):
                f.seek(_gguf_read(f, "<Q"), 1)
        else:
            raise ValueError(f"unsupported nested GGUF array type {elem_type}")
        return None, count
    fmt = _GGUF_SCALAR_FMT.get(vtype)
    if fmt is None:
        raise ValueError(f"unknown GGUF value type {vtype}")
    return _gguf_read(f, fmt), None


def _parse(path: str) -> GgufHeader:
    metadata: dict = {}
    array_lengths: dict[str, int] = {}
    tensor_names: list[str] = []
    experts: dict[int, float] = {}
    lookup_mb = 0.0
    with open(path, "rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = _gguf_read(f, "<I")
        if version < 2:
            # v1 used 32-bit lengths; only v2+ is supported, as the old parsers did.
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = _gguf_read(f, "<Q")
        kv_count = _gguf_read(f, "<Q")
        for _ in range(kv_count):
            key = _gguf_read_string(f)
            vtype = _gguf_read(f, "<I")
            value, array_length = _gguf_consume_value(f, vtype)
            if value is not None:
                metadata[key] = value
            if array_length is not None:
                array_lengths[key] = array_length
        # The tensor table follows the metadata: name, n_dims, dims[n_dims],
        # ggml type, offset. Best-effort — a truncated/synthetic table leaves
        # tensor_names/experts empty but never discards the metadata.
        if tensor_count > 0:
            try:
                entries: list[tuple[int, str]] = []
                for _ in range(tensor_count):
                    name = _gguf_read_string(f)
                    n_dims = _gguf_read(f, "<I")
                    f.seek(8 * n_dims + 4, 1)  # dims (u64 each) + ggml type (u32)
                    offset = _gguf_read(f, "<Q")
                    entries.append((offset, name))
                    tensor_names.append(name)
                alignment = int(metadata.get("general.alignment") or 32)
                data_start = -(-f.tell() // alignment) * alignment
                data_size = os.fstat(f.fileno()).st_size - data_start
            except (struct.error, ValueError, OSError):
                return GgufHeader(metadata, array_lengths, (), {})
            ordered = sorted(entries)
            for i, (offset, name) in enumerate(ordered):
                end = ordered[i + 1][0] if i + 1 < len(ordered) else data_size
                size_mb = max(0, end - offset) / (1024 * 1024)
                if _LOOKUP_TENSOR.match(name):
                    lookup_mb += size_mb
                    continue
                match = _EXPERT_TENSOR.match(name)
                if not match:
                    continue
                layer = int(match.group(1))
                experts[layer] = experts.get(layer, 0.0) + size_mb
    return GgufHeader(metadata, array_lengths, tuple(tensor_names), experts, lookup_mb)


@functools.lru_cache(maxsize=256)
def _read_gguf_header_cached(path: str, _mtime_ns: int, _size: int) -> GgufHeader | None:
    try:
        return _parse(path)
    except (OSError, ValueError, struct.error, IndexError):
        return None


def read_gguf_header(path: str | Path) -> GgufHeader | None:
    """Parse ``path``'s GGUF header, memoized while the file is unchanged.

    Returns ``None`` when the file is missing, not a GGUF, or unparseable —
    callers fall back to whatever "unknown" meant for them before.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return _read_gguf_header_cached(str(path), st.st_mtime_ns, st.st_size)
