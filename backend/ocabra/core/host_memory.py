"""Host RAM as a model load sees it, and what a llama.cpp load needs from it.

``MemAvailable`` (``/proc/meminfo``) is the figure that matters: it already
counts reclaimable page cache as available, which is where most of a
memory-mapped GGUF lives. A container memory limit (cgroup v2
``memory.max``) caps both numbers when it is set.

What a llama.cpp load needs, measured on Mapache (GTX 1660, MoE experts in
RAM, 2026-11-18): the process RSS ends up at the size of the whole GGUF with or
without ``--mlock`` — the GPU-offloaded tensors are read through the mapping
too — but only the CPU-side tensors (llama-fit-params' ``Host`` figure: the
experts left in RAM and the token embeddings) must stay resident; the rest is
cache the kernel reclaims under pressure. With ``--mlock`` nothing is
reclaimable: the whole file is pinned.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Kept free for the OS, other processes and the worker's own growth.
RAM_MARGIN_MB = 1024


def _cgroup_limit_mb() -> int | None:
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw) // (1024 * 1024)
    except ValueError:
        return None


def _cgroup_used_mb() -> int | None:
    try:
        return int(Path("/sys/fs/cgroup/memory.current").read_text().strip()) // (1024 * 1024)
    except (OSError, ValueError):
        return None


def read_host_memory() -> dict[str, int] | None:
    """``{"total_mb", "available_mb"}`` for this host (or container), or None."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = int(rest.split()[0]) // 1024
    except (OSError, ValueError):
        return None
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    total, available = values["MemTotal"], values["MemAvailable"]
    limit = _cgroup_limit_mb()
    if limit is not None and limit < total:
        used = _cgroup_used_mb() or 0
        total = limit
        available = max(0, min(available, limit - used))
    return {"total_mb": total, "available_mb": available}


def file_mb(path: str | os.PathLike[str] | None) -> int:
    try:
        return int(Path(path).stat().st_size // (1024 * 1024)) if path else 0
    except OSError:
        return 0


_SPLIT_PART = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")


def lookup_tables_mb(path: str | os.PathLike[str] | None) -> int:
    """MiB of row-lookup tables (token embeddings, per-layer embeddings) in a
    GGUF, across every part of a split one: they live in host RAM but a token
    reads a few KB of them, so they need not be resident (see
    ``gguf_reader._LOOKUP_TENSOR``). qwen3.8-flash-next-coder keeps a 27 GB
    per-layer table in the second part; counting it as resident made the
    editor say the model did not fit the node's RAM (2026-11-24).
    """
    if not path:
        return 0
    from ocabra.core.gguf_reader import read_gguf_header

    first = Path(path)
    match = _SPLIT_PART.match(first.name)
    if match:
        count = int(match.group("count"))
        parts = [
            first.with_name(f"{match.group('stem')}-{i:05d}-of-{count:05d}.gguf")
            for i in range(1, count + 1)
        ]
    else:
        parts = [first]
    total = 0.0
    for part in parts:
        header = read_gguf_header(part)
        if header is not None:
            total += header.lookup_mb
    return int(total)


def llama_ram_need_mb(
    host_projection_mb: int, *, model_file_mb: int, mlock: bool, lookup_mb: int = 0
) -> int:
    """RAM a llama.cpp load must keep resident.

    ``host_projection_mb``: what llama-fit-params puts on the CPU side. With
    ``--mlock`` the whole mapped file is pinned on top of the host buffers
    that are not file-backed (those are small: KV and compute stay on GPU).
    Without it, ``lookup_mb`` of row-lookup tables (``lookup_tables_mb``)
    inside that CPU side is paged in on demand and does not count.
    """
    need = max(0, int(host_projection_mb))
    if mlock and model_file_mb:
        return max(need, int(model_file_mb))
    return max(0, need - max(0, int(lookup_mb)))
