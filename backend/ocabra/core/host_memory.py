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


def llama_ram_need_mb(host_projection_mb: int, *, model_file_mb: int, mlock: bool) -> int:
    """RAM a llama.cpp load must keep resident.

    ``host_projection_mb``: what llama-fit-params puts on the CPU side. With
    ``--mlock`` the whole mapped file is pinned on top of the host buffers
    that are not file-backed (those are small: KV and compute stay on GPU).
    """
    need = max(0, int(host_projection_mb))
    if mlock and model_file_mb:
        need = max(need, int(model_file_mb))
    return need
