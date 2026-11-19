"""Filesystem helpers for backends that must not block the event loop.

``get_vram_estimate_mb`` runs on hot paths (scheduler, model list, admission)
and several backends summed weight sizes with ``rglob`` + ``stat`` directly in
the coroutine — over a big HF snapshot that is thousands of syscalls on the
event loop (ASY-04). :func:`total_file_size` does the walk in a thread.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


def _total_file_size_sync(root: Path, patterns: tuple[str, ...]) -> int:
    if not root.is_dir():
        return 0
    for pattern in patterns:
        total = sum(path.stat().st_size for path in root.rglob(pattern) if path.is_file())
        if total:
            return total
    return 0


async def total_file_size(root: Path, *patterns: str) -> int:
    """Bytes of the files under ``root`` matching the first pattern that has any.

    Patterns are tried in order (``"*.safetensors", "*.bin"`` = safetensors,
    falling back to .bin). Returns 0 when ``root`` is not a directory. Sizes
    follow symlinks, as HF snapshots are links into ``blobs/``.
    """
    return await asyncio.to_thread(_total_file_size_sync, Path(root), patterns or ("*",))
