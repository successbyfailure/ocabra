"""Short-lived cache for the directory walks that model listings repeat.

``/api/tags``, ``/api/ps`` and ``/ocabra/models`` used to walk the models
directory once per model on every request, on the event loop. The results are
the same from one request to the next unless a model is downloaded or deleted,
so they are cached here by path, and the walk itself runs in whatever thread
calls these functions (the async callers hand them to ``asyncio.to_thread``).

What is cached is only filesystem facts (file lists and byte totals), never a
response: access control and federation are still applied per request by the
callers.

Invalidation: an entry is reused while the directory's own mtime is unchanged
and it is younger than ``TTL_SECONDS``. The mtime catches files added or
removed directly in the directory; the TTL bounds everything the mtime cannot
see (a file growing during a download, a change in a nested subdirectory).
Thirty seconds is shorter than a model download takes to finish and settle,
and the openresty front proxy polls ``/api/tags`` once a minute, so a poll
never sees anything older than half its own interval.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path

TTL_SECONDS = 30.0

#: Suffixes counted as model weights by the Ollama-compatible listings.
WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".gguf"})

_lock = threading.Lock()
# key -> (stored_at_monotonic, dir_mtime_ns, value)
_cache: dict[tuple, tuple[float, int, object]] = {}


def _dir_mtime_ns(path: Path) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _lookup(key: tuple, mtime_ns: int):
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < TTL_SECONDS and hit[1] == mtime_ns:
            return True, hit[2]
    return False, None


def _store(key: tuple, mtime_ns: int, value) -> None:
    with _lock:
        _cache[key] = (time.monotonic(), mtime_ns, value)


def invalidate() -> None:
    """Forget every cached walk (tests, or after a known download/delete)."""
    with _lock:
        _cache.clear()


def _walk_weights_size(model_dir: Path, suffixes: frozenset[str]) -> int:
    return sum(
        path.stat().st_size
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix in suffixes
    )


def weights_size_bytes(model_dir: Path, suffixes: Iterable[str] = WEIGHT_SUFFIXES) -> int:
    """Total bytes of the weight files under ``model_dir`` (0 if it is missing).

    Blocking: call it from a worker thread.
    """
    mtime_ns = _dir_mtime_ns(model_dir)
    if mtime_ns is None:
        return 0
    suffix_set = frozenset(suffixes)
    key = ("weights", str(model_dir), suffix_set)
    found, value = _lookup(key, mtime_ns)
    if found:
        return int(value)
    try:
        size = _walk_weights_size(model_dir, suffix_set)
    except OSError:
        # A file vanished mid-walk (delete in progress): report nothing and
        # do not cache, so the next request walks the settled tree.
        return 0
    _store(key, mtime_ns, size)
    return size


def glob_files(base: Path, pattern: str) -> list[Path]:
    """``list(base.rglob(pattern))`` (walk order kept), cached. Missing ``base`` gives ``[]``.

    Paths are returned as they were found; a caller that is about to use one
    should still check it exists (it may have been deleted within the TTL).
    Blocking: call it from a worker thread.
    """
    mtime_ns = _dir_mtime_ns(base)
    if mtime_ns is None:
        return []
    key = ("glob", str(base), pattern)
    found, value = _lookup(key, mtime_ns)
    if found:
        return list(value)
    paths = tuple(base.rglob(pattern))
    _store(key, mtime_ns, paths)
    return list(paths)
