"""Small helpers shared by backends that launch a worker and wait for /health.

Several backends (whisper, tts, acestep) used to poll ``/health`` until their
startup timeout even when the worker had already died, so a crash at import
time cost the full timeout (minutes) and the error said nothing about why.
:func:`worker_exit_error` builds the error to raise as soon as the process is
gone, with the tail of the worker's log.
"""

from __future__ import annotations

import os


def read_log_tail(path: str, limit: int = 4000) -> str:
    """Return the last ``limit`` characters of ``path`` (empty on any error)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:  # noqa: PTH123
            if size > limit:
                f.seek(size - limit)
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace").strip()
    return text if len(text) <= limit else text[-limit:]


def worker_exit_error(label: str, model_id: str, returncode: int, log_path: str) -> RuntimeError:
    """Error for a worker that exited during startup, with the log tail."""
    tail = read_log_tail(log_path)
    message = (
        f"{label} worker for model '{model_id}' exited during startup "
        f"(exit code {returncode}). Log: {log_path}"
    )
    if tail:
        message += f"\n--- last log lines ---\n{tail}"
    return RuntimeError(message)
