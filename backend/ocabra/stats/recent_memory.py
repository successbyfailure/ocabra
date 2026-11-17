"""The last few requests of a zero-data node, in memory only.

In «0 data» nothing about a request reaches the database (core/privacy.py),
so the dashboard, Logs and Stats had nothing to show. Decided with the user
on 2026-11-17: keep the **last 10** requests' metrics in this process's
memory — model, kind, status, timings, tokens, speed, node — and never who
asked (user, key, group, address, user agent) or the error text (it can quote
the input). Gone on restart, and cleared when the privacy mode changes.
"""

from __future__ import annotations

from collections import deque
from typing import Any

KEPT = 10

_rows: deque[dict[str, Any]] = deque(maxlen=KEPT)


def record(row: dict[str, Any]) -> None:
    """Keep *row* (newest first when read). Identity and error are dropped."""
    _rows.append(
        {
            **row,
            "error": None,
            "userId": None,
            "username": None,
            "groupId": None,
            "groupName": None,
            "apiKeyName": None,
            "clientAddr": None,
            "userAgent": None,
        }
    )


def recent(limit: int = KEPT) -> list[dict[str, Any]]:
    return list(reversed(_rows))[:limit]


def clear() -> None:
    _rows.clear()


def summary() -> dict[str, Any]:
    """Aggregates over what is kept (the «stats of the last 10»)."""
    rows = list(_rows)
    n = len(rows)
    durations = [r["durationMs"] for r in rows if isinstance(r.get("durationMs"), int)]
    speeds = [
        r["predictedTokensPerSecond"]
        for r in rows
        if isinstance(r.get("predictedTokensPerSecond"), (int, float))
    ]
    return {
        "kept": KEPT,
        "requests": n,
        "errors": sum(1 for r in rows if (r.get("statusCode") or 0) >= 400),
        "avgDurationMs": round(sum(durations) / len(durations)) if durations else None,
        "inputTokens": sum(int(r.get("inputTokens") or 0) for r in rows),
        "outputTokens": sum(int(r.get("outputTokens") or 0) for r in rows),
        "avgPredictedTokensPerSecond": round(sum(speeds) / len(speeds), 1) if speeds else None,
    }
