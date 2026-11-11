"""Peticiones recién terminadas, en memoria, para el dashboard en vivo.

Las peticiones muy cortas (un embedding, una transcripción pequeña, un 429)
pueden empezar y terminar entre dos refrescos del dashboard (1 s) y no verse
nunca. El middleware de stats deja aquí cada petición al terminar, con sus
datos finales, y el dashboard las muestra unos segundos.

Solo memoria y solo lo reciente: no es un histórico (eso es request_stats).
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any

_MAX = 200
_finished: deque[dict[str, Any]] = deque(maxlen=_MAX)
_lock = Lock()


def record_finished(entry: dict[str, Any]) -> None:
    """Apunta una petición terminada (``entry`` ya en el formato de la API)."""
    item = dict(entry)
    item["_endedMonotonic"] = time.monotonic()
    item["endedAt"] = time.time()
    with _lock:
        _finished.append(item)


def recent_finished(window_s: float = 20.0, limit: int = 30) -> list[dict[str, Any]]:
    """Las terminadas en los últimos ``window_s`` segundos, la más reciente primero."""
    now = time.monotonic()
    with _lock:
        items = [i for i in _finished if now - i["_endedMonotonic"] <= window_s]
    items.reverse()
    out = []
    for item in items[:limit]:
        clean = {k: v for k, v in item.items() if not k.startswith("_")}
        clean["endedAgoMs"] = int((now - item["_endedMonotonic"]) * 1000)
        out.append(clean)
    return out


def clear() -> None:
    with _lock:
        _finished.clear()
