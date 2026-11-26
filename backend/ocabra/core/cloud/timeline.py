"""Cuánto tarda un pod en estar listo, por etapas.

Se guarda en ``cloud_nodes.timeline`` (marcas ISO) y se resume en duraciones:

- ``wait_s``: esperando máquina (petición creada → pod lanzado), si vino de una
  petición que tuvo que reintentar.
- ``boot_s``: lanzado → RunPod lo da por ``running`` (arranque del contenedor).
- ``connect_s``: ``running`` → se une a oCabra como peer (túnel + invitación).
- ``prepare_s``: unido → backends instalados y modelos descargados/cargados.
- ``total_s``: desde el primer momento conocido hasta listo.
- ``items``: lo que tardó cada elemento de la precarga (backend o modelo).

Las marcas las pone el bucle de reconciliación (cada 30 s), así que la
precisión de cada etapa es de unos 30 s.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

ACTIVE = ("installing", "downloading", "registering", "loading")
DONE = ("loaded", "ready", "error")


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def stamp(timeline: dict | None, key: str, when: datetime) -> dict:
    """Devuelve una copia con ``key`` marcada si no lo estaba (JSONB se reasigna)."""
    out = dict(timeline or {})
    out.setdefault(key, _iso(when))
    return out


def track_items(timeline: dict | None, items: list[dict] | None, when: datetime) -> dict:
    """Marca inicio y fin de cada elemento de la precarga, y ``ready_at`` al acabar todos."""
    out = dict(timeline or {})
    tracked = {key: dict(value) for key, value in (out.get("items") or {}).items()}
    for item in items or []:
        key = str(item.get("model_id") or "")
        status = item.get("status")
        if not key:
            continue
        entry = tracked.setdefault(key, {})
        if status in ACTIVE or status in DONE:
            entry.setdefault("started_at", _iso(when))
        if status in DONE:
            entry.setdefault("finished_at", _iso(when))
            entry["status"] = status
    if tracked:
        out["items"] = tracked
    if items and all(item.get("status") in DONE for item in items) and out.get("joined_at"):
        out.setdefault("ready_at", _iso(when))
        if any(item.get("status") == "error" for item in items):
            out["ready_with_errors"] = True
    return out


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 1)


def summary(timeline: dict | None) -> dict[str, Any]:
    t = timeline or {}
    first = t.get("requested_at") or t.get("launched_at")
    return {
        "wait_s": _seconds(t.get("requested_at"), t.get("launched_at")),
        "boot_s": _seconds(t.get("launched_at"), t.get("running_at")),
        "connect_s": _seconds(t.get("running_at"), t.get("joined_at")),
        "prepare_s": _seconds(t.get("joined_at"), t.get("ready_at")),
        "total_s": _seconds(first, t.get("ready_at")),
        "ready_with_errors": bool(t.get("ready_with_errors")),
        "items": {
            key: _seconds(value.get("started_at"), value.get("finished_at"))
            for key, value in (t.get("items") or {}).items()
        },
    }
