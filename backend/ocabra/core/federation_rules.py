"""Federation mode «Mi servidor, mis reglas»: conditions on what peers ask of us.

A peer linked in **swarm** mode may do everything (that is the swarm: all
permissions on). A peer in **rules** mode is served under this node's rules:
node-wide defaults (``FederationManager.default_rules``) plus that peer's own
adjustments (``PeerState.rules``). Decided by the user on 2026-11-16; see
docs/tasks/cloud-storage-privacy-plan.md, section 3.

Where each rule is enforced:

* **On the tunnel** (``gate``, before the request reaches our API — we know
  for sure which peer it is): time window, admin writes, downloads, requests
  per hour, energy per month, models offered, «only what is warm».
* **In the API** (the request authenticates as the peer's service user):
  «me first» (the peer's requests take the batch class, so local ones go
  ahead and may preempt them) and «may it evict my models» (loads it triggers
  run without pressure eviction).

A request refused for a quota or because the model is not warm gets a **503**
(and 404 for a model not offered): the peer that sent it falls back to itself,
as it always did (decision 3: keep the current behaviour for now).
"""

from __future__ import annotations

import contextvars
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

@dataclass
class PeerRules:
    local_first: bool = False
    """Our own requests go first: the peer's get the batch class."""
    allow_loads: bool = True
    """False: serve only models already loaded here («only what is warm»)."""
    allow_evict: bool = True
    """False: a load the peer triggers may not push out a model in use here."""
    max_requests_per_hour: int | None = None
    max_wh_per_month: float | None = None
    """Energy spent serving this peer this calendar month."""
    allow_downloads: bool = True
    allow_admin: bool = True
    """False: the remote panel is read-only for this peer."""
    offered_models: list[str] | None = None
    """None = every model; otherwise only these (model ids or public names)."""
    hours: list[int] | None = None
    """[start, end) local hours when the peer is served; None = always."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PeerRules:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ALL_ALLOWED = PeerRules()

# Rules of the peer the current request comes from (set by the stats
# middleware from the authenticated service user); read by the model loader.
current_peer_rules: contextvars.ContextVar[PeerRules | None] = contextvars.ContextVar(
    "current_peer_rules", default=None
)

# What «rules» mode starts from before the node's own defaults: a neighbour
# using our machine, not a co-owner of it.
RULES_MODE_BASE: dict[str, Any] = {
    "local_first": True,
    "allow_loads": True,
    "allow_evict": False,
    "allow_downloads": False,
    "allow_admin": False,
}

DEFAULTS_KEY = "federation_default_rules"
# This node's defaults, kept here by FederationManager (load/set) so code
# without the manager at hand (API serializers) can compute effective rules.
NODE_DEFAULTS: dict[str, Any] = {}


def effective_rules(peer: Any, node_defaults: dict[str, Any] | None = None) -> PeerRules:
    """The rules *peer* is served under: everything in swarm (and manual) mode."""
    if getattr(peer, "mode", None) != "rules":
        return ALL_ALLOWED
    defaults = NODE_DEFAULTS if node_defaults is None else node_defaults
    merged = {**RULES_MODE_BASE, **defaults, **(getattr(peer, "rules", None) or {})}
    return PeerRules.from_dict(merged)


def clean_rules(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a rules dict from the API (unknown keys and bad values out)."""
    out: dict[str, Any] = {}
    for f in fields(PeerRules):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name in ("max_requests_per_hour", "max_wh_per_month"):
            # 0, "0", "" or None = no limit (forms send strings).
            number = float(value) if value not in (None, "") else 0.0
            if number <= 0:
                out[f.name] = None
            elif f.name == "max_requests_per_hour":
                out[f.name] = max(1, int(number))
            else:
                out[f.name] = number
        elif f.name == "offered_models":
            out[f.name] = None if value is None else [str(v) for v in value if str(v).strip()]
        elif f.name == "hours":
            if value is None:
                out[f.name] = None
            else:
                start, end = (int(v) % 24 for v in value)
                out[f.name] = [start, end]
        else:
            out[f.name] = bool(value)
    return out


def within_hours(hours: list[int] | None, now: datetime | None = None) -> bool:
    if not hours:
        return True
    start, end = hours
    h = (now or datetime.now().astimezone()).hour
    if start == end:
        return True
    return start <= h < end if start < end else (h >= start or h < end)


# ── per-peer usage ───────────────────────────────────────────────


@dataclass
class _Usage:
    hits: deque = field(default_factory=deque)  # monotonic times, last hour
    energy_checked_at: float = 0.0
    energy_wh: float = 0.0


_usage: dict[str, _Usage] = {}


def requests_last_hour(peer_id: str, now: float | None = None) -> int:
    u = _usage.setdefault(peer_id, _Usage())
    now = now if now is not None else time.monotonic()
    while u.hits and now - u.hits[0] > 3600:
        u.hits.popleft()
    return len(u.hits)


def _count_request(peer_id: str, now: float | None = None) -> None:
    _usage.setdefault(peer_id, _Usage()).hits.append(now if now is not None else time.monotonic())


async def energy_this_month(fm: Any, peer: Any, *, max_age_s: float = 60.0) -> float:
    """Wh spent this month on requests *peer* sent us (cached a minute)."""
    import sqlalchemy as sa

    from ocabra.db.stats import RequestStat

    u = _usage.setdefault(peer.peer_id, _Usage())
    if time.monotonic() - u.energy_checked_at < max_age_s:
        return u.energy_wh
    now = datetime.now(UTC)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    async with fm.session_factory() as session:
        wh = (
            await session.execute(
                sa.select(sa.func.coalesce(sa.func.sum(RequestStat.energy_wh), 0.0)).where(
                    RequestStat.federation_inbound_from == peer.name,
                    RequestStat.started_at >= month_start,
                )
            )
        ).scalar_one()
    u.energy_wh = float(wh or 0.0)
    u.energy_checked_at = time.monotonic()
    return u.energy_wh


# ── the tunnel gate ──────────────────────────────────────────────


def _refuse(status: int, code: str, message: str) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"message": message, "type": "federation_rules", "code": code}}


def _requested_model(body: bytes) -> str | None:
    if not body or body[:1] not in (b"{", b" "):
        return None
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    model = data.get("model") if isinstance(data, dict) else None
    return str(model) if model else None


_DOWNLOAD_PATHS = ("/ocabra/downloads", "/ocabra/registry")


async def gate(
    fm: Any, peer: Any, method: str, path: str, body: bytes
) -> tuple[int, dict[str, Any]] | None:
    """``(status, body)`` to refuse *peer*'s request, or ``None`` to serve it."""
    rules = effective_rules(peer, fm.default_rules)
    if rules == ALL_ALLOWED:
        return None
    clean = path.split("?", 1)[0]
    method = method.upper()
    if not within_hours(rules.hours):
        start, end = rules.hours or (0, 0)
        return _refuse(
            503, "outside_hours", f"This node serves the federation from {start}:00 to {end}:00"
        )
    if method != "GET" and clean.startswith("/ocabra/"):
        is_download = clean.startswith(_DOWNLOAD_PATHS) or (
            clean.startswith("/ocabra/backends/") and clean.endswith("/install")
        )
        if is_download and not rules.allow_downloads:
            return _refuse(403, "downloads_not_allowed", "This node does not take downloads from you")
        if not is_download and not rules.allow_admin:
            return _refuse(403, "read_only", "This node's settings are read-only for you")
    if not (clean.startswith("/v1/") or clean.startswith("/api/")):
        return None

    if rules.max_requests_per_hour and (
        requests_last_hour(peer.peer_id) >= rules.max_requests_per_hour
    ):
        return _refuse(
            503,
            "hourly_quota",
            f"Hourly quota for your node reached ({rules.max_requests_per_hour} requests)",
        )
    if rules.max_wh_per_month and await energy_this_month(fm, peer) >= rules.max_wh_per_month:
        return _refuse(
            503, "energy_quota", f"Monthly energy quota reached ({rules.max_wh_per_month:g} Wh)"
        )

    model = _requested_model(body)
    if model:
        canonical = await _canonical(fm, model)
        if rules.offered_models is not None and not (
            {model, canonical} & set(rules.offered_models)
        ):
            return _refuse(404, "model_not_offered", f"The model '{model}' is not offered to you")
        if not rules.allow_loads and not await _is_loaded(fm, canonical or model):
            return _refuse(
                503, "model_not_warm", f"'{model}' is not loaded here and you may not load it"
            )
    _count_request(peer.peer_id)
    return None


async def _canonical(fm: Any, model: str) -> str | None:
    registry = getattr(fm, "profile_registry", None)
    if registry is None:
        return None
    try:
        profile = await registry.get(model)
        if profile is None:
            alias = await registry.resolve_alias(model)
            profile = alias[0] if alias else None
    except Exception:  # noqa: BLE001 — best effort
        return None
    return getattr(profile, "base_model_id", None) if profile is not None else None


async def _is_loaded(fm: Any, model_id: str) -> bool:
    mm = getattr(fm, "model_manager", None)
    if mm is None:
        return True  # can't tell: don't refuse on a guess
    try:
        states = [await mm.get_state(model_id)]
        states += [s for s in await mm.list_states() if s.model_id.startswith(f"{model_id}::")]
    except Exception:  # noqa: BLE001
        return True
    return any(
        s is not None and str(getattr(s.status, "value", s.status)).lower() == "loaded"
        for s in states
    )
