"""Federation manager for multi-node peer-to-peer inference.

Handles peer registration, heartbeat polling, API key encryption,
and peer state tracking. Provides methods for proxy routing and
load balancing across federated nodes.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from cryptography.fernet import Fernet

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from ocabra.config import Settings
    from ocabra.core.model_manager import ModelManager

logger = structlog.get_logger(__name__)

# Peers in tunnel mode have no URL: their httpx client talks to a
# TunnelTransport, which ignores the host. This placeholder keeps the
# ``f"{base}{path}"`` call sites identical for both modes.
_TUNNEL_BASE_URL = "http://federation-tunnel.invalid"

TUNNEL_MODES = frozenset({"tunnel_in", "tunnel_out"})

# Set (by middleware in main.py) for requests that arrived *from* a peer. Such
# a request is served here or not at all: forwarding it to yet another peer
# could bounce it between nodes whose heartbeats disagree. ``get_remote_models``
# returns nothing while it is set, which disables every remote path at once.
federation_hop: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "federation_hop", default=None
)

# Maximum consecutive heartbeat failures before marking a peer offline.
_MAX_FAILURES = 3
# Maximum backoff interval (seconds) for offline peer retry.
_MAX_BACKOFF_S = 300


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a Fernet-compatible key from an arbitrary secret string."""
    digest = hashlib.sha256(secret.encode()).digest()
    return base64.urlsafe_b64encode(digest)


@dataclass
class PeerState:
    """Runtime state for a single federation peer."""

    peer_id: str
    name: str
    url: str
    api_key: str  # decrypted
    access_level: str  # "inference" | "full"
    enabled: bool = True
    online: bool = False
    last_heartbeat: datetime | None = None
    gpus: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    load: dict[str, Any] = field(default_factory=dict)
    consecutive_failures: int = 0
    _next_retry_at: float = 0.0
    # "http" (we call url) | "tunnel_in" (peer dials us) | "tunnel_out" (we dial it)
    connection_mode: str = "http"
    mode: str = "manual"  # manual | swarm | rules
    notes: str | None = None
    remote_node_id: str | None = None
    tunnel_token_hash: str | None = None
    # How we serve the peer's requests over the tunnel (None = we don't).
    local_api_key: str | None = None
    local_user_id: str | None = None
    local_access_level: str | None = None
    # Outgoing join request awaiting the other side's decision.
    join_state: str | None = None  # pending | denied | expired | None
    join_request_id: str | None = None
    join_poll_secret: str | None = None
    join_verification_code: str | None = None
    tunnel_session: Any = None  # TunnelConnection | None
    tunnel_client: httpx.AsyncClient | None = None
    dialer: Any = None  # TunnelDialer | None (tunnel_out only)

    @property
    def is_tunnel(self) -> bool:
        return self.connection_mode in TUNNEL_MODES

    @property
    def tunnel_connected(self) -> bool:
        return self.tunnel_session is not None and not self.tunnel_session.closed

    @property
    def base_url(self) -> str:
        return _TUNNEL_BASE_URL if self.is_tunnel else self.url


def hash_tunnel_token(token: str) -> str:
    """SHA-256 hex of a tunnel token (only the hash is stored)."""
    return hashlib.sha256(token.encode()).hexdigest()


def new_tunnel_token() -> str:
    """Fresh random tunnel token, prefixed so it is recognisable in configs."""
    return "oct_" + secrets.token_urlsafe(32)


class FederationManager:
    """Manages federation peer state, heartbeat polling, and proxy routing."""

    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._peers: dict[str, PeerState] = {}
        self._heartbeat_task: asyncio.Task | None = None
        self._fernet = Fernet(_derive_fernet_key(settings.jwt_secret))
        self._node_id = settings.federation_node_id or str(uuid.uuid4())
        self._node_name = settings.federation_node_name or f"node-{self._node_id[:8]}"
        self._started_at: float = 0.0
        self._http_client: httpx.AsyncClient | None = None
        # Our own API, used to serve peers' requests arriving over a tunnel.
        self._local_client: httpx.AsyncClient | None = None
        self.join_service: Any = None  # FederationJoinService, set in start()
        self.gpu_manager: Any = None  # set by main; used to describe us when joining

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Load peers from DB, poll them once, then start the heartbeat loop.

        The initial poll runs synchronously so federated resolution works for
        the very first request — otherwise every peer is treated as offline
        for the first ``federation_heartbeat_interval`` seconds and we'd
        fall back to local even when a peer is ready.
        """
        self._started_at = time.monotonic()
        self._http_client = httpx.AsyncClient(
            verify=self._settings.federation_verify_ssl,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._local_client = httpx.AsyncClient(
            base_url=str(getattr(self._settings, "federation_local_url", "") or "http://127.0.0.1:8000"),
            timeout=httpx.Timeout(None, connect=10.0),
        )
        await self._ensure_persistent_node_id()
        await self._load_peers_from_db()
        from ocabra.core.federation_join import FederationJoinService

        self.join_service = FederationJoinService(self)
        for peer in self._peers.values():
            self._start_dialer_if_ready(peer)
        self.join_service.start()
        # Initial poll — block startup briefly so the first request sees fresh
        # peer state. Don't fail startup if a peer is unreachable.
        try:
            await self._poll_all_peers()
        except Exception as exc:
            logger.warning("federation_initial_poll_error", error=str(exc))
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="federation-heartbeat-loop",
        )
        logger.info(
            "federation_manager_started",
            node_id=self._node_id,
            node_name=self._node_name,
            peer_count=len(self._peers),
            online_after_initial_poll=len(self.get_online_peers()),
        )

    async def stop(self) -> None:
        """Cancel heartbeat task and close HTTP client."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self.join_service is not None:
            await self.join_service.stop()
        for peer in self._peers.values():
            await self._stop_dialer(peer)
            await self._close_tunnel(peer)
        if self._local_client is not None:
            await self._local_client.aclose()
            self._local_client = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("federation_manager_stopped")

    # ── Properties ───────────────────────────────────────────────

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_name(self) -> str:
        return self._node_name

    @property
    def uptime_seconds(self) -> float:
        if self._started_at == 0.0:
            return 0.0
        return time.monotonic() - self._started_at

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    @property
    def http_client(self) -> httpx.AsyncClient | None:
        return self._http_client

    async def _ensure_persistent_node_id(self) -> None:
        """Keep ``node_id`` stable across restarts (peers know us by it).

        Before join requests existed an empty ``FEDERATION_NODE_ID`` meant a new
        random id per restart, which was harmless. Now it identifies us to the
        nodes we joined, so the generated id is persisted as a config override.
        """
        if self._settings.federation_node_id:
            return
        try:
            from ocabra.db.server_config import save_override

            async with self._session_factory() as session:
                await save_override(session, "federation_node_id", self._node_id)
                await session.commit()
            self._settings.federation_node_id = self._node_id
            logger.info("federation_node_id_persisted", node_id=self._node_id)
        except Exception as exc:  # noqa: BLE001 — not fatal: we just keep the ephemeral id
            logger.warning("federation_node_id_persist_failed", error=str(exc))

    # ── Encryption helpers ───────────────────────────────────────

    def encrypt_api_key(self, plaintext: str) -> str:
        """Encrypt an API key for DB storage."""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt_api_key(self, ciphertext: str) -> str:
        """Decrypt an API key from DB storage."""
        return self._fernet.decrypt(ciphertext.encode()).decode()

    # ── DB persistence ───────────────────────────────────────────

    async def _load_peers_from_db(self) -> None:
        """Load all enabled peers from the database into _peers."""
        import sqlalchemy as sa

        from ocabra.db.federation import FederationPeer

        async with self._session_factory() as session:
            result = await session.execute(sa.select(FederationPeer))
            rows = result.scalars().all()

        self._peers.clear()
        for row in rows:
            try:
                api_key = self.decrypt_api_key(row.api_key_encrypted)
            except Exception:
                logger.warning(
                    "federation_peer_key_decrypt_failed",
                    peer_id=str(row.id),
                    name=row.name,
                )
                continue
            self._peers[str(row.id)] = PeerState(
                peer_id=str(row.id),
                name=row.name,
                url=row.url.rstrip("/"),
                api_key=api_key,
                access_level=row.access_level,
                enabled=row.enabled,
                **self._row_extras(row),
            )
        logger.info("federation_peers_loaded", count=len(self._peers))

    def _row_extras(self, row: Any) -> dict[str, Any]:
        """Tunnel/join fields of a FederationPeer row as PeerState kwargs."""

        def _dec(value: str | None) -> str | None:
            if not value:
                return None
            try:
                return self.decrypt_api_key(value)
            except Exception:  # noqa: BLE001
                logger.warning("federation_peer_secret_decrypt_failed", peer=row.name)
                return None

        local_user_id = getattr(row, "local_user_id", None)
        return {
            "connection_mode": getattr(row, "connection_mode", None) or "http",
            "mode": getattr(row, "mode", None) or "manual",
            "notes": getattr(row, "notes", None),
            "remote_node_id": getattr(row, "remote_node_id", None),
            "tunnel_token_hash": getattr(row, "tunnel_token_hash", None),
            "local_api_key": _dec(getattr(row, "local_api_key_encrypted", None)),
            "local_user_id": str(local_user_id) if local_user_id else None,
            "local_access_level": getattr(row, "local_access_level", None),
            "join_state": getattr(row, "join_state", None),
            "join_request_id": getattr(row, "join_request_id", None),
            "join_poll_secret": _dec(getattr(row, "join_poll_secret_encrypted", None)),
        }

    async def add_peer(
        self,
        name: str,
        url: str,
        api_key: str,
        access_level: str = "inference",
        *,
        connection_mode: str = "http",
        notes: str | None = None,
        tunnel_token: str | None = None,
        mode: str = "manual",
        remote_node_id: str | None = None,
        local_api_key: str | None = None,
        local_user_id: str | None = None,
        local_access_level: str | None = None,
        join_state: str | None = None,
        join_request_id: str | None = None,
        join_poll_secret: str | None = None,
        enabled: bool = True,
    ) -> PeerState:
        """Add a new peer to the DB and runtime state.

        Args:
            name: Human-readable peer name (unique).
            url: Base URL of the peer node (empty for tunnel peers).
            api_key: API key for authenticating with the peer (empty for
                tunnel peers: the peer authenticates *us* nothing, it dials in).
            access_level: Access level ('inference' or 'full').
            connection_mode: ``"http"``, ``"tunnel_in"`` or ``"tunnel_out"``.
            notes: Free text shown in the UI.
            tunnel_token: For ``tunnel_in`` peers, the token the peer will
                present (only its hash is stored). For ``tunnel_out`` pass the
                token we present as ``api_key``.
            mode, remote_node_id, local_*, join_*: see :class:`PeerState`.

        Returns:
            The newly created PeerState.
        """
        token_hash = hash_tunnel_token(tunnel_token) if tunnel_token else None
        from ocabra.db.federation import FederationPeer

        peer_id = uuid.uuid4()
        encrypted_key = self.encrypt_api_key(api_key)

        async with self._session_factory() as session:
            row = FederationPeer(
                id=peer_id,
                name=name,
                url=url.rstrip("/"),
                api_key_encrypted=encrypted_key,
                access_level=access_level,
                enabled=enabled,
                connection_mode=connection_mode,
                tunnel_token_hash=token_hash,
                notes=notes,
                mode=mode,
                remote_node_id=remote_node_id,
                local_api_key_encrypted=(
                    self.encrypt_api_key(local_api_key) if local_api_key else None
                ),
                local_user_id=uuid.UUID(local_user_id) if local_user_id else None,
                local_access_level=local_access_level,
                join_state=join_state,
                join_request_id=join_request_id,
                join_poll_secret_encrypted=(
                    self.encrypt_api_key(join_poll_secret) if join_poll_secret else None
                ),
            )
            session.add(row)
            await session.commit()

        state = PeerState(
            peer_id=str(peer_id),
            name=name,
            url=url.rstrip("/"),
            api_key=api_key,
            access_level=access_level,
            enabled=enabled,
            connection_mode=connection_mode,
            mode=mode,
            notes=notes,
            remote_node_id=remote_node_id,
            tunnel_token_hash=token_hash,
            local_api_key=local_api_key,
            local_user_id=local_user_id,
            local_access_level=local_access_level,
            join_state=join_state,
            join_request_id=join_request_id,
            join_poll_secret=join_poll_secret,
        )
        self._peers[str(peer_id)] = state
        logger.info("federation_peer_added", peer_id=str(peer_id), name=name)
        return state

    async def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer from DB and runtime state.

        Args:
            peer_id: UUID string of the peer to remove.

        Returns:
            True if the peer was found and removed, False otherwise.
        """
        import sqlalchemy as sa

        from ocabra.db.federation import FederationPeer

        parsed_id = uuid.UUID(peer_id)
        async with self._session_factory() as session:
            result = await session.execute(
                sa.delete(FederationPeer).where(FederationPeer.id == parsed_id)
            )
            await session.commit()
            deleted = result.rowcount > 0

        removed = self._peers.pop(peer_id, None)
        if removed is not None:
            await self._stop_dialer(removed)
            await self._close_tunnel(removed, reason="peer removed")
            if self.join_service is not None:
                await self.join_service.on_peer_removed(removed)
        if deleted or removed:
            logger.info("federation_peer_removed", peer_id=peer_id)
            return True
        return False

    async def update_peer(self, peer_id: str, **kwargs: Any) -> PeerState | None:
        """Update a peer in DB and runtime state.

        Accepts keyword arguments: name, url, api_key, access_level, enabled.

        Args:
            peer_id: UUID string of the peer to update.
            **kwargs: Fields to update.

        Returns:
            Updated PeerState or None if peer not found.
        """
        import sqlalchemy as sa

        from ocabra.db.federation import FederationPeer

        parsed_id = uuid.UUID(peer_id)
        db_updates: dict[str, Any] = {}

        if "name" in kwargs and kwargs["name"] is not None:
            db_updates["name"] = kwargs["name"]
        if "url" in kwargs and kwargs["url"] is not None:
            db_updates["url"] = kwargs["url"].rstrip("/")
        if "api_key" in kwargs and kwargs["api_key"] is not None:
            db_updates["api_key_encrypted"] = self.encrypt_api_key(kwargs["api_key"])
        if "access_level" in kwargs and kwargs["access_level"] is not None:
            db_updates["access_level"] = kwargs["access_level"]
        if "enabled" in kwargs and kwargs["enabled"] is not None:
            db_updates["enabled"] = kwargs["enabled"]
        if "notes" in kwargs:
            db_updates["notes"] = kwargs["notes"] or None
        if kwargs.get("tunnel_token"):
            db_updates["tunnel_token_hash"] = hash_tunnel_token(kwargs["tunnel_token"])
        # Internal fields (join / tunnel bookkeeping), set by the join service.
        for plain in ("mode", "remote_node_id", "local_access_level", "join_state", "join_request_id"):
            if plain in kwargs:
                db_updates[plain] = kwargs[plain]
        if "local_api_key" in kwargs:
            db_updates["local_api_key_encrypted"] = (
                self.encrypt_api_key(kwargs["local_api_key"]) if kwargs["local_api_key"] else None
            )
        if "local_user_id" in kwargs:
            db_updates["local_user_id"] = (
                uuid.UUID(kwargs["local_user_id"]) if kwargs["local_user_id"] else None
            )
        if "join_poll_secret" in kwargs:
            db_updates["join_poll_secret_encrypted"] = (
                self.encrypt_api_key(kwargs["join_poll_secret"])
                if kwargs["join_poll_secret"]
                else None
            )

        if not db_updates:
            return self._peers.get(peer_id)

        async with self._session_factory() as session:
            result = await session.execute(
                sa.update(FederationPeer)
                .where(FederationPeer.id == parsed_id)
                .values(**db_updates)
            )
            await session.commit()
            if result.rowcount == 0:
                return None

        peer = self._peers.get(peer_id)
        if peer is None:
            return None

        if "name" in db_updates:
            peer.name = db_updates["name"]
        if "url" in db_updates:
            peer.url = db_updates["url"]
        if "api_key" in kwargs and kwargs["api_key"] is not None:
            peer.api_key = kwargs["api_key"]
        if "access_level" in db_updates:
            peer.access_level = db_updates["access_level"]
        if "enabled" in db_updates:
            peer.enabled = db_updates["enabled"]
            if not peer.enabled:
                await self._close_tunnel(peer, reason="peer disabled")
        if "notes" in db_updates:
            peer.notes = db_updates["notes"]
        if "tunnel_token_hash" in db_updates:
            peer.tunnel_token_hash = db_updates["tunnel_token_hash"]
            # A rotated token invalidates the live session too.
            await self._close_tunnel(peer, reason="token rotated")
        for plain in (
            "mode",
            "remote_node_id",
            "local_access_level",
            "join_state",
            "join_request_id",
            "local_api_key",
            "local_user_id",
            "join_poll_secret",
        ):
            if plain in kwargs:
                setattr(peer, plain, kwargs[plain])
        if peer.connection_mode == "tunnel_out":
            # url / token / enabled / join_state changes: (re)start or stop dialing.
            if any(k in kwargs for k in ("url", "api_key", "enabled", "join_state")):
                await self._stop_dialer(peer)
                self._start_dialer_if_ready(peer)

        logger.info("federation_peer_updated", peer_id=peer_id)
        return peer

    # ── Heartbeat ────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Periodically poll each enabled peer for heartbeat."""
        interval = max(5, self._settings.federation_heartbeat_interval)
        logger.info("federation_heartbeat_loop_started", interval_s=interval)
        while True:
            await asyncio.sleep(interval)
            try:
                logger.debug("federation_heartbeat_polling", peer_count=len(self._peers))
                await self._poll_all_peers()
            except Exception as exc:
                logger.warning("federation_heartbeat_loop_error", error=str(exc))

    async def _poll_all_peers(self) -> None:
        """Poll all enabled peers concurrently."""
        now = time.monotonic()
        tasks = []
        for peer in self._peers.values():
            if not peer.enabled:
                continue
            if peer.is_tunnel and not peer.tunnel_connected:
                # Nothing to poll: the peer has to dial us. Offline right away,
                # no failure counting/backoff (that is for flaky HTTP peers).
                peer.online = False
                continue
            # Skip offline peers until their backoff expires.
            if not peer.online and peer._next_retry_at > now:
                logger.debug(
                    "federation_peer_backoff_skip",
                    peer=peer.name,
                    retry_in_s=round(peer._next_retry_at - now, 1),
                )
                continue
            tasks.append(self._poll_peer(peer))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("federation_poll_task_error", error=str(r))
        elif self._peers:
            logger.debug("federation_no_peers_to_poll", total=len(self._peers))

    async def _poll_peer(self, peer: PeerState) -> None:
        """Send a single heartbeat request to a peer."""
        client = self._client_for(peer)
        if client is None:
            return
        url = f"{peer.base_url}/ocabra/federation/heartbeat"
        try:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {peer.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            peer.online = True
            peer.consecutive_failures = 0
            peer._next_retry_at = 0.0
            peer.last_heartbeat = datetime.now(UTC)
            peer.gpus = data.get("gpus", [])
            peer.models = data.get("models", [])
            peer.load = data.get("load", {})
            logger.info(
                "federation_peer_heartbeat_ok",
                peer=peer.name,
                models_count=len(peer.models),
                gpus_count=len(peer.gpus),
                loaded_model_ids=[
                    m.get("model_id")
                    for m in peer.models
                    if str(m.get("status", "")).upper() == "LOADED"
                ],
            )
        except Exception as exc:
            peer.consecutive_failures += 1
            if peer.consecutive_failures >= _MAX_FAILURES:
                if peer.online:
                    logger.warning(
                        "federation_peer_offline",
                        peer=peer.name,
                        failures=peer.consecutive_failures,
                    )
                peer.online = False
                # Exponential backoff: 30s, 60s, 120s, ... capped at _MAX_BACKOFF_S
                backoff = min(
                    30 * (2 ** (peer.consecutive_failures - _MAX_FAILURES)),
                    _MAX_BACKOFF_S,
                )
                peer._next_retry_at = time.monotonic() + backoff
            logger.warning(
                "federation_peer_heartbeat_failed",
                peer=peer.name,
                # ``str(exc)`` is empty for ``asyncio.TimeoutError`` and
                # bare httpx connect errors; the typed name keeps the log
                # meaningful instead of ``error=''``.
                error=str(exc) or type(exc).__name__,
                failures=peer.consecutive_failures,
            )

    async def test_peer_connection(self, peer_id: str) -> dict[str, Any]:
        """Test connectivity to a specific peer with a single heartbeat.

        Args:
            peer_id: UUID string of the peer to test.

        Returns:
            Dict with success, node_id, node_name, latency_ms, and error fields.
        """
        peer = self._peers.get(peer_id)
        if peer is None:
            return {"success": False, "error": "Peer not found"}

        client = self._client_for(peer)
        if client is None:
            return {"success": False, "error": "HTTP client not initialized"}
        if peer.is_tunnel and not peer.tunnel_connected:
            return {
                "success": False,
                "node_id": None,
                "node_name": None,
                "latency_ms": None,
                "error": "El nodo no tiene el túnel abierto (no se ha conectado a este oCabra)",
            }

        url = f"{peer.base_url}/ocabra/federation/heartbeat"
        start = time.monotonic()
        try:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {peer.api_key}"},
            )
            resp.raise_for_status()
            elapsed_ms = (time.monotonic() - start) * 1000
            data = resp.json()
            return {
                "success": True,
                "node_id": data.get("node_id"),
                "node_name": data.get("node_name"),
                "latency_ms": round(elapsed_ms, 1),
                "error": None,
            }
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            return {
                "success": False,
                "node_id": None,
                "node_name": None,
                "latency_ms": round(elapsed_ms, 1),
                "error": str(exc),
            }

    # ── Transport / tunnel ───────────────────────────────────────

    def _client_for(self, peer: PeerState) -> httpx.AsyncClient | None:
        """httpx client for *peer*: the shared one, or the peer's tunnel client."""
        if self._http_client is None:
            return None
        if not peer.is_tunnel:
            return self._http_client
        if peer.tunnel_client is None:
            from ocabra.core.federation_tunnel import TunnelTransport

            peer.tunnel_client = httpx.AsyncClient(
                transport=TunnelTransport(lambda: peer.tunnel_session),
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return peer.tunnel_client

    def _require_client(self, peer: PeerState) -> httpx.AsyncClient:
        client = self._client_for(peer)
        if client is None:
            raise RuntimeError("FederationManager not started")
        return client

    async def authenticate_tunnel(
        self, hello: dict[str, Any]
    ) -> tuple[PeerState, dict[str, Any]] | None:
        """Match the token in a tunnel HELLO against tunnel peers.

        Returns ``(peer, welcome_payload)`` or ``None`` (unknown token, or the
        peer is disabled).
        """
        token = hello.get("token")
        if not isinstance(token, str) or not token:
            return None
        presented = hash_tunnel_token(token)
        for peer in self._peers.values():
            if peer.connection_mode != "tunnel_in" or not peer.tunnel_token_hash:
                continue
            if hmac.compare_digest(peer.tunnel_token_hash, presented):
                if not peer.enabled:
                    return None
                return peer, {
                    "node_id": self._node_id,
                    "node_name": self._node_name,
                    "version": getattr(self._settings, "app_version", None),
                    "peer_name": peer.name,
                }
        return None

    async def attach_tunnel(self, peer: PeerState, session: Any) -> None:
        """Make *session* the live tunnel for *peer* and refresh its state now."""
        old = peer.tunnel_session
        peer.tunnel_session = session
        if old is not None and old is not session:
            await old.close(code=4000, reason="replaced by a new connection")
        peer.consecutive_failures = 0
        peer._next_retry_at = 0.0
        logger.info(
            "federation_tunnel_attached",
            peer=peer.name,
            remote_node=session.remote_info.get("node_name"),
            remote_version=session.remote_info.get("version"),
        )
        remote_id = session.remote_info.get("node_id")
        if remote_id and remote_id != peer.remote_node_id:
            try:
                await self.update_peer(peer.peer_id, remote_node_id=str(remote_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning("federation_remote_node_id_update_failed", error=str(exc))
        if peer.connection_mode == "tunnel_in" and self.join_service is not None:
            await self.join_service.on_tunnel_in_connected(peer)
        # Don't wait up to a heartbeat interval to start routing to it.
        asyncio.create_task(self._poll_peer(peer), name=f"federation-tunnel-poll-{peer.name}")

    def detach_tunnel(self, peer: PeerState, session: Any) -> None:
        """Forget *session* if it is still the peer's live tunnel."""
        if peer.tunnel_session is session:
            peer.tunnel_session = None
            peer.online = False
            logger.info("federation_tunnel_detached", peer=peer.name)

    def local_server_for(self, peer: PeerState) -> Any:
        """How we serve *peer*'s requests arriving over its tunnel."""
        from ocabra.core.federation_tunnel import LocalServer

        if self._local_client is None:
            return None
        return LocalServer(
            client=self._local_client,
            api_key=lambda: peer.local_api_key if peer.enabled else None,
            access_level=lambda: peer.local_access_level or "inference",
            node_id=self._node_id,
        )

    def tunnel_hello(self, peer: PeerState) -> dict[str, Any]:
        """HELLO we send when dialing *peer* (token = stored api_key)."""
        return {
            "token": peer.api_key,
            "node_id": self._node_id,
            "node_name": self._node_name,
            "version": getattr(self._settings, "app_version", None),
        }

    def _start_dialer_if_ready(self, peer: PeerState) -> None:
        """Start dialing a ``tunnel_out`` peer once it has a token and is enabled."""
        if (
            peer.connection_mode != "tunnel_out"
            or not peer.enabled
            or peer.join_state is not None
            or not peer.api_key
            or not peer.url
            or (peer.dialer is not None and peer.dialer.state not in ("stopped", "idle"))
        ):
            return
        from ocabra.core.federation_tunnel import TunnelDialer

        async def _connected(conn: Any) -> None:
            await self.attach_tunnel(peer, conn)

        def _disconnected(conn: Any) -> None:
            self.detach_tunnel(peer, conn)

        peer.dialer = TunnelDialer(
            url=peer.url,
            hello=lambda: self.tunnel_hello(peer),
            server=lambda: self.local_server_for(peer),
            on_connected=_connected,
            on_disconnected=_disconnected,
            verify_ssl=self._settings.federation_verify_ssl,
            name=f"federation-dialer-{peer.name}",
        )
        peer.dialer.start()

    async def _stop_dialer(self, peer: PeerState) -> None:
        if peer.dialer is not None:
            await peer.dialer.stop()
            peer.dialer = None

    async def _close_tunnel(self, peer: PeerState, reason: str = "") -> None:
        session = peer.tunnel_session
        peer.tunnel_session = None
        if session is not None:
            await session.close(code=4001, reason=reason[:100])
        if peer.tunnel_client is not None:
            await peer.tunnel_client.aclose()
            peer.tunnel_client = None
        if peer.is_tunnel:
            peer.online = False

    # ── Query methods ────────────────────────────────────────────

    def get_all_peers(self) -> list[PeerState]:
        """Return all peers regardless of state."""
        return list(self._peers.values())

    def get_online_peers(self) -> list[PeerState]:
        """Return only peers that are online and enabled."""
        return [p for p in self._peers.values() if p.online and p.enabled]

    def get_remote_models(self) -> dict[str, list[PeerState]]:
        """Return a mapping of model_id to list of peers that have it loaded.

        Only considers online, enabled peers with models in LOADED status.
        The status comparison is case-insensitive — ``ModelStatus.LOADED.value``
        is lowercase (``"loaded"``) but the federation plan and the inventory
        tests use uppercase. Accept either.
        """
        if federation_hop.get() is not None:
            return {}
        model_map: dict[str, list[PeerState]] = {}
        for peer in self.get_online_peers():
            for model in peer.models:
                model_id = model.get("model_id", "")
                status = str(model.get("status", "")).upper()
                if model_id and status == "LOADED":
                    model_map.setdefault(model_id, []).append(peer)
        return model_map

    def find_best_peer(self, model_id: str) -> PeerState | None:
        """Find the peer with the lowest load that has the given model loaded.

        Load scoring: active_requests * 10 + gpu_utilization_avg_pct.
        Lower is better.

        Args:
            model_id: The canonical model identifier.

        Returns:
            The best PeerState, or None if no peer has the model.
        """
        candidates = self.get_remote_models().get(model_id, [])
        if not candidates:
            return None

        def _score(peer: PeerState) -> float:
            active = peer.load.get("active_requests", 0)
            gpu_util = peer.load.get("gpu_utilization_avg_pct", 0.0)
            return active * 10 + gpu_util

        return min(candidates, key=_score)

    def select_target(self, model_id: str, local_available: bool = False) -> str | PeerState | None:
        """Select the best target for a model request.

        Strategy:
        1. If model is available both locally and on peers, score all
           candidates. Local gets a -5 bias (prefer local to avoid latency).
        2. If only remote, pick the best peer.
        3. If only local, return 'local'.

        Args:
            model_id: The canonical model identifier.
            local_available: Whether the model is loaded locally.

        Returns:
            'local' string, a PeerState, or None if unavailable anywhere.
        """
        remote_peers = self.get_remote_models().get(model_id, [])

        if not remote_peers and local_available:
            return "local"
        if not remote_peers and not local_available:
            return None
        if remote_peers and not local_available:
            return self.find_best_peer(model_id)

        # Both local and remote available — score them.
        def _peer_score(peer: PeerState) -> float:
            active = peer.load.get("active_requests", 0)
            gpu_util = peer.load.get("gpu_utilization_avg_pct", 0.0)
            return active * 10 + gpu_util

        best_peer = min(remote_peers, key=_peer_score)
        peer_score = _peer_score(best_peer)
        local_score = -5.0  # Local bias
        if local_score <= peer_score:
            return "local"
        return best_peer

    # ── Proxy methods ───────────────────────────────────────────

    def _build_proxy_headers(
        self,
        peer: PeerState,
        headers: dict[str, str] | None,
        *,
        drop_content_type: bool = False,
    ) -> dict[str, str]:
        """Sanitise incoming headers before forwarding to a peer.

        Always strips:
            - ``authorization``: the peer must see its own bearer token.
            - ``cookie`` / ``set-cookie``: the user's local session cookie is a
              JWT signed by this node's secret; the peer would reject it and
              never check the bearer token (the auth dep tries cookies first).
            - ``host`` / ``content-length``: rewritten by httpx for the new URL.

        Optionally strips ``content-type`` (used by ``proxy_multipart`` so
        httpx can rebuild the boundary with the new body).
        """
        always_strip = {"authorization", "cookie", "set-cookie", "host", "content-length"}
        if drop_content_type:
            always_strip.add("content-type")
        out: dict[str, str] = {}
        if headers:
            for k, v in headers.items():
                if k.lower() in always_strip:
                    continue
                out[k] = v
        out["Authorization"] = f"Bearer {peer.api_key}"
        from ocabra.core.federation_tunnel import FEDERATION_HOP_HEADER

        out[FEDERATION_HOP_HEADER] = self._node_id
        return out

    async def proxy_request(
        self,
        peer: PeerState,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout: float | None = None,  # noqa: ASYNC109 — el timeout se delega a httpx, que lo gestiona el mismo
    ) -> httpx.Response:
        """Proxy a non-streaming request to a federation peer.

        Args:
            peer: Target peer state.
            path: API path (e.g. '/v1/chat/completions').
            body: Request JSON body.
            headers: Original request headers (auth/cookie are stripped and
                rewritten so the peer's auth dep sees only its own bearer).
            timeout: Optional timeout override.

        Returns:
            The httpx.Response from the peer.
        """
        client = self._require_client(peer)
        proxy_timeout = timeout or self._settings.federation_proxy_timeout_s
        proxy_headers = self._build_proxy_headers(peer, headers)
        url = f"{peer.base_url}{path}"
        return await client.post(
            url,
            json=body,
            headers=proxy_headers,
            timeout=proxy_timeout,
        )

    async def proxy_stream(
        self,
        peer: PeerState,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[bytes]:
        """Proxy a streaming SSE request to a federation peer.

        Args:
            peer: Target peer state.
            path: API path.
            body: Request JSON body.
            headers: Original request headers (auth/cookie stripped).

        Yields:
            Raw bytes from the SSE stream.
        """
        client = self._require_client(peer)
        proxy_headers = self._build_proxy_headers(peer, headers)
        url = f"{peer.base_url}{path}"
        async with client.stream(
            "POST",
            url,
            json=body,
            headers=proxy_headers,
            timeout=self._settings.federation_proxy_timeout_s,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk

    async def proxy_multipart(
        self,
        peer: PeerState,
        path: str,
        files: dict[str, Any],
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 — el timeout se delega a httpx, que lo gestiona el mismo
    ) -> httpx.Response:
        """Proxy a multipart/form-data request to a federation peer.

        Used by endpoints like ``/v1/audio/transcriptions`` that accept file
        uploads. The original ``Content-Type`` header is dropped so httpx can
        rebuild the multipart boundary; only safe forwarded headers are kept.

        Args:
            peer: Target peer state.
            path: API path (e.g. ``/v1/audio/transcriptions``).
            files: ``{"file": (filename, bytes, content_type)}`` mapping.
            data: Form fields (model, language, etc.).
            headers: Optional original headers to inherit (auth/content-type
                are stripped and rewritten).
            timeout: Optional timeout override.

        Returns:
            The httpx.Response from the peer.
        """
        client = self._require_client(peer)
        proxy_timeout = timeout or self._settings.federation_proxy_timeout_s
        proxy_headers = self._build_proxy_headers(peer, headers, drop_content_type=True)
        url = f"{peer.base_url}{path}"
        return await client.post(
            url,
            files=files,
            data=data,
            headers=proxy_headers,
            timeout=proxy_timeout,
        )


# ---------------------------------------------------------------------------
# Federated resolution helper
# ---------------------------------------------------------------------------


# HTTP statuses the peer returns when it can't serve a model for this token
# or is temporarily broken. On these we fall back to local processing instead
# of bubbling the peer's error to the client.
#
#   401 / 403 / 404 — federation api_key lacks access on the peer (typical
#     when the peer hasn't granted the api_key access to the model the peer
#     itself reports as LOADED in its heartbeat).
#   500 / 502 / 503 / 504 — peer-side failure that the local node can probably
#     handle by itself. We'd rather pay a cold-start than surface a 5xx
#     produced by *somebody else's node*.
#
# The user's request is one indivisible thing from their point of view; if
# we routed it to a peer and the peer can't serve it, the right answer is to
# try locally before giving up.
PEER_FALLBACK_STATUSES = frozenset({401, 403, 404, 500, 502, 503, 504})


def should_fallback_to_local(status_code: int) -> bool:
    """True when a peer's response should trigger a local retry."""
    return status_code in PEER_FALLBACK_STATUSES


async def resolve_federated(
    model_id: str,
    model_manager: ModelManager,
    federation_manager: FederationManager | None,
    profile_registry: Any = None,
) -> tuple[str, PeerState | None]:
    """Decide whether to handle a request locally or proxy to a remote peer.

    The resolution favours **warm** workers across the cluster: if a peer
    already has the model loaded, the request is proxied there instead of
    paying a cold-start cost locally. Order:

        1. Loaded locally → load balancer decides (local bias of -5).
        2. Not loaded locally but loaded on at least one peer → ``("remote",
           best_peer)``. Avoids spinning a local worker when a cluster
           neighbour already has the model warm.
        3. Not loaded anywhere remote, but the model is registered locally
           (CONFIGURED / UNLOADED / LOADING / ERROR) → ``("local", None)``
           — fall back to a local load.
        4. Not found anywhere → raise ``HTTPException(404)``.

    Args:
        model_id: The ``model`` field as the client sent it. Can be either a
            canonical model id (``ollama/qwen3:8b``,
            ``diffusers/image/checkpoints/sd_xl_base_1.0.safetensors``) or a
            short profile id (``sdxl-base``, ``acestep/turbo``).
        profile_registry: Optional :class:`ProfileRegistry`. When provided,
            *model_id* is first resolved through the registry; if it matches a
            profile, the profile's ``base_model_id`` is used for the local /
            remote checks. Without this, callers passing a profile id would
            hit step 4 and 404 even when the underlying base model exists.
    """
    from fastapi import HTTPException

    from ocabra.core.model_manager import ModelStatus
    from ocabra.core.model_ref import _split_canonical_model_ref, build_model_ref

    # 0. Normalise profile id → canonical base model id when a registry is
    # provided. Endpoints that accept profile ids (chat, images, audio,
    # embeddings, completions, rerank) pass the registry so a profile id
    # short-circuits to its base_model_id before federation lookup.
    if profile_registry is not None:
        try:
            profile = await profile_registry.get(model_id)
            if profile is None:
                alias_target = await profile_registry.resolve_alias(model_id)
                profile = alias_target[0] if alias_target is not None else None
        except Exception:  # noqa: BLE001
            profile = None
        if profile is not None:
            if getattr(profile, "profile_kind", "execution") == "router":
                # Same first rule as RouterResolver.pick: a target that is
                # already warm (here, or on a peer) beats an earlier cold one.
                # Taking the first existing target sent every router request
                # to its #1 even when #2 was loaded somewhere in the cluster.
                remote_loaded = (
                    federation_manager.get_remote_models() if federation_manager is not None else {}
                )
                candidates = []
                for target_id in profile.routing_targets or []:
                    candidate = await profile_registry.get(target_id)
                    if candidate is None and "/" in str(target_id):
                        # deployment id target (``whisper_cpp/<model_key>``)
                        listed = await profile_registry.list_by_model(str(target_id))
                        candidate = listed[0] if listed else None
                    if candidate is not None and candidate.base_model_id is not None:
                        candidates.append(candidate)
                target = None
                for candidate in candidates:
                    local_state = await model_manager.get_state(candidate.base_model_id)
                    if (
                        local_state is not None and local_state.status == ModelStatus.LOADED
                    ) or remote_loaded.get(candidate.base_model_id):
                        target = candidate
                        break
                if target is None and candidates:
                    target = candidates[0]
                if target is not None:
                    model_id = target.base_model_id
            elif profile.base_model_id is not None:
                model_id = profile.base_model_id

    # Look up local state — also try the canonical ``ollama/<id>`` form for
    # bare Ollama names.
    state = await model_manager.get_state(model_id)
    if state is None and _split_canonical_model_ref(model_id) is None:
        canonical = build_model_ref("ollama", model_id)
        canonical_state = await model_manager.get_state(canonical)
        if canonical_state is not None:
            state = canonical_state
            model_id = canonical

    local_loaded = state is not None and state.status == ModelStatus.LOADED
    local_known = state is not None

    # Look up remote candidates that have the model LOADED right now. Try the
    # ollama/<id> canonical form too when the input wasn't canonical.
    remote_peers: list[PeerState] = []
    if federation_manager is not None:
        remote_models = federation_manager.get_remote_models()
        remote_peers = list(remote_models.get(model_id, []))
        if not remote_peers and _split_canonical_model_ref(model_id) is None:
            canonical = build_model_ref("ollama", model_id)
            canonical_peers = remote_models.get(canonical, [])
            if canonical_peers:
                remote_peers = list(canonical_peers)
                model_id = canonical

    # 1. Loaded locally → let the load balancer pick (it knows the local bias).
    if local_loaded:
        if federation_manager is not None:
            target = federation_manager.select_target(model_id, local_available=True)
            if target != "local" and target is not None:
                logger.info(
                    "federation_resolve",
                    decision="remote",
                    reason="local_loaded_peer_lower_load",
                    model_id=model_id,
                    peer=getattr(target, "name", None),
                )
                return ("remote", target)
        logger.info(
            "federation_resolve",
            decision="local",
            reason="local_loaded",
            model_id=model_id,
        )
        return ("local", None)

    # 2. Not loaded locally but a peer has it warm → proxy to the peer.
    if remote_peers and federation_manager is not None:
        best_peer = federation_manager.find_best_peer(model_id)
        if best_peer is not None:
            logger.info(
                "federation_resolve",
                decision="remote",
                reason="local_cold_peer_warm",
                model_id=model_id,
                peer=best_peer.name,
            )
            return ("remote", best_peer)

    # 3. Nobody has it loaded, but it's registered locally → load it locally.
    if local_known:
        logger.info(
            "federation_resolve",
            decision="local",
            reason="no_peer_has_model",
            model_id=model_id,
            local_status=(
                getattr(state.status, "value", state.status) if state is not None else None
            ),
            peer_loaded_models=(
                {
                    p.name: [
                        m.get("model_id")
                        for m in p.models
                        if str(m.get("status", "")).upper() == "LOADED"
                    ]
                    for p in federation_manager.get_online_peers()
                }
                if federation_manager is not None
                else {}
            ),
        )
        return ("local", None)

    # 4. Not found anywhere.
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"The model '{model_id}' does not exist.",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


class FederationHopMiddleware:
    """Pure ASGI middleware: mark requests that came from a federation peer.

    Peers add ``x-ocabra-federation-hop`` (their node id) to every request they
    send us, over HTTP or tunnel. While handling such a request
    :data:`federation_hop` is set, so it is never forwarded to another peer. A
    client spoofing the header only forces local handling — harmless.
    """

    _HEADER = b"x-ocabra-federation-hop"

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            for key, value in scope.get("headers") or ():
                if key == self._HEADER:
                    token = federation_hop.set(value.decode("latin-1") or "peer")
                    try:
                        await self.app(scope, receive, send)
                    finally:
                        federation_hop.reset(token)
                    return
        await self.app(scope, receive, send)
