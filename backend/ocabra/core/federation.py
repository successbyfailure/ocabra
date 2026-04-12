"""Federation manager for multi-node peer-to-peer inference.

Handles peer registration, heartbeat polling, API key encryption,
and peer state tracking. Provides methods for proxy routing and
load balancing across federated nodes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from cryptography.fernet import Fernet

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from ocabra.config import Settings

logger = structlog.get_logger(__name__)

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

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Load peers from DB, start heartbeat loop."""
        self._started_at = time.monotonic()
        self._http_client = httpx.AsyncClient(
            verify=self._settings.federation_verify_ssl,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        await self._load_peers_from_db()
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="federation-heartbeat-loop",
        )
        logger.info(
            "federation_manager_started",
            node_id=self._node_id,
            node_name=self._node_name,
            peer_count=len(self._peers),
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
            )
        logger.info("federation_peers_loaded", count=len(self._peers))

    async def add_peer(
        self,
        name: str,
        url: str,
        api_key: str,
        access_level: str = "inference",
    ) -> PeerState:
        """Add a new peer to the DB and runtime state.

        Args:
            name: Human-readable peer name (unique).
            url: Base URL of the peer node.
            api_key: API key for authenticating with the peer.
            access_level: Access level ('inference' or 'full').

        Returns:
            The newly created PeerState.
        """
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
                enabled=True,
            )
            session.add(row)
            await session.commit()

        state = PeerState(
            peer_id=str(peer_id),
            name=name,
            url=url.rstrip("/"),
            api_key=api_key,
            access_level=access_level,
            enabled=True,
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

        logger.info("federation_peer_updated", peer_id=peer_id)
        return peer

    # ── Heartbeat ────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Periodically poll each enabled peer for heartbeat."""
        interval = max(5, self._settings.federation_heartbeat_interval)
        logger.info("federation_heartbeat_loop_started", interval_s=interval)
        while True:
            try:
                await self._poll_all_peers()
            except Exception as exc:
                logger.warning("federation_heartbeat_loop_error", error=str(exc))
            await asyncio.sleep(interval)

    async def _poll_all_peers(self) -> None:
        """Poll all enabled peers concurrently."""
        now = time.monotonic()
        tasks = []
        for peer in self._peers.values():
            if not peer.enabled:
                continue
            # Skip offline peers until their backoff expires.
            if not peer.online and peer._next_retry_at > now:
                continue
            tasks.append(self._poll_peer(peer))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_peer(self, peer: PeerState) -> None:
        """Send a single heartbeat request to a peer."""
        if self._http_client is None:
            return
        url = f"{peer.url}/ocabra/federation/heartbeat"
        try:
            resp = await self._http_client.get(
                url,
                headers={"Authorization": f"Bearer {peer.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            peer.online = True
            peer.consecutive_failures = 0
            peer._next_retry_at = 0.0
            peer.last_heartbeat = datetime.now(timezone.utc)
            peer.gpus = data.get("gpus", [])
            peer.models = data.get("models", [])
            peer.load = data.get("load", {})
            logger.debug("federation_peer_heartbeat_ok", peer=peer.name)
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
            logger.debug(
                "federation_peer_heartbeat_failed",
                peer=peer.name,
                error=str(exc),
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

        if self._http_client is None:
            return {"success": False, "error": "HTTP client not initialized"}

        url = f"{peer.url}/ocabra/federation/heartbeat"
        start = time.monotonic()
        try:
            resp = await self._http_client.get(
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
        """
        model_map: dict[str, list[PeerState]] = {}
        for peer in self.get_online_peers():
            for model in peer.models:
                model_id = model.get("model_id", "")
                status = model.get("status", "")
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

    async def proxy_request(
        self,
        peer: PeerState,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout: float | None = None,
    ) -> httpx.Response:
        """Proxy a non-streaming request to a federation peer.

        Args:
            peer: Target peer state.
            path: API path (e.g. '/v1/chat/completions').
            body: Request JSON body.
            headers: Original request headers (Authorization will be replaced).
            timeout: Optional timeout override.

        Returns:
            The httpx.Response from the peer.
        """
        if self._http_client is None:
            raise RuntimeError("FederationManager not started")
        proxy_timeout = timeout or self._settings.federation_proxy_timeout_s
        proxy_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        proxy_headers["Authorization"] = f"Bearer {peer.api_key}"
        url = f"{peer.url}{path}"
        return await self._http_client.post(
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
            headers: Original request headers.

        Yields:
            Raw bytes from the SSE stream.
        """
        if self._http_client is None:
            raise RuntimeError("FederationManager not started")
        proxy_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        proxy_headers["Authorization"] = f"Bearer {peer.api_key}"
        url = f"{peer.url}{path}"
        async with self._http_client.stream(
            "POST",
            url,
            json=body,
            headers=proxy_headers,
            timeout=self._settings.federation_proxy_timeout_s,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk


# ---------------------------------------------------------------------------
# Federated resolution helper
# ---------------------------------------------------------------------------


async def resolve_federated(
    model_id: str,
    model_manager: ModelManager,
    federation_manager: FederationManager | None,
) -> tuple[str, PeerState | None]:
    """Decide whether to handle a request locally or proxy to a remote peer.

    Resolution order:
        1. Model loaded locally -> ``("local", None)``
        2. Model exists locally (can be loaded) -> ``("local", None)``
        3. Model available on a remote peer -> ``("remote", best_peer)``
        4. Not found anywhere -> raise ``HTTPException(404)``
    """
    from fastapi import HTTPException

    from ocabra.core.model_manager import ModelStatus

    # 1. Check if loaded locally
    state = await model_manager.get_state(model_id)
    if state is not None and state.status == ModelStatus.LOADED:
        # Model is loaded — still ask the load balancer if a peer is better
        if federation_manager is not None:
            target = federation_manager.select_target(model_id, local_available=True)
            if target != "local" and target is not None:
                return ("remote", target)
        return ("local", None)

    # 2. Model exists locally but not loaded — prefer local (will be loaded)
    if state is not None and state.status in (
        ModelStatus.CONFIGURED,
        ModelStatus.UNLOADED,
        ModelStatus.LOADING,
        ModelStatus.ERROR,
    ):
        return ("local", None)

    # 3. Check remote peers
    if federation_manager is not None:
        remote_models = federation_manager.get_remote_models()
        remote_peers = remote_models.get(model_id, [])
        if remote_peers:
            best_peer = federation_manager.find_best_peer(model_id)
            if best_peer is not None:
                return ("remote", best_peer)

    # 4. Not found
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
