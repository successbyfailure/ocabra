"""Joining a federation: requests, invites and the "swarm" agreement.

Flow (B wants to join A; only A needs a reachable URL):

1. B → ``POST A/ocabra/federation/join-requests`` (no auth) with its identity.
   A answers with a ``request_id``, a ``poll_secret`` and a short
   ``verification_code`` both admins can compare by another channel.
2. A's admin accepts or denies it in the UI. An **invite** code (created by
   A's admin, e.g. for a cloud pod) skips this step.
3. B polls ``GET A/.../join-requests/{id}/status`` with its secret; once
   accepted it receives the tunnel token and dials A.

On acceptance each side creates a local service user ``federation:<peer>``
with an API key: that key runs the *other* side's requests on our API (the
tunnel's :class:`~ocabra.core.federation_tunnel.LocalServer`), so stats show
what the peer used and revoking the user cuts it off.

Modes: ``swarm`` ("Soy parte del enjambre") = mutual, full control both ways.
``rules`` ("Mi server mis reglas") will let each side choose; not yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import secrets
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from ocabra.core.federation import FederationManager, PeerState

logger = structlog.get_logger(__name__)

JOIN_REQUEST_TTL = timedelta(hours=24)
MAX_PENDING_REQUESTS = 20
MAX_PENDING_PER_IP = 3
# Public endpoint: at most this many join attempts per IP per window.
_RATE_WINDOW_S = 600
_RATE_MAX = 10

SUPPORTED_MODES = frozenset({"swarm"})


class JoinError(Exception):
    """A join operation failed; ``status`` is the HTTP status to report."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def new_poll_secret() -> str:
    return "ocjs_" + secrets.token_urlsafe(24)


def new_invite_code() -> str:
    return "ocinv_" + secrets.token_urlsafe(18)


def new_verification_code() -> str:
    raw = secrets.token_hex(3).upper()
    return f"{raw[0:2]}-{raw[2:4]}-{raw[4:6]}"


def _service_username(peer_name: str) -> str:
    return f"federation:{peer_name}"


def _access_levels_for(mode: str) -> tuple[str, str]:
    """``(what we may do on them, what they may do on us)`` for a mode."""
    if mode == "swarm":
        return "full", "full"
    return "inference", "inference"


# ── Service identity ─────────────────────────────────────────────


async def create_service_identity(session_factory: Any, peer_name: str) -> tuple[str, str]:
    """Create (or reuse) the local user ``federation:<peer>`` and a fresh key.

    The user gets ``system_admin`` so it sees every model; what a peer can
    actually reach is limited by the tunnel's path allowlist (inference, model
    load/unload/download, GPUs — never users/config/keys). Its password is
    random and never revealed, so nobody can log in as it.

    Returns:
        ``(user_id, raw_api_key)``.
    """
    from sqlalchemy import select

    from ocabra.core.auth_manager import hash_api_key, hash_password
    from ocabra.db.auth import ApiKey, User

    username = _service_username(peer_name)
    raw_key = f"sk-ocabra-{secrets.token_urlsafe(32)}"
    async with session_factory() as session:
        user = (
            await session.execute(select(User).where(User.username == username))
        ).scalar_one_or_none()
        if user is None:
            user = User(
                id=uuid.uuid4(),
                username=username,
                hashed_password=hash_password(secrets.token_urlsafe(32)),
                role="system_admin",
                is_active=True,
            )
            session.add(user)
            await session.flush()
        else:
            user.is_active = True
            user.role = "system_admin"
        session.add(
            ApiKey(
                user_id=user.id,
                name="federation tunnel",
                key_hash=hash_api_key(raw_key),
                key_prefix=raw_key[:16],
                group_id=None,
            )
        )
        await session.commit()
        user_id = str(user.id)
    logger.info("federation_service_identity_created", username=username)
    return user_id, raw_key


async def delete_service_identity(session_factory: Any, user_id: str | None) -> None:
    """Delete a peer's service user (its keys go with it, cascade)."""
    if not user_id:
        return
    import sqlalchemy as sa

    from ocabra.db.auth import User

    async with session_factory() as session:
        await session.execute(sa.delete(User).where(User.id == uuid.UUID(user_id)))
        await session.commit()


# ── Service ──────────────────────────────────────────────────────


class FederationJoinService:
    """Join requests and invites (hub side) plus outgoing joins (requester side)."""

    def __init__(self, fm: FederationManager) -> None:
        self.fm = fm
        self._pollers: dict[str, asyncio.Task] = {}
        self._bootstrap_task: asyncio.Task | None = None
        self._rate: dict[str, list[float]] = {}

    # -- lifecycle

    def start(self) -> None:
        for peer in self.fm.get_all_peers():
            if peer.connection_mode == "tunnel_out" and peer.join_state == "pending":
                self._start_poller(peer)
        s = self.fm.settings
        if getattr(s, "federation_join_url", "") and getattr(s, "federation_join_invite", ""):
            self._bootstrap_task = asyncio.create_task(
                self._bootstrap_join(), name="federation-join-bootstrap"
            )

    async def stop(self) -> None:
        tasks = list(self._pollers.values())
        if self._bootstrap_task is not None:
            tasks.append(self._bootstrap_task)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._pollers.clear()
        self._bootstrap_task = None

    # -- helpers

    def _hub_info(self) -> dict[str, Any]:
        return {
            "node_id": self.fm.node_id,
            "node_name": self.fm.node_name,
            "version": getattr(self.fm.settings, "app_version", None),
        }

    def _unique_peer_name(self, wanted: str) -> str:
        base = (wanted or "nodo").strip()[:200] or "nodo"
        taken = {p.name for p in self.fm.get_all_peers()}
        if base not in taken:
            return base
        n = 2
        while f"{base}-{n}" in taken:
            n += 1
        return f"{base}-{n}"

    def _check_rate(self, ip: str | None) -> None:
        key = ip or "?"
        now = time.monotonic()
        hits = [t for t in self._rate.get(key, []) if now - t < _RATE_WINDOW_S]
        if len(hits) >= _RATE_MAX:
            raise JoinError("Demasiadas solicitudes desde esta dirección; prueba más tarde", 429)
        hits.append(now)
        self._rate[key] = hits

    # ═══ Hub side: requests and invites ══════════════════════════

    async def create_request(
        self, payload: dict[str, Any], *, source_ip: str | None
    ) -> dict[str, Any]:
        """Register an incoming join request (public endpoint).

        With a valid ``invite_code`` the request is accepted on the spot and the
        response already carries the tunnel token.
        """
        import sqlalchemy as sa

        from ocabra.db.federation import FederationInvite, FederationJoinRequest

        self._check_rate(source_ip)
        mode = str(payload.get("mode") or "swarm")
        if mode not in SUPPORTED_MODES:
            raise JoinError(f"Modo de federación no soportado: {mode}", 400)
        node_id = str(payload.get("node_id") or "").strip()
        node_name = str(payload.get("node_name") or "").strip()
        if not node_id or not node_name:
            raise JoinError("node_id y node_name son obligatorios", 400)
        if node_id == self.fm.node_id:
            raise JoinError("Un nodo no puede unirse a sí mismo", 400)
        if any(p.remote_node_id == node_id for p in self.fm.get_all_peers()):
            raise JoinError("Este nodo ya forma parte de la federación", 409)

        invite_code = str(payload.get("invite_code") or "").strip()
        now = datetime.now(UTC)
        async with self.fm.session_factory() as session:
            invite = None
            if invite_code:
                invite = (
                    await session.execute(
                        sa.select(FederationInvite).where(
                            FederationInvite.code_hash == _hash(invite_code)
                        )
                    )
                ).scalar_one_or_none()
                if (
                    invite is None
                    or invite.revoked
                    or invite.uses >= invite.max_uses
                    or (invite.expires_at is not None and invite.expires_at <= now)
                ):
                    raise JoinError("Invitación no válida o caducada", 403)
            elif not getattr(self.fm.settings, "federation_accept_join_requests", False):
                raise JoinError("Este nodo no acepta solicitudes de unión", 403)

            pending = (
                (
                    await session.execute(
                        sa.select(FederationJoinRequest).where(
                            FederationJoinRequest.status == "pending"
                        )
                    )
                )
                .scalars()
                .all()
            )
            # Expire stale ones on the way.
            live = []
            for req in pending:
                if req.expires_at is not None and req.expires_at <= now:
                    req.status = "expired"
                else:
                    live.append(req)
            # A node re-asking replaces its own pending request.
            for req in live:
                if req.remote_node_id == node_id:
                    req.status = "expired"
            live = [r for r in live if r.remote_node_id != node_id]
            if invite is None:
                if len(live) >= MAX_PENDING_REQUESTS:
                    raise JoinError("Hay demasiadas solicitudes pendientes", 429)
                if sum(1 for r in live if r.source_ip == source_ip) >= MAX_PENDING_PER_IP:
                    raise JoinError("Demasiadas solicitudes pendientes desde esta dirección", 429)

            poll_secret = new_poll_secret()
            gpus = payload.get("gpus") if isinstance(payload.get("gpus"), list) else None
            req = FederationJoinRequest(
                id=uuid.uuid4(),
                status="pending",
                mode=mode,
                remote_node_id=node_id[:255],
                remote_node_name=node_name[:255],
                remote_version=(str(payload.get("version") or "")[:64] or None),
                message=(str(payload.get("message") or "")[:2000] or None),
                source_ip=(source_ip or None),
                gpus=gpus[:16] if gpus else None,
                verification_code=new_verification_code(),
                poll_secret_hash=_hash(poll_secret),
                invite_id=invite.id if invite is not None else None,
                expires_at=now + JOIN_REQUEST_TTL,
            )
            session.add(req)
            if invite is not None:
                invite.uses += 1
            await session.commit()
            request_id = str(req.id)
            verification_code = req.verification_code

        logger.info(
            "federation_join_request_received",
            request_id=request_id,
            remote=node_name,
            source_ip=source_ip,
            via_invite=invite is not None,
        )
        result: dict[str, Any] = {
            "request_id": request_id,
            "poll_secret": poll_secret,
            "verification_code": verification_code,
            "status": "pending",
            "mode": mode,
            "hub": self._hub_info(),
        }
        if invite is not None:
            peer = await self.accept(request_id)
            result["status"] = "accepted"
            result["tunnel_token"] = await self._token_for(request_id)
            result["peer_name"] = peer.name
        return result

    async def _token_for(self, request_id: str) -> str | None:
        from ocabra.db.federation import FederationJoinRequest

        async with self.fm.session_factory() as session:
            req = await session.get(FederationJoinRequest, uuid.UUID(request_id))
            if req is None or not req.token_encrypted:
                return None
            return self.fm.decrypt_api_key(req.token_encrypted)

    async def request_status(self, request_id: str, poll_secret: str) -> dict[str, Any]:
        """Status of a join request for the node that made it (public endpoint)."""
        from ocabra.db.federation import FederationJoinRequest

        try:
            parsed = uuid.UUID(request_id)
        except ValueError as exc:
            raise JoinError("Solicitud no encontrada", 404) from exc
        async with self.fm.session_factory() as session:
            req = await session.get(FederationJoinRequest, parsed)
            if req is None or not hmac.compare_digest(req.poll_secret_hash, _hash(poll_secret)):
                raise JoinError("Solicitud no encontrada", 404)
            status = req.status
            if (
                status == "pending"
                and req.expires_at is not None
                and req.expires_at <= datetime.now(UTC)
            ):
                req.status = status = "expired"
                await session.commit()
            out: dict[str, Any] = {
                "request_id": request_id,
                "status": status,
                "mode": req.mode,
                "verification_code": req.verification_code,
                "hub": self._hub_info(),
            }
            if status == "accepted" and req.token_encrypted:
                out["tunnel_token"] = self.fm.decrypt_api_key(req.token_encrypted)
        return out

    async def list_requests(self, *, include_decided: bool = False) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        from ocabra.db.federation import FederationJoinRequest

        now = datetime.now(UTC)
        async with self.fm.session_factory() as session:
            q = sa.select(FederationJoinRequest).order_by(FederationJoinRequest.created_at.desc())
            if not include_decided:
                q = q.where(FederationJoinRequest.status == "pending")
            rows = (await session.execute(q.limit(100))).scalars().all()
            out = []
            for r in rows:
                status = r.status
                if status == "pending" and r.expires_at is not None and r.expires_at <= now:
                    status = "expired"
                    if not include_decided:
                        continue
                out.append(
                    {
                        "id": str(r.id),
                        "status": status,
                        "mode": r.mode,
                        "remote_node_id": r.remote_node_id,
                        "remote_node_name": r.remote_node_name,
                        "remote_version": r.remote_version,
                        "message": r.message,
                        "source_ip": r.source_ip,
                        "gpus": r.gpus or [],
                        "verification_code": r.verification_code,
                        "via_invite": r.invite_id is not None,
                        "peer_id": str(r.peer_id) if r.peer_id else None,
                        "created_at": r.created_at,
                        "expires_at": r.expires_at,
                        "decided_at": r.decided_at,
                    }
                )
        return out

    async def accept(
        self, request_id: str, *, name: str | None = None, notes: str | None = None
    ) -> PeerState:
        """Accept a pending request: create the ``tunnel_in`` peer and its token."""
        from ocabra.core.federation import new_tunnel_token
        from ocabra.db.federation import FederationJoinRequest

        async with self.fm.session_factory() as session:
            req = await session.get(FederationJoinRequest, uuid.UUID(request_id))
            if req is None:
                raise JoinError("Solicitud no encontrada", 404)
            if req.status != "pending":
                raise JoinError(f"La solicitud ya está '{req.status}'", 409)
            if req.expires_at is not None and req.expires_at <= datetime.now(UTC):
                req.status = "expired"
                await session.commit()
                raise JoinError("La solicitud ha caducado", 409)
            mode = req.mode
            remote_node_id = req.remote_node_id
            remote_name = req.remote_node_name

        peer_name = self._unique_peer_name(name or remote_name)
        our_level, their_level = _access_levels_for(mode)
        token = new_tunnel_token()
        user_id, local_key = await create_service_identity(self.fm.session_factory, peer_name)
        try:
            peer = await self.fm.add_peer(
                name=peer_name,
                url="",
                api_key="",
                access_level=our_level,
                connection_mode="tunnel_in",
                tunnel_token=token,
                notes=notes,
                mode=mode,
                remote_node_id=remote_node_id,
                local_api_key=local_key,
                local_user_id=user_id,
                local_access_level=their_level,
            )
        except Exception:
            await delete_service_identity(self.fm.session_factory, user_id)
            raise

        async with self.fm.session_factory() as session:
            req = await session.get(FederationJoinRequest, uuid.UUID(request_id))
            if req is not None:
                req.status = "accepted"
                req.peer_id = uuid.UUID(peer.peer_id)
                req.token_encrypted = self.fm.encrypt_api_key(token)
                req.decided_at = datetime.now(UTC)
                await session.commit()
        logger.info("federation_join_request_accepted", request_id=request_id, peer=peer_name)
        return peer

    async def deny(self, request_id: str) -> None:
        from ocabra.db.federation import FederationJoinRequest

        async with self.fm.session_factory() as session:
            req = await session.get(FederationJoinRequest, uuid.UUID(request_id))
            if req is None:
                raise JoinError("Solicitud no encontrada", 404)
            if req.status != "pending":
                raise JoinError(f"La solicitud ya está '{req.status}'", 409)
            req.status = "denied"
            req.decided_at = datetime.now(UTC)
            await session.commit()
        logger.info("federation_join_request_denied", request_id=request_id)

    async def on_tunnel_in_connected(self, peer: PeerState) -> None:
        """The requester picked up its token and connected: forget the token."""
        import sqlalchemy as sa

        from ocabra.db.federation import FederationJoinRequest

        try:
            async with self.fm.session_factory() as session:
                await session.execute(
                    sa.update(FederationJoinRequest)
                    .where(
                        FederationJoinRequest.peer_id == uuid.UUID(peer.peer_id),
                        FederationJoinRequest.token_encrypted.is_not(None),
                    )
                    .values(token_encrypted=None)
                )
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("federation_join_token_wipe_failed", peer=peer.name, error=str(exc))

    async def on_peer_removed(self, peer: PeerState) -> None:
        """Peer deleted: stop polling for it and remove its service user."""
        task = self._pollers.pop(peer.peer_id, None)
        if task is not None:
            task.cancel()
        try:
            await delete_service_identity(self.fm.session_factory, peer.local_user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("federation_service_identity_delete_failed", error=str(exc))

    # -- invites

    async def create_invite(
        self,
        *,
        label: str,
        mode: str = "swarm",
        max_uses: int = 1,
        expires_in_hours: float | None = 24,
    ) -> dict[str, Any]:
        """Create an invite. The code is returned only here."""
        from ocabra.db.federation import FederationInvite

        if mode not in SUPPORTED_MODES:
            raise JoinError(f"Modo de federación no soportado: {mode}", 400)
        code = new_invite_code()
        expires_at = (
            datetime.now(UTC) + timedelta(hours=expires_in_hours) if expires_in_hours else None
        )
        async with self.fm.session_factory() as session:
            inv = FederationInvite(
                id=uuid.uuid4(),
                label=label[:255],
                code_hash=_hash(code),
                mode=mode,
                max_uses=max(1, int(max_uses)),
                uses=0,
                expires_at=expires_at,
                revoked=False,
            )
            session.add(inv)
            await session.commit()
            out = self._invite_out(inv)
        out["code"] = code
        return out

    @staticmethod
    def _invite_out(inv: Any) -> dict[str, Any]:
        now = datetime.now(UTC)
        usable = (
            not inv.revoked
            and inv.uses < inv.max_uses
            and (inv.expires_at is None or inv.expires_at > now)
        )
        return {
            "id": str(inv.id),
            "label": inv.label,
            "mode": inv.mode,
            "max_uses": inv.max_uses,
            "uses": inv.uses,
            "expires_at": inv.expires_at,
            "revoked": inv.revoked,
            "usable": usable,
            "created_at": inv.created_at,
        }

    async def list_invites(self) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        from ocabra.db.federation import FederationInvite

        async with self.fm.session_factory() as session:
            rows = (
                (
                    await session.execute(
                        sa.select(FederationInvite)
                        .order_by(FederationInvite.created_at.desc())
                        .limit(100)
                    )
                )
                .scalars()
                .all()
            )
            return [self._invite_out(r) for r in rows]

    async def revoke_invite(self, invite_id: str) -> None:
        from ocabra.db.federation import FederationInvite

        async with self.fm.session_factory() as session:
            inv = await session.get(FederationInvite, uuid.UUID(invite_id))
            if inv is None:
                raise JoinError("Invitación no encontrada", 404)
            inv.revoked = True
            await session.commit()

    # ═══ Requester side: joining someone else ════════════════════

    async def _local_gpus(self) -> list[dict[str, Any]]:
        gpu_manager = getattr(self.fm, "gpu_manager", None)
        if gpu_manager is None:
            return []
        try:
            states = await gpu_manager.get_all_states()
            return [{"name": g.name, "total_vram_mb": g.total_vram_mb} for g in states]
        except Exception:  # noqa: BLE001
            return []

    async def join(
        self,
        hub_url: str,
        *,
        message: str | None = None,
        invite_code: str | None = None,
        name: str | None = None,
        mode: str = "swarm",
        gpus: list[dict[str, Any]] | None = None,
    ) -> PeerState:
        """Ask the node at *hub_url* to let us join; returns the new peer.

        The peer is ``tunnel_out`` with ``join_state='pending'`` until the hub
        decides (polled in the background), or connected right away when the
        invite auto-accepts.
        """
        hub_url = hub_url.strip().rstrip("/")
        if not hub_url.startswith(("http://", "https://")):
            raise JoinError("La URL debe empezar por https:// (o http://)", 400)
        if mode not in SUPPORTED_MODES:
            raise JoinError(f"Modo de federación no soportado: {mode}", 400)
        if any(
            p.connection_mode == "tunnel_out" and p.url.rstrip("/") == hub_url
            for p in self.fm.get_all_peers()
        ):
            raise JoinError("Ya hay un enlace (o una solicitud) con ese nodo", 409)

        body = {
            "node_id": self.fm.node_id,
            "node_name": self.fm.node_name,
            "version": getattr(self.fm.settings, "app_version", None),
            "gpus": gpus if gpus is not None else await self._local_gpus(),
            "message": message,
            "mode": mode,
            "invite_code": invite_code or None,
        }
        client = self.fm.http_client
        if client is None:
            raise JoinError("La federación no está arrancada", 503)
        try:
            resp = await client.post(
                f"{hub_url}/ocabra/federation/join-requests", json=body, timeout=20.0
            )
        except httpx.HTTPError as exc:
            raise JoinError(f"No se pudo contactar con {hub_url}: {exc}", 502) from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail")
            except Exception:  # noqa: BLE001
                detail = resp.text[:300]
            # Keep the hub's 4xx (invalid invite, not accepting, already joined)
            # so callers can tell "won't work" from "try again later".
            status = resp.status_code if 400 <= resp.status_code < 500 else 502
            raise JoinError(f"El otro nodo respondió {resp.status_code}: {detail}", status)
        data = resp.json()
        hub = data.get("hub") or {}

        peer = await self.fm.add_peer(
            name=self._unique_peer_name(name or hub.get("node_name") or hub_url.split("//")[-1]),
            url=hub_url,
            api_key="",
            access_level=_access_levels_for(mode)[0],
            connection_mode="tunnel_out",
            mode=mode,
            remote_node_id=hub.get("node_id"),
            join_state="pending",
            join_request_id=str(data.get("request_id")),
            join_poll_secret=str(data.get("poll_secret")),
        )
        peer.join_verification_code = data.get("verification_code")
        logger.info(
            "federation_join_requested", hub=hub_url, peer=peer.name, status=data.get("status")
        )
        if data.get("status") == "accepted" and data.get("tunnel_token"):
            await self._finalize_outgoing(peer, str(data["tunnel_token"]), hub)
        else:
            self._start_poller(peer)
        return peer

    def _start_poller(self, peer: PeerState) -> None:
        old = self._pollers.get(peer.peer_id)
        if old is not None and not old.done():
            return
        self._pollers[peer.peer_id] = asyncio.create_task(
            self._poll_until_decided(peer), name=f"federation-join-poll-{peer.name}"
        )

    async def check_outgoing(self, peer: PeerState) -> str | None:
        """Ask the hub once about our pending request; act on the answer.

        Returns the new status, or ``None`` if the hub couldn't be reached.
        """
        client = self.fm.http_client
        if client is None or not peer.join_request_id or not peer.join_poll_secret:
            return None
        try:
            resp = await client.get(
                f"{peer.url}/ocabra/federation/join-requests/{peer.join_request_id}/status",
                headers={"X-Join-Secret": peer.join_poll_secret},
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            logger.debug("federation_join_poll_unreachable", hub=peer.url, error=str(exc))
            return None
        if resp.status_code == 404:
            await self.fm.update_peer(peer.peer_id, join_state="expired")
            return "expired"
        if resp.status_code >= 400:
            return None
        data = resp.json()
        status = data.get("status")
        peer.join_verification_code = data.get("verification_code") or peer.join_verification_code
        if status == "accepted" and data.get("tunnel_token"):
            await self._finalize_outgoing(peer, str(data["tunnel_token"]), data.get("hub") or {})
        elif status in ("denied", "expired"):
            await self.fm.update_peer(peer.peer_id, join_state=status)
        return status

    async def _poll_until_decided(self, peer: PeerState) -> None:
        interval = max(2, int(getattr(self.fm.settings, "federation_join_poll_interval_s", 10)))
        try:
            while peer.join_state == "pending" and peer.peer_id in {
                p.peer_id for p in self.fm.get_all_peers()
            }:
                status = await self.check_outgoing(peer)
                if status in ("accepted", "denied", "expired"):
                    return
                await asyncio.sleep(interval)
        finally:
            self._pollers.pop(peer.peer_id, None)

    async def _finalize_outgoing(self, peer: PeerState, token: str, hub: dict[str, Any]) -> None:
        """Accepted: create the service identity for the hub and start dialing."""
        _, their_level = _access_levels_for(peer.mode)
        user_id, local_key = await create_service_identity(self.fm.session_factory, peer.name)
        await self.fm.update_peer(
            peer.peer_id,
            api_key=token,
            local_api_key=local_key,
            local_user_id=user_id,
            local_access_level=their_level,
            remote_node_id=hub.get("node_id") or peer.remote_node_id,
            join_poll_secret=None,
            join_state=None,  # last: this is what starts the dialer
        )
        logger.info("federation_join_accepted", hub=peer.url, peer=peer.name)

    async def _bootstrap_join(self) -> None:
        """Unattended node (cloud pod): join the configured hub, retrying."""
        s = self.fm.settings
        url = str(s.federation_join_url).rstrip("/")
        delay = 5.0
        while True:
            if any(
                p.connection_mode == "tunnel_out" and p.url.rstrip("/") == url
                for p in self.fm.get_all_peers()
            ):
                return
            try:
                await self.join(url, invite_code=str(s.federation_join_invite))
                return
            except JoinError as exc:
                if exc.status in (400, 401, 403, 404, 409):
                    logger.error("federation_bootstrap_join_failed", hub=url, error=str(exc))
                    return
                logger.warning("federation_bootstrap_join_retry", hub=url, error=str(exc))
            await asyncio.sleep(delay)
            delay = min(delay * 2, 120.0)
