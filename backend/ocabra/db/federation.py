"""SQLAlchemy models for federation: peers, join requests, invites."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ocabra.database import Base


class FederationPeer(Base):
    """A node we federate with. See migration 0041 for the connection modes."""

    __tablename__ = "federation_peers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    # http: API key for the peer. tunnel_out: the tunnel token we present.
    api_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    # What *we* may do on the peer.
    access_level: Mapped[str] = mapped_column(String(50), nullable=False, default="inference")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # "http" | "tunnel_in" (peer dials us) | "tunnel_out" (we dial the peer)
    connection_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="http")
    tunnel_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    # Service key we use to run the peer's requests on our API (tunnel modes).
    local_api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    local_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # What the *peer* may do on us (None = we don't serve it).
    local_access_level: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # "manual" | "swarm" (mutual, full control) | "rules" («Mi servidor, mis reglas»)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="manual")
    # «rules» mode: our adjustments for this peer over the node defaults.
    rules: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    remote_node_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Outgoing join request still being decided by the other side.
    join_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    join_request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    join_poll_secret_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class FederationInvite(Base):
    """Pre-approved join code: joining with it skips manual acceptance."""

    __tablename__ = "federation_invites"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="swarm")
    max_uses: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    uses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FederationJoinRequest(Base):
    """A node asking to join us. Accepting creates a ``tunnel_in`` peer."""

    __tablename__ = "federation_join_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # pending | accepted | denied | expired
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="swarm")
    remote_node_id: Mapped[str] = mapped_column(String(255), nullable=False)
    remote_node_name: Mapped[str] = mapped_column(String(255), nullable=False)
    remote_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gpus: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    verification_code: Mapped[str] = mapped_column(String(16), nullable=False)
    poll_secret_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    invite_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("federation_invites.id", ondelete="SET NULL"), nullable=True
    )
    peer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("federation_peers.id", ondelete="SET NULL"), nullable=True
    )
    # Tunnel token for the requester, held until it first connects.
    token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
