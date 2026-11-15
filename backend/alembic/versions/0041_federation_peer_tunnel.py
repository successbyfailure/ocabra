"""federation: reverse tunnel, join requests and invites

Revision ID: 0041
Revises: 0040
Create Date: 2026-11-15

A peer can now be reached three ways (``connection_mode``):

* ``http`` (the original): we call ``url`` with ``api_key``.
* ``tunnel_in``: the peer dials *us* over a WebSocket; we keep the SHA-256 of
  the token it presents (``tunnel_token_hash``). It needs no reachable URL.
* ``tunnel_out``: *we* dial the peer at ``url``, presenting the token stored
  (encrypted) in ``api_key_encrypted``.

Over a tunnel the link is symmetric: ``local_api_key_encrypted`` is the key of
the local service user (``local_user_id``, ``federation:<peer>``) we use to run
the peer's requests on our own API, and ``local_access_level`` what we let it
do. ``mode`` labels the agreement (``manual``, ``swarm`` = mutual full control).

``join_*`` track an outgoing join request while the other side decides.

New tables: ``federation_join_requests`` (incoming requests to join us) and
``federation_invites`` (pre-approved join codes, e.g. for cloud nodes).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0041"
down_revision: str | None = "0040"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    cols = [
        sa.Column("connection_mode", sa.String(16), nullable=False, server_default="http"),
        sa.Column("tunnel_token_hash", sa.String(64), nullable=True),
        sa.Column("local_api_key_encrypted", sa.Text, nullable=True),
        sa.Column(
            "local_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("local_access_level", sa.String(50), nullable=True),
        sa.Column("mode", sa.String(16), nullable=False, server_default="manual"),
        sa.Column("remote_node_id", sa.String(255), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("join_state", sa.String(16), nullable=True),
        sa.Column("join_request_id", sa.String(64), nullable=True),
        sa.Column("join_poll_secret_encrypted", sa.Text, nullable=True),
    ]
    for col in cols:
        op.add_column("federation_peers", col)
    op.create_index(
        "ix_federation_peers_tunnel_token_hash",
        "federation_peers",
        ["tunnel_token_hash"],
        unique=True,
    )

    op.create_table(
        "federation_invites",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("code_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("mode", sa.String(16), nullable=False, server_default="swarm"),
        sa.Column("max_uses", sa.Integer, nullable=False, server_default="1"),
        sa.Column("uses", sa.Integer, nullable=False, server_default="0"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "federation_join_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("mode", sa.String(16), nullable=False, server_default="swarm"),
        sa.Column("remote_node_id", sa.String(255), nullable=False),
        sa.Column("remote_node_name", sa.String(255), nullable=False),
        sa.Column("remote_version", sa.String(64), nullable=True),
        sa.Column("message", sa.Text, nullable=True),
        sa.Column("source_ip", sa.String(64), nullable=True),
        sa.Column("gpus", postgresql.JSONB, nullable=True),
        sa.Column("verification_code", sa.String(16), nullable=False),
        sa.Column("poll_secret_hash", sa.String(64), nullable=False),
        sa.Column(
            "invite_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("federation_invites.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "peer_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("federation_peers.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("token_encrypted", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_federation_join_requests_status", "federation_join_requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_federation_join_requests_status", table_name="federation_join_requests")
    op.drop_table("federation_join_requests")
    op.drop_table("federation_invites")
    op.drop_index("ix_federation_peers_tunnel_token_hash", table_name="federation_peers")
    for name in (
        "join_poll_secret_encrypted",
        "join_request_id",
        "join_state",
        "notes",
        "remote_node_id",
        "mode",
        "local_access_level",
        "local_user_id",
        "local_api_key_encrypted",
        "tunnel_token_hash",
        "connection_mode",
    ):
        op.drop_column("federation_peers", name)
