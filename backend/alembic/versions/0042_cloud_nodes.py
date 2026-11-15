"""cloud_providers + cloud_nodes: temporary federation nodes on rented GPUs

Revision ID: 0042
Revises: 0041
Create Date: 2026-11-15

``cloud_providers`` holds the encrypted API key and the defaults for each
provider (only ``runpod`` for now). ``cloud_nodes`` is one row per pod we
launched: what it costs, when it must die (TTL / idle), and which federation
peer it became once it joined us with its single-use invite.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0042"
down_revision: str | None = "0041"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cloud_providers",
        sa.Column("provider", sa.String(32), primary_key=True),
        sa.Column("api_key_encrypted", sa.Text, nullable=True),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "cloud_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("provider_pod_id", sa.String(128), nullable=True, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        # creating | starting | running | stopping | terminated | error
        sa.Column("status", sa.String(16), nullable=False, server_default="creating"),
        sa.Column("gpu_type_id", sa.String(128), nullable=False),
        sa.Column("gpu_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("cloud_type", sa.String(16), nullable=False, server_default="COMMUNITY"),
        sa.Column("image", sa.String(512), nullable=False),
        sa.Column("cost_per_hr", sa.Float, nullable=True),
        sa.Column("max_runtime_minutes", sa.Integer, nullable=True),
        sa.Column("idle_shutdown_minutes", sa.Integer, nullable=True),
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
        sa.Column("preload_models", postgresql.JSONB, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminate_reason", sa.String(64), nullable=True),
    )
    op.create_index("ix_cloud_nodes_status", "cloud_nodes", ["status"])


def downgrade() -> None:
    op.drop_index("ix_cloud_nodes_status", table_name="cloud_nodes")
    op.drop_table("cloud_nodes")
    op.drop_table("cloud_providers")
