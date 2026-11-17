"""cloud_model_recipes + cloud_nodes.privacy_mode

Revision ID: 0047
Revises: 0046
Create Date: 2026-11-17

Recipes: how to give a cloud node one model (source + tuned config), captured
from the nodes so a model set up by hand in a pod is offered again at the next
launch. privacy_mode: what a node was launched with, for «relaunch».
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0047"
down_revision: str | None = "0046"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("cloud_nodes", sa.Column("privacy_mode", sa.String(length=16), nullable=True))
    op.create_table(
        "cloud_model_recipes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("node_model_id", sa.String(length=512), nullable=False, unique=True),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("backend_type", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=512), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("steps", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("last_node_name", sa.String(length=255), nullable=True),
        sa.Column("last_gpu_type_id", sa.String(length=128), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "captured_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade() -> None:
    op.drop_table("cloud_model_recipes")
    op.drop_column("cloud_nodes", "privacy_mode")
