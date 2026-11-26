"""peticiones de máquina en la nube que se reintentan hasta que hay capacidad

Revision ID: 0058
Revises: 0057
Create Date: 2026-11-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0058"
down_revision: str | None = "0057"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cloud_launch_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="waiting"),
        sa.Column("gpu_type_ids", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("launch", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("retry_interval_s", sa.Integer(), nullable=False, server_default="120"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "cloud_node_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cloud_nodes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("launched_gpu_type_id", sa.String(128), nullable=True),
        sa.Column("purpose", sa.String(16), nullable=False, server_default="manual"),
        sa.Column("benchmark_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_cloud_launch_requests_status", "cloud_launch_requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_cloud_launch_requests_status", table_name="cloud_launch_requests")
    op.drop_table("cloud_launch_requests")
