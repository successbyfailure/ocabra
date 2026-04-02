"""add model_load_stats table

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-02 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_load_stats",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", sa.String(length=512), nullable=False),
        sa.Column("backend_type", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("gpu_count", sa.Integer(), nullable=True),
        sa.Column("gpu_indices", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_load_stats_model_id", "model_load_stats", ["model_id"])
    op.create_index("ix_model_load_stats_backend_type", "model_load_stats", ["backend_type"])
    op.create_index(
        "ix_model_load_stats_started_at_model",
        "model_load_stats",
        ["started_at", "model_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_load_stats_started_at_model", table_name="model_load_stats")
    op.drop_index("ix_model_load_stats_backend_type", table_name="model_load_stats")
    op.drop_index("ix_model_load_stats_model_id", table_name="model_load_stats")
    op.drop_table("model_load_stats")
