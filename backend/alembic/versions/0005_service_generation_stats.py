"""add service_generation_stats table

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-03 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "service_generation_stats",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("service_id", sa.String(length=64), nullable=False),
        sa.Column("service_type", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("gpu_index", sa.Integer(), nullable=True),
        sa.Column("vram_peak_mb", sa.Integer(), nullable=True),
        sa.Column("evicted", sa.Boolean(), nullable=False, server_default="false"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_svc_gen_stats_service_id", "service_generation_stats", ["service_id"])
    op.create_index("ix_svc_gen_stats_service_type", "service_generation_stats", ["service_type"])
    op.create_index("ix_svc_gen_stats_started_at", "service_generation_stats", ["started_at"])
    op.create_index(
        "ix_svc_gen_stats_started_at_svc",
        "service_generation_stats",
        ["started_at", "service_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_svc_gen_stats_started_at_svc", table_name="service_generation_stats")
    op.drop_index("ix_svc_gen_stats_started_at", table_name="service_generation_stats")
    op.drop_index("ix_svc_gen_stats_service_type", table_name="service_generation_stats")
    op.drop_index("ix_svc_gen_stats_service_id", table_name="service_generation_stats")
    op.drop_table("service_generation_stats")
