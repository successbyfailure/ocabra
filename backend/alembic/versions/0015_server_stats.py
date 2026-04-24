"""add server_stats table for CPU and total power tracking

Revision ID: 0015
Revises: 0014
Create Date: 2026-04-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "server_stats",
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cpu_power_w", sa.Float(), nullable=True),
        sa.Column("cpu_temp_c", sa.Float(), nullable=True),
        sa.Column("total_gpu_power_w", sa.Float(), nullable=True),
        sa.Column("total_power_w", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("recorded_at"),
    )
    op.create_index("ix_server_stats_recorded_at", "server_stats", ["recorded_at"])


def downgrade() -> None:
    op.drop_index("ix_server_stats_recorded_at", table_name="server_stats")
    op.drop_table("server_stats")
