"""add user_id column to request_stats

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-04 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_stats",
        sa.Column("user_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_request_stats_user_id",
        "request_stats",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_request_stats_user_id", "request_stats", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_request_stats_user_id", table_name="request_stats")
    op.drop_constraint("fk_request_stats_user_id", "request_stats", type_="foreignkey")
    op.drop_column("request_stats", "user_id")
