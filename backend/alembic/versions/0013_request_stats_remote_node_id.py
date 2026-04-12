"""add remote_node_id to request_stats for federation proxy tracking

Revision ID: 0013
Revises: 0012
Create Date: 2026-04-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_stats",
        sa.Column("remote_node_id", sa.String(256), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("request_stats", "remote_node_id")
