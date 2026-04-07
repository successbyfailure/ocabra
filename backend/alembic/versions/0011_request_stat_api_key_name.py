"""add api_key_name to request_stats

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_stats",
        sa.Column("api_key_name", sa.String(256), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("request_stats", "api_key_name")
