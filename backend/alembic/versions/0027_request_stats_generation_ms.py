"""add request_stats.generation_ms (pure engine generation time)

Revision ID: 0027
Revises: 0026
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0027"
down_revision: str | None = "0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_stats",
        sa.Column("generation_ms", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("request_stats", "generation_ms")
