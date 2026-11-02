"""add request_stats runtime telemetry

Revision ID: 0032
Revises: 0031
Create Date: 2026-11-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0032"
down_revision: str | None = "0031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_stats",
        sa.Column("runtime_telemetry", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("request_stats", "runtime_telemetry")
