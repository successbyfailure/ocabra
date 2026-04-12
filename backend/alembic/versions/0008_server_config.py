"""add server_config table for persisted settings overrides

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-04 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # server_config was originally created in 0001 with (key VARCHAR PK, value JSONB).
    # This migration reshapes it to (key VARCHAR(128) PK, value TEXT, updated_at TIMESTAMPTZ).
    op.drop_table("server_config")
    op.create_table(
        "server_config",
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    # Restore original schema from 0001
    op.drop_table("server_config")
    op.create_table(
        "server_config",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
