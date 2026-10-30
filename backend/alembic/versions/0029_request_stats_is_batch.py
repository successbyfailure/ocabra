"""add request_stats.is_batch (tags requests dispatched by the Batches API)

Revision ID: 0029
Revises: 0028
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0029"
down_revision: str | None = "0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_stats",
        sa.Column("is_batch", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_request_stats_is_batch", "request_stats", ["is_batch"])


def downgrade() -> None:
    op.drop_index("ix_request_stats_is_batch", table_name="request_stats")
    op.drop_column("request_stats", "is_batch")
