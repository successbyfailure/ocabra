"""add request_stats speculative-decoding fields (draft_n, draft_n_accepted, predicted/prompt tokens per second)

Revision ID: 0028
Revises: 0027
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0028"
down_revision: str | None = "0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_stats",
        sa.Column("draft_n", sa.Integer(), nullable=True),
    )
    op.add_column(
        "request_stats",
        sa.Column("draft_n_accepted", sa.Integer(), nullable=True),
    )
    op.add_column(
        "request_stats",
        sa.Column("predicted_tokens_per_second", sa.Float(), nullable=True),
    )
    op.add_column(
        "request_stats",
        sa.Column("prompt_tokens_per_second", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("request_stats", "prompt_tokens_per_second")
    op.drop_column("request_stats", "predicted_tokens_per_second")
    op.drop_column("request_stats", "draft_n_accepted")
    op.drop_column("request_stats", "draft_n")
