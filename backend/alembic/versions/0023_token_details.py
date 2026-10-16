"""request_stats: cached_tokens + reasoning_tokens

Revision ID: 0023
Revises: 0022
Create Date: 2026-10-16

Two nullable Integer columns on ``request_stats`` to persist the OpenAI
usage detail payload that until now was reforwarded to the client but
silently discarded in our analytics:

- ``cached_tokens``: subset of ``input_tokens`` that hit the backend's
  prompt cache (``usage.prompt_tokens_details.cached_tokens``).
- ``reasoning_tokens``: subset of ``output_tokens`` the backend spent on
  hidden reasoning (``usage.completion_tokens_details.reasoning_tokens``).
  When the backend doesn't emit the detail we fall back to counting words
  of ``choices[].message.reasoning_content``.

Both nullable with no backfill: old rows keep NULL and the aggregators
treat NULL as "unreported" (distinct from 0, which means "reported, zero").
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "request_stats",
        sa.Column("cached_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "request_stats",
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("request_stats", "reasoning_tokens")
    op.drop_column("request_stats", "cached_tokens")
