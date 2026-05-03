"""add subagent links to agents

Revision ID: 0018
Revises: 0017
Create Date: 2026-05-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("agents", sa.Column("subagent_slugs", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("agents", "subagent_slugs")
