"""federation_peers.rules: «Mi servidor, mis reglas» adjustments per peer

Revision ID: 0045
Revises: 0044
Create Date: 2026-11-16

A peer in ``mode='rules'`` is served under the node's default rules (stored in
``server_config`` as ``federation_default_rules``) plus these adjustments.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0045"
down_revision: str | None = "0044"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "federation_peers",
        sa.Column("rules", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("federation_peers", "rules")
