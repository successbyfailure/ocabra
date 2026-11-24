"""benchmarks en nodos de la federación

Revision ID: 0056
Revises: 0055
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0056"
down_revision: str | None = "0055"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("benchmark_runs", sa.Column("node", sa.Text(), nullable=True))
    op.add_column("benchmark_schedules", sa.Column("node", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("benchmark_schedules", "node")
    op.drop_column("benchmark_runs", "node")
