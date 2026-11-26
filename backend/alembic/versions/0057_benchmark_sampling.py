"""benchmarks con muestreo del fabricante

Revision ID: 0057
Revises: 0056
Create Date: 2026-11-26
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0057"
down_revision: str | None = "0056"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # null = el muestreo que mande cada benchmark (lo de siempre).
    op.add_column("benchmark_runs", sa.Column("sampling", sa.Text(), nullable=True))
    op.add_column("benchmark_schedules", sa.Column("sampling", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("benchmark_schedules", "sampling")
    op.drop_column("benchmark_runs", "sampling")
