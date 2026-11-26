"""línea de tiempo de los pods y baterías de benchmarks

Revision ID: 0060
Revises: 0059
Create Date: 2026-11-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0060"
down_revision: str | None = "0059"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("cloud_nodes", sa.Column("timeline", postgresql.JSONB(), nullable=True))
    # Ejecuciones pedidas juntas (una batería): con pod propio comparten máquina.
    op.add_column(
        "benchmark_runs", sa.Column("batch_id", postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_index("ix_benchmark_runs_batch_id", "benchmark_runs", ["batch_id"])
    # Plan con varias pruebas: [{app, suite, suite_args}] además de la principal.
    op.add_column("benchmark_schedules", sa.Column("battery", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("benchmark_schedules", "battery")
    op.drop_index("ix_benchmark_runs_batch_id", table_name="benchmark_runs")
    op.drop_column("benchmark_runs", "batch_id")
    op.drop_column("cloud_nodes", "timeline")
