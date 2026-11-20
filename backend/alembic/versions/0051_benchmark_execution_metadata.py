"""fases y entorno reproducible de benchmarks

Revision ID: 0051
Revises: 0050
Create Date: 2026-11-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0051"
down_revision: str | None = "0050"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("benchmark_runs", sa.Column("phase", sa.Text(), nullable=True))
    op.execute("UPDATE benchmark_runs SET phase = status")
    op.alter_column(
        "benchmark_runs", "phase", nullable=False, server_default=sa.text("'queued'")
    )
    op.add_column(
        "benchmark_attempts",
        sa.Column(
            "environment",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("benchmark_attempts", "environment")
    op.drop_column("benchmark_runs", "phase")
