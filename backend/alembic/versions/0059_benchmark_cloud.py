"""benchmarks que lanzan su propio pod en la nube

Revision ID: 0059
Revises: 0058
Create Date: 2026-11-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0059"
down_revision: str | None = "0058"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("benchmark_runs", sa.Column("cloud", postgresql.JSONB(), nullable=True))
    op.add_column(
        "benchmark_runs",
        sa.Column("cloud_request_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "benchmark_runs", sa.Column("cloud_node_id", postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.add_column("benchmark_schedules", sa.Column("cloud", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("benchmark_schedules", "cloud")
    op.drop_column("benchmark_runs", "cloud_node_id")
    op.drop_column("benchmark_runs", "cloud_request_id")
    op.drop_column("benchmark_runs", "cloud")
