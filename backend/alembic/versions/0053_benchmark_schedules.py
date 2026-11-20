"""planes programados e históricos de benchmarks

Revision ID: 0053
Revises: 0052
Create Date: 2026-11-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0053"
down_revision: str | None = "0052"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "benchmark_schedules",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("app", sa.Text(), nullable=False),
        sa.Column("suite", sa.Text(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=True),
        sa.Column("models", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("suite_args", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("auto_pause", sa.Boolean(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("weekdays", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("time_of_day", sa.Text(), nullable=False),
        sa.Column("timezone", sa.Text(), nullable=False),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["last_run_id"], ["benchmark_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_schedules_app", "benchmark_schedules", ["app"])
    op.create_index("ix_benchmark_schedules_enabled", "benchmark_schedules", ["enabled"])
    op.create_index("ix_benchmark_schedules_next_run_at", "benchmark_schedules", ["next_run_at"])
    op.create_index("ix_benchmark_schedules_due", "benchmark_schedules", ["enabled", "next_run_at"])
    op.add_column(
        "benchmark_runs",
        sa.Column("schedule_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "benchmark_runs",
        sa.Column("is_baseline", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.create_foreign_key(
        "fk_benchmark_runs_schedule_id",
        "benchmark_runs",
        "benchmark_schedules",
        ["schedule_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_benchmark_runs_schedule_id", "benchmark_runs", ["schedule_id"])
    op.create_index(
        "uq_benchmark_runs_baseline_suite",
        "benchmark_runs",
        ["app", "suite"],
        unique=True,
        postgresql_where=sa.text("is_baseline"),
    )


def downgrade() -> None:
    op.drop_index("uq_benchmark_runs_baseline_suite", table_name="benchmark_runs")
    op.drop_index("ix_benchmark_runs_schedule_id", table_name="benchmark_runs")
    op.drop_constraint("fk_benchmark_runs_schedule_id", "benchmark_runs", type_="foreignkey")
    op.drop_column("benchmark_runs", "is_baseline")
    op.drop_column("benchmark_runs", "schedule_id")
    op.drop_index("ix_benchmark_schedules_due", table_name="benchmark_schedules")
    op.drop_index("ix_benchmark_schedules_next_run_at", table_name="benchmark_schedules")
    op.drop_index("ix_benchmark_schedules_enabled", table_name="benchmark_schedules")
    op.drop_index("ix_benchmark_schedules_app", table_name="benchmark_schedules")
    op.drop_table("benchmark_schedules")
