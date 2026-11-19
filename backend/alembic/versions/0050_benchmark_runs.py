"""ejecuciones y reintentos de benchmarks

Revision ID: 0050
Revises: 0049
Create Date: 2026-11-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0050"
down_revision: str | None = "0049"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("app", sa.Text(), nullable=False),
        sa.Column("suite", sa.Text(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=True),
        sa.Column("models", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("suite_args", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("auto_pause", sa.Boolean(), nullable=False),
        sa.Column("pause_reason", sa.Text(), nullable=True),
        sa.Column("current_model_index", sa.Integer(), nullable=False),
        sa.Column("current_model", sa.Text(), nullable=True),
        sa.Column("created_by", sa.UUID(), nullable=True),
        sa.Column("workspace_path", sa.Text(), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("contaminated_attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("external_activity_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_runs_app", "benchmark_runs", ["app"])
    op.create_index("ix_benchmark_runs_status", "benchmark_runs", ["status"])
    op.create_index("ix_benchmark_runs_created_at", "benchmark_runs", ["created_at"])

    op.create_table(
        "benchmark_attempts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("workspace_path", sa.Text(), nullable=False),
        sa.Column("log_path", sa.Text(), nullable=False),
        sa.Column("return_code", sa.Integer(), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("external_requests", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["benchmark_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_attempts_run_id", "benchmark_attempts", ["run_id"])
    op.create_index("ix_benchmark_attempts_status", "benchmark_attempts", ["status"])
    op.create_index(
        "ix_benchmark_attempts_run_model", "benchmark_attempts", ["run_id", "model"]
    )
    op.create_index(
        "uq_benchmark_attempts_run_number",
        "benchmark_attempts",
        ["run_id", "model", "attempt_number"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("benchmark_attempts")
    op.drop_table("benchmark_runs")
