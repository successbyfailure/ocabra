"""add trtllm_compile_jobs table

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-31 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "trtllm_compile_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_model", sa.Text(), nullable=False),
        sa.Column("engine_name", sa.Text(), nullable=False),
        sa.Column("gpu_indices", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("dtype", sa.String(length=32), nullable=False),
        sa.Column("config", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("phase", sa.String(length=32), nullable=True),
        sa.Column("progress_pct", sa.Integer(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("engine_dir", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trtllm_compile_jobs_status", "trtllm_compile_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_trtllm_compile_jobs_status", table_name="trtllm_compile_jobs")
    op.drop_table("trtllm_compile_jobs")
