"""add openai_files and openai_batches tables

Revision ID: 0016
Revises: 0015
Create Date: 2026-04-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "openai_files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="uploaded"),
        sa.Column("status_details", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_openai_files_user_id", "openai_files", ["user_id"])
    op.create_index("ix_openai_files_purpose", "openai_files", ["purpose"])

    op.create_table(
        "openai_batches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column(
            "input_file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("openai_files.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("completion_window", sa.Text(), nullable=False, server_default="24h"),
        sa.Column("status", sa.Text(), nullable=False, server_default="validating"),
        sa.Column(
            "output_file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("openai_files.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "error_file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("openai_files.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("errors", postgresql.JSONB(), nullable=True),
        sa.Column("request_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("request_completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("request_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("batch_metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("in_progress_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finalizing_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelling_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expired_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_openai_batches_user_id", "openai_batches", ["user_id"])
    op.create_index("ix_openai_batches_status", "openai_batches", ["status"])


def downgrade() -> None:
    op.drop_index("ix_openai_batches_status", table_name="openai_batches")
    op.drop_index("ix_openai_batches_user_id", table_name="openai_batches")
    op.drop_table("openai_batches")
    op.drop_index("ix_openai_files_purpose", table_name="openai_files")
    op.drop_index("ix_openai_files_user_id", table_name="openai_files")
    op.drop_table("openai_files")
