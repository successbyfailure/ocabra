"""runtime attribution and hidden profile aliases

Revision ID: 0025
Revises: 0024
Create Date: 2026-10-19
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_profile_aliases",
        sa.Column("alias_id", sa.String(length=512), primary_key=True),
        sa.Column(
            "profile_id",
            sa.String(length=512),
            sa.ForeignKey(
                "model_profiles.profile_id",
                ondelete="CASCADE",
                onupdate="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("preset_key", sa.String(length=64), nullable=True),
        sa.Column("hidden", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_model_profile_aliases_profile_id",
        "model_profile_aliases",
        ["profile_id"],
    )
    op.add_column(
        "request_stats", sa.Column("runtime_preset", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "request_stats", sa.Column("runtime_key", sa.String(length=128), nullable=True)
    )
    op.add_column(
        "request_stats", sa.Column("effective_context", sa.Integer(), nullable=True)
    )
    op.create_index(
        "ix_request_stats_runtime_preset",
        "request_stats",
        ["runtime_preset"],
    )


def downgrade() -> None:
    op.drop_index("ix_request_stats_runtime_preset", table_name="request_stats")
    op.drop_column("request_stats", "effective_context")
    op.drop_column("request_stats", "runtime_key")
    op.drop_column("request_stats", "runtime_preset")
    op.drop_index(
        "ix_model_profile_aliases_profile_id", table_name="model_profile_aliases"
    )
    op.drop_table("model_profile_aliases")
