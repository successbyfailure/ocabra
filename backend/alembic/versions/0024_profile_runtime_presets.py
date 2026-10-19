"""profile runtime presets and selection policy

Revision ID: 0024
Revises: 0023
Create Date: 2026-10-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_profiles",
        sa.Column("runtime_policy", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_table(
        "profile_runtime_presets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("profile_id", sa.String(length=512), nullable=False),
        sa.Column("preset_key", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("context_mode", sa.String(length=16), nullable=False, server_default="fixed"),
        sa.Column("context_tokens", sa.Integer(), nullable=True),
        sa.Column("parallelism", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("kv_cache_dtype", sa.String(length=16), nullable=False, server_default="q8"),
        sa.Column("placement_policy", sa.String(length=16), nullable=False, server_default="auto"),
        sa.Column("preferred_gpus", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("execution_policy", sa.String(length=16), nullable=False, server_default="auto"),
        sa.Column("load_policy", sa.String(length=16), nullable=False, server_default="on_demand"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "(context_mode = 'fixed' AND context_tokens > 0) OR "
            "(context_mode = 'max' AND context_tokens IS NULL)",
            name="ck_profile_runtime_preset_context",
        ),
        sa.CheckConstraint("parallelism > 0", name="ck_profile_runtime_preset_parallelism"),
        sa.CheckConstraint("priority >= 0", name="ck_profile_runtime_preset_priority"),
        sa.CheckConstraint(
            "NOT is_default OR enabled",
            name="ck_profile_runtime_preset_default_enabled",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["model_profiles.profile_id"],
            ondelete="CASCADE",
            onupdate="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("profile_id", "preset_key", name="uq_profile_runtime_preset_key"),
    )
    op.create_index(
        "ix_profile_runtime_presets_profile_id",
        "profile_runtime_presets",
        ["profile_id"],
    )
    op.create_index(
        "uq_profile_runtime_preset_default",
        "profile_runtime_presets",
        ["profile_id"],
        unique=True,
        postgresql_where=sa.text("is_default"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_profile_runtime_preset_default",
        table_name="profile_runtime_presets",
    )
    op.drop_index("ix_profile_runtime_presets_profile_id", table_name="profile_runtime_presets")
    op.drop_table("profile_runtime_presets")
    op.drop_column("model_profiles", "runtime_policy")
