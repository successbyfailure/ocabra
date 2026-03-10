"""initial schema

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", sa.String(512), nullable=False),
        sa.Column("display_name", sa.String(512)),
        sa.Column("backend_type", sa.String(64), nullable=False),
        sa.Column("load_policy", sa.String(32), nullable=False, server_default="on_demand"),
        sa.Column("auto_reload", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("preferred_gpu", sa.Integer),
        sa.Column("extra_config", postgresql.JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id"),
    )
    op.create_index("ix_model_configs_model_id", "model_configs", ["model_id"])

    op.create_table(
        "eviction_schedules",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", sa.String(512)),
        sa.Column("cron_expr", sa.String(128), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("label", sa.Text),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_eviction_schedules_model_id", "eviction_schedules", ["model_id"])

    op.create_table(
        "request_stats",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", sa.String(512), nullable=False),
        sa.Column("gpu_index", sa.Integer),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column("energy_wh", sa.Float),
        sa.Column("error", sa.Text),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_request_stats_model_id", "request_stats", ["model_id"])
    op.create_index("ix_request_stats_started_at", "request_stats", ["started_at"])

    op.create_table(
        "gpu_stats",
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gpu_index", sa.Integer, nullable=False),
        sa.Column("utilization_pct", sa.Float),
        sa.Column("vram_used_mb", sa.Integer),
        sa.Column("power_draw_w", sa.Float),
        sa.Column("temperature_c", sa.Float),
        sa.PrimaryKeyConstraint("recorded_at", "gpu_index"),
    )
    op.create_index(
        "ix_gpu_stats_recorded_at_gpu", "gpu_stats", ["recorded_at", "gpu_index"]
    )

    op.create_table(
        "server_config",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", postgresql.JSONB, nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("server_config")
    op.drop_table("gpu_stats")
    op.drop_table("request_stats")
    op.drop_table("eviction_schedules")
    op.drop_table("model_configs")
