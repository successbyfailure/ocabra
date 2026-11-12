"""Let each runtime preset own its GPU/CPU placement.

Revision ID: 0037
Revises: 0036
Create Date: 2026-11-12

Adds ``profile_runtime_presets.placement`` (JSONB, nullable): split policy,
split mode, tensor split, GPU layers and MoE expert layers on CPU for that
preset. Existing rows stay NULL, which keeps today's behaviour (the preset
inherits those options from the model's base configuration).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0037"
down_revision = "0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "profile_runtime_presets",
        sa.Column("placement", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("profile_runtime_presets", "placement")
