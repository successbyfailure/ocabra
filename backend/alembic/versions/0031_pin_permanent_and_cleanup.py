"""warm/pin rework: add model_configs.pin_permanent, drop auto_reload and
profile_runtime_presets.load_policy (2026-10-30 design review)

Revision ID: 0031
Revises: 0030
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0031"
down_revision: str | None = "0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_configs",
        sa.Column("pin_permanent", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.drop_column("model_configs", "auto_reload")
    op.drop_column("profile_runtime_presets", "load_policy")


def downgrade() -> None:
    op.add_column(
        "profile_runtime_presets",
        sa.Column("load_policy", sa.String(16), nullable=False, server_default="on_demand"),
    )
    op.add_column(
        "model_configs",
        sa.Column("auto_reload", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.drop_column("model_configs", "pin_permanent")
