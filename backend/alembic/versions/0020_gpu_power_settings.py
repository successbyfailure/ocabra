"""persist GPU power caps + persistence mode

Revision ID: 0020
Revises: 0019
Create Date: 2026-06-05

Adds ``gpu_power_settings``: one row per physical GPU (keyed by NVML UUID,
which is stable across reboots / driver reloads / index reordering) that
records the configured power limit (W) and persistence mode. On API startup
these rows are re-applied via hw-monitor's Redis channel so caps survive
container restarts and host reboots.

The UUID is the canonical identifier; ``last_known_index`` / ``last_known_name``
are stored for human-readable display and debug logs but should never be used
as join keys.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "gpu_power_settings",
        sa.Column("gpu_uuid", sa.String(length=80), nullable=False),
        sa.Column("power_limit_w", sa.Integer(), nullable=True),
        sa.Column("persistence_mode", sa.Boolean(), nullable=True),
        sa.Column("last_known_index", sa.Integer(), nullable=True),
        sa.Column("last_known_name", sa.String(length=128), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("gpu_uuid"),
    )


def downgrade() -> None:
    op.drop_table("gpu_power_settings")
