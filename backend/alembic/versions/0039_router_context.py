"""Let each router choose which context it advertises.

Revision ID: 0039
Revises: 0038
Create Date: 2026-11-14

Adds ``model_profiles.router_context`` (JSONB, nullable), only meaningful on
router profiles: ``{"mode": "max" | "common" | "fixed", "tokens": int}``.
NULL keeps today's behaviour (``max``: the largest context any target can
serve).
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0039"
down_revision = "0038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_profiles", sa.Column("router_context", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("model_profiles", "router_context")
