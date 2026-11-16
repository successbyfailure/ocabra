"""cloud_nodes: network volume and data center a node was launched with

Revision ID: 0044
Revises: 0043
Create Date: 2026-11-16

A node launched with a network volume (model/backend cache) is pinned to the
volume's data center. Recorded so a volume is never mounted on two live pods
at once and can't be deleted while one uses it.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0044"
down_revision: str | None = "0043"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("cloud_nodes", sa.Column("network_volume_id", sa.String(64), nullable=True))
    op.add_column("cloud_nodes", sa.Column("data_center_id", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("cloud_nodes", "data_center_id")
    op.drop_column("cloud_nodes", "network_volume_id")
