"""add group_id to api_keys and request_stats

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("group_id", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_api_keys_group_id_groups",
        "api_keys",
        "groups",
        ["group_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_api_keys_group_id", "api_keys", ["group_id"])

    op.add_column(
        "request_stats",
        sa.Column("group_id", sa.UUID(), nullable=True),
    )
    op.create_index("ix_request_stats_group_id", "request_stats", ["group_id"])


def downgrade() -> None:
    op.drop_index("ix_request_stats_group_id", table_name="request_stats")
    op.drop_column("request_stats", "group_id")

    op.drop_index("ix_api_keys_group_id", table_name="api_keys")
    op.drop_constraint("fk_api_keys_group_id_groups", "api_keys", type_="foreignkey")
    op.drop_column("api_keys", "group_id")
