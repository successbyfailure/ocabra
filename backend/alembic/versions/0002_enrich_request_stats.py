"""enrich request stats

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-23 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("request_stats", sa.Column("backend_type", sa.String(length=64), nullable=True))
    op.add_column("request_stats", sa.Column("request_kind", sa.String(length=64), nullable=True))
    op.add_column("request_stats", sa.Column("endpoint_path", sa.String(length=256), nullable=True))
    op.add_column("request_stats", sa.Column("status_code", sa.Integer(), nullable=True))

    op.create_index("ix_request_stats_backend_type", "request_stats", ["backend_type"])
    op.create_index("ix_request_stats_request_kind", "request_stats", ["request_kind"])


def downgrade() -> None:
    op.drop_index("ix_request_stats_request_kind", table_name="request_stats")
    op.drop_index("ix_request_stats_backend_type", table_name="request_stats")

    op.drop_column("request_stats", "status_code")
    op.drop_column("request_stats", "endpoint_path")
    op.drop_column("request_stats", "request_kind")
    op.drop_column("request_stats", "backend_type")
