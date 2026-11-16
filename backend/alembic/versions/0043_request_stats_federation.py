"""request_stats: name of the serving node, who sent it to us, and why

Revision ID: 0043
Revises: 0042
Create Date: 2026-11-16

``remote_node_id`` alone left history anonymous once a peer was deleted (its
UUID was all that remained). ``remote_node_name`` keeps the name.
``federation_inbound_from`` marks requests a peer sent *us* (served here for
another node), so traffic can be read both ways. ``federation_route`` is why
the request went where it did (``local_cold_peer_warm``, ``forced``…, or
``fallback_local`` when the chosen peer failed and it was served here).

Backfills the name for rows whose peer still exists.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0043"
down_revision: str | None = "0042"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("request_stats", sa.Column("remote_node_name", sa.String(256), nullable=True))
    op.add_column(
        "request_stats", sa.Column("federation_inbound_from", sa.String(256), nullable=True)
    )
    op.add_column("request_stats", sa.Column("federation_route", sa.String(64), nullable=True))
    op.execute(
        """
        UPDATE request_stats r SET remote_node_name = p.name
        FROM federation_peers p
        WHERE r.remote_node_id IS NOT NULL AND r.remote_node_id = p.id::text
        """
    )
    op.create_index(
        "ix_request_stats_federation",
        "request_stats",
        ["started_at"],
        postgresql_where=sa.text(
            "remote_node_id IS NOT NULL OR federation_inbound_from IS NOT NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index("ix_request_stats_federation", table_name="request_stats")
    op.drop_column("request_stats", "federation_route")
    op.drop_column("request_stats", "federation_inbound_from")
    op.drop_column("request_stats", "remote_node_name")
