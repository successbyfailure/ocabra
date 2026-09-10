"""routing profiles + estimator work size metadata

Revision ID: 0022
Revises: 0021
Create Date: 2026-09-10

Bloque 20 — Router profiles, duration estimator and session registry.

Adds three columns that together support the routing/estimation system:

- ``model_profiles.routing_targets`` (jsonb): ordered list of profile_ids to
  delegate to when this profile is a router. NULL for plain profiles.
- ``request_stats.work_size_meta`` (jsonb): per-family work descriptor
  (``audio_seconds`` for whisper, ``frames`` for flashvsr, ``steps`` for
  diffusers, ...) so the estimator can fit per-model regressions instead of
  falling back to coarse duration percentiles for non-LLM families.
- ``request_stats.via_router_profile_id`` (varchar 512): when a router
  redirected the request to a different target, this records the router's
  profile_id. NULL for direct calls. Indexed so the routing analytics view
  can aggregate cheaply.

All three are nullable with no backfill: old rows keep NULL and the
estimator/router gracefully ignore them.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_profiles",
        sa.Column("routing_targets", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "request_stats",
        sa.Column("work_size_meta", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "request_stats",
        sa.Column("via_router_profile_id", sa.String(length=512), nullable=True),
    )
    op.create_index(
        "ix_request_stats_via_router_profile_id",
        "request_stats",
        ["via_router_profile_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_request_stats_via_router_profile_id",
        table_name="request_stats",
    )
    op.drop_column("request_stats", "via_router_profile_id")
    op.drop_column("request_stats", "work_size_meta")
    op.drop_column("model_profiles", "routing_targets")
