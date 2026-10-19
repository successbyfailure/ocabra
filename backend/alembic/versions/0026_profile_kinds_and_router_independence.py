"""separate execution profiles from model-independent routers

Revision ID: 0026
Revises: 0025
Create Date: 2026-10-19
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_profiles",
        sa.Column(
            "profile_kind",
            sa.String(length=16),
            nullable=False,
            server_default="execution",
        ),
    )
    op.alter_column("model_profiles", "base_model_id", nullable=True)
    # Normalise ``jsonb null`` values first — Postgres treats ``'null'::jsonb``
    # as a distinct value from SQL NULL, and jsonb_array_length() cannot
    # short-circuit against it inside a composite WHERE because the planner
    # may evaluate operands in either order (observed 2026-10-19: migration
    # crashed with "cannot get array length of a scalar" on a lone row with
    # routing_targets = 'null'::jsonb left behind by an older write path).
    op.execute(
        "UPDATE model_profiles SET routing_targets = NULL "
        "WHERE jsonb_typeof(routing_targets) = 'null'"
    )
    op.execute(
        """
        UPDATE model_profiles
        SET profile_kind = 'router', base_model_id = NULL,
            is_default = FALSE, load_overrides = NULL,
            runtime_policy = NULL
        WHERE jsonb_typeof(routing_targets) = 'array'
          AND routing_targets != '[]'::jsonb
        """
    )
    op.create_check_constraint(
        "ck_model_profile_kind_shape",
        "model_profiles",
        "(profile_kind = 'execution' AND base_model_id IS NOT NULL "
        "AND (routing_targets IS NULL OR routing_targets = '[]'::jsonb)) OR "
        "(profile_kind = 'router' AND base_model_id IS NULL "
        "AND jsonb_typeof(routing_targets) = 'array' "
        "AND jsonb_array_length(routing_targets) > 0)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_model_profile_kind_shape",
        "model_profiles",
        type_="check",
    )
    op.execute(
        """
        UPDATE model_profiles AS router
        SET base_model_id = target.base_model_id
        FROM model_profiles AS target
        WHERE router.profile_kind = 'router'
          AND target.profile_id = router.routing_targets ->> 0
        """
    )
    op.alter_column("model_profiles", "base_model_id", nullable=False)
    op.drop_column("model_profiles", "profile_kind")
