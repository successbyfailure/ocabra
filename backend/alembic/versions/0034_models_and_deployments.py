"""logical models + deployments (model/backends unification, Fase 1)

Revision ID: 0034
Revises: 0033
Create Date: 2026-11-07

Schema:
- ``models``: one row per logical model (``model_key`` = public name), with its
  default backend and upstream registry metadata.
- ``model_configs`` becomes the deployment table: ``model_key`` (FK, NOT NULL
  after the data step) + ``artifact`` (what file it serves); unique
  ``(model_key, backend_type)``; ``model_id`` is always ``<backend>/<model_key>``.
- ``model_profiles.backend`` / ``model_profile_aliases.backend``: optional pin
  of a variant / alias to one backend.
- ``request_stats.requested_model``: the name the client sent (``model_id``
  becomes the model key).
- The FKs from ``model_profiles.base_model_id`` and ``agents.base_model_id`` to
  ``model_configs.model_id`` gain ON UPDATE CASCADE so deployments can be renamed.

Data: :func:`ocabra.core.model_unification.apply_plan` on the plan computed from
this very database (preview it with ``backend/scripts/model_unification_plan.py``).
It refuses to run on blocking anomalies (name collisions...).

downgrade() reverts the **schema** only. The data step renames ids, converts
profiles into aliases and rewrites ACL/stats — it is not reversible in place;
restore the pg_dump taken before the upgrade to roll the data back.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0034"
down_revision: str | None = "0033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _recreate_fk(table: str, name: str, column: str, *, on_update: str | None) -> None:
    op.drop_constraint(name, table, type_="foreignkey")
    op.create_foreign_key(
        name,
        table,
        "model_configs",
        [column],
        ["model_id"],
        ondelete="CASCADE",
        onupdate=on_update,
    )


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("model_key", sa.String(512), primary_key=True),
        sa.Column("display_name", sa.String(512)),
        sa.Column("default_backend", sa.String(64), nullable=False),
        sa.Column("registry_metadata", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.add_column("model_configs", sa.Column("model_key", sa.String(512), nullable=True))
    op.add_column("model_configs", sa.Column("artifact", JSONB, nullable=True))
    op.add_column("model_profiles", sa.Column("backend", sa.String(64), nullable=True))
    op.add_column("model_profile_aliases", sa.Column("backend", sa.String(64), nullable=True))
    op.add_column("request_stats", sa.Column("requested_model", sa.String(512), nullable=True))

    _recreate_fk("model_profiles", "model_profiles_base_model_id_fkey", "base_model_id", on_update="CASCADE")
    _recreate_fk("agents", "agents_base_model_id_fkey", "base_model_id", on_update="CASCADE")

    from ocabra.core.model_unification import apply_plan, build_plan, snapshot

    bind = op.get_bind()
    apply_plan(bind, build_plan(snapshot(bind)))

    op.alter_column("model_configs", "model_key", nullable=False)
    op.create_foreign_key(
        "model_configs_model_key_fkey",
        "model_configs",
        "models",
        ["model_key"],
        ["model_key"],
        ondelete="CASCADE",
        onupdate="CASCADE",
    )
    op.create_index("ix_model_configs_model_key", "model_configs", ["model_key"])
    op.create_unique_constraint("uq_model_configs_model_key_backend", "model_configs", ["model_key", "backend_type"])


def downgrade() -> None:
    op.drop_constraint("uq_model_configs_model_key_backend", "model_configs", type_="unique")
    op.drop_index("ix_model_configs_model_key", table_name="model_configs")
    op.drop_constraint("model_configs_model_key_fkey", "model_configs", type_="foreignkey")
    _recreate_fk("agents", "agents_base_model_id_fkey", "base_model_id", on_update=None)
    _recreate_fk("model_profiles", "model_profiles_base_model_id_fkey", "base_model_id", on_update=None)
    op.drop_column("request_stats", "requested_model")
    op.drop_column("model_profile_aliases", "backend")
    op.drop_column("model_profiles", "backend")
    op.drop_column("model_configs", "artifact")
    op.drop_column("model_configs", "model_key")
    op.drop_table("models")
