"""Per-model federation strategy replaces the per-rung execution_policy

Revision ID: 0046
Revises: 0045
Create Date: 2026-11-17

Where a model runs is decided by the model (``server_config`` key
``federation_model_policies``, core/federation_policy.py), not by each rung of
its context ladder. Models whose rungs were all «Solo local» become
``local_only``; ``remote`` («prefer a peer that has it warm») had no users.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

revision: str = "0046"
down_revision: str | None = "0045"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KEY = "federation_model_policies"


def upgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            """
            SELECT p.base_model_id, array_agg(DISTINCT r.execution_policy)
            FROM profile_runtime_presets r
            JOIN model_profiles p ON p.profile_id = r.profile_id
            WHERE p.base_model_id IS NOT NULL
            GROUP BY p.base_model_id
            """
        )
    ).fetchall()
    policies = {
        model_id: {"strategy": "local_only", "nodes": [], "fallback": True}
        for model_id, values in rows
        if "local" in (values or [])
    }
    if policies:
        existing = conn.execute(
            sa.text("SELECT value FROM server_config WHERE key = :k"), {"k": _KEY}
        ).scalar_one_or_none()
        merged = {**policies, **(json.loads(existing) if existing else {})}
        conn.execute(
            sa.text(
                """
                INSERT INTO server_config (key, value, updated_at) VALUES (:k, :v, :t)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                """
            ),
            {"k": _KEY, "v": json.dumps(merged), "t": datetime.now(UTC)},
        )
    op.drop_column("profile_runtime_presets", "execution_policy")


def downgrade() -> None:
    op.add_column(
        "profile_runtime_presets",
        sa.Column("execution_policy", sa.String(length=16), nullable=False, server_default="auto"),
    )
    conn = op.get_bind()
    existing = conn.execute(
        sa.text("SELECT value FROM server_config WHERE key = :k"), {"k": _KEY}
    ).scalar_one_or_none()
    local = [
        m
        for m, p in (json.loads(existing) if existing else {}).items()
        if p.get("strategy") == "local_only"
    ]
    if local:
        conn.execute(
            sa.text(
                """
                UPDATE profile_runtime_presets r SET execution_policy = 'local'
                FROM model_profiles p
                WHERE p.profile_id = r.profile_id AND p.base_model_id = ANY(:ids)
                """
            ),
            {"ids": local},
        )
