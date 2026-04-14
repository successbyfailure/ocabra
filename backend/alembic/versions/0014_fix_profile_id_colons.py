"""fix profile_id slugs to preserve Ollama ':' tag separator

Profile IDs for Ollama models were generated stripping the ':' character
(e.g. 'qwen3-embedding8b' instead of 'qwen3-embedding:8b'), breaking
clients that use the standard Ollama name format.  This migration
renames affected profile_ids to include the ':'.

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-13
"""

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _derive_slug(raw_name: str) -> str:
    """Regenerate profile slug preserving ':'."""
    slug = re.sub(
        r"[^a-z0-9\-.:]",
        "",
        raw_name.lower().replace("/", "-").replace("_", "-").replace(" ", "-"),
    )
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def upgrade() -> None:
    conn = op.get_bind()

    rows = conn.execute(
        sa.text(
            "SELECT profile_id, base_model_id FROM model_profiles"
        )
    ).fetchall()

    for profile_id, base_model_id in rows:
        if not base_model_id or ":" not in base_model_id:
            continue

        # Extract the backend_model_id part (after the first '/')
        parts = base_model_id.split("/", 1)
        if len(parts) != 2:
            continue
        backend_model_id = parts[1]

        # Only fix if the backend_model_id contains ':' but the profile_id doesn't
        if ":" not in backend_model_id or ":" in profile_id:
            continue

        new_slug = _derive_slug(backend_model_id)
        if not new_slug or new_slug == profile_id:
            continue

        # Check the new slug doesn't already exist
        existing = conn.execute(
            sa.text("SELECT 1 FROM model_profiles WHERE profile_id = :pid"),
            {"pid": new_slug},
        ).fetchone()
        if existing:
            continue

        conn.execute(
            sa.text(
                "UPDATE model_profiles SET profile_id = :new_id WHERE profile_id = :old_id"
            ),
            {"new_id": new_slug, "old_id": profile_id},
        )


def downgrade() -> None:
    # Downgrade strips colons from profile_ids (reverting to old behaviour)
    conn = op.get_bind()

    rows = conn.execute(
        sa.text("SELECT profile_id FROM model_profiles WHERE profile_id LIKE '%:%'")
    ).fetchall()

    for (profile_id,) in rows:
        old_slug = profile_id.replace(":", "")
        if not old_slug or old_slug == profile_id:
            continue
        existing = conn.execute(
            sa.text("SELECT 1 FROM model_profiles WHERE profile_id = :pid"),
            {"pid": old_slug},
        ).fetchone()
        if existing:
            continue
        conn.execute(
            sa.text(
                "UPDATE model_profiles SET profile_id = :new_id WHERE profile_id = :old_id"
            ),
            {"new_id": old_slug, "old_id": profile_id},
        )
