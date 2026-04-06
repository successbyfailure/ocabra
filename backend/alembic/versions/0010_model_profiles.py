"""add model_profiles table with default profiles for existing models

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-06
"""

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Backend type → profile category mapping
_BACKEND_CATEGORY: dict[str, str] = {
    "vllm": "llm",
    "llama_cpp": "llm",
    "sglang": "llm",
    "tensorrt_llm": "llm",
    "bitnet": "llm",
    "ollama": "llm",
    "whisper": "stt",
    "tts": "tts",
    "voxtral": "tts",
    "diffusers": "image",
    "acestep": "music",
}


def _slugify(value: str) -> str:
    """Convert a string to a clean slug: lowercase, no slashes, hyphens instead of spaces."""
    slug = value.lower().strip()
    slug = slug.replace("/", "-").replace("_", "-").replace(" ", "-")
    slug = re.sub(r"[^a-z0-9\-.]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    return slug


def upgrade() -> None:
    op.create_table(
        "model_profiles",
        sa.Column("profile_id", sa.String(512), primary_key=True),
        sa.Column(
            "base_model_id",
            sa.String(512),
            sa.ForeignKey("model_configs.model_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("display_name", sa.String(512), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(32), nullable=False, server_default="llm"),
        sa.Column("load_overrides", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("request_defaults", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("assets", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Seed default profiles from existing model_configs
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT model_id, display_name, backend_type FROM model_configs")
    ).fetchall()

    for model_id, display_name, backend_type in rows:
        category = _BACKEND_CATEGORY.get(backend_type, "llm")

        # Derive slug from display_name or the part after backend prefix
        raw_name = display_name or model_id
        # Strip backend prefix if present (e.g. "vllm/org/model" -> "org/model")
        if "/" in model_id:
            parts = model_id.split("/", 1)
            if parts[0] in _BACKEND_CATEGORY:
                raw_name = display_name or parts[1]
        profile_id = _slugify(raw_name)
        if not profile_id:
            profile_id = _slugify(model_id)

        conn.execute(
            sa.text(
                "INSERT INTO model_profiles "
                "(profile_id, base_model_id, display_name, category, enabled, is_default) "
                "VALUES (:pid, :mid, :dn, :cat, true, true) "
                "ON CONFLICT (profile_id) DO NOTHING"
            ),
            {
                "pid": profile_id,
                "mid": model_id,
                "dn": display_name,
                "cat": category,
            },
        )

        # For whisper models with ::diarize variant, create diarized profile
        if backend_type == "whisper" and "::diarize" in model_id:
            diarize_pid = f"{profile_id}-diarized"
            diarize_dn = f"{display_name or raw_name} (Diarized)"
            conn.execute(
                sa.text(
                    "INSERT INTO model_profiles "
                    "(profile_id, base_model_id, display_name, category, "
                    "load_overrides, enabled, is_default) "
                    "VALUES (:pid, :mid, :dn, :cat, :lo, true, false) "
                    "ON CONFLICT (profile_id) DO NOTHING"
                ),
                {
                    "pid": diarize_pid,
                    "mid": model_id,
                    "dn": diarize_dn,
                    "cat": "stt",
                    "lo": '{"diarization_enabled": true}',
                },
            )


def downgrade() -> None:
    op.drop_table("model_profiles")
