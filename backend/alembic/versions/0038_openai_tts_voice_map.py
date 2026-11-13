"""openai_tts_voice_map table

Revision ID: 0038
Revises: 0037
Create Date: 2026-11-13

Compatibility shim for TTS clients that only know OpenAI's official
model names (``tts-1``, ``tts-1-hd``, ``gpt-4o-mini-tts``) and voice names
(``alloy``, ``echo``...), with no way to send anything oCabra-specific.

Each row maps one ``(openai_model, voice_name)`` pair straight to a
target ``profile_id`` — deliberately independent of routers/
``routing_targets``; the target can be any enabled profile oCabra already
knows about. No rows at all = unchanged behaviour (those model names stay
unrecognised, same 404 as today).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0038"
down_revision: str | None = "0037"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "openai_tts_voice_map",
        sa.Column("openai_model", sa.String(length=128), primary_key=True),
        sa.Column("voice_name", sa.String(length=64), primary_key=True),
        sa.Column(
            "target_profile_id",
            sa.String(length=512),
            sa.ForeignKey("model_profiles.profile_id", ondelete="CASCADE", onupdate="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("openai_tts_voice_map")
