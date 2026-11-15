"""openai_image_model_map table

Revision ID: 0040
Revises: 0039
Create Date: 2026-11-15

Compatibility shim for image clients that only know OpenAI's official model
names (``gpt-image-1``, ``gpt-image-1-mini``, ``dall-e-3``, ``dall-e-2``).
Each row maps one OpenAI name to a target ``profile_id``. No rows = unchanged
behaviour (those names stay unrecognised, 404).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0040"
down_revision: str | None = "0039"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "openai_image_model_map",
        sa.Column("openai_model", sa.String(length=128), primary_key=True),
        sa.Column(
            "target_profile_id",
            sa.String(length=512),
            sa.ForeignKey("model_profiles.profile_id", ondelete="CASCADE", onupdate="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("openai_image_model_map")
