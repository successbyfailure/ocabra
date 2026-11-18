"""users.sessions_valid_after

Revision ID: 0048
Revises: 0047
Create Date: 2026-11-18

Sessions (JWT cookies) issued before this moment are no longer accepted. Set
when the password changes, so a stolen session does not survive it (SEG-19).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0048"
down_revision: str | None = "0047"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("sessions_valid_after", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("users", "sessions_valid_after")
