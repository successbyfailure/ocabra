"""request_stats: client address + user agent

Revision ID: 0021
Revises: 0020
Create Date: 2026-07-08

Adds ``client_addr`` (real client IP, from X-Forwarded-For / X-Real-IP behind
Caddy) and ``user_agent`` to ``request_stats`` so anonymous requests (keyless
Ollama calls, allowed when ``require_api_key_ollama=false``) can be traced back
to the client that made them.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("request_stats", sa.Column("client_addr", sa.String(length=64), nullable=True))
    op.add_column("request_stats", sa.Column("user_agent", sa.String(length=512), nullable=True))


def downgrade() -> None:
    op.drop_column("request_stats", "user_agent")
    op.drop_column("request_stats", "client_addr")
