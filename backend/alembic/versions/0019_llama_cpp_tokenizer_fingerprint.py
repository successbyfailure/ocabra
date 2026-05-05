"""llama_cpp tokenizer fingerprint

Revision ID: 0019
Revises: 0018
Create Date: 2026-05-05

Plan: docs/tasks/llama-cpp-parity-plan.md — Sprint 17.4.

Adds three nullable integer columns to ``model_configs`` so the local scanner
can persist GGUF tokenizer fingerprints (vocab size + bos/eos token IDs) and
the API can validate speculative-decoding draft compatibility.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("model_configs", sa.Column("vocab_size", sa.Integer(), nullable=True))
    op.add_column("model_configs", sa.Column("bos_id", sa.Integer(), nullable=True))
    op.add_column("model_configs", sa.Column("eos_id", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("model_configs", "eos_id")
    op.drop_column("model_configs", "bos_id")
    op.drop_column("model_configs", "vocab_size")
