"""Drop llama.cpp split_strategy values the backend never understood.

Revision ID: 0036
Revises: 0035
Create Date: 2026-11-12

The old model editor stored ``split_strategy: "single"`` (its "one GPU"
radio; the real switch is ``split_mode: "none"``). The backend only knows
``evenly`` and ``favor_main`` and silently ignored anything else, so removing
those values changes nothing at load time. It does let the new editor, which
validates options, save those models.
"""

from alembic import op

revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE model_configs
        SET extra_config = jsonb_set(
            extra_config, '{llama_cpp}', (extra_config->'llama_cpp') - 'split_strategy'
        )
        WHERE extra_config ? 'llama_cpp'
          AND (extra_config->'llama_cpp') ? 'split_strategy'
          AND coalesce(extra_config->'llama_cpp'->>'split_strategy', '')
              NOT IN ('evenly', 'favor_main')
        """
    )


def downgrade() -> None:
    pass
