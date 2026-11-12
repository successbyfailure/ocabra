"""Drop the runtime presets of Ollama models.

Revision ID: 0035
Revises: 0034
Create Date: 2026-11-12

Ollama runs as one shared daemon with a global context and parallelism (the
per-GPU pools the presets switched between were retired). A preset ladder on
an Ollama profile has nothing to choose from any more, so the presets go and
the API refuses new ones. Aliases that pointed at those presets lose their
preset key and resolve to the profile itself.

Data-only and not reversible: the deleted presets are not recreated on
downgrade.
"""

from alembic import op

revision = "0035"
down_revision = "0034"
branch_labels = None
depends_on = None

_OLLAMA_PROFILES = """
    SELECT p.profile_id
    FROM model_profiles p
    JOIN model_configs c ON c.model_id = p.base_model_id
    WHERE c.backend_type = 'ollama'
"""


def upgrade() -> None:
    op.execute(
        f"UPDATE model_profile_aliases SET preset_key = NULL "
        f"WHERE profile_id IN ({_OLLAMA_PROFILES})"
    )
    op.execute(f"DELETE FROM profile_runtime_presets WHERE profile_id IN ({_OLLAMA_PROFILES})")


def downgrade() -> None:
    # The presets are gone for good; nothing to restore.
    pass
