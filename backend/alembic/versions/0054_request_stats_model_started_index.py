"""índice compuesto (model_id, started_at) en request_stats (REF-12)

Las agregaciones de stats que filtran por modelo y rango de tiempo
(``get_performance_stats``, ``get_overview_stats``, ``get_request_stats`` con
``model_id``) no tenían un índice que cubriera las dos columnas a la vez: había
uno por ``model_id`` y otro por ``started_at`` sueltos. Este los combina.

La migración SOLO añade el índice; no borra ni toca datos. La retención de
filas viejas se hace en caliente (``stats/retention.py``), desactivada por
defecto, para que el operador la vea y la ajuste antes de borrar nada.

Revision ID: 0054
Revises: 0053
Create Date: 2026-11-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0054"
down_revision: str | None = "0053"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_request_stats_model_started",
        "request_stats",
        ["model_id", "started_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_request_stats_model_started", table_name="request_stats")
