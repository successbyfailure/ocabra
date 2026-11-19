"""openai_batches.input_file_id sin clave foránea

Revision ID: 0049
Revises: 0048
Create Date: 2026-11-19

``input_file_id`` apuntaba a ``openai_files`` con ON DELETE RESTRICT y NOT
NULL: borrar el fichero de entrada de un batch ya terminado (``DELETE
/v1/files/{id}``) chocaba con la clave y daba 500. Como en OpenAI, el batch
conserva el id aunque el fichero ya no exista: se quita la clave foránea y la
columna sigue obligatoria. El borrado sigue negándose (409) mientras el batch
está en curso, eso lo comprueba el endpoint.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0049"
down_revision: str | None = "0048"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("openai_batches_input_file_id_fkey", "openai_batches", type_="foreignkey")


def downgrade() -> None:
    # NOT VALID: los batches cuyo fichero ya se borró no cumplirían la clave;
    # así la recupera sin comprobar las filas que ya hay (ni borrarlas).
    op.execute(
        "ALTER TABLE openai_batches ADD CONSTRAINT openai_batches_input_file_id_fkey "
        "FOREIGN KEY (input_file_id) REFERENCES openai_files(id) "
        "ON DELETE RESTRICT NOT VALID"
    )
