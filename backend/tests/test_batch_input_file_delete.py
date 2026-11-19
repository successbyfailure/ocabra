"""Borrar el fichero de entrada de un batch terminado daba 500.

``openai_batches.input_file_id`` era NOT NULL con ON DELETE RESTRICT: el
``DELETE /v1/files/{id}`` chocaba con la clave (VHS-transcripts, tras cada
batch). La 0049 quita la clave foránea; el batch conserva el id.
"""

from __future__ import annotations

from pathlib import Path

from ocabra.db.openai_batches import OpenAIBatch


def test_input_file_id_has_no_foreign_key_and_stays_required():
    column = OpenAIBatch.__table__.c.input_file_id
    assert not column.foreign_keys
    assert column.nullable is False


def test_output_and_error_files_still_set_null_on_delete():
    for name in ("output_file_id", "error_file_id"):
        (fk,) = OpenAIBatch.__table__.c[name].foreign_keys
        assert fk.ondelete == "SET NULL"


def test_the_migration_drops_that_constraint():
    path = (
        Path(__file__).resolve().parent.parent / "alembic/versions/0049_batch_input_file_no_fk.py"
    )
    source = path.read_text()
    assert 'down_revision: str | None = "0048"' in source
    assert "openai_batches_input_file_id_fkey" in source
