"""Promover estado de ejecución/config fuera del cajón de sastre extra_config (REF-20)

``extra_config`` mezclaba configuración con estado de ejecución. Esta migración
asciende cuatro valores a columnas reales de ``model_configs`` y las rellena a
partir del ``extra_config`` existente:

- ``runtime_state`` (JSONB)         <- extra_config["_runtime"]
- ``default_reasoning_effort`` (str) <- extra_config["default_reasoning_effort"]
- ``context_length`` (int)          <- contexto en frío (nivel raíz o sección)
- ``vram_estimate_mb`` (int)        <- override por backend (llama_cpp/vllm)

NO borra las claves de ``extra_config``: ~20 sitios de ``api/`` y ``backends/``
siguen leyéndolas, así que las columnas son por ahora espejos denormalizados que
``ModelManager`` mantiene en sync al escribir la fila (ver REF-20 en
docs/REVISION-2026-11-16.md para la migración diferida de los lectores). Por eso
la migración es reversible sin pérdida de datos: ``downgrade`` solo quita las
columnas y ``extra_config`` queda intacto.

Revision ID: 0055
Revises: 0054
Create Date: 2026-11-24
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0055"
down_revision: str | None = "0054"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Auto-contenido (las migraciones no importan código de la app): réplica de
# model_manager_helpers.promoted_columns_from_extra_config y de los mapeos de
# config_schema / capabilities vigentes al crear esta migración.
_VRAM_OVERRIDE_SECTIONS = {"llama_cpp": "llama_cpp", "llamacpp_mks": "llama_cpp", "vllm": "vllm"}
_CONTEXT_SECTIONS = ("vllm", "sglang", "strata", "llama_cpp", "bitnet", "tensorrt_llm", "whisper")
_CONTEXT_KEYS = ("context_length", "max_model_len", "ctx_size")


def _snake_to_camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part.title() for part in rest)


def _context_length(extra: dict) -> int | None:
    for key in _CONTEXT_KEYS:
        value = extra.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    for section_name in _CONTEXT_SECTIONS:
        section = extra.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in _CONTEXT_KEYS:
            for candidate in (key, _snake_to_camel(key)):
                value = section.get(candidate)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    return value
    return None


def _vram_override(extra: dict, backend_type: str) -> int | None:
    candidates: list = []
    section_name = _VRAM_OVERRIDE_SECTIONS.get(backend_type)
    if section_name:
        section = extra.get(section_name)
        if isinstance(section, dict):
            candidates += [section.get("vram_estimate_mb"), section.get("vramEstimateMb")]
    candidates += [extra.get("vram_estimate_mb"), extra.get("vramEstimateMb")]
    for value in candidates:
        if isinstance(value, bool):
            continue
        try:
            as_int = int(value)
        except (TypeError, ValueError):
            continue
        if as_int > 0:
            return as_int
    return None


def _promoted(extra_config, backend_type: str) -> dict:
    extra = extra_config if isinstance(extra_config, dict) else {}
    runtime = extra.get("_runtime")
    raw_effort = extra.get("default_reasoning_effort") or extra.get("defaultReasoningEffort")
    return {
        "runtime_state": runtime if isinstance(runtime, dict) else None,
        "default_reasoning_effort": (str(raw_effort).strip().lower() if raw_effort else None),
        "context_length": _context_length(extra),
        "vram_estimate_mb": _vram_override(extra, backend_type),
    }


def upgrade() -> None:
    op.add_column("model_configs", sa.Column("runtime_state", JSONB()))
    op.add_column("model_configs", sa.Column("default_reasoning_effort", sa.String(length=16)))
    op.add_column("model_configs", sa.Column("context_length", sa.Integer()))
    op.add_column("model_configs", sa.Column("vram_estimate_mb", sa.Integer()))

    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT model_id, backend_type, extra_config FROM model_configs")
    ).fetchall()
    update = sa.text(
        """
        UPDATE model_configs
        SET runtime_state = CAST(:rs AS jsonb),
            default_reasoning_effort = :dre,
            context_length = :ctx,
            vram_estimate_mb = :vram
        WHERE model_id = :mid
        """
    )
    for model_id, backend_type, extra_config in rows:
        cols = _promoted(extra_config, backend_type or "")
        if not any(value is not None for value in cols.values()):
            continue
        runtime_state = cols["runtime_state"]
        conn.execute(
            update,
            {
                "mid": model_id,
                "rs": json.dumps(runtime_state) if runtime_state is not None else None,
                "dre": cols["default_reasoning_effort"],
                "ctx": cols["context_length"],
                "vram": cols["vram_estimate_mb"],
            },
        )


def downgrade() -> None:
    # extra_config conserva todas las claves, así que esto no pierde datos.
    op.drop_column("model_configs", "vram_estimate_mb")
    op.drop_column("model_configs", "context_length")
    op.drop_column("model_configs", "default_reasoning_effort")
    op.drop_column("model_configs", "runtime_state")
