"""BUG-45: hw-monitor's copy of ``GpuStat``/``ServerStat`` must match the backend.

hw-monitor is its own image (``hw-monitor/Dockerfile`` copies only
``hw_monitor.py``), so it redefines the two tables instead of importing
``ocabra.db.stats``. A migration that touches either table without updating
that copy breaks hw-monitor's per-minute INSERT, and the failure only shows up
as a ``persist_error`` log line. This test makes the drift fail CI instead.

The hw-monitor module is read with ``ast`` rather than imported: importing it
needs pynvml and reconfigures structlog for the whole test process.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ocabra.db.stats import GpuStat, ServerStat

HW_MONITOR = Path(__file__).resolve().parents[2] / "hw-monitor" / "hw_monitor.py"


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    raise AssertionError(f"unexpected node {ast.dump(node)}")


def _hw_monitor_tables() -> dict[str, dict]:
    tree = ast.parse(HW_MONITOR.read_text(encoding="utf-8"))
    tables: dict[str, dict] = {}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        if cls.name not in {"GpuStat", "ServerStat"}:
            continue
        info: dict = {"columns": {}, "indexes": set(), "tablename": None}
        for stmt in cls.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                if name == "__tablename__":
                    info["tablename"] = ast.literal_eval(stmt.value)
                elif name == "__table_args__":
                    for idx in stmt.value.elts:  # type: ignore[attr-defined]
                        assert _call_name(idx) == "Index"
                        args = [ast.literal_eval(a) for a in idx.args]
                        info["indexes"].add((args[0], tuple(args[1:])))
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                call = stmt.value
                assert isinstance(call, ast.Call) and _call_name(call) == "mapped_column"
                kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
                pk = bool(kwargs.get("primary_key", False))
                optional = "None" in ast.unparse(stmt.annotation)
                nullable = kwargs.get("nullable", optional and not pk)
                info["columns"][stmt.target.id] = (_call_name(call.args[0]), pk, nullable)
        tables[cls.name] = info
    return tables


def _backend_table(model) -> dict:
    table = model.__table__
    return {
        "tablename": table.name,
        "columns": {
            c.name: (type(c.type).__name__, bool(c.primary_key), bool(c.nullable))
            for c in table.columns
        },
        "indexes": {(i.name, tuple(c.name for c in i.columns)) for i in table.indexes},
    }


@pytest.mark.parametrize("model", [GpuStat, ServerStat], ids=lambda m: m.__name__)
def test_hw_monitor_table_matches_backend(model) -> None:
    hw = _hw_monitor_tables()
    assert model.__name__ in hw, f"{model.__name__} no longer defined in {HW_MONITOR}"
    assert hw[model.__name__] == _backend_table(model), (
        f"hw-monitor/hw_monitor.py's {model.__name__} drifted from ocabra/db/stats.py; "
        "update both together (and the Alembic migration)."
    )
