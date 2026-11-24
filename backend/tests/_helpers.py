"""Factorías compartidas por los tests (ver DUP-10 de docs/REVISION-2026-11-16).

Se importa como ``from tests._helpers import …`` (igual que ``tests._app_routes``).
De momento solo ``make_fake_proc``: los tests de backend repetían el mismo mock
de proceso de worker (asyncio subprocess) con variaciones mínimas (pid y, en
algunos, un ``stderr`` legible).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def make_fake_proc(
    *, pid: int = 4242, returncode: int | None = None, with_stderr: bool = False
) -> MagicMock:
    """Mock de un proceso de worker lanzado con ``create_subprocess_exec``.

    ``terminate``/``kill`` son ``MagicMock``; ``wait`` un ``AsyncMock`` que
    devuelve 0. Con ``with_stderr`` añade ``stderr.read`` como ``AsyncMock`` que
    devuelve ``b""`` (lo usan los backends que leen la salida de error).
    """
    proc = MagicMock()
    proc.pid = pid
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    if with_stderr:
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=b"")
    return proc
