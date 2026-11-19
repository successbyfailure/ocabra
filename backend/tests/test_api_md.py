"""DOC-05: ``docs/API.md`` lists every route of the app, and nothing else.

By 2026-11-16 it tabulated 133 of 266 routes (no federation, agents, backends,
cloud, sessions, video). The internal ``/ocabra/*`` API is registered with
``include_in_schema=False``, so Swagger does not list it either: API.md is the
only index. A table row is ``| METHODS | `path` | … |``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._app_routes import effective_routes

API_MD = Path(__file__).resolve().parents[2] / "docs" / "API.md"
_ROW = re.compile(r"^\|\s*(?P<methods>[A-Z, ]+?)\s*\|\s*`(?P<path>/[^`]*)`\s*\|")

pytestmark = pytest.mark.skipif(not API_MD.exists(), reason="no docs/ (image build)")


def _documented() -> set[tuple[str, str]]:
    pairs = set()
    for line in API_MD.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match:
            for method in match["methods"].split(","):
                pairs.add((method.strip(), match["path"]))
    return pairs


def _app_pairs() -> set[tuple[str, str]]:
    from ocabra.main import app

    pairs = set()
    for path, methods in effective_routes(app):
        for method in methods - {"HEAD"} or {"WS"}:
            pairs.add((method, path))
    return pairs


def test_cada_ruta_de_la_app_esta_en_api_md():
    faltan = sorted(_app_pairs() - _documented(), key=lambda p: (p[1], p[0]))
    assert not faltan, (
        "rutas sin documentar en docs/API.md (añade una fila a su familia): "
        + ", ".join(f"{m} {p}" for m, p in faltan)
    )


def test_api_md_no_documenta_rutas_que_no_existen():
    sobran = sorted(_documented() - _app_pairs(), key=lambda p: (p[1], p[0]))
    assert not sobran, "docs/API.md lista rutas que ya no existen: " + ", ".join(
        f"{m} {p}" for m, p in sobran
    )
