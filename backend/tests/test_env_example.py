"""INF-09: ``.env.example`` must describe the settings that really exist.

``Settings`` uses ``extra="ignore"``, so a variable that nothing reads (a
retired integration, a typo) is dropped without a word, and a setting added to
``config.py`` without a line in ``.env.example`` is invisible to operators. By
2026-11-16 the two had drifted to 313 fields against 213 variables. These tests
keep them in step:

- every key in ``.env.example`` is a setting, or a variable that compose,
  setup.sh or a build script reads;
- every setting is in ``.env.example`` (APP_VERSION excepted: it is ignored);
- a commented ``# KEY=value`` shows the real default: the one in config.py, or
  the ``${KEY:-value}`` that docker-compose passes instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from ocabra.config import Settings

REPO = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO / ".env.example"
# Where the variables that are not settings are read.
CONSUMERS = [
    REPO / "docker-compose.yml",
    REPO / "docker-compose.fat.yml",
    REPO / "setup.sh",
    *sorted((REPO / "backend" / "scripts").glob("*.sh")),
]
# Settings deliberately left out of .env.example.
NOT_DOCUMENTED = {"APP_VERSION"}  # the validator ignores it: the version is the code's

_LINE = re.compile(r"^(?P<commented>#\s*)?(?P<key>[A-Z][A-Z0-9_]+)=(?P<value>.*)$")

pytestmark = pytest.mark.skipif(not ENV_EXAMPLE.exists(), reason="no .env.example (image build)")


def _entries() -> list[tuple[str, str, bool]]:
    out = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = _LINE.match(line.strip())
        if match:
            out.append((match["key"], match["value"].strip(), bool(match["commented"])))
    return out


def _consumers_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in CONSUMERS if p.exists())


def _compose_defaults(key: str) -> set[str]:
    return set(re.findall(r"\$\{" + re.escape(key) + r":-([^}]*)\}", _consumers_text()))


def test_cada_clave_del_env_example_la_lee_alguien():
    fields = {name.upper() for name in Settings.model_fields}
    text = _consumers_text()
    nadie = sorted(
        {key for key, _, _ in _entries() if key not in fields and key not in text}
    )
    assert not nadie, f"variables de .env.example que nada lee: {nadie}"


def test_cada_ajuste_esta_en_el_env_example():
    keys = {key for key, _, _ in _entries()}
    faltan = sorted(
        name.upper()
        for name in Settings.model_fields
        if name.upper() not in keys and name.upper() not in NOT_DOCUMENTED
    )
    assert not faltan, f"ajustes de config.py sin documentar en .env.example: {faltan}"


def test_sin_claves_repetidas():
    keys = [key for key, _, _ in _entries()]
    repetidas = sorted({k for k in keys if keys.count(k) > 1})
    assert not repetidas, f"claves repetidas en .env.example: {repetidas}"


def _parse(field, raw: str):
    if raw == "":
        return None
    annotation = field.annotation
    origin = getattr(annotation, "__origin__", None)
    if origin in (list, dict) or annotation in (list, dict):
        return json.loads(raw)
    return TypeAdapter(annotation).validate_python(raw)


def test_los_valores_comentados_son_los_por_defecto():
    fields = {name.upper(): field for name, field in Settings.model_fields.items()}
    mal = []
    for key, value, commented in _entries():
        if not commented:
            continue
        compose = _compose_defaults(key)
        if value in compose:
            continue
        field = fields.get(key)
        if field is None:
            if compose:
                mal.append(f"{key}={value} (compose: {sorted(compose)})")
            continue
        if field.default_factory is not None:
            continue
        default = field.default
        if _parse(field, value) != default and not (value == "" and default in ("", None)):
            mal.append(f"{key}={value} (config.py: {default!r})")
    assert not mal, f"valores comentados que no son el defecto: {mal}"
