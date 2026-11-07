import asyncio
import os
from collections.abc import AsyncGenerator
from urllib.parse import urlsplit, urlunsplit

# ── Base de datos de tests separada (DEBE ir antes de importar ``ocabra``) ────
# Los tests se ejecutan dentro del contenedor ``api``, cuyo DATABASE_URL apunta
# al Postgres de produccion. Hasta 2026-11-07 los tests de integracion escribian
# alli: ``test_patch_config_applies_runtime_values`` persistia en
# ``server_config`` un idle_timeout de 123 s, download_dir=/tmp/test-downloads,
# diffusers_offload_mode=model, etc., que produccion aplicaba en cada arranque;
# y los tests del model manager dejaban filas ``test/*`` en model_load_stats.
# Aqui se redirige SIEMPRE a ``<db>_test`` (o a OCABRA_TEST_DATABASE_URL) antes
# de que ``ocabra.config.settings`` y el engine de ``ocabra.database`` se creen.
_TEST_DB_SUFFIX = "_test"


def _derive_test_database_url() -> str:
    explicit = os.environ.get("OCABRA_TEST_DATABASE_URL")
    if explicit:
        return explicit
    base = (
        os.environ.get("DATABASE_URL") or "postgresql+asyncpg://ocabra:ocabra@postgres:5432/ocabra"
    )
    parts = urlsplit(base)
    name = parts.path.lstrip("/") or "ocabra"
    if not name.endswith(_TEST_DB_SUFFIX):
        name += _TEST_DB_SUFFIX
    return urlunsplit(parts._replace(path=f"/{name}"))


os.environ["DATABASE_URL"] = _derive_test_database_url()

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

try:
    from fastapi.testclient import TestClient
    from httpx import ASGITransport, AsyncClient

    from ocabra.main import app

    _APP_AVAILABLE = True
except ImportError:
    _APP_AVAILABLE = False


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if _APP_AVAILABLE:

    @pytest_asyncio.fixture
    async def async_client() -> AsyncGenerator:
        from ocabra.api._deps_auth import UserContext, get_current_user

        _admin_ctx = UserContext(
            user_id=None,
            username="__test__",
            role="system_admin",
            group_ids=[],
            accessible_model_ids=set(),
            is_anonymous=False,
        )
        app.dependency_overrides[get_current_user] = lambda: _admin_ctx
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.fixture
    def sync_client():
        return TestClient(app)


# ── Tests que necesitan la base de datos real ────────────────────────────────
# Unos pocos tests son de integracion a proposito: escriben en un Postgres de
# verdad porque lo que validan es justo la consulta agregada que leen las
# paginas de stats. Ese Postgres es la base ``<db>_test`` del mismo servidor
# (ver arriba), que se crea y se migra con Alembic al arrancar la sesion. El
# contenedor de Postgres no publica puerto al host, asi que fuera de Docker el
# hostname ni resuelve y el test moria con un `socket.gaierror` crudo que
# parecia un fallo de codigo. Con esta guarda se saltan con un motivo legible
# fuera de Docker y siguen ejecutandose donde si hay base de datos.
_DB_REACHABLE: bool | None = None


def _provision_test_database(url: str) -> None:
    """Crea la base de tests si falta y la lleva a ``alembic head``.

    Se niega a tocar cualquier base cuyo nombre no termine en ``_test``: es la
    ultima linea de defensa para que un DATABASE_URL mal derivado no vuelva a
    ejecutar migraciones o escrituras de tests contra produccion.
    """
    import subprocess
    import sys
    from pathlib import Path

    import asyncpg

    parts = urlsplit(url.replace("+asyncpg", ""))
    db_name = parts.path.lstrip("/")
    if not db_name.endswith(_TEST_DB_SUFFIX):
        raise RuntimeError(f"Refusing to run DB tests against non-test database {db_name!r}")

    async def _ensure_exists() -> None:
        admin_url = urlunsplit(parts._replace(path="/postgres"))
        conn = await asyncpg.connect(admin_url)
        try:
            exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
            if not exists:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
        finally:
            await conn.close()

    asyncio.run(_ensure_exists())
    backend_dir = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=backend_dir,
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"alembic upgrade head failed on {db_name!r}:\n{result.stderr[-4000:]}")


def _db_reachable() -> bool:
    global _DB_REACHABLE
    if _DB_REACHABLE is None:
        import socket

        from ocabra.config import settings

        try:
            parsed = urlsplit(settings.database_url.replace("+asyncpg", ""))
            socket.getaddrinfo(parsed.hostname, parsed.port or 5432)
            _DB_REACHABLE = True
        except Exception:
            _DB_REACHABLE = False
        if _DB_REACHABLE:
            _provision_test_database(settings.database_url)
    return _DB_REACHABLE


requires_db = pytest.mark.skipif(
    not _db_reachable(),
    reason="necesita Postgres (no publica puerto al host; correr dentro de Docker)",
)


@pytest_asyncio.fixture(autouse=True)
async def _dispose_db_engine_between_tests():
    """Devuelve el pool de conexiones al final de cada test.

    ``ocabra.database.engine`` es un singleton de modulo y su pool queda ligado
    al event loop del primer test que lo usa. pytest-asyncio crea un loop nuevo
    por test, asi que a partir del segundo test que tocaba la base de datos
    asyncpg reutilizaba una conexion de un loop ya cerrado y el test moria con
    "got Future attached to a different loop" / "Event loop is closed" — un
    fallo que parecia de la consulta y en realidad era de infraestructura, y que
    volvia el resultado dependiente del orden (los tests pasaban sueltos y
    fallaban en conjunto). Desechar el pool es barato cuando no hay conexiones
    abiertas, que es el caso de la gran mayoria de los tests.
    """
    yield
    from ocabra.database import engine

    await engine.dispose()
