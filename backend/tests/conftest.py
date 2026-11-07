import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

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
# Unos pocos tests son de integracion a proposito: escriben en el Postgres de
# verdad (con indices/ids reservados) porque lo que validan es justo la consulta
# agregada que leen las paginas de stats. El contenedor de Postgres no publica
# puerto al host, asi que fuera de Docker el hostname ni resuelve y el test
# moria con un `socket.gaierror` crudo que parecia un fallo de codigo. Con esta
# guarda se saltan con un motivo legible fuera de Docker y siguen ejecutandose
# donde si hay base de datos.
_DB_REACHABLE: bool | None = None


def _db_reachable() -> bool:
    global _DB_REACHABLE
    if _DB_REACHABLE is None:
        import socket
        from urllib.parse import urlparse

        from ocabra.config import settings

        try:
            parsed = urlparse(settings.database_url.replace("+asyncpg", ""))
            socket.getaddrinfo(parsed.hostname, parsed.port or 5432)
            _DB_REACHABLE = True
        except Exception:
            _DB_REACHABLE = False
    return _DB_REACHABLE


requires_db = pytest.mark.skipif(
    not _db_reachable(),
    reason="necesita el Postgres real (no publica puerto al host; correr dentro de Docker)",
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
