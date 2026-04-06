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
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client

    @pytest.fixture
    def sync_client():
        return TestClient(app)
