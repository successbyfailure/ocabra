"""Shared health-check client of the backends (INF-14)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ocabra.backends._health_http import health_client


@pytest.mark.asyncio
async def test_one_client_per_loop_without_keepalive():
    a = health_client()
    assert health_client() is a
    pool = a._transport._pool
    assert pool._max_keepalive_connections == 0


def test_a_new_loop_gets_its_own_client():
    first = asyncio.run(_get())
    second = asyncio.run(_get())
    assert first is not second


async def _get():
    return health_client()


@pytest.mark.asyncio
async def test_a_patched_httpx_client_is_honoured():
    health_client()
    fake = MagicMock()
    with patch.object(httpx, "AsyncClient", return_value=fake):
        assert health_client() is fake
    assert isinstance(health_client(), httpx.AsyncClient)
