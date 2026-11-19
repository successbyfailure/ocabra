"""One HTTP client per event loop for the backends' worker health checks (INF-14).

Every ``health_check`` built and tore down its own ``httpx.AsyncClient`` (SSL
context, pool, transport) to send one GET to ``127.0.0.1:<port>/health``, and
the model manager polls every loaded worker. They share this one instead.

No keep-alive on purpose: a worker that dies and is replaced on the same port
would leave a pooled connection to the dead process, and a stale connection
failing a health check could get a healthy worker restarted. Each check opens
its connection, as before; what is no longer rebuilt is the client. The
timeout stays per call (``client.get(url, timeout=...)``).

Inference requests keep their own clients on purpose (worker_pool, the
backends' ``forward_*``): this is only for health checks.
"""

from __future__ import annotations

import asyncio

import httpx

_clients: dict[asyncio.AbstractEventLoop, tuple[object, httpx.AsyncClient]] = {}


def health_client() -> httpx.AsyncClient:
    """The shared health-check client of the running event loop."""
    loop = asyncio.get_running_loop()
    for other in [lp for lp in _clients if lp.is_closed()]:
        del _clients[other]
    factory = httpx.AsyncClient  # looked up each time: tests patch it
    cached = _clients.get(loop)
    if cached is not None and cached[0] is factory and not cached[1].is_closed:
        return cached[1]
    client = factory(
        timeout=5.0,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=0),
    )
    _clients[loop] = (factory, client)
    return client
