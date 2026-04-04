"""oCabra Generation Services Gateway.

Listens on GATEWAY_PORT (default 9000). Routes incoming requests by Host header:
  - DIRECTORY_HOST → serves the directory page (all services overview)
  - known service host → transparent HTTP/WS proxy with on-demand startup
  - anything else → 404

Internal API endpoints (prefix /_gw/) are served for all hosts:
  GET  /_gw/status/{service_id}    — live service status (used by loading page JS)
  GET  /_gw/services               — list all services (used by directory page JS)
  POST /_gw/services/{id}/start    — start a service
  POST /_gw/services/{id}/unload   — stop/unload a service
  PATCH /_gw/services/{id}/enable  — enable or disable a service
"""
from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response

from config import (
    DIRECTORY_HOST,
    GATEWAY_PORT,
    GATEWAY_SERVICE_TOKEN,
    OCABRA_API_URL,
    SERVICE_BY_HOST,
    STARTUP_TIMEOUT_S,
)
from pages import directory_page, disabled_page, loading_page, not_found_page
from proxy import proxy_http, proxy_websocket

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# ---------------------------------------------------------------------------
# Helpers — talk to the ocabra API
# ---------------------------------------------------------------------------

def _service_headers() -> dict:
    """Headers to authenticate gateway→ocabra internal calls."""
    if GATEWAY_SERVICE_TOKEN:
        return {"X-Gateway-Token": GATEWAY_SERVICE_TOKEN}
    return {}


async def _ocabra(method: str, path: str, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    headers.update(_service_headers())
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await client.request(method, f"{OCABRA_API_URL}{path}", headers=headers, **kwargs)


async def _get_service_state(service_id: str) -> dict | None:
    try:
        r = await _ocabra("GET", f"/ocabra/services/{service_id}")
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


async def _start_service_bg(service_id: str) -> None:
    """Fire-and-forget: ask ocabra to start the container."""
    try:
        await _ocabra("POST", f"/ocabra/services/{service_id}/start")
    except Exception as exc:
        logger.warning("start_service_failed service=%s error=%s", service_id, exc)


# ---------------------------------------------------------------------------
# Internal gateway API  (/_gw/*)
# ---------------------------------------------------------------------------

@app.post("/_gw/auth/login")
async def gw_auth_login(request: Request) -> JSONResponse:
    """Proxy login to the ocabra API and return the JWT in the JSON body.

    The gateway directory page uses this to authenticate users via localStorage
    instead of relying on the HTTP-only cookie (which is scoped to the API domain).
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON"}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{OCABRA_API_URL}/ocabra/auth/login", json=body)
        if r.status_code == 200:
            data = r.json()
            return JSONResponse({"access_token": data.get("access_token", ""), "user": data.get("user", {})})
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


@app.get("/_gw/status/{service_id}")
async def gw_status(service_id: str) -> JSONResponse:
    """Return live state of a service. Used by the loading page to poll.

    Calls POST /refresh so the response reflects the real health of the container
    immediately — no need to wait for the 30-second background health loop.
    """
    try:
        r = await _ocabra("POST", f"/ocabra/services/{service_id}/refresh")
        if r.status_code == 200:
            state = r.json()
            return JSONResponse({
                "service_alive": state.get("service_alive", False),
                "status":        state.get("status", "unknown"),
                "detail":        state.get("detail"),
                "enabled":       state.get("enabled", True),
            })
    except Exception:
        pass
    return JSONResponse({"service_alive": False, "status": "unknown", "detail": "ocabra unreachable"})


@app.get("/_gw/services")
async def gw_list_services() -> JSONResponse:
    """Return state of all generation services. Used by the directory page."""
    try:
        r = await _ocabra("GET", "/ocabra/services")
        if r.status_code == 200:
            return JSONResponse(r.json())
    except Exception as exc:
        logger.warning("list_services_failed error=%s", exc)
    return JSONResponse([])


@app.post("/_gw/services/{service_id}/start")
async def gw_start(service_id: str) -> JSONResponse:
    try:
        r = await _ocabra("POST", f"/ocabra/services/{service_id}/start")
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


@app.post("/_gw/services/{service_id}/unload")
async def gw_unload(service_id: str) -> JSONResponse:
    try:
        r = await _ocabra("POST", f"/ocabra/services/{service_id}/unload")
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


@app.patch("/_gw/services/{service_id}/enable")
async def gw_enable(service_id: str, request: Request) -> JSONResponse:
    body = await request.json()
    try:
        r = await _ocabra("PATCH", f"/ocabra/services/{service_id}", json={"enabled": body.get("enabled", True)})
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


# ---------------------------------------------------------------------------
# WebSocket catch-all
# ---------------------------------------------------------------------------

@app.websocket("/{path:path}")
async def ws_catch_all(websocket: WebSocket, path: str) -> None:
    host = websocket.headers.get("host", "").split(":")[0].lower()
    service = SERVICE_BY_HOST.get(host)

    if service is None:
        await websocket.close(code=4004, reason="Unknown host")
        return

    state = await _get_service_state(service["service_id"])
    if state and not state.get("service_alive", False):
        await websocket.close(code=4503, reason="Service not ready — load the page first")
        return

    await proxy_websocket(websocket, service["upstream"], service["service_id"])


# ---------------------------------------------------------------------------
# HTTP catch-all
# ---------------------------------------------------------------------------

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def http_catch_all(request: Request, path: str) -> Response:
    host = request.headers.get("host", "").split(":")[0].lower()

    # ── Directory page ──────────────────────────────────────────────────────
    if DIRECTORY_HOST and host == DIRECTORY_HOST:
        return HTMLResponse(directory_page())

    # ── Generation service ──────────────────────────────────────────────────
    service = SERVICE_BY_HOST.get(host)
    if service is None:
        return HTMLResponse(not_found_page(host), status_code=404)

    service_id = service["service_id"]
    state = await _get_service_state(service_id)

    # ocabra is unreachable — let it through and fail naturally
    if state is None:
        logger.warning("ocabra_unreachable service=%s, attempting blind proxy", service_id)
        try:
            return await proxy_http(request, service["upstream"], service_id, public_url=service["ui_url"])
        except Exception as exc:
            return HTMLResponse(
                f"<h1>Gateway error</h1><p>{exc}</p>",
                status_code=502,
            )

    # Service disabled
    if not state.get("enabled", True):
        return HTMLResponse(disabled_page(service["display_name"]), status_code=503)

    # Service alive → proxy transparently
    if state.get("service_alive", False):
        try:
            return await proxy_http(
                request,
                service["upstream"],
                service_id,
                public_url=service["ui_url"],
            )
        except Exception as exc:
            logger.error("proxy_error service=%s error=%s", service_id, exc)
            return HTMLResponse(f"<h1>Proxy error</h1><p>{exc}</p>", status_code=502)

    # ── Service is DOWN → trigger start, show loading page ─────────────────
    # Only trigger start for browser navigations (not background XHR/assets
    # from a stale tab still polling after the service went down).
    accept = request.headers.get("accept", "")
    is_browser_nav = "text/html" in accept

    if is_browser_nav:
        logger.info("service_down_starting service=%s", service_id)
        asyncio.create_task(_start_service_bg(service_id))
        return HTMLResponse(
            loading_page(service_id, service["display_name"], STARTUP_TIMEOUT_S),
            status_code=503,
            headers={"Retry-After": "5"},
        )

    # Non-browser requests while service is down → 503 + Retry-After
    return Response(
        content=f"Service {service_id!r} is not running",
        status_code=503,
        headers={"Retry-After": "10"},
        media_type="text/plain",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=GATEWAY_PORT, log_level="info")
