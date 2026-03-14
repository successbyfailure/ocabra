from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

router = APIRouter(tags=["services"])

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


class ServiceTouchRequest(BaseModel):
    runtime_loaded: bool | None = None
    active_model_ref: str | None = None
    detail: str | None = None


class ServiceRuntimePatch(BaseModel):
    runtime_loaded: bool
    active_model_ref: str | None = None
    detail: str | None = None


@router.get("/services")
async def list_services(request: Request) -> list[dict]:
    sm = request.app.state.service_manager
    states = await sm.list_states()
    return [state.to_dict() for state in states]


@router.get("/services/{service_id}")
async def get_service(service_id: str, request: Request) -> dict:
    sm = request.app.state.service_manager
    state = await sm.get_state(service_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")
    return state.to_dict()


@router.post("/services/{service_id}/refresh")
async def refresh_service(service_id: str, request: Request) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.refresh(service_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.to_dict()


@router.post("/services/{service_id}/touch")
async def touch_service(service_id: str, body: ServiceTouchRequest, request: Request) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.touch(
            service_id,
            runtime_loaded=body.runtime_loaded,
            active_model_ref=body.active_model_ref,
            detail=body.detail,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.to_dict()


@router.patch("/services/{service_id}/runtime")
async def patch_service_runtime(
    service_id: str,
    body: ServiceRuntimePatch,
    request: Request,
) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.mark_runtime(
            service_id,
            runtime_loaded=body.runtime_loaded,
            active_model_ref=body.active_model_ref,
            detail=body.detail,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.to_dict()


@router.post("/services/{service_id}/unload")
async def unload_service(service_id: str, request: Request) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.unload(service_id, reason="manual")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return state.to_dict()


@router.api_route(
    "/services/{service_id}/proxy",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
@router.api_route(
    "/services/{service_id}/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_service_http(
    service_id: str,
    request: Request,
    path: str = "",
) -> Response:
    sm = request.app.state.service_manager
    state = await sm.get_state(service_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")

    upstream_path = _resolve_upstream_path(
        state.ui_base_path,
        request.headers.get("x-forwarded-uri"),
        path,
    )
    if state.service_type == "hunyuan3d" and upstream_path == "/":
        upstream_path = "//"
    url = f"{state.base_url}{upstream_path}"
    query = request.url.query
    if query:
        url = f"{url}?{query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() != "x-forwarded-uri"
    }
    body = await request.body()

    requested_runtime_loaded = _should_mark_runtime_loaded(
        request.method,
        upstream_path,
        unload_path=state.unload_path,
    )
    await sm.touch(
        service_id,
        runtime_loaded=requested_runtime_loaded,
        detail=f"{request.method} {upstream_path}",
    )

    try:
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=False) as client:
            upstream = await client.request(
                request.method,
                url,
                headers=headers,
                content=body,
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }
    _rewrite_location_header(response_headers, state.base_url, state.ui_base_path)
    await _sync_runtime_state_from_response(
        sm,
        state,
        upstream_path,
        upstream,
        requested_runtime_loaded=requested_runtime_loaded,
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


def _resolve_upstream_path(ui_base_path: str, forwarded_uri: str | None, path: str) -> str:
    if forwarded_uri:
        path_only = forwarded_uri.split("?", 1)[0] or "/"
        if path_only.startswith(ui_base_path):
            remainder = path_only[len(ui_base_path):]
            return remainder or "/"
        return path_only
    if not path:
        return "/"
    return f"/{path.lstrip('/')}"


def _should_mark_runtime_loaded(
    method: str,
    upstream_path: str,
    *,
    unload_path: str | None = None,
) -> bool | None:
    normalized = upstream_path.rstrip("/") or "/"
    if unload_path and normalized == (unload_path.rstrip("/") or "/"):
        return False
    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        return True
    return None


def _rewrite_location_header(
    response_headers: dict[str, str],
    base_url: str,
    ui_base_path: str,
) -> None:
    location = response_headers.get("location")
    if not location:
        return
    if location.startswith(base_url):
        suffix = location[len(base_url):] or "/"
        if suffix.startswith("//"):
            normalized_suffix = suffix
        else:
            normalized_suffix = "/" + suffix.lstrip("/")
        response_headers["location"] = f"{ui_base_path.rstrip('/')}{normalized_suffix}"


async def _sync_runtime_state_from_response(
    sm,
    state,
    upstream_path: str,
    upstream: httpx.Response,
    *,
    requested_runtime_loaded: bool | None,
) -> None:
    normalized = upstream_path.rstrip("/") or "/"
    if upstream.is_error:
        return

    if state.unload_path and normalized == (state.unload_path.rstrip("/") or "/"):
        await sm.mark_runtime(
            state.service_id,
            runtime_loaded=False,
            active_model_ref=None,
            detail=f"{upstream.request.method} {normalized}",
        )
        return

    if normalized == "/runtime/status":
        try:
            payload = upstream.json()
        except ValueError:
            return
        runtime_loaded = payload.get("runtime_loaded")
        if isinstance(runtime_loaded, bool):
            await sm.mark_runtime(
                state.service_id,
                runtime_loaded=runtime_loaded,
                active_model_ref=state.active_model_ref,
                detail=f"{upstream.request.method} {normalized}",
            )
        return

    if requested_runtime_loaded is not None:
        await sm.mark_runtime(
            state.service_id,
            runtime_loaded=requested_runtime_loaded,
            active_model_ref=state.active_model_ref,
            detail=f"{upstream.request.method} {normalized}",
        )


@router.websocket("/services/{service_id}/ws")
@router.websocket("/services/{service_id}/ws/{path:path}")
async def proxy_service_ws(
    websocket: WebSocket,
    service_id: str,
    path: str = "",
) -> None:
    try:
        import websockets
        from websockets.exceptions import ConnectionClosed
    except ImportError as exc:
        await websocket.close(code=1011, reason="websockets dependency not installed")
        raise RuntimeError("websockets dependency not installed") from exc

    sm = websocket.app.state.service_manager
    state = await sm.get_state(service_id)
    if state is None:
        await websocket.close(code=4404, reason="service not found")
        return

    forwarded_uri = websocket.headers.get("x-forwarded-uri")
    upstream_path = _resolve_upstream_path(state.ui_base_path, forwarded_uri, path)
    upstream_url = _build_ws_url(state.base_url, upstream_path, websocket.url.query)

    subprotocol_header = websocket.headers.get("sec-websocket-protocol", "")
    subprotocols = [item.strip() for item in subprotocol_header.split(",") if item.strip()]
    upstream_headers = [
        (key, value)
        for key, value in websocket.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() != "x-forwarded-uri"
    ]

    await websocket.accept(subprotocol=subprotocols[0] if subprotocols else None)

    try:
        async with websockets.connect(
            upstream_url,
            additional_headers=upstream_headers,
            subprotocols=subprotocols or None,
            open_timeout=15,
        ) as upstream:
            await sm.touch(
                service_id,
                runtime_loaded=True,
                detail=f"WS {upstream_path}",
            )

            async def client_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    if message.get("text") is not None:
                        await upstream.send(message["text"])
                        await sm.touch(service_id, runtime_loaded=True)
                    elif message.get("bytes") is not None:
                        await upstream.send(message["bytes"])
                        await sm.touch(service_id, runtime_loaded=True)

            async def upstream_to_client() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)
                    await sm.touch(service_id, runtime_loaded=True)

            tasks = [
                asyncio.create_task(client_to_upstream(), name=f"{service_id}-ws-client"),
                asyncio.create_task(upstream_to_client(), name=f"{service_id}-ws-upstream"),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, ConnectionClosed)):
                    raise exc
    except (WebSocketDisconnect, ConnectionClosed):
        pass
    except Exception:
        await websocket.close(code=1011)
        raise


def _build_ws_url(base_url: str, upstream_path: str, query: str) -> str:
    if base_url.startswith("https://"):
        ws_base = "wss://" + base_url[len("https://"):]
    elif base_url.startswith("http://"):
        ws_base = "ws://" + base_url[len("http://"):]
    else:
        ws_base = base_url
    url = f"{ws_base}{upstream_path}"
    if query:
        url = f"{url}?{query}"
    return url
