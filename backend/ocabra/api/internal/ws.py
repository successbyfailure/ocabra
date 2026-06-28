import asyncio
import contextlib
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ocabra.core.auth_manager import AuthError, decode_access_token
from ocabra.redis_client import get_redis

router = APIRouter(tags=["websocket"])

CHANNEL_EVENT_MAP = {
    "gpu:stats": "gpu_stats",
    "model:events": "model_event",
    "service:events": "service_event",
    "download:progress": "download_progress",
    "system:alerts": "system_alert",
}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Real-time event stream over WebSocket.

    Requires a valid ``ocabra_session`` cookie. Forwards Redis pub/sub
    events (gpu_stats, model_event, service_event, download_progress,
    system_alert) as JSON frames to the client.
    """
    token = websocket.cookies.get("ocabra_session")
    if not token:
        await websocket.close(code=1008)
        return
    try:
        decode_access_token(token)
    except AuthError:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    pubsub = get_redis().pubsub()

    async def redis_listener() -> None:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            channel = message["channel"]
            event_type = CHANNEL_EVENT_MAP.get(channel, "unknown")
            try:
                data = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            await websocket.send_text(json.dumps({"type": event_type, "data": data}))

    listener_task: asyncio.Task | None = None
    try:
        await pubsub.subscribe(*CHANNEL_EVENT_MAP)
        listener_task = asyncio.create_task(redis_listener())
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        # Always release the dedicated Redis pub/sub connection, even when the
        # socket is already dead. ``aclose()`` resets and returns the connection
        # to the pool; calling ``unsubscribe()`` first would raise on a broken
        # connection and skip the close, leaking one Redis connection per
        # dropped WebSocket (which eventually exhausts the pool → "Too many
        # connections" on unrelated operations like loading an Ollama model).
        if listener_task is not None:
            listener_task.cancel()
            with contextlib.suppress(BaseException):
                await listener_task
        with contextlib.suppress(Exception):
            await pubsub.aclose()
