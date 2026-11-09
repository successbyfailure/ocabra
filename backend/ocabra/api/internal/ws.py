import asyncio
import contextlib
import json
import time

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


_ACTIVITY_PUSH_SECONDS = 1.0
# Un solo calculo por tick aunque haya varias pestañas suscritas.
_activity_cache: dict[str, object] = {"at": 0.0, "data": None}
_activity_lock = asyncio.Lock()


async def _shared_models_activity(websocket: WebSocket) -> dict:
    from ocabra.api.internal.models import build_models_activity

    async with _activity_lock:
        now = time.monotonic()
        if _activity_cache["data"] is None or now - float(_activity_cache["at"]) >= 0.8:
            _activity_cache["data"] = await build_models_activity(websocket)
            _activity_cache["at"] = now
        return _activity_cache["data"]  # type: ignore[return-value]


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Real-time event stream over WebSocket.

    Requires a valid ``ocabra_session`` cookie. Forwards Redis pub/sub
    events (gpu_stats, model_event, service_event, download_progress,
    system_alert) as JSON frames to the client. A client that sends
    ``{"subscribe": "models_activity"}`` also gets a ``models_activity``
    frame every second (same payload as ``GET /ocabra/models/activity``)
    until it sends ``{"unsubscribe": "models_activity"}``.
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

    # Actividad en vivo de los modelos (dashboard): solo a quien la pide con
    # {"subscribe": "models_activity"}, para no calcularla para cada pestaña.
    activity_task: asyncio.Task | None = None

    async def activity_pusher() -> None:
        while True:
            try:
                data = await _shared_models_activity(websocket)
                await websocket.send_text(json.dumps({"type": "models_activity", "data": data}))
            except (WebSocketDisconnect, RuntimeError):
                return
            except Exception:  # noqa: BLE001 — un tick fallido no cierra el socket
                pass
            await asyncio.sleep(_ACTIVITY_PUSH_SECONDS)

    listener_task: asyncio.Task | None = None
    try:
        await pubsub.subscribe(*CHANNEL_EVENT_MAP)
        listener_task = asyncio.create_task(redis_listener())
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue
            try:
                message = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(message, dict):
                continue
            if message.get("subscribe") == "models_activity" and activity_task is None:
                activity_task = asyncio.create_task(activity_pusher())
            elif message.get("unsubscribe") == "models_activity" and activity_task is not None:
                activity_task.cancel()
                activity_task = None
    except WebSocketDisconnect:
        pass
    finally:
        if activity_task is not None:
            activity_task.cancel()
            with contextlib.suppress(BaseException):
                await activity_task
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
