import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ocabra.redis_client import get_redis

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        try:
            self._connections.remove(ws)
        except ValueError:
            pass

    async def broadcast(self, message: str) -> None:
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()

CHANNEL_EVENT_MAP = {
    "gpu:stats": "gpu_stats",
    "model:events": "model_event",
    "service:events": "service_event",
    "download:progress": "download_progress",
}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    pubsub = get_redis().pubsub()
    await pubsub.subscribe("gpu:stats", "model:events", "service:events")

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
            await manager.broadcast(json.dumps({"type": event_type, "data": data}))

    listener_task = asyncio.create_task(redis_listener())
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        listener_task.cancel()
        manager.disconnect(websocket)
        await pubsub.unsubscribe()
        await pubsub.aclose()
