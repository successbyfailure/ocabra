import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis

from ocabra.config import settings

_redis: aioredis.Redis | None = None


async def init_redis() -> None:
    global _redis
    _redis = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis


async def publish(channel: str, data: Any) -> None:
    """Publish a JSON-serializable payload to a Redis channel."""
    await get_redis().publish(channel, json.dumps(data))


async def set_key(key: str, data: Any, ttl: int | None = None) -> None:
    """Set a Redis key with optional TTL in seconds."""
    payload = json.dumps(data)
    if ttl:
        await get_redis().setex(key, ttl, payload)
    else:
        await get_redis().set(key, payload)


async def get_key(key: str) -> Any | None:
    """Get and JSON-decode a Redis key."""
    value = await get_redis().get(key)
    return json.loads(value) if value else None


async def delete_key(key: str) -> None:
    await get_redis().delete(key)


async def lpush(queue: str, data: Any) -> None:
    await get_redis().lpush(queue, json.dumps(data))


async def rpop(queue: str) -> Any | None:
    value = await get_redis().rpop(queue)
    return json.loads(value) if value else None


@asynccontextmanager
async def subscribe(channel: str) -> AsyncGenerator[aioredis.client.PubSub, None]:
    """Context manager for subscribing to a Redis channel."""
    pubsub = get_redis().pubsub()
    await pubsub.subscribe(channel)
    try:
        yield pubsub
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
