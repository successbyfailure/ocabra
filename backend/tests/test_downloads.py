import asyncio
from pathlib import Path

import pytest

from ocabra.api.internal.downloads import DownloadManager


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, dict] = {}

    async def keys(self, pattern: str) -> list[str]:
        if pattern == "download:job:*":
            return [k for k in self.store if k.startswith("download:job:")]
        return []


@pytest.mark.asyncio
async def test_download_manager_enqueue_progress_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_redis = FakeRedis()
    queue: list[dict] = []
    published: list[tuple[str, dict]] = []

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = ttl
        fake_redis.store[key] = data

    async def fake_get_key(key: str):
        return fake_redis.store.get(key)

    async def fake_lpush(_queue_name: str, data: dict) -> None:
        queue.insert(0, data)

    async def fake_rpop(_queue_name: str):
        if queue:
            return queue.pop()
        await asyncio.sleep(0)
        return None

    async def fake_publish(channel: str, data: dict) -> None:
        published.append((channel, data))

    async def fake_hf_download(repo_id: str, target_dir: Path, progress_callback, artifact=None):
        _ = repo_id, target_dir, artifact
        progress_callback(25.0, 10.0)
        await asyncio.sleep(0)
        progress_callback(80.0, 8.0)
        return Path("/tmp/model")

    monkeypatch.setattr("ocabra.api.internal.downloads.set_key", fake_set_key)
    monkeypatch.setattr("ocabra.api.internal.downloads.get_key", fake_get_key)
    monkeypatch.setattr("ocabra.api.internal.downloads.lpush", fake_lpush)
    monkeypatch.setattr("ocabra.api.internal.downloads.rpop", fake_rpop)
    monkeypatch.setattr("ocabra.api.internal.downloads.publish", fake_publish)
    monkeypatch.setattr("ocabra.api.internal.downloads.get_redis_safe", lambda: fake_redis)

    manager = DownloadManager()
    monkeypatch.setattr(manager._hf_registry, "download", fake_hf_download)

    job = await manager.enqueue(source="huggingface", model_ref="org/model")
    await manager._execute_job(job)

    stored = await manager.get_job(job.job_id)
    assert stored is not None
    assert stored.status == "completed"
    assert stored.progress_pct == 100.0
    assert stored.artifact is None
    assert any(ch == f"download:progress:{job.job_id}" for ch, _ in published)


@pytest.mark.asyncio
async def test_download_manager_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_redis = FakeRedis()
    queue: list[dict] = []

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = ttl
        fake_redis.store[key] = data

    async def fake_get_key(key: str):
        return fake_redis.store.get(key)

    async def fake_lpush(_queue_name: str, data: dict) -> None:
        queue.insert(0, data)

    async def fake_rpop(_queue_name: str):
        if queue:
            return queue.pop()
        return None

    async def fake_publish(_channel: str, _data: dict) -> None:
        return None

    monkeypatch.setattr("ocabra.api.internal.downloads.set_key", fake_set_key)
    monkeypatch.setattr("ocabra.api.internal.downloads.get_key", fake_get_key)
    monkeypatch.setattr("ocabra.api.internal.downloads.lpush", fake_lpush)
    monkeypatch.setattr("ocabra.api.internal.downloads.rpop", fake_rpop)
    monkeypatch.setattr("ocabra.api.internal.downloads.publish", fake_publish)
    monkeypatch.setattr("ocabra.api.internal.downloads.get_redis_safe", lambda: fake_redis)

    manager = DownloadManager()
    job = await manager.enqueue(source="ollama", model_ref="llama3.2:3b")
    await manager.cancel(job.job_id)

    stored = await manager.get_job(job.job_id)
    assert stored is not None
    assert stored.status == "cancelled"
    assert stored.completed_at is not None
