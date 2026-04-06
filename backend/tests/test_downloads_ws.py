from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api.internal.downloads import DownloadJob, DownloadManager
from ocabra.api.internal.ws import router as ws_router


def _sample_job(job_id: str = "job-1") -> DownloadJob:
    return DownloadJob(
        job_id=job_id,
        source="huggingface",
        model_ref="Qwen/Qwen3-8B-Instruct",
        artifact=None,
        register_config=None,
        status="downloading",
        progress_pct=42.5,
        speed_mb_s=12.0,
        eta_seconds=90,
        error=None,
        started_at=datetime(2026, 3, 23, tzinfo=UTC),
        completed_at=None,
    )


class _FakePubSub:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.subscribed_channels: tuple[str, ...] = ()
        self.unsubscribed_channels: tuple[str, ...] = ()
        self.closed = False

    async def subscribe(self, *channels: str) -> None:
        self.subscribed_channels = channels

    async def listen(self):
        yield {
            "type": "message",
            "channel": "download:progress",
            "data": self._payload,
        }
        while True:
            await asyncio.sleep(3600)

    async def unsubscribe(self, *channels: str) -> None:
        self.unsubscribed_channels = channels

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_download_manager_publishes_global_and_job_channels() -> None:
    manager = DownloadManager()
    job = _sample_job()

    with patch("ocabra.api.internal.downloads.publish", new=AsyncMock()) as publish_mock:
        await manager._publish_job_update(job)

    channels = [call.args[0] for call in publish_mock.await_args_list]
    assert channels == ["download:progress", f"download:progress:{job.job_id}"]


def test_websocket_forwards_global_download_progress() -> None:
    from ocabra.core.auth_manager import create_access_token

    app = FastAPI()
    app.include_router(ws_router, prefix="/ocabra")

    token = create_access_token(user_id="test-user", role="user")

    job = DownloadJob(
        job_id="job-2",
        source="ollama",
        model_ref="llama3.2:3b",
        artifact=None,
        register_config=None,
        status="queued",
        progress_pct=0.0,
        speed_mb_s=None,
        eta_seconds=None,
        error=None,
        started_at=datetime(2026, 3, 23, tzinfo=UTC),
        completed_at=None,
    )
    payload = job.model_dump_json()
    fake_pubsub = _FakePubSub(payload)
    fake_redis = SimpleNamespace(pubsub=lambda: fake_pubsub)

    with patch("ocabra.api.internal.ws.get_redis", return_value=fake_redis):
        with TestClient(app, cookies={"ocabra_session": token}) as client:
            with client.websocket_connect("/ocabra/ws") as websocket:
                message = websocket.receive_json()

    assert message["type"] == "download_progress"
    assert message["data"]["job_id"] == job.job_id
    assert "download:progress" in fake_pubsub.subscribed_channels
    assert fake_pubsub.closed is True
