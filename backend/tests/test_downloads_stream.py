"""El SSE de progreso de una descarga termina solo al llegar a un estado final,
y leer descargas (lista, detalle, stream) pide el mismo rol."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest

from ocabra.api.internal import downloads as downloads_api
from ocabra.schemas.registry import DownloadJob


def _job(status: str, pct: float = 0.0) -> DownloadJob:
    return DownloadJob(
        job_id="job-1",
        source="huggingface",
        model_ref="Qwen/Qwen3-8B",
        status=status,
        progress_pct=pct,
        speed_mb_s=None,
        eta_seconds=None,
        error=None,
        started_at=datetime.now(UTC),
        completed_at=None,
    )


class _FakePubSub:
    def __init__(self, messages: list[dict | None]) -> None:
        self._messages = list(messages)
        self.reads = 0

    async def get_message(self, **_kwargs):
        self.reads += 1
        if self._messages:
            return self._messages.pop(0)
        return None


def _patch(monkeypatch, pubsub: _FakePubSub, stored: list[DownloadJob | None]) -> None:
    @asynccontextmanager
    async def fake_subscribe(channel):
        yield pubsub

    async def fake_get_job(job_id):
        return stored[0] if len(stored) == 1 else stored.pop(0)

    monkeypatch.setattr(downloads_api, "subscribe", fake_subscribe)
    monkeypatch.setattr(downloads_api.download_manager, "get_job", fake_get_job)


async def _collect(limit: int = 50) -> list[dict]:
    events = []
    async for chunk in downloads_api._download_event_stream("job-1", "download:progress:job-1"):
        assert chunk.startswith("data: ") and chunk.endswith("\n\n")
        events.append(json.loads(chunk[len("data: ") :]))
        if len(events) >= limit:
            raise AssertionError("stream did not end")
    return events


def _msg(job: DownloadJob) -> dict:
    return {"type": "message", "data": job.model_dump_json()}


async def test_stream_ends_on_terminal_update(monkeypatch) -> None:
    pubsub = _FakePubSub(
        [_msg(_job("downloading", 40)), None, _msg(_job("completed", 100)), _msg(_job("failed"))]
    )
    _patch(monkeypatch, pubsub, [_job("queued")])

    events = await _collect()

    assert [e["status"] for e in events] == ["queued", "downloading", "completed"]


async def test_stream_of_finished_job_sends_snapshot_and_ends(monkeypatch) -> None:
    pubsub = _FakePubSub([])
    _patch(monkeypatch, pubsub, [_job("cancelled")])

    events = await _collect()

    assert [e["status"] for e in events] == ["cancelled"]
    assert pubsub.reads == 0


async def test_stream_notices_a_missed_terminal_update(monkeypatch) -> None:
    """If the terminal publish is lost, the periodic re-read still ends it."""
    monkeypatch.setattr(downloads_api, "_STREAM_RECHECK_TICKS", 3)
    pubsub = _FakePubSub([])
    _patch(monkeypatch, pubsub, [_job("downloading"), _job("failed")])

    events = await _collect()

    assert [e["status"] for e in events] == ["downloading", "failed"]
    assert pubsub.reads == 3


@pytest.mark.parametrize(
    "endpoint",
    [
        downloads_api.list_downloads,
        downloads_api.get_download,
        downloads_api.stream_download_progress,
    ],
)
def test_reading_downloads_needs_the_same_role(endpoint) -> None:
    import inspect

    dep = inspect.signature(endpoint).parameters["_user"].default.dependency
    # ``require_role`` closes over the minimum role.
    roles = [c.cell_contents for c in (dep.__closure__ or []) if isinstance(c.cell_contents, str)]
    assert roles == ["user"]
