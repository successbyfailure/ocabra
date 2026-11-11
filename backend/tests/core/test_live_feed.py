"""Feed de peticiones recién terminadas del dashboard (stats/live_feed.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ocabra.stats import live_feed


def test_recent_finished_window_order_and_limit(monkeypatch):
    live_feed.clear()
    clock = {"t": 1000.0}
    monkeypatch.setattr(live_feed.time, "monotonic", lambda: clock["t"])
    for i in range(5):
        live_feed.record_finished({"requestId": str(i)})
        clock["t"] += 1
    clock["t"] = 1023.0
    # Ventana de 20 s: solo las dos últimas (terminadas a los 1003 y 1004).
    assert [i["requestId"] for i in live_feed.recent_finished(window_s=20.0)] == ["4", "3"]
    # Ventana amplia: manda el límite, la más reciente primero.
    items = live_feed.recent_finished(window_s=60.0, limit=3)
    assert [i["requestId"] for i in items] == ["4", "3", "2"]
    assert items[0]["endedAgoMs"] == 19_000
    assert all(not k.startswith("_") for k in items[0])


@pytest.mark.asyncio
async def test_record_stat_feeds_finished_requests_even_without_db():
    """El middleware apunta la petición antes de escribir en la BD (y aunque
    esta falle), sin IP ni user agent: lo ve cualquier usuario."""
    from ocabra.stats.collector import _record_stat

    live_feed.clear()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace()),
        state=SimpleNamespace(
            auth_user=SimpleNamespace(
                user_id=None,
                is_anonymous=False,
                username="luis",
                api_key_name="opencode",
                key_group_id=None,
            ),
            inflight_request_id="abc",
            runtime_telemetry=None,
        ),
        headers={"user-agent": "secreto/1.0"},
        client=SimpleNamespace(host="10.0.0.7"),
    )
    await _record_stat(
        request=request,
        model_id="gemma4:12b",
        started_at=datetime.now(UTC),
        duration_ms=3.0,
        error_message="model_at_capacity",
        status_code=429,
        endpoint_path="/v1/embeddings",
        request_kind="embedding",
        input_tokens=None,
        output_tokens=None,
    )
    [item] = live_feed.recent_finished()
    assert item["statusCode"] == 429
    assert item["username"] == "luis" and item["apiKeyName"] == "opencode"
    assert item["requestKind"] == "embedding"
    assert "10.0.0.7" not in str(item) and "secreto" not in str(item)


@pytest.mark.asyncio
async def test_finished_transcription_carries_audio_seconds():
    from ocabra.stats.collector import _record_stat

    live_feed.clear()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace()),
        state=SimpleNamespace(auth_user=None, inflight_request_id="t", runtime_telemetry=None),
        headers={},
        client=None,
    )
    await _record_stat(
        request=request,
        model_id="whisper-large-v3-turbo",
        started_at=datetime.now(UTC),
        duration_ms=51_000.0,
        error_message=None,
        status_code=200,
        endpoint_path="/v1/audio/transcriptions",
        request_kind="audio_transcription",
        input_tokens=None,
        output_tokens=1200,
        work_size_meta={"audio_seconds": 720.0},
    )
    [item] = live_feed.recent_finished()
    assert item["audioSeconds"] == 720.0
