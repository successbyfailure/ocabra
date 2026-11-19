"""``ocabra`` live status of an unfinished batch (api/openai/batches.py)."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ocabra.api.openai import batches as api
from ocabra.core.batch_processor import BatchProcessor


def _batch(status="in_progress"):
    return SimpleNamespace(id=uuid.uuid4(), status=status, created_at=datetime.now(UTC))


class _Session:
    def __init__(self, ahead=0):
        self.ahead = ahead

    async def execute(self, stmt):
        return SimpleNamespace(scalar_one=lambda: self.ahead)


def _request(processor, states=None, typical=None, started=None):
    mm = SimpleNamespace(
        _states=states or {},
        get_expected_load_seconds=lambda m: _aw(typical),
        load_started_at=lambda key: started,
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(batch_processor=processor, model_manager=mm))
    )


async def _aw(v):
    return v


def _processor(batch_id, **live):
    p = BatchProcessor.__new__(BatchProcessor)
    p._live = {}
    if live is not None:
        entry = BatchProcessor._new_live()
        entry.update(live)
        p._live[str(batch_id)] = entry
    return p


@pytest.mark.asyncio
async def test_a_batch_behind_others_is_queued():
    b = _batch("validating")
    p = BatchProcessor.__new__(BatchProcessor)
    p._live = {}
    out = await api._live_field(_Session(ahead=2), b, _request(p))
    assert out["phase"] == "queued" and out["queue_position"] == 2


@pytest.mark.asyncio
async def test_lines_waiting_for_their_model_to_load():
    b = _batch()
    model = "llamacpp_mks/gemma4:12b"
    p = _processor(b.id, in_flight=4, dispatch_models={model}, started=True)
    states = {
        model + "::abc": SimpleNamespace(
            model_id=model + "::abc", status=SimpleNamespace(value="loading")
        )
    }
    out = await api._live_field(
        _Session(), b, _request(p, states, typical=80, started=time.time() - 30)
    )
    assert out["phase"] == "loading_model" and out["in_flight"] == 4
    [m] = out["models"]
    assert m["status"] == "loading" and 45 <= m["expected_wait_seconds"] <= 50
    assert "about" in out["detail"]


@pytest.mark.asyncio
async def test_yielding_to_interactive_traffic():
    b = _batch()
    model = "llamacpp_mks/qwen3.8:27b"
    p = _processor(
        b.id,
        waiting=10,
        preemptions=3,
        cooldown_models={model},
        retry_at=time.monotonic() + 4,
        started=True,
    )
    states = {model: SimpleNamespace(model_id=model, status=SimpleNamespace(value="loaded"))}
    out = await api._live_field(_Session(), b, _request(p, states))
    assert out["phase"] == "yielding_to_interactive"
    assert out["waiting"] == 10 and out["preemptions"] == 3 and out["retry_in_seconds"] in (3, 4)


@pytest.mark.asyncio
async def test_finished_batches_have_no_live_status():
    for status in ("completed", "failed", "expired", "cancelled"):
        assert await api._live_field(_Session(), _batch(status), _request(None)) is None
    out = await api._live_field(_Session(), _batch("finalizing"), _request(None))
    assert out["phase"] == "finalizing"


class _FileSession:
    def __init__(self, file):
        self.file = file

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, model, key):
        return self.file


@pytest.mark.asyncio
@pytest.mark.parametrize(("lines", "code"), [(5, "batch_too_large"), (0, "empty_batch")])
async def test_create_refuses_too_many_or_no_requests(tmp_path, monkeypatch, lines, code):
    from fastapi import HTTPException

    from ocabra.config import settings

    path = tmp_path / "in.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "custom_id": f"r{i}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {"model": "m", "input": "x"},
                }
            )
            + "\n"
            for i in range(lines)
        )
        + "\n"
    )
    uid = uuid.uuid4()
    file = SimpleNamespace(user_id=uid, purpose="batch", storage_path=str(path))
    monkeypatch.setattr(api, "AsyncSessionLocal", lambda: _FileSession(file))
    monkeypatch.setattr(settings, "batch_max_requests", 3)
    body = api.BatchCreateBody(input_file_id=f"file-{uuid.uuid4()}", endpoint="/v1/embeddings")
    user = SimpleNamespace(is_anonymous=False, user_id=str(uid), is_admin=False, role="user")
    with pytest.raises(HTTPException) as exc:
        await api.create_batch(
            body, user, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        )
    assert exc.value.status_code == 400 and exc.value.detail["error"]["code"] == code
