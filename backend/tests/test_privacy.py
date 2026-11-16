"""Privacy modes (core/privacy.py) and file retention (core/files_retention.py)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from ocabra.config import settings
from ocabra.core import privacy
from tests.conftest import requires_db


@pytest.fixture
def zero_data(monkeypatch):
    monkeypatch.setattr(settings, "privacy_mode", "zero_data")


def test_mode_defaults_to_optimal(monkeypatch):
    monkeypatch.setattr(settings, "privacy_mode", "nonsense")
    assert privacy.mode() == "optimal" and not privacy.zero_data()


def test_preview_is_empty_in_zero_data(zero_data):
    assert privacy.preview("the secret answer") is None


def test_preview_kept_in_optimal():
    assert privacy.preview("x" * 500) == "x" * 200


async def test_header_tells_the_mode(zero_data):
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message):
        sent.append(message)

    await privacy.PrivacyHeaderMiddleware(app)({"type": "http"}, None, send)
    assert (b"x-ocabra-privacy", b"zero-data") in sent[0]["headers"]


async def test_batches_and_files_refused_in_zero_data(zero_data):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await privacy.require_storage_allowed()
    assert exc.value.status_code == 403
    assert exc.value.detail["error"]["code"] == "zero_data_mode"


async def test_batches_and_files_allowed_in_optimal():
    await privacy.require_storage_allowed()  # no exception


def test_images_inline_in_zero_data(zero_data):
    from ocabra.api.openai.images import _default_response_format

    assert _default_response_format("dall-e-3", None) == "b64_json"
    assert _default_response_format("x", "url") == "b64_json"


def test_images_by_url_in_optimal():
    from ocabra.api.openai.images import _default_response_format

    assert _default_response_format("dall-e-3", None) == "url"


def test_no_tracing_in_zero_data(zero_data, monkeypatch):
    from ocabra.integrations import langfuse_tracer

    monkeypatch.setattr(settings, "langfuse_enabled", True)
    assert langfuse_tracer._get_client() is None


# ── stats ────────────────────────────────────────────────────────


@requires_db
@pytest.mark.parametrize("mode, rows", [("zero_data", 0), ("optimal", 1)])
async def test_zero_data_writes_no_request_row(mode, rows, monkeypatch):
    monkeypatch.setattr(settings, "privacy_mode", mode)
    from types import SimpleNamespace

    from ocabra.database import AsyncSessionLocal
    from ocabra.db.stats import RequestStat
    from ocabra.stats import collector, live_feed

    marker = f"zz-privacy-{uuid.uuid4().hex[:6]}"
    from ocabra.api._deps_auth import UserContext

    user = UserContext(
        user_id=None,
        username="alice",
        role="user",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
        api_key_name="k",
    )
    request = SimpleNamespace(
        state=SimpleNamespace(auth_user=user),
        headers={},
        client=SimpleNamespace(host="1.2.3.4"),
        app=SimpleNamespace(state=SimpleNamespace()),
    )
    seen = []
    orig = live_feed.record_finished
    live_feed.record_finished = lambda item: seen.append(item)
    try:
        await collector._record_stat(
            request=request,
            model_id=marker,
            started_at=datetime.now(UTC),
            duration_ms=10,
            error_message="input echoed: my secret",
            status_code=400,
            endpoint_path="/v1/chat/completions",
            request_kind="chat",
            input_tokens=1,
            output_tokens=1,
        )
    finally:
        live_feed.record_finished = orig
    async with AsyncSessionLocal() as s:
        n = (
            await s.execute(
                sa.select(sa.func.count())
                .select_from(RequestStat)
                .where(RequestStat.requested_model == marker)
            )
        ).scalar_one()
    async with AsyncSessionLocal() as s:
        await s.execute(sa.delete(RequestStat).where(RequestStat.requested_model == marker))
        await s.commit()
    # The control (optimal) proves the call really reaches the write.
    assert n == rows
    if mode == "zero_data" and seen:  # the feed still counts it, without who or what
        assert seen[0]["username"] is None and seen[0]["apiKeyName"] is None
        assert seen[0]["error"] is None


# ── file retention ───────────────────────────────────────────────


@requires_db
async def test_retention_purges_finished_batches_and_old_uploads(tmp_path):
    from ocabra.core.files_retention import purge_expired
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from ocabra.db.openai_batches import OpenAIBatch, OpenAIFile

    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    recent = now - timedelta(hours=2)
    uid = uuid.uuid4()

    def mkfile(name, created):
        path = tmp_path / name
        path.write_text("data")
        return OpenAIFile(
            id=uuid.uuid4(),
            user_id=uid,
            filename=name,
            bytes=4,
            purpose="batch",
            storage_path=str(path),
            created_at=created,
        )

    f_done_in, f_done_out = mkfile("done_in", old), mkfile("done_out", old)
    f_live_in = mkfile("live_in", old)  # old, but a running batch still reads it
    f_recent_in = mkfile("recent_in", old)
    f_loose_old, f_loose_new = mkfile("loose_old", old), mkfile("loose_new", recent)
    async with AsyncSessionLocal() as s:
        s.add(User(id=uid, username=f"zz-ret-{uid.hex[:6]}", hashed_password="x", role="user"))
        await s.flush()
        s.add_all([f_done_in, f_done_out, f_live_in, f_recent_in, f_loose_old, f_loose_new])
        await s.flush()
        s.add_all(
            [
                OpenAIBatch(
                    user_id=uid,
                    endpoint="/v1/chat/completions",
                    input_file_id=f_done_in.id,
                    output_file_id=f_done_out.id,
                    status="completed",
                    completed_at=old,
                ),
                OpenAIBatch(
                    user_id=uid,
                    endpoint="/v1/chat/completions",
                    input_file_id=f_live_in.id,
                    status="in_progress",
                ),
                OpenAIBatch(
                    user_id=uid,
                    endpoint="/v1/chat/completions",
                    input_file_id=f_recent_in.id,
                    status="completed",
                    completed_at=recent,
                ),
            ]
        )
        await s.commit()
    try:
        out = await purge_expired(AsyncSessionLocal, 24 * 7, now=now)
        async with AsyncSessionLocal() as s:
            left = set(
                (
                    await s.execute(sa.select(OpenAIFile.filename).where(OpenAIFile.user_id == uid))
                ).scalars()
            )
            statuses = sorted(
                (
                    await s.execute(sa.select(OpenAIBatch.status).where(OpenAIBatch.user_id == uid))
                ).scalars()
            )
        assert out == {"batches": 1, "files": 3}
        assert left == {"live_in", "recent_in", "loose_new"}
        assert statuses == ["completed", "in_progress"]  # the recent one and the running one
        assert not (tmp_path / "done_out").exists() and not (tmp_path / "loose_old").exists()
        assert (tmp_path / "live_in").exists()
        assert await purge_expired(AsyncSessionLocal, 0, now=now) == {"batches": 0, "files": 0}
    finally:
        async with AsyncSessionLocal() as s:
            await s.execute(sa.delete(OpenAIBatch).where(OpenAIBatch.user_id == uid))
            await s.execute(sa.delete(OpenAIFile).where(OpenAIFile.user_id == uid))
            await s.execute(sa.delete(User).where(User.id == uid))
            await s.commit()


def test_in_flight_rows_drop_who_asked_in_zero_data(zero_data):
    from tests.test_vram_eviction import _add_loaded_model, _make_mm

    mm = _make_mm()
    _add_loaded_model(mm, "m")
    rid = mm.try_begin_request("m", 4, user_id="u1", username="alice", api_key_name="k")
    active = mm._active_requests[rid]
    assert active.username is None and active.api_key_name is None
    assert active.user_id == "u1"  # fair share between users still needs it
