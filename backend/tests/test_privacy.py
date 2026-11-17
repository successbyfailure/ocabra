"""Privacy modes (core/privacy.py) and file retention (core/files_retention.py)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from ocabra.config import settings
from ocabra.core import privacy
from tests.conftest import requires_db


@pytest.fixture
def zero_data(monkeypatch):
    monkeypatch.setattr(settings, "privacy_mode", "zero_data")


def test_mode_defaults_to_functional(monkeypatch):
    monkeypatch.setattr(settings, "privacy_mode", "nonsense")
    assert privacy.mode() == "functional" and not privacy.zero_data()


def test_preview_is_empty_in_zero_data(zero_data):
    assert privacy.preview("the secret answer") is None


def test_preview_kept_in_functional():
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


async def test_batches_and_files_allowed_in_functional():
    await privacy.require_storage_allowed()  # no exception


def test_images_inline_in_zero_data(zero_data):
    from ocabra.api.openai.images import _default_response_format

    assert _default_response_format("dall-e-3", None) == "b64_json"
    assert _default_response_format("x", "url") == "b64_json"


def test_images_by_url_in_functional():
    from ocabra.api.openai.images import _default_response_format

    assert _default_response_format("dall-e-3", None) == "url"


def test_no_tracing_in_zero_data(zero_data, monkeypatch):
    from ocabra.integrations import langfuse_tracer

    monkeypatch.setattr(settings, "langfuse_enabled", True)
    assert langfuse_tracer._get_client() is None


# ── stats ────────────────────────────────────────────────────────


@requires_db
@pytest.mark.parametrize("mode, rows", [("zero_data", 0), ("functional", 1)])
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
    # The control (functional) proves the call really reaches the write.
    assert n == rows
    if mode == "zero_data" and seen:  # the feed still counts it, without who or what
        assert seen[0]["username"] is None and seen[0]["apiKeyName"] is None
        assert seen[0]["error"] is None


# ── file retention ───────────────────────────────────────────────


@requires_db
async def test_retention_purges_finished_batches_and_old_uploads(tmp_path, monkeypatch):
    import os

    from ocabra.core.files_retention import purge_expired
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from ocabra.db.openai_batches import OpenAIBatch, OpenAIFile

    monkeypatch.setattr(settings, "openai_files_dir", str(tmp_path))
    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    recent = now - timedelta(hours=2)
    uid = uuid.uuid4()
    # On disk with no row: an old one goes, a fresh one stays.
    (tmp_path / "orphan_old").write_text("x")
    os.utime(tmp_path / "orphan_old", (old.timestamp(), old.timestamp()))
    (tmp_path / "orphan_new").write_text("x")

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
        assert out == {"batches": 1, "files": 3, "orphans": 1}
        assert not (tmp_path / "orphan_old").exists() and (tmp_path / "orphan_new").exists()
        assert left == {"live_in", "recent_in", "loose_new"}
        assert statuses == ["completed", "in_progress"]  # the recent one and the running one
        assert not (tmp_path / "done_out").exists() and not (tmp_path / "loose_old").exists()
        assert (tmp_path / "live_in").exists()
        assert await purge_expired(AsyncSessionLocal, 0, now=now) == {
            "batches": 0,
            "files": 0,
            "orphans": 0,
        }
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


def test_zero_data_temp_files_go_to_memory(monkeypatch):
    """Uploads above 1 MB and the media workers process use ``tempfile``:
    in zero data they must land in RAM, never on the node's disk."""
    import os
    import subprocess
    import sys
    import tempfile

    if not os.path.isdir("/dev/shm"):
        pytest.skip("no /dev/shm here")
    # Own directory: the real one may already exist on this host, owned by
    # another user (a container run as root), and the test is not about that.
    monkeypatch.setattr(privacy, "MEMORY_TEMP_DIR", f"/dev/shm/ocabra-tmp-test-{os.getpid()}")
    monkeypatch.setattr(settings, "privacy_mode", "zero_data")
    try:
        status = privacy.apply_temp_dir()
        assert status["in_memory"] and tempfile.gettempdir() == privacy.MEMORY_TEMP_DIR
        # A spooled upload that rolls over to a real file (what FastAPI does
        # past 1 MB) lands in memory too.
        spool = tempfile.SpooledTemporaryFile(max_size=10)
        spool.write(b"x" * 100)
        spool.rollover()
        assert os.readlink(f"/proc/self/fd/{spool.fileno()}").startswith("/dev/shm/")
        spool.close()
        # Workers are subprocesses: they inherit it.
        out = subprocess.run(
            [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert out == privacy.MEMORY_TEMP_DIR
    finally:
        monkeypatch.setattr(settings, "privacy_mode", "functional")
        status = privacy.apply_temp_dir()
        import shutil

        shutil.rmtree(privacy.MEMORY_TEMP_DIR, ignore_errors=True)
    assert not status["in_memory"]


async def test_zero_data_keeps_the_last_10_requests_in_memory_without_identity(monkeypatch):
    """«0 data»: nothing reaches the database, but the last 10 requests'
    metrics are shown — never who asked or the error text."""
    from ocabra.api.internal import stats as stats_api
    from ocabra.stats import recent_memory

    recent_memory.clear()
    for i in range(12):
        recent_memory.record(
            {
                "id": str(i),
                "modelId": "m",
                "statusCode": 500 if i == 11 else 200,
                "durationMs": 100 * (i + 1),
                "inputTokens": 10,
                "outputTokens": 5,
                "predictedTokensPerSecond": 50.0,
                "username": "javi",
                "apiKeyName": "clave",
                "error": "el prompt decía…",
                "clientAddr": "10.0.0.1",
            }
        )
    rows = recent_memory.recent()
    assert [r["id"] for r in rows] == [str(i) for i in range(11, 1, -1)]  # newest first, 10 kept
    assert all(r["username"] is None and r["apiKeyName"] is None for r in rows)
    assert all(r["error"] is None and r["clientAddr"] is None for r in rows)
    summary = recent_memory.summary()
    assert summary["requests"] == 10 and summary["errors"] == 1
    assert summary["inputTokens"] == 100 and summary["avgPredictedTokensPerSecond"] == 50.0

    monkeypatch.setattr(settings, "privacy_mode", "zero_data")
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model_manager=None)))
    out = await stats_api.recent_requests(req, limit=3, user=SimpleNamespace(is_admin=True))
    assert [r["id"] for r in out["requests"]] == ["11", "10", "9"]
    assert out["zeroData"]["kept"] == 10
    recent_memory.clear()


async def test_record_stat_in_zero_data_fills_the_memory_buffer(monkeypatch):
    """The real path: ``_record_stat`` in zero data keeps the metrics (no DB)."""
    from ocabra.api._deps_auth import UserContext
    from ocabra.stats import collector, recent_memory

    monkeypatch.setattr(settings, "privacy_mode", "zero_data")
    recent_memory.clear()
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
    await collector._record_stat(
        request=request,
        model_id="gemma4:26b",
        started_at=datetime.now(UTC),
        duration_ms=1234,
        error_message="input echoed: my secret",
        status_code=200,
        endpoint_path="/v1/chat/completions",
        request_kind="chat",
        input_tokens=40,
        output_tokens=12,
        predicted_tokens_per_second=61.5,
    )
    (row,) = recent_memory.recent()
    assert row["requestedModel"] == "gemma4:26b" and row["durationMs"] == 1234
    assert row["outputTokens"] == 12 and row["predictedTokensPerSecond"] == 61.5
    assert row["username"] is None and row["apiKeyName"] is None and row["error"] is None
    recent_memory.clear()
