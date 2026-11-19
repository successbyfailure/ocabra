"""GET /v1/batches looks up the owners of a page in one query (INF-13).

The payload must be exactly what the per-batch path builds (``user`` and the
``ocabra`` live field of unfinished batches included).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ocabra.api.openai import batches as api
from ocabra.core.batch_processor import BatchProcessor
from ocabra.db.openai_batches import OpenAIBatch

ALICE, BOB, GONE = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
NAMES = {ALICE: "alice", BOB: "bob"}  # GONE: owner no longer in users


def _batch(owner, status, minutes_ago):
    return OpenAIBatch(
        id=uuid.uuid4(),
        user_id=owner,
        endpoint="/v1/chat/completions",
        input_file_id=uuid.uuid4(),
        completion_window="24h",
        status=status,
        request_total=10,
        request_completed=3,
        request_failed=1,
        total_tokens=0,
        created_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
    )


class _Session:
    """Answers the three queries the listing can run; counts them."""

    def __init__(self, rows):
        self.rows = rows
        self.queries: list[str] = []

    async def execute(self, stmt):
        sql = str(stmt)
        self.queries.append(sql)
        params = stmt.compile().params
        if "count(" in sql:  # _live_field: batches queued ahead
            return SimpleNamespace(scalar_one=lambda: 2)
        if "FROM openai_batches" in sql:  # the page itself
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self.rows))
        if "users.id," in sql:  # _usernames: the whole page at once
            ids = next(v for v in params.values() if isinstance(v, list | tuple | set))
            pairs = [(uid, NAMES[uid]) for uid in ids if uid in NAMES]
            return SimpleNamespace(all=lambda: pairs)
        uid = next(iter(params.values()))  # one owner (per-batch path)
        return SimpleNamespace(scalar_one_or_none=lambda: NAMES.get(uid))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _request(rows):
    processor = BatchProcessor.__new__(BatchProcessor)
    processor._live = {}
    running = rows[1]  # in_progress with a live snapshot
    entry = BatchProcessor._new_live()
    entry.update(in_flight=2, dispatch_models=["m"])
    processor._live[str(running.id)] = entry
    mm = SimpleNamespace(
        _states={},
        get_expected_load_seconds=lambda m: None,
        load_started_at=lambda key: None,
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(batch_processor=processor, model_manager=mm))
    )


@pytest.mark.asyncio
async def test_list_payload_is_identical_and_owners_take_one_query(monkeypatch):
    rows = [
        _batch(ALICE, "validating", 1),  # queued: no snapshot -> count query
        _batch(BOB, "in_progress", 2),  # running: live snapshot
        _batch(ALICE, "completed", 3),  # finished: ocabra None
        _batch(GONE, "failed", 4),  # owner deleted: username None
        _batch(BOB, "finalizing", 5),
    ]
    request = _request(rows)

    reference = [await api._with_username(_Session(rows), r, request) for r in rows]

    session = _Session(rows)
    monkeypatch.setattr(api, "AsyncSessionLocal", lambda: session)
    user = SimpleNamespace(user_id=str(uuid.uuid4()), role="system_admin")
    out = await api.list_batches(user=user, request=request, after=None, limit=20)

    assert out["data"] == reference
    assert [d["user"]["username"] for d in out["data"]] == ["alice", "bob", "alice", None, "bob"]
    assert out["data"][0]["ocabra"]["phase"] == "queued"
    assert out["data"][1]["ocabra"]["phase"] == "running"
    assert out["data"][2]["ocabra"] is None
    assert out["first_id"] == reference[0]["id"] and out["last_id"] == reference[-1]["id"]
    assert out["has_more"] is False
    # page + owners + the queued batch's count; before, one owner query per batch
    owner_queries = [q for q in session.queries if "FROM users" in q]
    assert len(owner_queries) == 1
    assert len(session.queries) == 3
