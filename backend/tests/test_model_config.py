import pytest

from ocabra.db.model_config import (
    global_schedule_payload_to_rows,
    global_schedule_rows_to_payload,
    replace_global_schedules,
)


def test_global_schedule_payload_roundtrip_cross_midnight() -> None:
    rows = global_schedule_payload_to_rows("night-window", [1, 2, 3], "22:00", "06:00", True)

    assert [row.action for row in rows] == ["evict_all", "reload"]
    assert rows[0].cron_expr == "0 22 * * 1,2,3"
    assert rows[1].cron_expr == "0 6 * * 2,3,4"

    payload = global_schedule_rows_to_payload(rows)
    assert payload == [
        {
            "id": "night-window",
            "days": [1, 2, 3],
            "start": "22:00",
            "end": "06:00",
            "enabled": True,
        }
    ]


@pytest.mark.asyncio
async def test_replace_global_schedules_replaces_existing_rows() -> None:
    existing_rows = global_schedule_payload_to_rows("old-window", [0, 6], "01:00", "03:00", True)
    new_payloads = [
        {
            "id": "new-window",
            "days": [1, 2],
            "start": "04:00",
            "end": "08:00",
            "enabled": False,
        }
    ]

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def scalars(self):
            return self

        def all(self):
            return list(self._rows)

    class FakeSession:
        def __init__(self, rows):
            self.rows = list(rows)
            self.deleted = []
            self.added = []

        async def execute(self, _query):
            return FakeResult(self.rows)

        async def delete(self, row):
            self.deleted.append(row)
            self.rows.remove(row)

        def add_all(self, rows):
            self.added.extend(rows)
            self.rows.extend(rows)

    session = FakeSession(existing_rows)
    await replace_global_schedules(session, new_payloads)

    assert len(session.deleted) == 2
    assert [row.label for row in session.added] == ["new-window", "new-window"]
    assert [row.action for row in session.added] == ["evict_all", "reload"]
    assert global_schedule_rows_to_payload(session.rows) == [
        {
            "id": "new-window",
            "days": [1, 2],
            "start": "04:00",
            "end": "08:00",
            "enabled": False,
        }
    ]
