"""SEG-17: stored secrets on their own key, not on the session secret."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from ocabra.config import settings
from ocabra.core import secrets_box


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "old-session-secret")
    monkeypatch.setattr(settings, "ocabra_secrets_key", "dedicated-key")
    old = Fernet(secrets_box.derive_key("old-session-secret"))
    new = Fernet(secrets_box.derive_key("dedicated-key"))
    return old, new


def test_new_secrets_go_on_the_dedicated_key_and_old_ones_still_read(keys):
    old, new = keys
    box = secrets_box.fernet()

    assert new.decrypt(box.encrypt(b"peer-key")) == b"peer-key"
    assert box.decrypt(old.encrypt(b"legacy")) == b"legacy"


def test_without_a_dedicated_key_it_works_as_before(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "old-session-secret")
    monkeypatch.setattr(settings, "ocabra_secrets_key", "")
    old = Fernet(secrets_box.derive_key("old-session-secret"))

    assert old.decrypt(secrets_box.fernet().encrypt(b"x")) == b"x"


def test_the_secrets_key_is_never_stored_in_the_database():
    from ocabra.db.server_config import _ENV_ONLY_FIELDS

    assert "ocabra_secrets_key" in _ENV_ONLY_FIELDS


class _Session:
    """Just enough of an AsyncSession for rotate_stored_secrets."""

    def __init__(self, tables: dict[str, list[tuple]]):
        self.tables = tables
        self.updates: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, params=None):
        sql = str(statement)
        if "information_schema" in sql:
            return _Result(scalar=1 if params["t"] in self.tables else 0)
        if sql.startswith("select"):
            table = sql.rsplit(" from ", 1)[1].strip()
            return _Result(rows=self.tables[table])
        self.updates.append((sql, dict(params)))
        return _Result()

    async def commit(self):
        return None


class _Result:
    def __init__(self, rows=None, scalar=None):
        self._rows, self._scalar = rows or [], scalar

    def all(self):
        return self._rows

    def scalar(self):
        return self._scalar


@pytest.mark.asyncio
async def test_rotation_moves_old_secrets_to_the_new_key_and_keeps_the_rest(keys):
    old, new = keys
    already = new.encrypt(b"already-moved").decode()
    legacy = old.encrypt(b"peer-api-key").decode()
    garbage = "not-a-fernet-token"
    session = _Session(
        {
            "federation_peers": [("p1", legacy, already, None)],
            "mcp_servers": [("m1", garbage, None)],
        }
    )

    rotated = await secrets_box.rotate_stored_secrets(lambda: session)

    assert rotated == 1
    (sql, params), = session.updates
    assert "api_key_encrypted = :api_key_encrypted" in sql and "local_api" not in sql
    assert new.decrypt(params["api_key_encrypted"].encode()) == b"peer-api-key"


@pytest.mark.asyncio
async def test_rotation_does_nothing_without_a_dedicated_key(monkeypatch):
    monkeypatch.setattr(settings, "ocabra_secrets_key", "")

    assert await secrets_box.rotate_stored_secrets(lambda: pytest.fail("no DB access")) == 0
