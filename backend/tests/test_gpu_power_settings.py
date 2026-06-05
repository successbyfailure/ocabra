"""Tests for persisted GPU power settings (DB layer + reapply logic)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_session_factory(rows: list):
    """Return a factory mimicking ``AsyncSessionLocal`` over an in-memory row list."""
    session = AsyncMock()
    session.commit = AsyncMock()
    upserts: list[dict] = []
    session._upserts = upserts

    async def execute(stmt):
        # ``upsert_setting`` builds a postgres INSERT...ON CONFLICT statement
        # which we can't run on a fake session; instead, capture the values
        # being upserted so the test can assert on them.
        try:
            compiled_params = stmt.compile().params  # type: ignore[attr-defined]
            upserts.append(dict(compiled_params))
        except Exception:
            pass
        result = MagicMock()
        result.scalars.return_value.all.return_value = rows
        return result

    session.execute = execute

    @asynccontextmanager
    async def factory():
        yield session

    factory.session = session  # type: ignore[attr-defined]
    return factory


@pytest.mark.asyncio
async def test_reapply_skips_when_no_pynvml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing pynvml → empty summary, no crash."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no nvml here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from ocabra.db.gpu_power_settings import reapply_persisted_settings

    factory = _make_session_factory([])
    summary = await reapply_persisted_settings(factory)
    assert summary == {"applied": 0, "skipped_no_match": 0, "errors": 0}


@pytest.mark.asyncio
async def test_reapply_no_rows_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """No persisted rows → no publishes, no warnings, zero counters."""
    fake_pynvml = SimpleNamespace(
        NVMLError=Exception,
        nvmlInit=MagicMock(),
        nvmlDeviceGetCount=MagicMock(return_value=1),
        nvmlDeviceGetHandleByIndex=MagicMock(return_value=MagicMock()),
        nvmlDeviceGetUUID=MagicMock(return_value="GPU-uuid-A"),
        nvmlDeviceGetName=MagicMock(return_value="RTX 3090"),
        NVML_FEATURE_ENABLED=1,
        NVML_FEATURE_DISABLED=0,
    )
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)

    publish_mock = AsyncMock()
    monkeypatch.setattr(
        "ocabra.redis_client.publish", publish_mock, raising=True
    )

    from ocabra.db.gpu_power_settings import reapply_persisted_settings

    factory = _make_session_factory([])
    summary = await reapply_persisted_settings(factory)
    assert summary == {"applied": 0, "skipped_no_match": 0, "errors": 0}
    publish_mock.assert_not_called()


@pytest.mark.asyncio
async def test_reapply_publishes_for_matching_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saved row whose UUID matches a live GPU → publish to gpu:set_power_limit."""
    fake_handle = MagicMock()
    fake_pynvml = SimpleNamespace(
        NVMLError=Exception,
        nvmlInit=MagicMock(),
        nvmlDeviceGetCount=MagicMock(return_value=2),
        nvmlDeviceGetHandleByIndex=MagicMock(return_value=fake_handle),
        nvmlDeviceGetUUID=MagicMock(side_effect=["GPU-A", "GPU-B"]),
        nvmlDeviceGetName=MagicMock(return_value="RTX 3090"),
        nvmlDeviceSetPersistenceMode=MagicMock(),
        NVML_FEATURE_ENABLED=1,
        NVML_FEATURE_DISABLED=0,
    )
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)

    publish_mock = AsyncMock()
    get_key_mock = AsyncMock(return_value="1")  # hw-monitor heartbeat present
    monkeypatch.setattr("ocabra.redis_client.publish", publish_mock, raising=True)
    monkeypatch.setattr("ocabra.redis_client.get_key", get_key_mock, raising=True)

    rows = [
        SimpleNamespace(
            gpu_uuid="GPU-B",
            power_limit_w=250,
            persistence_mode=True,
            last_known_index=1,
            last_known_name="RTX 3090",
        ),
    ]
    factory = _make_session_factory(rows)

    from ocabra.db.gpu_power_settings import reapply_persisted_settings

    summary = await reapply_persisted_settings(factory)
    assert summary == {"applied": 1, "skipped_no_match": 0, "errors": 0}
    publish_mock.assert_awaited_once()
    channel, payload = publish_mock.await_args.args
    assert channel == "gpu:set_power_limit"
    # GPU-B is index 1 in our fake inventory.
    assert payload == {"gpu_index": 1, "limit_w": 250}
    # Persistence mode set inline via NVML.
    fake_pynvml.nvmlDeviceSetPersistenceMode.assert_called_once_with(fake_handle, 1)


@pytest.mark.asyncio
async def test_reapply_counts_unknown_uuid_as_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saved row whose UUID is no longer in the inventory → skipped_no_match."""
    fake_pynvml = SimpleNamespace(
        NVMLError=Exception,
        nvmlInit=MagicMock(),
        nvmlDeviceGetCount=MagicMock(return_value=1),
        nvmlDeviceGetHandleByIndex=MagicMock(return_value=MagicMock()),
        nvmlDeviceGetUUID=MagicMock(return_value="GPU-A"),
        nvmlDeviceGetName=MagicMock(return_value="RTX 3090"),
        NVML_FEATURE_ENABLED=1,
        NVML_FEATURE_DISABLED=0,
    )
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)

    publish_mock = AsyncMock()
    get_key_mock = AsyncMock(return_value="1")
    monkeypatch.setattr("ocabra.redis_client.publish", publish_mock, raising=True)
    monkeypatch.setattr("ocabra.redis_client.get_key", get_key_mock, raising=True)

    rows = [
        SimpleNamespace(
            gpu_uuid="GPU-NOT-PRESENT",
            power_limit_w=250,
            persistence_mode=None,
            last_known_index=3,
            last_known_name="Removed",
        ),
    ]
    factory = _make_session_factory(rows)

    from ocabra.db.gpu_power_settings import reapply_persisted_settings

    summary = await reapply_persisted_settings(factory)
    assert summary == {"applied": 0, "skipped_no_match": 1, "errors": 0}
    publish_mock.assert_not_called()


@pytest.mark.asyncio
async def test_reapply_waits_for_hw_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """When hw-monitor heartbeat is absent the reapply gives up and counts errors."""
    fake_pynvml = SimpleNamespace(
        NVMLError=Exception,
        nvmlInit=MagicMock(),
        nvmlDeviceGetCount=MagicMock(return_value=1),
        nvmlDeviceGetHandleByIndex=MagicMock(return_value=MagicMock()),
        nvmlDeviceGetUUID=MagicMock(return_value="GPU-A"),
        nvmlDeviceGetName=MagicMock(return_value="RTX 3090"),
        NVML_FEATURE_ENABLED=1,
        NVML_FEATURE_DISABLED=0,
    )
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)

    publish_mock = AsyncMock()
    get_key_mock = AsyncMock(return_value=None)  # no heartbeat
    monkeypatch.setattr("ocabra.redis_client.publish", publish_mock, raising=True)
    monkeypatch.setattr("ocabra.redis_client.get_key", get_key_mock, raising=True)

    # Patch _wait_for_hw_monitor's internal sleep to be near-instant
    import ocabra.db.gpu_power_settings as mod

    monkeypatch.setattr(mod, "_wait_for_hw_monitor", AsyncMock(return_value=False))

    rows = [
        SimpleNamespace(
            gpu_uuid="GPU-A",
            power_limit_w=250,
            persistence_mode=None,
            last_known_index=0,
            last_known_name="RTX 3090",
        ),
    ]
    factory = _make_session_factory(rows)

    summary = await mod.reapply_persisted_settings(factory)
    assert summary == {"applied": 0, "skipped_no_match": 0, "errors": 1}
    publish_mock.assert_not_called()
