"""BUG-35: the immediate poll after a tunnel attaches was a bare
``asyncio.create_task`` whose result was dropped. The loop keeps only a weak
reference, so it could be collected mid-flight and its exception was lost.
It now goes through ``background_tasks.spawn`` (strong ref + logged failure).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ocabra.core import background_tasks
from ocabra.core.federation import FederationManager


@pytest.mark.asyncio
async def test_tunnel_attach_poll_is_kept_alive_and_its_failure_logged(monkeypatch):
    release = asyncio.Event()
    polled: list[str] = []

    async def poll(peer):
        polled.append(peer.name)
        await release.wait()
        raise RuntimeError("peer exploded")

    warnings: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        background_tasks.logger,
        "warning",
        lambda event, **kw: warnings.append((event, kw)),
    )
    fm = SimpleNamespace(_poll_peer=poll, join_service=None)
    peer = SimpleNamespace(
        name="pod-1",
        peer_id="p1",
        tunnel_session=None,
        consecutive_failures=3,
        _next_retry_at=1.0,
        connection_mode="tunnel_out",
        remote_node_id=None,
    )
    session = SimpleNamespace(remote_info={})

    before = background_tasks.pending_count()
    await FederationManager.attach_tunnel(fm, peer, session)
    # Held by a strong reference until it finishes.
    assert background_tasks.pending_count() == before + 1
    await asyncio.sleep(0)
    assert polled == ["pod-1"]

    release.set()
    for _ in range(10):
        await asyncio.sleep(0)
    assert background_tasks.pending_count() == before
    assert any(
        event == "background_task_failed" and "peer exploded" in kw.get("error", "")
        for event, kw in warnings
    )
