"""Tests for :class:`MCPRegistry`: cache TTL, invalidation, Fernet helpers."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from ocabra.agents.mcp_client import MCPClientInterface, MCPTool, MCPToolResult
from ocabra.agents.mcp_registry import MCPRegistry, MCPServerRuntime


class FakeClient(MCPClientInterface):
    """Minimal in-memory client used to exercise the registry plumbing."""

    def __init__(self, tools: list[MCPTool] | None = None, healthy: bool = True) -> None:
        self._tools = tools or []
        self._healthy = healthy
        self.calls: list[tuple[str, dict]] = []
        self.close_called = 0
        self.list_tools_calls = 0

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        self.close_called += 1

    async def list_tools(self) -> list[MCPTool]:
        self.list_tools_calls += 1
        return list(self._tools)

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        *,
        timeout_seconds: float,
        extra_headers=None,
    ) -> MCPToolResult:
        self.calls.append((name, arguments))
        return MCPToolResult(content=[{"type": "text", "text": "ok"}])

    async def health_check(self) -> bool:
        return self._healthy


@pytest.fixture
def registry() -> MCPRegistry:
    return MCPRegistry(
        session_factory=None,
        fernet_secret="a" * 64,
        cache_ttl_seconds=300,
    )


# ── Fernet helpers ──────────────────────────────────────────


def test_roundtrip_text(registry: MCPRegistry) -> None:
    ct = registry.encrypt_text("hello world")
    assert ct != "hello world"
    assert registry.decrypt_text(ct) == "hello world"


def test_roundtrip_json(registry: MCPRegistry) -> None:
    ct = registry.encrypt_json({"header_name": "X-API-Key", "value": "top-secret"})
    assert "top-secret" not in (ct or "")
    assert registry.decrypt_json(ct) == {
        "header_name": "X-API-Key",
        "value": "top-secret",
    }


def test_encrypt_json_none(registry: MCPRegistry) -> None:
    assert registry.encrypt_json(None) is None
    assert registry.decrypt_json(None) is None


def test_decrypt_rejects_tampered(registry: MCPRegistry) -> None:
    ct = registry.encrypt_text("hello")
    with pytest.raises(ValueError):
        registry.decrypt_text(ct + "garbage")


# ── Cache / TTL ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_tools_caches_result(registry: MCPRegistry) -> None:
    client = FakeClient(tools=[MCPTool(name="t1", description="", input_schema={})])
    registry._clients["gh"] = client
    registry._runtime["gh"] = MCPServerRuntime(
        id=uuid.uuid4(),
        alias="gh",
        display_name="GitHub",
        transport="http",
    )
    t1 = await registry.get_tools("gh")
    t2 = await registry.get_tools("gh")
    assert [t.name for t in t1] == ["t1"]
    assert [t.name for t in t2] == ["t1"]
    # Cache hit → only the first get_tools hits list_tools.
    assert client.list_tools_calls == 1


@pytest.mark.asyncio
async def test_invalidate_forces_refresh(registry: MCPRegistry) -> None:
    client = FakeClient(tools=[MCPTool(name="t1", description="", input_schema={})])
    registry._clients["gh"] = client
    registry._runtime["gh"] = MCPServerRuntime(
        id=uuid.uuid4(),
        alias="gh",
        display_name="GitHub",
        transport="http",
    )
    await registry.get_tools("gh")
    registry.invalidate("gh")
    await registry.get_tools("gh")
    assert client.list_tools_calls == 2


@pytest.mark.asyncio
async def test_ttl_zero_always_refreshes() -> None:
    registry = MCPRegistry(
        session_factory=None,
        fernet_secret="a" * 64,
        cache_ttl_seconds=0,
    )
    client = FakeClient(tools=[MCPTool(name="t1", description="", input_schema={})])
    registry._clients["gh"] = client
    registry._runtime["gh"] = MCPServerRuntime(
        id=uuid.uuid4(), alias="gh", display_name="GitHub", transport="http"
    )
    await registry.get_tools("gh")
    await registry.get_tools("gh")
    # TTL=0 means stale on every call → two fetches.
    assert client.list_tools_calls == 2


@pytest.mark.asyncio
async def test_force_refresh_bypasses_cache(registry: MCPRegistry) -> None:
    client = FakeClient(tools=[MCPTool(name="t1", description="", input_schema={})])
    registry._clients["gh"] = client
    registry._runtime["gh"] = MCPServerRuntime(
        id=uuid.uuid4(), alias="gh", display_name="GitHub", transport="http"
    )
    await registry.get_tools("gh")
    await registry.get_tools("gh", force_refresh=True)
    assert client.list_tools_calls == 2


# ── Health check ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_marks_runtime(registry: MCPRegistry) -> None:
    ok_runtime = MCPServerRuntime(
        id=uuid.uuid4(), alias="gh", display_name="GitHub", transport="http"
    )
    registry._runtime["gh"] = ok_runtime
    registry._clients["gh"] = FakeClient(healthy=True)
    assert await registry.health_check("gh") is True
    assert ok_runtime.health_status == "healthy"

    bad_runtime = MCPServerRuntime(
        id=uuid.uuid4(), alias="bad", display_name="Bad", transport="http"
    )
    registry._runtime["bad"] = bad_runtime
    registry._clients["bad"] = FakeClient(healthy=False)
    assert await registry.health_check("bad") is False
    assert bad_runtime.health_status == "unhealthy"


# ── Unregister + stop close clients ─────────────────────────


@pytest.mark.asyncio
async def test_unregister_closes_client(registry: MCPRegistry) -> None:
    client = FakeClient()
    registry._clients["gh"] = client
    registry._runtime["gh"] = MCPServerRuntime(
        id=uuid.uuid4(), alias="gh", display_name="GitHub", transport="http"
    )
    registry._cache["gh"] = object()  # type: ignore[assignment]
    await registry.unregister("gh")
    assert "gh" not in registry._clients
    assert "gh" not in registry._runtime
    assert "gh" not in registry._cache
    assert client.close_called == 1


@pytest.mark.asyncio
async def test_stop_closes_every_client() -> None:
    reg = MCPRegistry(session_factory=None, fernet_secret="a" * 64)
    c1, c2 = FakeClient(), FakeClient()
    reg._clients = {"a": c1, "b": c2}
    reg._started = True
    await reg.stop()
    assert c1.close_called == 1
    assert c2.close_called == 1
    assert reg._clients == {}


# ── call_tool entrypoint ────────────────────────────────────


@pytest.mark.asyncio
async def test_call_tool_delegates_to_client(registry: MCPRegistry) -> None:
    client = FakeClient()
    registry._clients["gh"] = client
    registry._runtime["gh"] = MCPServerRuntime(
        id=uuid.uuid4(), alias="gh", display_name="GitHub", transport="http"
    )
    await registry.call_tool("gh", "get_issue", {"id": 1}, timeout_seconds=5.0)
    assert client.calls == [("get_issue", {"id": 1})]


@pytest.mark.asyncio
async def test_call_tool_unknown_alias_raises(registry: MCPRegistry) -> None:
    with pytest.raises(KeyError):
        await registry.call_tool("nope", "x", {}, timeout_seconds=1.0)
