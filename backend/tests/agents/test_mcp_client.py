"""Tests for the MCP client abstractions.

The upstream ``mcp`` SDK is not required at runtime for these tests: we patch
it with a fake ``ClientSession`` that exposes the two methods the registry
cares about (``list_tools`` and ``call_tool``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from ocabra.agents.mcp_client import (
    MCPTool,
    MCPToolResult,
    _BaseClient,
    _flatten_content_blocks,
    _HeaderMerger,
)

# ── Header merger ───────────────────────────────────────────


def test_header_merger_keeps_static_headers():
    merger = _HeaderMerger({"Authorization": "Bearer static", "x-static": "1"})
    merged = merger.merge({"authorization": "Bearer INJECTED", "x-extra": "2"})
    # Static header wins
    assert merged["authorization"] == "Bearer static"
    assert merged["x-static"] == "1"
    assert merged["x-extra"] == "2"


def test_header_merger_case_insensitive():
    merger = _HeaderMerger({"X-Auth": "A"})
    merged = merger.merge({"x-auth": "B"})
    assert merged == {"x-auth": "A"}


def test_header_merger_handles_empty():
    merger = _HeaderMerger(None)
    assert merger.merge(None) == {}
    assert merger.merge({"X-New": "1"}) == {"x-new": "1"}


# ── Flatten helpers ─────────────────────────────────────────


class _FakeTextContent:
    def __init__(self, text):
        self.text = text

    def model_dump(self, mode="python"):
        return {"type": "text", "text": self.text}


def test_flatten_uses_model_dump_when_available():
    blocks = _flatten_content_blocks([_FakeTextContent("hi")])
    assert blocks == [{"type": "text", "text": "hi"}]


def test_flatten_passes_through_dicts():
    blocks = _flatten_content_blocks([{"type": "text", "text": "a"}, _FakeTextContent("b")])
    assert blocks == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_flatten_handles_none():
    assert _flatten_content_blocks(None) == []


# ── _BaseClient happy path with a fake session ──────────────


@dataclass
class _FakeTool:
    name: str
    description: str = ""
    inputSchema: dict | None = None


class _FakeListToolsResult:
    def __init__(self, tools):
        self.tools = tools


class _FakeCallResult:
    def __init__(self, content, is_error=False):
        self.content = content
        self.isError = is_error

    def model_dump(self, mode="python"):
        return {"content": [{"type": "text", "text": "ok"}], "isError": self.isError}


class _FakeSession:
    def __init__(self):
        self.list_tools = AsyncMock(
            return_value=_FakeListToolsResult(
                [
                    _FakeTool(
                        name="get_issue",
                        description="Fetch a GitHub issue",
                        inputSchema={
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                        },
                    )
                ]
            )
        )
        self.call_tool = AsyncMock(
            return_value=_FakeCallResult(
                content=[_FakeTextContent("ok")],
                is_error=False,
            )
        )


class _ProbeClient(_BaseClient):
    """_BaseClient subclass with a pre-installed fake session; skips .connect()."""

    async def connect(self) -> None:
        # Pretend we already opened a session.
        pass


@pytest.mark.asyncio
async def test_base_client_list_tools_maps_into_dataclasses():
    client = _ProbeClient()
    client._session = _FakeSession()
    tools = await client.list_tools()
    assert len(tools) == 1
    assert isinstance(tools[0], MCPTool)
    assert tools[0].name == "get_issue"
    assert tools[0].input_schema["type"] == "object"


@pytest.mark.asyncio
async def test_base_client_call_tool_returns_flattened_result():
    client = _ProbeClient()
    client._session = _FakeSession()
    result = await client.call_tool("get_issue", {"id": 1}, timeout_seconds=5.0)
    assert isinstance(result, MCPToolResult)
    assert result.is_error is False
    assert result.content == [{"type": "text", "text": "ok"}]


@pytest.mark.asyncio
async def test_base_client_call_tool_times_out():
    async def _slow(**_kwargs):
        await asyncio.sleep(10)

    client = _ProbeClient()
    sess = _FakeSession()
    sess.call_tool = AsyncMock(side_effect=_slow)
    client._session = sess

    result = await client.call_tool("slow", {}, timeout_seconds=0.01)
    assert result.is_error is True
    assert "timeout" in result.content[0]["text"]


@pytest.mark.asyncio
async def test_base_client_health_check_ok():
    client = _ProbeClient()
    client._session = _FakeSession()
    assert await client.health_check() is True


@pytest.mark.asyncio
async def test_base_client_health_check_failure():
    client = _ProbeClient()
    session = MagicMock()
    session.list_tools = AsyncMock(side_effect=RuntimeError("boom"))
    client._session = session
    assert await client.health_check() is False
