"""Tests for :class:`ocabra.agents.executor.AgentExecutor` (Stream B / Fase 2+3).

The DB writes performed by the executor (request_stats hop rows + tool_call_stats)
are stubbed out via patches because the test suite does not provision a real
PostgreSQL backing store.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.agents.executor import (
    AgentExecutor,
    WorkerInvoker,
)
from ocabra.agents.mcp_client import MCPTool, MCPToolResult
from ocabra.agents.resolver import AgentMCPBindingSpec, AgentSpec
from ocabra.api._deps_auth import UserContext

# ── Test doubles ─────────────────────────────────────────────


class FakeRegistry:
    """Async stand-in for :class:`MCPRegistry`."""

    def __init__(
        self,
        tools_by_alias: dict[str, list[MCPTool]] | None = None,
        responses: dict[tuple[str, str], MCPToolResult] | None = None,
        raise_on_call: dict[tuple[str, str], BaseException] | None = None,
    ) -> None:
        self._tools = tools_by_alias or {}
        self._responses = responses or {}
        self._raises = raise_on_call or {}
        self.call_log: list[tuple[str, str, dict[str, Any]]] = []

    async def get_tools(self, alias: str, *, force_refresh: bool = False):
        if alias not in self._tools:
            raise KeyError(alias)
        return list(self._tools[alias])

    async def call_tool(
        self,
        alias: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        timeout_seconds: float,
        extra_headers=None,
    ) -> MCPToolResult:
        self.call_log.append((alias, tool_name, dict(args)))
        if (alias, tool_name) in self._raises:
            raise self._raises[(alias, tool_name)]
        return self._responses.get(
            (alias, tool_name),
            MCPToolResult(
                content=[{"type": "text", "text": f"{alias}.{tool_name} ok"}],
                is_error=False,
            ),
        )


class FakeWorker(WorkerInvoker):
    """Iterates pre-canned responses one per :meth:`forward` call."""

    def __init__(self, responses: list[dict[str, Any]], vision: bool = False) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._vision = vision

    @property
    def vision_capable(self) -> bool:
        return self._vision

    async def forward(self, body):
        self.calls.append(body)
        if not self._responses:
            raise AssertionError("FakeWorker exhausted")
        return self._responses.pop(0)


def _agent(
    *,
    bindings: list[AgentMCPBindingSpec] | None = None,
    require_approval: str = "never",
    max_hops: int = 4,
    system_prompt: str = "You are helpful.",
    base_model_id: str = "vllm/test-model",
    tool_choice: str = "auto",
) -> AgentSpec:
    return AgentSpec(
        id=uuid.uuid4(),
        slug="bot",
        display_name="Bot",
        description=None,
        base_model_id=base_model_id,
        profile_id=None,
        system_prompt=system_prompt,
        tool_choice_default=tool_choice,
        max_tool_hops=max_hops,
        tool_timeout_seconds=30,
        require_approval=require_approval,
        request_defaults=None,
        group_id=None,
        bindings=bindings or [],
    )


def _user() -> UserContext:
    return UserContext(
        user_id=str(uuid.uuid4()),
        username="tester",
        role="user",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )


def _assistant_response(*, content: str = "", tool_calls=None, finish="stop"):
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-x",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# Patch DB writes globally — no DB available in unit tests.


@pytest.fixture(autouse=True)
def _no_db_writes():
    with (
        patch("ocabra.agents.executor._persist_hop_request_stat", new=AsyncMock(return_value=None)),
        patch(
            "ocabra.agents.executor._persist_tool_call_stats",
            new=AsyncMock(return_value=None),
        ),
    ):
        yield


# ── Tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_tools_simple_completion():
    """Agent with 0 tools and no tool_calls returns the LLM response unchanged."""
    agent = _agent(bindings=[])
    worker = FakeWorker([_assistant_response(content="hi")])
    registry = FakeRegistry()
    executor = AgentExecutor(registry)

    result = await executor.run(agent, [{"role": "user", "content": "hello"}], {}, _user(), worker)

    assert result.hops_used == 0
    assert result.openai_response["choices"][0]["message"]["content"] == "hi"
    # The body sent to the worker must inject the system prompt.
    assert worker.calls[0]["messages"][0]["role"] == "system"
    assert worker.calls[0]["messages"][0]["content"].startswith("You are helpful.")


@pytest.mark.asyncio
async def test_one_hop_tool_call_success():
    binding = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="fs",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding])
    tool = MCPTool(name="read", description="Read", input_schema={"type": "object"})
    registry = FakeRegistry(
        tools_by_alias={"fs": [tool]},
        responses={("fs", "read"): MCPToolResult(content=[{"type": "text", "text": "OK"}])},
    )
    tc = {
        "id": "tc1",
        "type": "function",
        "function": {"name": "fs_read", "arguments": '{"path": "/tmp"}'},
    }
    worker = FakeWorker(
        [
            _assistant_response(tool_calls=[tc], finish="tool_calls"),
            _assistant_response(content="done"),
        ]
    )
    executor = AgentExecutor(registry)
    result = await executor.run(agent, [{"role": "user", "content": "read"}], {}, _user(), worker)

    assert result.hops_used == 1
    assert result.openai_response["choices"][0]["message"]["content"] == "done"
    assert len(result.tool_calls) == 1
    rec = result.tool_calls[0]
    assert rec.alias == "fs"
    assert rec.tool_name == "read"
    assert rec.status == "ok"
    # The second worker call must include the role=tool message.
    assert worker.calls[1]["messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_n_hops_until_max_then_hop_limit():
    binding = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="fs",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding], max_hops=2)
    tool = MCPTool(name="loop", description="", input_schema={"type": "object"})
    registry = FakeRegistry(tools_by_alias={"fs": [tool]})
    tc = {"id": "x", "function": {"name": "fs_loop", "arguments": "{}"}}
    # Worker keeps emitting tool_calls forever, but the loop must abort at max.
    responses = [_assistant_response(tool_calls=[tc], finish="tool_calls") for _ in range(5)]
    worker = FakeWorker(responses)
    executor = AgentExecutor(registry)
    result = await executor.run(agent, [{"role": "user", "content": "x"}], {}, _user(), worker)

    assert result.openai_response["choices"][0]["finish_reason"] == "tool_hop_limit"
    assert result.hops_used == 2


@pytest.mark.asyncio
async def test_tool_timeout_records_status_timeout():
    binding = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="fs",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding])
    tool = MCPTool(name="hang", description="", input_schema={"type": "object"})
    registry = FakeRegistry(
        tools_by_alias={"fs": [tool]},
        raise_on_call={("fs", "hang"): TimeoutError()},
    )
    tc = {"id": "x", "function": {"name": "fs_hang", "arguments": "{}"}}
    worker = FakeWorker(
        [
            _assistant_response(tool_calls=[tc], finish="tool_calls"),
            _assistant_response(content="recovered"),
        ]
    )
    executor = AgentExecutor(registry)
    result = await executor.run(agent, [{"role": "user", "content": "x"}], {}, _user(), worker)
    assert result.tool_calls[0].status == "timeout"


@pytest.mark.asyncio
async def test_schema_error_does_not_call_mcp():
    binding = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="fs",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding])
    tool = MCPTool(
        name="read",
        description="",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    registry = FakeRegistry(tools_by_alias={"fs": [tool]})
    tc = {"id": "x", "function": {"name": "fs_read", "arguments": "{}"}}  # missing required
    worker = FakeWorker(
        [
            _assistant_response(tool_calls=[tc], finish="tool_calls"),
            _assistant_response(content="ok"),
        ]
    )
    executor = AgentExecutor(registry)
    result = await executor.run(agent, [{"role": "user", "content": "x"}], {}, _user(), worker)
    assert result.tool_calls[0].status == "schema_error"
    # MCP must NOT have been called.
    assert registry.call_log == []


@pytest.mark.asyncio
async def test_parallel_tool_calls_executed_concurrently():
    binding = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="fs",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding])
    tool = MCPTool(name="ping", description="", input_schema={"type": "object"})
    registry = FakeRegistry(tools_by_alias={"fs": [tool]})
    tcs = [
        {"id": f"c{i}", "function": {"name": "fs_ping", "arguments": json.dumps({"i": i})}}
        for i in range(3)
    ]
    worker = FakeWorker(
        [
            _assistant_response(tool_calls=tcs, finish="tool_calls"),
            _assistant_response(content="done"),
        ]
    )
    executor = AgentExecutor(registry, max_concurrent_tool_calls=4)
    result = await executor.run(agent, [{"role": "user", "content": "x"}], {}, _user(), worker)
    assert len(result.tool_calls) == 3
    assert all(r.status == "ok" for r in result.tool_calls)
    # All three calls reached the registry.
    assert sorted(c[1] for c in registry.call_log) == ["ping", "ping", "ping"]


@pytest.mark.asyncio
async def test_require_approval_always_returns_after_first_tool_calls():
    binding = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="fs",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding], require_approval="always")
    tool = MCPTool(name="read", description="", input_schema={"type": "object"})
    registry = FakeRegistry(tools_by_alias={"fs": [tool]})
    tc = {"id": "x", "function": {"name": "fs_read", "arguments": "{}"}}
    worker = FakeWorker([_assistant_response(tool_calls=[tc], finish="tool_calls")])
    executor = AgentExecutor(registry)
    result = await executor.run(agent, [{"role": "user", "content": "x"}], {}, _user(), worker)
    # Tool calls returned to the caller; MCP not invoked.
    assert registry.call_log == []
    assert result.openai_response["choices"][0]["message"]["tool_calls"]


@pytest.mark.asyncio
async def test_require_approval_always_continuation_with_tool_results():
    """Second turn: client re-sends conversation including tool results."""
    binding = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="fs",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding], require_approval="always")
    tool = MCPTool(name="read", description="", input_schema={"type": "object"})
    registry = FakeRegistry(tools_by_alias={"fs": [tool]})
    # The model now produces a final answer (no tool_calls) given the results.
    worker = FakeWorker([_assistant_response(content="final")])
    executor = AgentExecutor(registry)
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "x", "function": {"name": "fs_read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "x", "content": "OK"},
    ]
    result = await executor.run(agent, messages, {}, _user(), worker)
    assert result.openai_response["choices"][0]["message"]["content"] == "final"


@pytest.mark.asyncio
async def test_require_approval_header_cannot_relax_agent():
    """Agent.require_approval=always must NOT be downgraded to 'never' via header."""
    binding = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="fs",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding], require_approval="always")
    tool = MCPTool(name="read", description="", input_schema={"type": "object"})
    registry = FakeRegistry(tools_by_alias={"fs": [tool]})
    tc = {"id": "x", "function": {"name": "fs_read", "arguments": "{}"}}
    worker = FakeWorker([_assistant_response(tool_calls=[tc], finish="tool_calls")])
    executor = AgentExecutor(registry)
    result = await executor.run(
        agent,
        [{"role": "user", "content": "x"}],
        {},
        _user(),
        worker,
        require_approval_override="never",
    )
    assert registry.call_log == []  # still in approval mode
    assert result.openai_response["choices"][0]["message"]["tool_calls"]


@pytest.mark.asyncio
async def test_caller_tool_collision_raises_400():
    """Caller tools share the prefixed namespace ``caller_*``.  Collision = 400.

    The agent advertises a tool named ``caller_thing`` (via an alias literally
    called ``caller``).  When the caller adds a tool whose namespaced name
    ``caller_thing`` matches it, the executor must reject with HTTP 400.
    """
    binding_caller_alias = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="caller",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding_caller_alias])
    tool_clash = MCPTool(name="thing", description="", input_schema={"type": "object"})
    registry = FakeRegistry(tools_by_alias={"caller": [tool_clash]})
    worker = FakeWorker([_assistant_response(content="ok")])
    executor = AgentExecutor(registry)
    caller_tools = [{"type": "function", "function": {"name": "thing", "parameters": {}}}]
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await executor.run(
            agent,
            [{"role": "user", "content": "x"}],
            {},
            _user(),
            worker,
            caller_tools=caller_tools,
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_per_request_allowed_tools_filters_inventory():
    binding = AgentMCPBindingSpec(
        server_id=uuid.uuid4(),
        server_alias="fs",
        server_group_id=None,
        allowed_tools=None,
        server_allowed_tools=None,
    )
    agent = _agent(bindings=[binding])
    tools = [
        MCPTool(name="read", description="", input_schema={"type": "object"}),
        MCPTool(name="write", description="", input_schema={"type": "object"}),
    ]
    registry = FakeRegistry(tools_by_alias={"fs": tools})
    worker = FakeWorker([_assistant_response(content="ok")])
    executor = AgentExecutor(registry)
    await executor.run(
        agent,
        [{"role": "user", "content": "x"}],
        {},
        _user(),
        worker,
        per_request_allowed_tools=["read"],
    )
    advertised = [t["function"]["name"] for t in worker.calls[0]["tools"]]
    assert advertised == ["fs_read"]


@pytest.mark.asyncio
async def test_system_prompt_prepended_to_existing_system_message():
    agent = _agent(system_prompt="Be terse.")
    worker = FakeWorker([_assistant_response(content="ok")])
    registry = FakeRegistry()
    executor = AgentExecutor(registry)
    await executor.run(
        agent,
        [
            {"role": "system", "content": "Speak like a pirate."},
            {"role": "user", "content": "hi"},
        ],
        {},
        _user(),
        worker,
    )
    sys_msg = worker.calls[0]["messages"][0]["content"]
    assert "Be terse." in sys_msg
    assert "Speak like a pirate." in sys_msg
