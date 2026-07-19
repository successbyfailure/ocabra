"""AgentExecutor — the tool-loop runtime for ``model="agent/<slug>"``.

This module is the entry point used by the OpenAI / Ollama chat endpoints once
:func:`ocabra.agents.resolver.resolve_agent` returns a non-``None``
:class:`AgentSpec`.  It owns:

* The OpenAI ↔ MCP tool translation (delegated to :mod:`ocabra.agents.translation`).
* Allowlist intersection (server, agent, per-request header).
* JSON-Schema validation of arguments before invoking the MCP tool.
* Parallel tool execution with a global concurrency cap.
* Per-hop request stats with ``parent_request_id`` linkage.
* Per-tool ``tool_call_stats`` rows (with redacted args).
* The two-turn ``require_approval=always`` handshake.
* Streaming variant emitting OpenAI SSE chunks plus ``ocabra.tool_result``
  custom events between hops.

Plan: docs/tasks/agents-mcp-plan.md — Fase 2 + Fase 3.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Literal

import structlog
from fastapi import HTTPException

from ocabra.agents.mcp_client import MCPTool, MCPToolResult
from ocabra.agents.mcp_registry import MCPRegistry
from ocabra.agents.resolver import AgentSpec
from ocabra.agents.translation import (
    DEFAULT_REDACT_FIELDS,
    mcp_result_to_openai_message,
    mcp_tool_to_openai,
    parse_openai_tool_call,
    redact_args,
    sanitize_openai_function_name,
    summarise_result,
)
from ocabra.api._deps_auth import UserContext
from ocabra.config import settings

logger = structlog.get_logger(__name__)

# Context-vars consumed by the StatsMiddleware to stamp the root request_stat
# row with the agent_id.  The executor sets these for the duration of ``run``
# so the middleware-managed root row inherits the link.  Hop child rows are
# inserted directly by the executor with explicit parent_request_id.
current_agent_id: contextvars.ContextVar[uuid.UUID | None] = contextvars.ContextVar(
    "current_agent_id", default=None
)
current_root_request_id: contextvars.ContextVar[uuid.UUID | None] = contextvars.ContextVar(
    "current_root_request_id", default=None
)

REQUIRE_APPROVAL_NEVER = "never"
REQUIRE_APPROVAL_ALWAYS = "always"
SUBAGENT_TOOL_PREFIX = "delegate_"
SUBAGENT_MAX_DEPTH = 8

# Interval between SSE keepalive comments emitted while a tool gather (which
# may include a long-running subagent) is in flight. Must be smaller than the
# typical proxy / browser idle timeout (~30 s for many CDN defaults).
_KEEPALIVE_INTERVAL_S = 10.0


# ── Result payloads ────────────────────────────────────────────


@dataclass
class ToolCallRecord:
    """Persisted record of one tool invocation inside the loop."""

    alias: str
    tool_name: str
    args_redacted: dict[str, Any]
    duration_ms: int
    status: str  # ok | timeout | schema_error | mcp_error
    error: str | None
    hop_index: int
    result_summary: str | None
    server_id: uuid.UUID | None = None
    # When ``alias == "agent"`` this carries the records of every tool the
    # subagent executed inside its own loop, so the parent can surface the
    # nested trace to the UI without leaking it into ``tool_call_stats`` (those
    # rows are already persisted by the child executor with parent_request_id).
    child_tool_calls: list["ToolCallRecord"] = field(default_factory=list)


@dataclass
class AgentExecutorResult:
    """Result of a non-streaming :meth:`AgentExecutor.run` invocation."""

    openai_response: dict[str, Any]
    hops_used: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


@dataclass
class ToolTarget:
    """Resolved execution target for one exposed tool."""

    kind: Literal["mcp", "subagent"]
    public_name: str
    alias: str
    tool_name: str
    mcp_tool: MCPTool | None = None
    subagent_slug: str | None = None


# ── Worker contract ────────────────────────────────────────────


class WorkerInvoker:
    """Minimal callable surface used by the executor to talk to the LLM.

    Stream B mocks this in tests.  In production, the chat endpoint wires up
    a concrete instance backed by ``WorkerPool.forward_request`` /
    ``forward_stream`` plus the same profile-resolution logic the
    non-agent path uses (see ``api/openai/chat.py``).
    """

    async def forward(self, body: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def forward_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        raise NotImplementedError
        # pragma: no cover - generator stub
        yield b""

    @property
    def vision_capable(self) -> bool:
        return False

    def acquire_run_lease(self):
        """Hold the underlying model busy for the duration of a run.

        Default implementation is a no-op (used by mocks and any invoker that
        does not need busy-tracking). Real implementations — see
        :class:`ocabra.agents.chat_glue.ProfileWorkerInvoker` — return an async
        context manager that increments the in-flight counter on entry and
        decrements it on exit, so the idle eviction watchdog cannot desalojate
        the worker mid-loop.
        """
        return contextlib.nullcontext()


# ── Internal helpers ───────────────────────────────────────────


def _intersect(*allowlists: list[str] | None) -> list[str] | None:
    """Intersect any number of allowlists.  ``None`` means "everything"."""
    effective: set[str] | None = None
    for allow in allowlists:
        if allow is None:
            continue
        as_set = set(allow)
        effective = as_set if effective is None else effective & as_set
    if effective is None:
        return None
    return sorted(effective)


def _validate_args(
    schema: dict[str, Any] | None,
    args: dict[str, Any],
) -> str | None:
    """Validate ``args`` against ``schema`` using :mod:`jsonschema`.

    Returns ``None`` on success or an error message string on failure.  Failures
    must be surfaced to the LLM as a tool message rather than raised, so the
    caller turns a non-None return into a :class:`MCPToolResult` with
    ``is_error=True`` and ``status="schema_error"``.
    """
    if not schema:
        return None
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - dep is in pyproject
        if schema.get("type") == "object" and not isinstance(args, dict):
            return "schema_error: expected object arguments"
        required = schema.get("required") or []
        if isinstance(required, list):
            for key in required:
                if key not in args:
                    return f"schema_error: '{key}' is a required property"
        return None
    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.ValidationError as exc:
        return f"schema_error: {exc.message}"
    except jsonschema.SchemaError as exc:
        return f"schema_definition_error: {exc.message}"
    return None


def _ensure_system_prompt(
    messages: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, Any]]:
    """Inject *system_prompt* into ``messages`` according to the plan rules.

    * If there is no leading system message, prepend one.
    * If there is, prepend ``system_prompt`` to it joined by ``\\n\\n``.
    * If ``system_prompt`` is empty, the messages are returned unchanged.
    """
    if not system_prompt:
        return list(messages)
    new_messages = list(messages)
    if (
        new_messages
        and isinstance(new_messages[0], dict)
        and new_messages[0].get("role") == "system"
    ):
        existing = new_messages[0]
        existing_content = existing.get("content") or ""
        # Preserve list-style content (vision parts) by stringifying for the prepend.
        if isinstance(existing_content, list):
            text_parts = [
                part.get("text", "")
                for part in existing_content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            existing_text = "\n".join(p for p in text_parts if p)
            new_text = f"{system_prompt}\n\n{existing_text}".rstrip()
            new_messages[0] = {**existing, "content": new_text}
        else:
            joined = f"{system_prompt}\n\n{existing_content}".rstrip()
            new_messages[0] = {**existing, "content": joined}
        return new_messages
    return [{"role": "system", "content": system_prompt}, *new_messages]


def _resolve_require_approval(agent_value: str, header_value: str | None) -> str:
    """Apply the override rule: header may not relax the agent's restrictivity.

    Restrictivity ranking: ``always`` > ``never``.  ``always`` -> ``never``
    via header is rejected (we silently keep the agent's value).  ``never`` ->
    ``always`` via header is accepted.
    """
    if not header_value:
        return agent_value
    header_value = header_value.strip().lower()
    if header_value not in (REQUIRE_APPROVAL_NEVER, REQUIRE_APPROVAL_ALWAYS):
        return agent_value
    if agent_value == REQUIRE_APPROVAL_ALWAYS and header_value == REQUIRE_APPROVAL_NEVER:
        # Caller cannot weaken the agent's restrictivity.
        return agent_value
    return header_value


def _build_caller_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Namespace a caller-provided tool with the ``caller_`` prefix."""
    fn = tool.get("function") or {}
    if not isinstance(fn, dict):
        return tool
    name = fn.get("name") or "tool"
    namespaced = sanitize_openai_function_name(f"caller_{name}")
    new_fn = {**fn, "name": namespaced}
    return {**tool, "function": new_fn}


def _make_subagent_tool_name(slug: str) -> str:
    """Return a stable OpenAI-safe function name for a child agent slug."""
    base = sanitize_openai_function_name(f"{SUBAGENT_TOOL_PREFIX}{slug}")
    if "." not in slug:
        return base
    suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"ocabra-subagent:{slug}").hex[:8]
    trimmed = base[: max(1, 64 - 9)]
    return f"{trimmed}_{suffix}"


def _subagent_parameters_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Task or question for the subagent.",
            },
            "context": {
                "type": "string",
                "description": "Optional supporting context for the subagent.",
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    }


def _build_subagent_tool(agent: AgentSpec, child_slug: str) -> dict[str, Any]:
    """Expose a child agent as a synthetic function tool."""
    child_name = agent.subagent_names.get(child_slug, child_slug)
    child_desc = agent.subagent_descriptions.get(child_slug)
    description = f"Delegate work to subagent '{child_name}' (agent/{child_slug})."
    if child_desc:
        description = f"{description} {child_desc}"
    return {
        "type": "function",
        "function": {
            "name": _make_subagent_tool_name(child_slug),
            "description": description,
            "parameters": _subagent_parameters_schema(),
        },
    }


def _subagent_args_to_messages(args: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate a subagent tool payload into a minimal chat history."""
    task = str(args.get("task") or "").strip()
    context = str(args.get("context") or "").strip()
    if context:
        content = f"{task}\n\nContext:\n{context}"
    else:
        content = task
    return [{"role": "user", "content": content}]


def _last_message_role(messages: list[dict[str, Any]]) -> str | None:
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict):
        return None
    return last.get("role")


def _extract_message(response: dict[str, Any]) -> dict[str, Any] | None:
    choices = response.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message")
    return msg if isinstance(msg, dict) else None


def _finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


# ── Stats persistence helpers ──────────────────────────────────


async def _persist_hop_request_stat(
    *,
    agent_id: uuid.UUID,
    parent_request_id: uuid.UUID | None,
    base_model_id: str,
    duration_ms: int,
    input_tokens: int | None,
    output_tokens: int | None,
    status_code: int,
    error_message: str | None,
    started_at: datetime,
) -> uuid.UUID | None:
    """Insert a child ``request_stats`` row representing one hop of the loop."""
    try:
        from ocabra.database import AsyncSessionLocal
        from ocabra.db.stats import RequestStat

        row_id = uuid.uuid4()
        async with AsyncSessionLocal() as session:
            row = RequestStat(
                id=row_id,
                model_id=base_model_id,
                backend_type=None,
                request_kind="agent_hop",
                endpoint_path="/v1/chat/completions",
                status_code=status_code,
                gpu_index=None,
                started_at=started_at,
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                energy_wh=None,
                error=error_message,
                user_id=None,
                group_id=None,
                api_key_name=None,
                remote_node_id=None,
                agent_id=agent_id,
                parent_request_id=parent_request_id,
            )
            session.add(row)
            await session.commit()
        return row_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_hop_stat_write_failed", error=str(exc))
        return None


async def _persist_tool_call_stats(
    *,
    agent_id: uuid.UUID,
    request_stat_id: uuid.UUID | None,
    records: Iterable[ToolCallRecord],
) -> None:
    """Bulk-insert ``tool_call_stats`` rows for one hop's tool batch."""
    records_list = list(records)
    if not records_list:
        return
    try:
        from ocabra.database import AsyncSessionLocal
        from ocabra.db.stats import ToolCallStat

        async with AsyncSessionLocal() as session:
            for rec in records_list:
                session.add(
                    ToolCallStat(
                        request_stat_id=request_stat_id,
                        agent_id=agent_id,
                        mcp_server_alias=rec.alias,
                        tool_name=rec.tool_name,
                        tool_args_redacted=rec.args_redacted,
                        result_summary=rec.result_summary,
                        duration_ms=rec.duration_ms,
                        status=rec.status,
                        error=rec.error,
                        hop_index=rec.hop_index,
                    )
                )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_tool_call_stats_write_failed", error=str(exc))


# ── Agent executor ─────────────────────────────────────────────


class AgentExecutor:
    """Orchestrate the tool-loop for one agent invocation.

    The executor is stateless across requests; one instance can be reused as
    long as the underlying ``MCPRegistry`` is the process-wide singleton.
    """

    def __init__(
        self,
        registry: MCPRegistry,
        *,
        subagent_runner: Callable[
            [
                str,
                dict[str, Any],
                UserContext,
                tuple[uuid.UUID, ...],
                dict[str, str] | None,
                list[str] | None,
            ],
            Awaitable[AgentExecutorResult],
        ]
        | None = None,
        max_concurrent_tool_calls: int | None = None,
        result_max_bytes: int | None = None,
        redact_fields: list[str] | tuple[str, ...] = DEFAULT_REDACT_FIELDS,
    ) -> None:
        self._registry = registry
        self._subagent_runner = subagent_runner
        self._max_concurrency = max(
            1,
            max_concurrent_tool_calls
            if max_concurrent_tool_calls is not None
            else settings.mcp_max_concurrent_tool_calls,
        )
        self._result_max_bytes = (
            result_max_bytes if result_max_bytes is not None else settings.mcp_result_max_bytes
        )
        self._redact_fields = tuple(redact_fields)

    # ── Tool inventory + allowlist ───────────────────────────

    async def _load_tools(
        self,
        agent: AgentSpec,
        per_request_allow: list[str] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, ToolTarget]]:
        """Return ``(openai_tools, name → ToolTarget)``.

        Names in ``openai_tools`` follow the ``{alias}_{tool_name}`` namespace.
        Empty ``MCPTool.input_schema`` is normalised to a permissive object
        schema downstream.
        """
        openai_tools: list[dict[str, Any]] = []
        lookup: dict[str, ToolTarget] = {}
        for binding in agent.bindings:
            try:
                tools = await self._registry.get_tools(binding.server_alias)
            except KeyError:
                logger.warning(
                    "agent_executor_alias_not_registered",
                    alias=binding.server_alias,
                    agent_slug=agent.slug,
                )
                continue
            allowed = _intersect(
                binding.server_allowed_tools,
                binding.allowed_tools,
                per_request_allow,
            )
            for tool in tools:
                if allowed is not None and tool.name not in allowed:
                    continue
                schema = mcp_tool_to_openai(binding.server_alias, tool)
                name = schema["function"]["name"]
                openai_tools.append(schema)
                lookup[name] = ToolTarget(
                    kind="mcp",
                    public_name=name,
                    alias=binding.server_alias,
                    tool_name=tool.name,
                    mcp_tool=tool,
                )
        for child_slug in agent.subagent_slugs:
            schema = _build_subagent_tool(agent, child_slug)
            name = schema["function"]["name"]
            openai_tools.append(schema)
            lookup[name] = ToolTarget(
                kind="subagent",
                public_name=name,
                alias="agent",
                tool_name=child_slug,
                subagent_slug=child_slug,
            )
        return openai_tools, lookup

    @staticmethod
    def _merge_caller_tools(
        agent_tools: list[dict[str, Any]],
        caller_tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Append namespaced caller tools, raising 400 on collision."""
        if not caller_tools:
            return agent_tools
        agent_names = {
            (t.get("function") or {}).get("name") for t in agent_tools if isinstance(t, dict)
        }
        merged = list(agent_tools)
        seen_caller: set[str] = set()
        for raw in caller_tools:
            if not isinstance(raw, dict):
                continue
            namespaced = _build_caller_tool(raw)
            name = (namespaced.get("function") or {}).get("name")
            if not name:
                continue
            if name in agent_names:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Caller tool '{name}' collides with an agent-provided tool. "
                        "Caller tools are namespaced as 'caller_*'; agent tools as "
                        "'{alias}_*'."
                    ),
                )
            if name in seen_caller:
                continue
            seen_caller.add(name)
            merged.append(namespaced)
        return merged

    # ── Single tool invocation ───────────────────────────────

    async def _invoke_tool(
        self,
        *,
        tool_call: dict[str, Any],
        lookup: dict[str, ToolTarget],
        timeout_seconds: float,
        per_request_headers: dict[str, str] | None,
        hop_index: int,
        vision_capable: bool,
        sem: asyncio.Semaphore,
        user_ctx: UserContext,
        agent_chain: tuple[uuid.UUID, ...],
        per_request_allowed_tools: list[str] | None,
    ) -> tuple[dict[str, Any], ToolCallRecord]:
        """Execute one tool_call and return ``(message, record)``.

        Errors are converted to tool messages so the LLM can react in the
        next hop.  Validation failures, MCP timeouts, and MCP errors are all
        captured in the :class:`ToolCallRecord` ``status`` field.
        """
        tool_call_id = str(tool_call.get("id") or "")
        async with sem:
            t_start = time.monotonic()
            try:
                public_name = (tool_call.get("function") or {}).get("name") or ""
                target = lookup.get(public_name)
                if target is None:
                    duration_ms = int((time.monotonic() - t_start) * 1000)
                    err = MCPToolResult(
                        content=[{"type": "text", "text": f"Unknown tool '{public_name}'"}],
                        is_error=True,
                    )
                    msg = mcp_result_to_openai_message(
                        tool_call_id, err, vision_capable=vision_capable
                    )
                    rec = ToolCallRecord(
                        alias="",
                        tool_name=public_name,
                        args_redacted={},
                        duration_ms=duration_ms,
                        status="schema_error",
                        error=f"unknown_tool:{public_name}",
                        hop_index=hop_index,
                        result_summary="unknown_tool",
                    )
                    return msg, rec

                if target.kind == "mcp":
                    alias, mcp_tool_name, mcp_tool = target.alias, target.tool_name, target.mcp_tool
                    try:
                        parsed_alias, parsed_name, args = parse_openai_tool_call(tool_call)
                    except ValueError as exc:
                        duration_ms = int((time.monotonic() - t_start) * 1000)
                        err = MCPToolResult(
                            content=[{"type": "text", "text": f"invalid_tool_call: {exc}"}],
                            is_error=True,
                        )
                        msg = mcp_result_to_openai_message(
                            tool_call_id, err, vision_capable=vision_capable
                        )
                        rec = ToolCallRecord(
                            alias=alias,
                            tool_name=mcp_tool_name,
                            args_redacted={},
                            duration_ms=duration_ms,
                            status="schema_error",
                            error=str(exc),
                            hop_index=hop_index,
                            result_summary="invalid_tool_call",
                        )
                        return msg, rec

                    redacted = redact_args(args, list(self._redact_fields))
                    schema_err = _validate_args(mcp_tool.input_schema if mcp_tool else None, args)
                    if schema_err is not None:
                        duration_ms = int((time.monotonic() - t_start) * 1000)
                        err = MCPToolResult(
                            content=[{"type": "text", "text": schema_err}], is_error=True
                        )
                        msg = mcp_result_to_openai_message(
                            tool_call_id, err, vision_capable=vision_capable
                        )
                        rec = ToolCallRecord(
                            alias=alias,
                            tool_name=mcp_tool_name,
                            args_redacted=redacted,
                            duration_ms=duration_ms,
                            status="schema_error",
                            error=schema_err,
                            hop_index=hop_index,
                            result_summary=schema_err,
                        )
                        return msg, rec

                    forwarded_headers: dict[str, str] | None = None
                    if per_request_headers:
                        prefix = f"x-mcp-{alias}-"
                        forwarded_headers = {
                            k[len(prefix) :]: v
                            for k, v in per_request_headers.items()
                            if k.lower().startswith(prefix)
                        }
                        if not forwarded_headers:
                            forwarded_headers = None

                    try:
                        result = await self._registry.call_tool(
                            alias,
                            mcp_tool_name,
                            args,
                            timeout_seconds=timeout_seconds,
                            extra_headers=forwarded_headers,
                        )
                    except KeyError:
                        duration_ms = int((time.monotonic() - t_start) * 1000)
                        err = MCPToolResult(
                            content=[{"type": "text", "text": f"alias '{alias}' not registered"}],
                            is_error=True,
                        )
                        msg = mcp_result_to_openai_message(
                            tool_call_id, err, vision_capable=vision_capable
                        )
                        rec = ToolCallRecord(
                            alias=alias,
                            tool_name=mcp_tool_name,
                            args_redacted=redacted,
                            duration_ms=duration_ms,
                            status="mcp_error",
                            error="alias_not_registered",
                            hop_index=hop_index,
                            result_summary=None,
                        )
                        return msg, rec
                    except TimeoutError:
                        duration_ms = int((time.monotonic() - t_start) * 1000)
                        err = MCPToolResult(
                            content=[{"type": "text", "text": "tool_timeout"}], is_error=True
                        )
                        msg = mcp_result_to_openai_message(
                            tool_call_id, err, vision_capable=vision_capable
                        )
                        rec = ToolCallRecord(
                            alias=alias,
                            tool_name=mcp_tool_name,
                            args_redacted=redacted,
                            duration_ms=duration_ms,
                            status="timeout",
                            error="tool_timeout",
                            hop_index=hop_index,
                            result_summary="tool_timeout",
                        )
                        return msg, rec
                    except Exception as exc:  # noqa: BLE001
                        duration_ms = int((time.monotonic() - t_start) * 1000)
                        err = MCPToolResult(
                            content=[{"type": "text", "text": f"mcp_error: {exc}"}],
                            is_error=True,
                        )
                        msg = mcp_result_to_openai_message(
                            tool_call_id, err, vision_capable=vision_capable
                        )
                        rec = ToolCallRecord(
                            alias=alias,
                            tool_name=mcp_tool_name,
                            args_redacted=redacted,
                            duration_ms=duration_ms,
                            status="mcp_error",
                            error=str(exc),
                            hop_index=hop_index,
                            result_summary=str(exc)[:512],
                        )
                        return msg, rec

                    duration_ms = int((time.monotonic() - t_start) * 1000)
                    summary = summarise_result(result, max_bytes=self._result_max_bytes)
                    msg = mcp_result_to_openai_message(
                        tool_call_id, result, vision_capable=vision_capable
                    )
                    if isinstance(msg.get("content"), str):
                        msg["content"] = summary if len(summary) <= self._result_max_bytes else summary
                    rec = ToolCallRecord(
                        alias=alias,
                        tool_name=mcp_tool_name,
                        args_redacted=redacted,
                        duration_ms=duration_ms,
                        status="mcp_error" if result.is_error else "ok",
                        error=summary if result.is_error else None,
                        hop_index=hop_index,
                        result_summary=summary,
                    )
                    rec.alias = parsed_alias
                    rec.tool_name = parsed_name
                    return msg, rec

                args_raw = (tool_call.get("function") or {}).get("arguments")
                if isinstance(args_raw, str):
                    args = json.loads(args_raw) if args_raw.strip() else {}
                elif isinstance(args_raw, dict):
                    args = dict(args_raw)
                else:
                    args = {}
                redacted = redact_args(args, list(self._redact_fields))
                schema_err = _validate_args(_subagent_parameters_schema(), args)
                if schema_err is not None:
                    duration_ms = int((time.monotonic() - t_start) * 1000)
                    err = MCPToolResult(content=[{"type": "text", "text": schema_err}], is_error=True)
                    msg = mcp_result_to_openai_message(tool_call_id, err, vision_capable=vision_capable)
                    rec = ToolCallRecord(
                        alias="agent",
                        tool_name=target.subagent_slug or "",
                        args_redacted=redacted,
                        duration_ms=duration_ms,
                        status="schema_error",
                        error=schema_err,
                        hop_index=hop_index,
                        result_summary=schema_err,
                    )
                    return msg, rec
                if self._subagent_runner is None or not target.subagent_slug:
                    raise RuntimeError("subagent runner not configured")
                if len(agent_chain) >= SUBAGENT_MAX_DEPTH:
                    raise RuntimeError("subagent_depth_limit")
                result = await self._subagent_runner(
                    target.subagent_slug,
                    args,
                    user_ctx,
                    agent_chain,
                    per_request_headers,
                    per_request_allowed_tools,
                )
                child_message = _extract_message(result.openai_response) or {}
                child_content = child_message.get("content") or ""
                if isinstance(child_content, list):
                    child_content = json.dumps(child_content, ensure_ascii=False)
                child_result = MCPToolResult(
                    content=[{"type": "text", "text": str(child_content)}],
                    is_error=False,
                )
                duration_ms = int((time.monotonic() - t_start) * 1000)
                summary = str(child_content)[: self._result_max_bytes]
                msg = mcp_result_to_openai_message(
                    tool_call_id, child_result, vision_capable=vision_capable
                )
                rec = ToolCallRecord(
                    alias="agent",
                    tool_name=target.subagent_slug,
                    args_redacted=redacted,
                    duration_ms=duration_ms,
                    status="ok",
                    error=None,
                    hop_index=hop_index,
                    result_summary=summary,
                    child_tool_calls=list(result.tool_calls),
                )
                return msg, rec
            except Exception as exc:  # noqa: BLE001
                logger.exception("agent_executor_unexpected_tool_error", error=str(exc))
                duration_ms = int((time.monotonic() - t_start) * 1000)
                err = MCPToolResult(
                    content=[{"type": "text", "text": f"internal_error: {exc}"}], is_error=True
                )
                msg = mcp_result_to_openai_message(tool_call_id, err, vision_capable=vision_capable)
                rec = ToolCallRecord(
                    alias="",
                    tool_name="",
                    args_redacted={},
                    duration_ms=duration_ms,
                    status="mcp_error",
                    error=str(exc),
                    hop_index=hop_index,
                    result_summary=None,
                )
                return msg, rec

    # ── Public API: non-streaming ────────────────────────────

    async def run(
        self,
        agent: AgentSpec,
        messages: list[dict[str, Any]],
        request_options: dict[str, Any],
        user_ctx: UserContext,
        worker: WorkerInvoker,
        per_request_headers: dict[str, str] | None = None,
        per_request_allowed_tools: list[str] | None = None,
        caller_tools: list[dict[str, Any]] | None = None,
        require_approval_override: str | None = None,
        _agent_chain: tuple[uuid.UUID, ...] | None = None,
    ) -> AgentExecutorResult:
        """Execute the tool-loop until either no tool_calls remain or hop cap."""
        agent_chain = (*(_agent_chain or ()), agent.id)

        agent_tools, lookup = await self._load_tools(agent, per_request_allowed_tools)
        all_tools = self._merge_caller_tools(agent_tools, caller_tools)

        require_approval = _resolve_require_approval(
            agent.require_approval, require_approval_override
        )

        # Resolve tool_choice precedence: per-request body > agent default.
        body_tool_choice = request_options.get("tool_choice")
        tool_choice = (
            body_tool_choice if body_tool_choice is not None else agent.tool_choice_default
        )

        loop_messages = _ensure_system_prompt(messages, agent.system_prompt)

        # Detect handshake continuation: when the agent is `always` and the
        # client has just sent tool results, we *skip* re-emitting them and
        # let the LLM continue from where it left off.
        # No-op here; the messages already contain the role=tool turns.

        # Apply request_defaults from the agent (caller body wins).
        defaults = agent.request_defaults or {}
        merged_options: dict[str, Any] = {**defaults, **request_options}
        merged_options.pop("tool_choice", None)  # we set it explicitly below
        merged_options.pop("tools", None)
        merged_options.pop("messages", None)
        merged_options.pop("model", None)
        merged_options.pop("stream", None)

        agent_id = agent.id
        root_id = current_root_request_id.get()
        records: list[ToolCallRecord] = []
        last_response: dict[str, Any] | None = None

        last_role = _last_message_role(loop_messages)
        # When require_approval=always and the last message is `role=tool`,
        # the client is continuing a previous handshake — proceed normally.
        del last_role  # documentation only

        # Hold the worker busy for the entire loop (LLM hops + tool execution)
        # so the idle eviction watchdog cannot desalojate the model between
        # hops. See ProfileWorkerInvoker.acquire_run_lease.
        async with worker.acquire_run_lease():
            for hop in range(agent.max_tool_hops + 1):
                body = {
                    **merged_options,
                    "model": agent.base_model_id or agent.profile_id or "",
                    "messages": loop_messages,
                    "stream": False,
                }
                if all_tools:
                    body["tools"] = all_tools
                    body["tool_choice"] = tool_choice
                t_hop_start = time.monotonic()
                started = datetime.now(UTC)
                try:
                    response = await worker.forward(body)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("agent_executor_worker_error", error=str(exc))
                    raise
                duration_ms = int((time.monotonic() - t_hop_start) * 1000)
                usage = (response or {}).get("usage") or {}
                await _persist_hop_request_stat(
                    agent_id=agent_id,
                    parent_request_id=root_id,
                    base_model_id=str(body["model"]),
                    duration_ms=duration_ms,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    status_code=200,
                    error_message=None,
                    started_at=started,
                )
                last_response = response

                assistant_msg = _extract_message(response) or {}
                tool_calls = assistant_msg.get("tool_calls") or []
                finish_reason = _finish_reason(response)

                # Always append the assistant turn so subsequent hops see it.
                loop_messages = [*loop_messages, assistant_msg]

                if not tool_calls or finish_reason in (None, "stop", "length", "content_filter"):
                    if not tool_calls:
                        return AgentExecutorResult(
                            openai_response=response,
                            hops_used=hop,
                            tool_calls=records,
                        )

                if hop >= agent.max_tool_hops:
                    # Hop budget exhausted — return whatever the LLM said with a hint.
                    response.setdefault("choices", [{}])
                    if response["choices"]:
                        response["choices"][0]["finish_reason"] = "tool_hop_limit"
                    return AgentExecutorResult(
                        openai_response=response,
                        hops_used=hop,
                        tool_calls=records,
                    )

                if require_approval == REQUIRE_APPROVAL_ALWAYS:
                    # Two-turn handshake: hand the unresolved tool_calls back to
                    # the caller and stop.
                    return AgentExecutorResult(
                        openai_response=response,
                        hops_used=hop,
                        tool_calls=records,
                    )

                # Execute tool_calls in parallel (auto mode).
                sem = asyncio.Semaphore(self._max_concurrency)
                tasks = [
                    self._invoke_tool(
                        tool_call=tc,
                        lookup=lookup,
                        timeout_seconds=float(agent.tool_timeout_seconds),
                        per_request_headers=per_request_headers,
                        hop_index=hop,
                        vision_capable=worker.vision_capable,
                        sem=sem,
                        user_ctx=user_ctx,
                        agent_chain=agent_chain,
                        per_request_allowed_tools=per_request_allowed_tools,
                    )
                    for tc in tool_calls
                ]
                results = await asyncio.gather(*tasks, return_exceptions=False)
                for tool_msg, record in results:
                    loop_messages.append(tool_msg)
                    records.append(record)
                await _persist_tool_call_stats(
                    agent_id=agent_id,
                    request_stat_id=root_id,
                    records=[r for _, r in results],
                )

        # Should not reach here — the loop returns inside the body.
        return AgentExecutorResult(
            openai_response=last_response or {},
            hops_used=agent.max_tool_hops,
            tool_calls=records,
        )

    # ── Public API: streaming ────────────────────────────────

    async def run_stream(
        self,
        agent: AgentSpec,
        messages: list[dict[str, Any]],
        request_options: dict[str, Any],
        user_ctx: UserContext,
        worker: WorkerInvoker,
        per_request_headers: dict[str, str] | None = None,
        per_request_allowed_tools: list[str] | None = None,
        caller_tools: list[dict[str, Any]] | None = None,
        require_approval_override: str | None = None,
        emit_ocabra_events: bool = True,
        _agent_chain: tuple[uuid.UUID, ...] | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream the tool-loop as OpenAI SSE chunks plus ``ocabra.tool_result``.

        ``emit_ocabra_events`` gates the non-standard ``event: ocabra.tool_*``
        frames: strict OpenAI clients parse every ``data:`` as a chat chunk and
        choke on them, so callers default this off for plain clients and set it
        (via ``X-Ocabra-Stream-Events: true``) only for oCabra-aware consumers
        like the Playground. The standard chat chunks (assistant content +
        tool_calls) and ``: keepalive`` comments are emitted either way.

        The strategy is intentionally conservative:

        1. We run the LLM in **non-streaming** mode for any hop that may
           produce tool_calls (because tool_calls are notoriously hard to
           reassemble from delta chunks reliably across vLLM/Ollama).
        2. After resolving the tools we emit one ``data: {...}`` chunk
           replaying the assistant content + tool_calls of that hop, plus a
           custom ``event: ocabra.tool_result`` after each tool execution.
        3. The **final** hop (no tool_calls) is forwarded as a true SSE stream
           so the client gets real token-by-token deltas.

        Clients that don't recognise ``event: ocabra.tool_result`` will simply
        ignore those frames; the standard ``data:`` chunks remain valid
        OpenAI-compat output.
        """
        agent_chain = (*(_agent_chain or ()), agent.id)
        agent_tools, lookup = await self._load_tools(agent, per_request_allowed_tools)
        all_tools = self._merge_caller_tools(agent_tools, caller_tools)
        require_approval = _resolve_require_approval(
            agent.require_approval, require_approval_override
        )
        body_tool_choice = request_options.get("tool_choice")
        tool_choice = (
            body_tool_choice if body_tool_choice is not None else agent.tool_choice_default
        )
        loop_messages = _ensure_system_prompt(messages, agent.system_prompt)
        defaults = agent.request_defaults or {}
        merged_options: dict[str, Any] = {**defaults, **request_options}
        for key in ("tool_choice", "tools", "messages", "model", "stream"):
            merged_options.pop(key, None)

        agent_id = agent.id
        root_id = current_root_request_id.get()

        request_id = f"chatcmpl-agent-{uuid.uuid4().hex[:24]}"
        created_ts = int(time.time())
        base_model = agent.base_model_id or agent.profile_id or ""

        # Hold the worker busy for the whole streaming run (LLM hops + tool
        # execution + keepalives) so the idle eviction watchdog cannot
        # desalojate the model between hops. See
        # ProfileWorkerInvoker.acquire_run_lease.
        async with worker.acquire_run_lease():
            for hop in range(agent.max_tool_hops + 1):
                body: dict[str, Any] = {
                    **merged_options,
                    "model": base_model,
                    "messages": loop_messages,
                    "stream": False,
                }
                if all_tools:
                    body["tools"] = all_tools
                    body["tool_choice"] = tool_choice

                t_hop_start = time.monotonic()
                started = datetime.now(UTC)
                response = await worker.forward(body)
                duration_ms = int((time.monotonic() - t_hop_start) * 1000)
                usage = (response or {}).get("usage") or {}
                await _persist_hop_request_stat(
                    agent_id=agent_id,
                    parent_request_id=root_id,
                    base_model_id=base_model,
                    duration_ms=duration_ms,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    status_code=200,
                    error_message=None,
                    started_at=started,
                )

                assistant_msg = _extract_message(response) or {}
                tool_calls = assistant_msg.get("tool_calls") or []
                content_text = assistant_msg.get("content") or ""

                loop_messages = [*loop_messages, assistant_msg]

                if not tool_calls:
                    # Final hop: emit the content as a single delta chunk + final.
                    yield _sse_chunk(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": base_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": content_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    yield _sse_chunk(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": base_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": _finish_reason(response) or "stop",
                                }
                            ],
                            "usage": usage or None,
                        }
                    )
                    yield b"data: [DONE]\n\n"
                    return

                if hop >= agent.max_tool_hops:
                    yield _sse_chunk(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": base_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": content_text},
                                    "finish_reason": "tool_hop_limit",
                                }
                            ],
                        }
                    )
                    yield b"data: [DONE]\n\n"
                    return

                # Replay the assistant turn (with tool_calls) so client UIs can
                # render them inline.
                yield _sse_chunk(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": base_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": content_text,
                                    "tool_calls": tool_calls,
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )

                if require_approval == REQUIRE_APPROVAL_ALWAYS:
                    yield _sse_chunk(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": base_model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        }
                    )
                    yield b"data: [DONE]\n\n"
                    return

                # Tell the client which tools are about to run so the UI can show
                # "in progress" cards before the (possibly long) execution finishes.
                if emit_ocabra_events:
                    for tc in tool_calls:
                        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                        yield _sse_event(
                            "ocabra.tool_started",
                            {
                                "hop": hop,
                                "tool_call_id": tc.get("id") if isinstance(tc, dict) else None,
                                "name": fn.get("name") if isinstance(fn, dict) else None,
                            },
                        )

                sem = asyncio.Semaphore(self._max_concurrency)
                tasks = [
                    self._invoke_tool(
                        tool_call=tc,
                        lookup=lookup,
                        timeout_seconds=float(agent.tool_timeout_seconds),
                        per_request_headers=per_request_headers,
                        hop_index=hop,
                        vision_capable=worker.vision_capable,
                        sem=sem,
                        user_ctx=user_ctx,
                        agent_chain=agent_chain,
                        per_request_allowed_tools=per_request_allowed_tools,
                    )
                    for tc in tool_calls
                ]
                # Subagent calls and slow MCP tools can keep gather() blocked for
                # tens of seconds while the SSE stream emits nothing, which makes
                # proxies and browsers drop the connection ("network error" in the
                # playground). Emit an SSE comment as a keepalive every
                # _KEEPALIVE_INTERVAL_S seconds while the gather is still running.
                # Lines starting with `:` are SSE comments — the OpenAI parser
                # ignores them, but they reset the proxy/browser inactivity timer.
                # ``asyncio.gather`` already returns a ``_GatheringFuture``;
                # wrap it through ``ensure_future`` (which accepts both coroutines
                # and futures) so we can drive it from ``wait_for`` below.
                gather_task = asyncio.ensure_future(
                    asyncio.gather(*tasks, return_exceptions=False)
                )
                while not gather_task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(gather_task),
                            timeout=_KEEPALIVE_INTERVAL_S,
                        )
                    except TimeoutError:
                        yield b": keepalive\n\n"
                results = gather_task.result()
                for tool_msg, record in results:
                    loop_messages.append(tool_msg)
                    tc_id = tool_msg.get("tool_call_id") or ""
                    if not emit_ocabra_events:
                        continue
                    yield _sse_event(
                        "ocabra.tool_result",
                        {
                            "hop": hop,
                            "tool_call_id": tc_id,
                            "alias": record.alias,
                            "tool_name": record.tool_name,
                            "status": record.status,
                            "duration_ms": record.duration_ms,
                            "error": record.error,
                            "result_summary": record.result_summary,
                            "child_tool_calls": [
                                {
                                    "alias": c.alias,
                                    "tool_name": c.tool_name,
                                    "status": c.status,
                                    "duration_ms": c.duration_ms,
                                    "error": c.error,
                                    "result_summary": c.result_summary,
                                    "hop_index": c.hop_index,
                                }
                                for c in record.child_tool_calls
                            ],
                        },
                    )
                await _persist_tool_call_stats(
                    agent_id=agent_id,
                    request_stat_id=root_id,
                    records=[r for _, r in results],
                )

        # If we exit the loop normally, the response is already terminated
        # inside the body.  This is just defensive.
        yield b"data: [DONE]\n\n"


def _sse_chunk(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    return (f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n").encode()


# ── Helper: scope context for stats ─────────────────────────────


@contextlib.contextmanager
def agent_request_scope(*, agent_id: uuid.UUID, root_request_id: uuid.UUID | None):
    """Set the contextvars consumed by ``stats.collector`` for this request."""
    tok_a = current_agent_id.set(agent_id)
    tok_r = current_root_request_id.set(root_request_id)
    try:
        yield
    finally:
        current_agent_id.reset(tok_a)
        current_root_request_id.reset(tok_r)
