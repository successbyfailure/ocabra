"""MCP client abstractions used by :mod:`ocabra.agents.mcp_registry`.

Three transports are supported:

* ``http``  — MCP over HTTP (``mcp.client.streamable_http``).
* ``sse``   — MCP over Server-Sent Events (``mcp.client.sse``).
* ``stdio`` — MCP over a local subprocess (``mcp.client.stdio``).

The :class:`MCPClientInterface` defines the minimal surface consumed by the
registry and (later) by the AgentExecutor.  Per-request headers (``x-mcp-*``)
are passed via ``extra_headers`` on :meth:`call_tool` but must **never**
override headers baked into the server config — that invariant is enforced by
:class:`_HeaderMerger` and the registry.

The ``mcp`` SDK is imported lazily so that a dev environment without the
package can still import this module and run the tests that mock the clients.

Plan: docs/tasks/agents-mcp-plan.md — "Contratos" / MCPClientInterface.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ── Data classes ─────────────────────────────────────────────


@dataclass
class MCPTool:
    """A single tool advertised by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class MCPToolResult:
    """Result of a single MCP ``tools/call`` invocation."""

    content: list[dict[str, Any]]
    is_error: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


# ── Header-merge helper ──────────────────────────────────────


class _HeaderMerger:
    """Merge ``extra_headers`` from the caller on top of ``static_headers``.

    Headers already present in ``static_headers`` are never overridden by the
    caller.  This is the mechanism that prevents a client from using
    ``x-mcp-{alias}-authorization`` to bypass server-side auth (see plan,
    "Seguridad — checklist obligatoria").
    """

    def __init__(self, static_headers: Mapping[str, str] | None) -> None:
        self._static: dict[str, str] = {
            k.lower(): v for k, v in (static_headers or {}).items()
        }

    @property
    def static(self) -> dict[str, str]:
        return dict(self._static)

    def merge(self, extra_headers: Mapping[str, str] | None) -> dict[str, str]:
        merged: dict[str, str] = dict(self._static)
        if not extra_headers:
            return merged
        for raw_key, raw_value in extra_headers.items():
            key_l = raw_key.lower()
            if key_l in self._static:
                # Static header wins — never let the caller override server auth.
                continue
            merged[key_l] = raw_value
        return merged


# ── Interface ────────────────────────────────────────────────


class MCPClientInterface(ABC):
    """Common surface for every MCP transport implementation."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the underlying session.  Idempotent."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources.  Idempotent."""

    @abstractmethod
    async def list_tools(self) -> list[MCPTool]:
        """Fetch the tool inventory (``tools/list``)."""

    @abstractmethod
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float,
        extra_headers: Mapping[str, str] | None = None,
    ) -> MCPToolResult:
        """Invoke a single tool and return its result."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the server is reachable, ``False`` otherwise."""


# ── SDK import helper ────────────────────────────────────────


def _load_mcp():
    """Import the ``mcp`` SDK lazily.  Raises a clear error if missing."""
    try:
        import mcp  # noqa: F401
        import mcp.client.session as session_mod
    except ImportError as exc:  # pragma: no cover - exercised in dev
        raise RuntimeError(
            "The 'mcp' SDK is required to talk to MCP servers. "
            "Install it with `pip install mcp>=0.9.0`."
        ) from exc
    return session_mod


def _flatten_content_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalise the SDK's content blocks to plain dicts.

    The SDK returns a list of ``mcp.types.*Content`` models (TextContent,
    ImageContent, ResourceLink, ...) but we store plain JSON-serialisable
    dicts to keep the registry result opaque to the rest of the codebase.
    """
    if content is None:
        return []
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            blocks.append(block)
            continue
        dump = getattr(block, "model_dump", None)
        if callable(dump):
            try:
                blocks.append(dump(mode="python"))
                continue
            except TypeError:
                blocks.append(dump())
                continue
        # Fallback: best-effort dict representation
        blocks.append({"type": "text", "text": str(block)})
    return blocks


# ── Concrete implementations ─────────────────────────────────


class _BaseClient(MCPClientInterface):
    """Shared plumbing for HTTP/SSE clients (server-driven async context)."""

    def __init__(self) -> None:
        self._exit_stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        async with self._lock:
            if self._exit_stack is None:
                return
            stack = self._exit_stack
            self._exit_stack = None
            self._session = None
            try:
                await stack.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning("mcp_client_close_failed", error=str(exc))

    async def list_tools(self) -> list[MCPTool]:
        await self.connect()
        session = self._session
        assert session is not None  # noqa: S101 — for type-narrow only
        response = await session.list_tools()
        result: list[MCPTool] = []
        for tool in getattr(response, "tools", []):
            result.append(
                MCPTool(
                    name=getattr(tool, "name", ""),
                    description=getattr(tool, "description", "") or "",
                    input_schema=getattr(tool, "inputSchema", None)
                    or getattr(tool, "input_schema", None)
                    or {},
                )
            )
        return result

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float,
        extra_headers: Mapping[str, str] | None = None,
    ) -> MCPToolResult:
        # NB: ``extra_headers`` is plumbed for per-request auth.  The HTTP/SSE
        # transports in the MCP SDK do not yet accept per-call header overrides
        # without reopening the session.  Stream B will implement re-connect
        # with merged headers when the agent passes non-empty extra_headers.
        # Here we simply connect with the static headers.
        await self.connect()
        session = self._session
        assert session is not None  # noqa: S101 — for type-narrow only
        try:
            raw = await asyncio.wait_for(
                session.call_tool(name=name, arguments=arguments),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            return MCPToolResult(
                content=[{"type": "text", "text": "tool_timeout"}],
                is_error=True,
                raw={"error": "timeout", "timeout_seconds": timeout_seconds},
            )
        content = _flatten_content_blocks(getattr(raw, "content", None))
        is_error = bool(getattr(raw, "isError", False) or getattr(raw, "is_error", False))
        dump = getattr(raw, "model_dump", None)
        raw_dict = dump(mode="python") if callable(dump) else {"content": content}
        return MCPToolResult(content=content, is_error=is_error, raw=raw_dict)

    async def health_check(self) -> bool:
        try:
            await self.list_tools()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("mcp_client_health_check_failed", error=str(exc))
            return False


class HttpMCPClient(_BaseClient):
    """MCP over streamable HTTP."""

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__()
        self._url = url
        self._merger = _HeaderMerger(headers)
        self._timeout_s = timeout_seconds

    async def connect(self) -> None:
        async with self._lock:
            if self._session is not None:
                return
            session_mod = _load_mcp()
            from mcp.client.streamable_http import streamablehttp_client

            stack = AsyncExitStack()
            try:
                read, write, *_ = await stack.enter_async_context(
                    streamablehttp_client(
                        self._url,
                        headers=self._merger.static,
                        timeout=self._timeout_s,
                    )
                )
                session = await stack.enter_async_context(
                    session_mod.ClientSession(read, write)
                )
                await session.initialize()
            except Exception:
                await stack.aclose()
                raise
            self._exit_stack = stack
            self._session = session


class SseMCPClient(_BaseClient):
    """MCP over Server-Sent Events."""

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__()
        self._url = url
        self._merger = _HeaderMerger(headers)
        self._timeout_s = timeout_seconds

    async def connect(self) -> None:
        async with self._lock:
            if self._session is not None:
                return
            session_mod = _load_mcp()
            from mcp.client.sse import sse_client

            stack = AsyncExitStack()
            try:
                read, write = await stack.enter_async_context(
                    sse_client(
                        url=self._url,
                        headers=self._merger.static,
                        timeout=self._timeout_s,
                    )
                )
                session = await stack.enter_async_context(
                    session_mod.ClientSession(read, write)
                )
                await session.initialize()
            except Exception:
                await stack.aclose()
                raise
            self._exit_stack = stack
            self._session = session


class StdioMCPClient(_BaseClient):
    """MCP over a local subprocess.

    The subprocess is spawned with a *sanitised* environment: nothing is
    inherited from the parent process (the ``env`` dict is passed as-is and
    merged with a minimal ``PATH`` fallback so the ``command`` resolves).

    Security: stdio is admin-only (enforced at the router level) and the
    subprocess ``cwd`` defaults to ``/data/mcp/<alias>/`` so filesystem scope
    is bounded.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        super().__init__()
        self._command = command
        self._args = list(args or [])
        self._env = dict(env or {})
        self._cwd = cwd

    async def connect(self) -> None:
        async with self._lock:
            if self._session is not None:
                return
            session_mod = _load_mcp()
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            sanitised_env: dict[str, str] = {}
            # Minimal PATH fallback so the command resolves.  Explicit PATH in
            # the caller ``env`` wins.  No other variables leak from parent.
            sanitised_env["PATH"] = self._env.get(
                "PATH", os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
            )
            sanitised_env.update(self._env)

            params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=sanitised_env,
                cwd=self._cwd,
            )
            stack = AsyncExitStack()
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(
                    session_mod.ClientSession(read, write)
                )
                await session.initialize()
            except Exception:
                await stack.aclose()
                raise
            self._exit_stack = stack
            self._session = session
