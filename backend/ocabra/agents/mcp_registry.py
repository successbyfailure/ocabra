"""Runtime registry of MCP servers.

Responsibilities:

* Load MCP server rows from the DB on startup.
* Maintain one :class:`MCPClientInterface` per registered alias (pool).
* Cache ``tools/list`` responses with a configurable TTL.
* Expose ``refresh``, ``invalidate``, and ``health_check`` helpers.
* Own the Fernet helpers that encrypt/decrypt ``auth_value`` and ``env``.

The registry is strictly Fase 1 scoped — it does **not** execute tool calls on
behalf of an agent (that lives in the AgentExecutor, Stream B).  What it does
expose is :meth:`call_tool`, which the executor will use as the single access
point.

Plan: docs/tasks/agents-mcp-plan.md — "Fase 1 — Schema + registry".
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from cryptography.fernet import Fernet, InvalidToken

from ocabra.agents.mcp_client import (
    HttpMCPClient,
    MCPClientInterface,
    MCPTool,
    MCPToolResult,
    SseMCPClient,
    StdioMCPClient,
)
from ocabra.config import settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from ocabra.db.mcp import MCPServer

logger = structlog.get_logger(__name__)


# ── Encryption helpers (shared pattern with federation) ──────


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a Fernet-compatible key from an arbitrary secret string."""
    digest = hashlib.sha256(secret.encode()).digest()
    return base64.urlsafe_b64encode(digest)


# ── Data classes ─────────────────────────────────────────────


@dataclass
class _CacheEntry:
    tools: list[MCPTool]
    fetched_at: float = field(default_factory=time.monotonic)

    def is_stale(self, ttl_seconds: int) -> bool:
        if ttl_seconds <= 0:
            return True
        return (time.monotonic() - self.fetched_at) > ttl_seconds


@dataclass
class MCPServerRuntime:
    """In-memory view of a registered MCP server."""

    id: UUID
    alias: str
    display_name: str
    transport: str
    health_status: str = "unknown"
    last_error: str | None = None
    allowed_tools: list[str] | None = None
    group_id: UUID | None = None


# ── Registry ─────────────────────────────────────────────────


class MCPRegistry:
    """Process-wide MCP server registry.

    Thread-safety: the registry is designed to run in a single asyncio event
    loop.  Internal state is guarded by an :class:`asyncio.Lock` so concurrent
    refreshes and CRUD mutations do not race.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        fernet_secret: str | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self._session_factory = session_factory
        secret = fernet_secret or settings.jwt_secret
        self._fernet = Fernet(_derive_fernet_key(secret))
        self._cache_ttl = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else settings.mcp_tools_cache_ttl_seconds
        )

        self._lock = asyncio.Lock()
        self._runtime: dict[str, MCPServerRuntime] = {}  # alias → runtime
        self._clients: dict[str, MCPClientInterface] = {}  # alias → client
        self._cache: dict[str, _CacheEntry] = {}  # alias → cache entry
        self._started = False

    # ── Lifecycle ───────────────────────────────────────────

    async def start(self) -> None:
        """Load all MCP server rows from the DB and build clients."""
        if self._started:
            return
        self._started = True
        if self._session_factory is None:
            logger.info("mcp_registry_started_without_db")
            return
        try:
            async with self._session_factory() as session:
                await self.reload_from_db(session)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp_registry_initial_reload_failed", error=str(exc))

    async def stop(self) -> None:
        """Close every open client.  Idempotent."""
        async with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
            self._runtime.clear()
            self._cache.clear()
            self._started = False
        for client in clients:
            try:
                await client.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("mcp_registry_client_close_failed", error=str(exc))

    # ── Fernet helpers ──────────────────────────────────────

    def encrypt_text(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt_text(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:  # noqa: BLE001
            raise ValueError("Unable to decrypt MCP server secret") from exc

    def encrypt_json(self, payload: Mapping[str, Any] | None) -> str | None:
        if payload is None:
            return None
        return self.encrypt_text(json.dumps(dict(payload)))

    def decrypt_json(self, ciphertext: str | None) -> dict[str, Any] | None:
        if not ciphertext:
            return None
        return json.loads(self.decrypt_text(ciphertext))

    # ── Runtime accessors ───────────────────────────────────

    def list_runtime(self) -> list[MCPServerRuntime]:
        return list(self._runtime.values())

    def get_runtime(self, alias: str) -> MCPServerRuntime | None:
        return self._runtime.get(alias)

    def get_client(self, alias: str) -> MCPClientInterface | None:
        return self._clients.get(alias)

    # ── DB reload / upsert ──────────────────────────────────

    async def reload_from_db(self, session: AsyncSession) -> None:
        """Rebuild the in-memory state from the current DB rows."""
        import sqlalchemy as sa

        from ocabra.db.mcp import MCPServer

        rows = (await session.execute(sa.select(MCPServer))).scalars().all()
        async with self._lock:
            # Close every previous client.
            to_close = list(self._clients.values())
            self._clients.clear()
            self._runtime.clear()
            self._cache.clear()

            for row in rows:
                try:
                    client = self._build_client(row)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "mcp_registry_client_build_failed",
                        alias=row.alias,
                        error=str(exc),
                    )
                    continue
                self._clients[row.alias] = client
                self._runtime[row.alias] = MCPServerRuntime(
                    id=row.id,
                    alias=row.alias,
                    display_name=row.display_name,
                    transport=row.transport,
                    health_status=row.health_status,
                    last_error=row.last_error,
                    allowed_tools=list(row.allowed_tools) if row.allowed_tools else None,
                    group_id=row.group_id,
                )
                if row.tools_cache:
                    try:
                        self._cache[row.alias] = _CacheEntry(
                            tools=[
                                MCPTool(
                                    name=t.get("name", ""),
                                    description=t.get("description", "") or "",
                                    input_schema=t.get("input_schema", {}) or {},
                                )
                                for t in row.tools_cache
                            ],
                            fetched_at=time.monotonic(),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "mcp_registry_cache_hydration_failed",
                            alias=row.alias,
                            error=str(exc),
                        )

        # Close old clients outside the lock so nothing blocks on network I/O.
        for c in to_close:
            try:
                await c.close()
            except Exception:  # noqa: BLE001
                pass

    async def register(self, row: MCPServer) -> None:
        """Attach a new MCP server to the registry or replace an existing one."""
        async with self._lock:
            old = self._clients.pop(row.alias, None)
            try:
                self._clients[row.alias] = self._build_client(row)
            except Exception:
                if old is not None:
                    self._clients[row.alias] = old
                raise
            self._runtime[row.alias] = MCPServerRuntime(
                id=row.id,
                alias=row.alias,
                display_name=row.display_name,
                transport=row.transport,
                health_status=row.health_status,
                last_error=row.last_error,
                allowed_tools=list(row.allowed_tools) if row.allowed_tools else None,
                group_id=row.group_id,
            )
            self._cache.pop(row.alias, None)
        if old is not None:
            try:
                await old.close()
            except Exception:  # noqa: BLE001
                pass

    async def unregister(self, alias: str) -> None:
        """Remove a server from the registry and close its client."""
        async with self._lock:
            runtime = self._runtime.pop(alias, None)
            client = self._clients.pop(alias, None)
            self._cache.pop(alias, None)
        if runtime is not None and client is not None:
            try:
                await client.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "mcp_registry_unregister_close_failed",
                    alias=alias,
                    error=str(exc),
                )

    # ── Tool discovery ──────────────────────────────────────

    async def get_tools(self, alias: str, *, force_refresh: bool = False) -> list[MCPTool]:
        """Return cached tools, refreshing when the TTL expires."""
        client = self._clients.get(alias)
        if client is None:
            raise KeyError(f"MCP server '{alias}' is not registered")
        entry = self._cache.get(alias)
        if entry is not None and not force_refresh and not entry.is_stale(self._cache_ttl):
            return list(entry.tools)
        tools = await client.list_tools()
        self._cache[alias] = _CacheEntry(tools=list(tools), fetched_at=time.monotonic())
        return list(tools)

    def invalidate(self, alias: str) -> None:
        """Drop any cached ``tools/list`` for *alias*."""
        self._cache.pop(alias, None)

    async def refresh(self, alias: str, *, session: AsyncSession | None = None) -> list[MCPTool]:
        """Force a re-fetch and update the DB cache columns when possible."""
        tools = await self.get_tools(alias, force_refresh=True)
        if session is not None:
            import sqlalchemy as sa

            from ocabra.db.mcp import MCPServer

            runtime = self._runtime.get(alias)
            if runtime is None:
                return tools
            await session.execute(
                sa.update(MCPServer)
                .where(MCPServer.id == runtime.id)
                .values(
                    tools_cache=[t.to_dict() for t in tools],
                    tools_cache_updated_at=sa.func.now(),
                    last_error=None,
                    health_status="healthy",
                )
            )
            runtime.health_status = "healthy"
            runtime.last_error = None
        return tools

    # ── Health check ────────────────────────────────────────

    async def health_check(self, alias: str) -> bool:
        """Ping a server and update its recorded status."""
        client = self._clients.get(alias)
        runtime = self._runtime.get(alias)
        if client is None or runtime is None:
            return False
        try:
            healthy = await client.health_check()
        except Exception as exc:  # noqa: BLE001
            runtime.health_status = "unhealthy"
            runtime.last_error = str(exc)
            return False
        runtime.health_status = "healthy" if healthy else "unhealthy"
        if healthy:
            runtime.last_error = None
        return healthy

    # ── Tool invocation ─────────────────────────────────────

    async def call_tool(
        self,
        alias: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> MCPToolResult:
        """Invoke ``tool_name`` on server ``alias``.

        Stream A does not call this from a tool-loop — it is exposed here so
        the CRUD ``/mcp-servers/{id}/test`` endpoint and Stream B share a
        single entrypoint.
        """
        client = self._clients.get(alias)
        if client is None:
            raise KeyError(f"MCP server '{alias}' is not registered")
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.mcp_default_tool_timeout_seconds
        )
        return await client.call_tool(
            tool_name,
            arguments,
            timeout_seconds=effective_timeout,
            extra_headers=extra_headers,
        )

    # ── Client factory ──────────────────────────────────────

    def _build_client(self, row: MCPServer) -> MCPClientInterface:
        static_headers: dict[str, str] = {}
        if row.auth_type in ("api_key", "bearer", "basic") and row.auth_value_encrypted:
            auth_payload = self.decrypt_json(row.auth_value_encrypted) or {}
            header_name = auth_payload.get("header_name") or "Authorization"
            raw_value = auth_payload.get("value") or ""
            if row.auth_type == "bearer":
                static_headers[header_name] = (
                    raw_value if raw_value.lower().startswith("bearer ") else f"Bearer {raw_value}"
                )
            elif row.auth_type == "basic":
                static_headers[header_name] = (
                    raw_value if raw_value.lower().startswith("basic ") else f"Basic {raw_value}"
                )
            else:  # api_key
                static_headers[header_name] = raw_value

        transport = row.transport
        if transport == "http":
            if not row.url:
                raise ValueError(f"MCP server '{row.alias}' has transport=http but no url")
            return HttpMCPClient(row.url, headers=static_headers)
        if transport == "sse":
            if not row.url:
                raise ValueError(f"MCP server '{row.alias}' has transport=sse but no url")
            return SseMCPClient(row.url, headers=static_headers)
        if transport == "stdio":
            if not row.command:
                raise ValueError(f"MCP server '{row.alias}' has transport=stdio but no command")
            env = self.decrypt_json(row.env_encrypted)
            return StdioMCPClient(
                command=row.command,
                args=list(row.args or []),
                env=env or {},
                cwd=f"/data/mcp/{row.alias}",
            )
        raise ValueError(f"Unsupported transport '{transport}'")


# ── Singleton accessor ───────────────────────────────────────


_registry: MCPRegistry | None = None


def get_registry() -> MCPRegistry | None:
    """Return the process-wide registry, or ``None`` if uninitialised."""
    return _registry


def set_registry(registry: MCPRegistry | None) -> None:
    """Install (or clear) the process-wide registry.  Used in lifespan/tests."""
    global _registry
    _registry = registry
