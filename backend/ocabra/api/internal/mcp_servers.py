"""Internal REST endpoints for MCP server CRUD + refresh/test.

Plan: docs/tasks/agents-mcp-plan.md — "API REST nueva" / MCP servers.

Role matrix:

* ``GET``          → ``model_manager``
* ``POST``/``PATCH`` (http/sse) → ``model_manager``
* ``POST``/``PATCH`` (stdio)    → ``system_admin``
* ``DELETE``       → ``model_manager`` (falls back to ``system_admin`` if
                     the target is a ``stdio`` server, since only admins could
                     have created it — kept consistent for least-privilege).
* ``POST refresh`` → ``model_manager``
* ``POST test``    → ``model_manager``
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import sqlalchemy as sa
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ocabra.agents.mcp_registry import MCPRegistry, get_registry
from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.agents import AgentMCPServer
from ocabra.db.mcp import MCPServer
from ocabra.schemas.mcp import (
    MCPServerCreate,
    MCPServerOut,
    MCPServerTestResult,
    MCPServerUpdate,
    MCPToolOut,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["mcp-servers"])


# ── Helpers ──────────────────────────────────────────────────


def _get_registry(request: Request) -> MCPRegistry:
    reg = getattr(request.app.state, "mcp_registry", None) or get_registry()
    if reg is None:
        raise HTTPException(
            status_code=503,
            detail="MCP registry is not available on this node",
        )
    return reg


def _serialize(row: MCPServer) -> MCPServerOut:
    return MCPServerOut(
        id=row.id,
        alias=row.alias,
        display_name=row.display_name,
        description=row.description,
        transport=row.transport,
        url=row.url,
        command=row.command,
        args=list(row.args) if row.args else None,
        has_env=bool(row.env_encrypted),
        has_auth=bool(row.auth_value_encrypted),
        auth_type=row.auth_type,
        oauth_config=row.oauth_config,
        allowed_tools=list(row.allowed_tools) if row.allowed_tools else None,
        group_id=row.group_id,
        tools_cache=([MCPToolOut(**t) for t in row.tools_cache] if row.tools_cache else None),
        tools_cache_updated_at=row.tools_cache_updated_at,
        last_error=row.last_error,
        health_status=row.health_status,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _require_admin_for_stdio(user: UserContext, transport: str | None) -> None:
    if transport == "stdio" and not user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="transport='stdio' requires role 'system_admin'",
        )
    if transport == "stdio" and not settings.mcp_stdio_allowed:
        raise HTTPException(
            status_code=400,
            detail="transport='stdio' is disabled by server configuration",
        )


async def _load_server_or_404(session, server_id: UUID) -> MCPServer:
    row = (
        await session.execute(sa.select(MCPServer).where(MCPServer.id == server_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")
    return row


# ── CRUD ─────────────────────────────────────────────────────


@router.get(
    "/mcp-servers",
    summary="List MCP servers",
    description=(
        "Return every MCP server with its health status and redacted config.  "
        "Secrets (``env``, ``auth_value``) are never included in the response; "
        "only the ``has_*`` flags are exposed."
    ),
    response_model=list[MCPServerOut],
)
async def list_mcp_servers(
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[MCPServerOut]:
    """List all MCP servers (redacted)."""
    async with AsyncSessionLocal() as session:
        rows = (
            (await session.execute(sa.select(MCPServer).order_by(MCPServer.alias))).scalars().all()
        )
    return [_serialize(r) for r in rows]


@router.post(
    "/mcp-servers",
    summary="Register a new MCP server",
    status_code=status.HTTP_201_CREATED,
    response_model=MCPServerOut,
    responses={
        400: {"description": "Invalid body (transport/auth mismatch, duplicate alias, ...)"},
        403: {"description": "stdio transport requires system_admin"},
    },
)
async def create_mcp_server(
    body: MCPServerCreate,
    request: Request,
    user: UserContext = Depends(require_role("model_manager")),
) -> MCPServerOut:
    """Create a new MCP server.  ``transport=stdio`` requires ``system_admin``."""
    _require_admin_for_stdio(user, body.transport)

    registry = _get_registry(request)

    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(sa.select(MCPServer.id).where(MCPServer.alias == body.alias))
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Alias '{body.alias}' is already registered",
            )
        auth_cipher = (
            registry.encrypt_json(body.auth_value.model_dump(exclude_none=True))
            if body.auth_value is not None
            else None
        )
        env_cipher = registry.encrypt_json(body.env) if body.env else None
        row = MCPServer(
            alias=body.alias,
            display_name=body.display_name,
            description=body.description,
            transport=body.transport,
            url=body.url,
            command=body.command,
            args=list(body.args) if body.args else None,
            env_encrypted=env_cipher,
            auth_type=body.auth_type,
            auth_value_encrypted=auth_cipher,
            oauth_config=body.oauth_config,
            allowed_tools=list(body.allowed_tools) if body.allowed_tools else None,
            group_id=body.group_id,
            health_status="unknown",
            created_by=UUID(user.user_id) if user.user_id else None,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)

    try:
        await registry.register(row)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mcp_server_register_failed", alias=row.alias, error=str(exc))

    return _serialize(row)


@router.get(
    "/mcp-servers/{server_id}",
    summary="Retrieve an MCP server",
    response_model=MCPServerOut,
    responses={404: {"description": "Server not found"}},
)
async def get_mcp_server(
    server_id: UUID,
    _user: UserContext = Depends(require_role("model_manager")),
) -> MCPServerOut:
    """Return the redacted config of a single MCP server."""
    async with AsyncSessionLocal() as session:
        row = await _load_server_or_404(session, server_id)
    return _serialize(row)


@router.patch(
    "/mcp-servers/{server_id}",
    summary="Update an MCP server",
    response_model=MCPServerOut,
    responses={
        400: {"description": "Invalid patch"},
        403: {"description": "stdio transport requires system_admin"},
        404: {"description": "Server not found"},
    },
)
async def update_mcp_server(
    server_id: UUID,
    body: MCPServerUpdate,
    request: Request,
    user: UserContext = Depends(require_role("model_manager")),
) -> MCPServerOut:
    """Apply a partial update.  Sensitive fields are re-encrypted."""
    registry = _get_registry(request)
    patch = body.model_dump(exclude_unset=True)

    async with AsyncSessionLocal() as session:
        row = await _load_server_or_404(session, server_id)

        next_transport = patch.get("transport", row.transport)
        # Guard: switching to stdio or editing a stdio server requires admin.
        if row.transport == "stdio" or next_transport == "stdio":
            _require_admin_for_stdio(user, "stdio")

        if "display_name" in patch:
            row.display_name = patch["display_name"]
        if "description" in patch:
            row.description = patch["description"]
        if "transport" in patch:
            row.transport = patch["transport"]
        if "url" in patch:
            row.url = patch["url"]
        if "command" in patch:
            row.command = patch["command"]
        if "args" in patch:
            row.args = list(patch["args"]) if patch["args"] else None
        if "env" in patch:
            row.env_encrypted = registry.encrypt_json(patch["env"]) if patch["env"] else None
        if "auth_type" in patch:
            row.auth_type = patch["auth_type"]
            if patch["auth_type"] == "none":
                row.auth_value_encrypted = None
        if "auth_value" in patch:
            payload = patch["auth_value"]
            row.auth_value_encrypted = registry.encrypt_json(payload) if payload else None
        if "oauth_config" in patch:
            row.oauth_config = patch["oauth_config"]
        if "allowed_tools" in patch:
            row.allowed_tools = list(patch["allowed_tools"]) if patch["allowed_tools"] else None
        if "group_id" in patch:
            row.group_id = patch["group_id"]

        # Any config change invalidates the runtime cache on this alias.
        row.tools_cache = None
        row.tools_cache_updated_at = None
        row.health_status = "unknown"
        row.last_error = None

        await session.commit()
        await session.refresh(row)

    registry.invalidate(row.alias)
    try:
        await registry.register(row)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mcp_server_update_register_failed", alias=row.alias, error=str(exc))

    return _serialize(row)


@router.delete(
    "/mcp-servers/{server_id}",
    summary="Delete an MCP server",
    responses={
        404: {"description": "Server not found"},
        409: {"description": "Server is referenced by one or more agents"},
    },
)
async def delete_mcp_server(
    server_id: UUID,
    request: Request,
    force: bool = Query(False, description="Delete even when agents reference the server"),
    user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    """Delete an MCP server.  Fails with 409 if referenced unless ``force=true``."""
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        row = await _load_server_or_404(session, server_id)

        # stdio deletion still requires admin to keep the privilege boundary symmetric.
        if row.transport == "stdio" and not user.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Deleting a stdio MCP server requires role 'system_admin'",
            )

        in_use = (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(AgentMCPServer)
                .where(AgentMCPServer.mcp_server_id == row.id)
            )
        ).scalar_one()
        if in_use and not force:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"MCP server '{row.alias}' is used by {in_use} agent(s); "
                    "pass ?force=true to delete anyway."
                ),
            )
        alias = row.alias
        await session.delete(row)
        await session.commit()

    await registry.unregister(alias)
    return {"ok": True, "alias": alias}


# ── Tools (refresh + list) ───────────────────────────────────


@router.post(
    "/mcp-servers/{server_id}/refresh",
    summary="Refresh tools/list for an MCP server",
    response_model=list[MCPToolOut],
    responses={
        404: {"description": "Server not found"},
        502: {"description": "Unable to reach the MCP server"},
    },
)
async def refresh_mcp_tools(
    server_id: UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[MCPToolOut]:
    """Force a ``tools/list`` refresh and persist the cache in the DB."""
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        row = await _load_server_or_404(session, server_id)
        alias = row.alias
        try:
            tools = await registry.refresh(alias, session=session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            row.last_error = str(exc)
            row.health_status = "unhealthy"
            await session.commit()
            raise HTTPException(
                status_code=502,
                detail=f"Unable to refresh MCP server '{alias}': {exc}",
            ) from exc
        await session.commit()
    return [MCPToolOut(**t.to_dict()) for t in tools]


@router.get(
    "/mcp-servers/{server_id}/tools",
    summary="List cached tools for an MCP server",
    response_model=list[MCPToolOut],
    responses={404: {"description": "Server not found"}},
)
async def list_mcp_tools(
    server_id: UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[MCPToolOut]:
    """Return the currently cached tools (may refresh on TTL expiry)."""
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        row = await _load_server_or_404(session, server_id)
        alias = row.alias
    try:
        tools = await registry.get_tools(alias)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Unable to list tools for '{alias}': {exc}",
        ) from exc
    return [MCPToolOut(**t.to_dict()) for t in tools]


@router.post(
    "/mcp-servers/{server_id}/test",
    summary="Test connectivity to an MCP server",
    response_model=MCPServerTestResult,
    responses={404: {"description": "Server not found"}},
)
async def test_mcp_server(
    server_id: UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> MCPServerTestResult:
    """Run a dry connection test and return ``{healthy, tools_count, error}``."""
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        row = await _load_server_or_404(session, server_id)
        alias = row.alias
    try:
        tools = await registry.refresh(alias)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        return MCPServerTestResult(healthy=False, tools_count=0, error=str(exc))
    return MCPServerTestResult(healthy=True, tools_count=len(tools), error=None)
