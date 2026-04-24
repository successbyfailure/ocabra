"""Agent + MCP runtime.

Stream A (Fase 1) ships the schema, the MCP registry, the MCP clients, and
the CRUD routers.  The AgentExecutor, the OpenAI↔MCP translation module, and
the agent resolver arrive in Stream B.

Plan: docs/tasks/agents-mcp-plan.md.
"""

from ocabra.agents.mcp_client import (
    MCPClientInterface,
    MCPTool,
    MCPToolResult,
)

__all__ = [
    "MCPClientInterface",
    "MCPTool",
    "MCPToolResult",
]
