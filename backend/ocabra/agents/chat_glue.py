"""Glue helpers between :mod:`ocabra.agents.executor` and the chat endpoints.

The executor is endpoint-agnostic and only knows about a :class:`WorkerInvoker`
contract.  This module provides:

* :class:`ProfileWorkerInvoker` — a concrete invoker backed by ``WorkerPool``
  and the same profile-resolution path the non-agent OpenAI endpoint uses.
* :func:`build_invoker_for_agent` — factory used by both the OpenAI and the
  Ollama chat handlers.
* :func:`extract_per_request_headers` — collects the relevant ``x-ocabra-*``
  and ``x-mcp-*`` headers from a Starlette request.

Plan: docs/tasks/agents-mcp-plan.md — Fase 2.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from fastapi import HTTPException, Request

from ocabra.agents.executor import (
    AgentExecutor,
    AgentExecutorResult,
    REQUIRE_APPROVAL_NEVER,
    WorkerInvoker,
)
from ocabra.agents.resolver import AgentSpec, resolve_agent
from ocabra.api.openai._deps import (
    _openai_error,
    compute_worker_key,
    raise_upstream_http_error,
    resolve_profile,
)

if TYPE_CHECKING:
    from ocabra.api._deps_auth import UserContext
    from ocabra.core.model_manager import ModelManager
    from ocabra.core.profile_registry import ProfileRegistry
    from uuid import UUID

logger = structlog.get_logger(__name__)


class ProfileWorkerInvoker(WorkerInvoker):
    """Invoker that forwards each hop to the agent's resolved base model."""

    def __init__(
        self,
        *,
        worker_pool,
        worker_key: str,
        backend_model_id: str,
        vision_capable: bool,
    ) -> None:
        self._worker_pool = worker_pool
        self._worker_key = worker_key
        self._backend_model_id = backend_model_id
        self._vision_capable = vision_capable

    @property
    def vision_capable(self) -> bool:
        return self._vision_capable

    async def forward(self, body: dict[str, Any]) -> dict[str, Any]:
        # Always rewrite the model field to the backend's expected id.
        local_body = {**body, "model": self._backend_model_id}
        try:
            return await self._worker_pool.forward_request(
                self._worker_key, "/v1/chat/completions", local_body
            )
        except httpx.HTTPStatusError as exc:
            raise_upstream_http_error(exc)
            raise

    async def forward_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        local_body = {**body, "model": self._backend_model_id}
        async for chunk in self._worker_pool.forward_stream(
            self._worker_key, "/v1/chat/completions", local_body
        ):
            yield chunk


async def build_invoker_for_agent(
    agent: AgentSpec,
    *,
    model_manager: ModelManager,
    profile_registry: ProfileRegistry,
    user: UserContext,
    worker_pool,
) -> ProfileWorkerInvoker:
    """Resolve the agent's base model/profile and wrap it in a worker invoker.

    Raises an HTTP error compatible with the OpenAI endpoint if the model is
    not loadable.
    """
    if agent.profile_id:
        profile, state = await resolve_profile(
            agent.profile_id,
            model_manager,
            profile_registry,
            user=user,
        )
        worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
        backend_model_id = state.backend_model_id
        vision_capable = bool(getattr(state.capabilities, "vision", False))
        return ProfileWorkerInvoker(
            worker_pool=worker_pool,
            worker_key=worker_key,
            backend_model_id=backend_model_id,
            vision_capable=vision_capable,
        )
    if agent.base_model_id:
        # Treat it like a profile id: many deployments use base_model_id as the
        # canonical id and the profile registry resolves it via the legacy
        # fallback.  When that fails we surface the underlying 404.
        try:
            profile, state = await resolve_profile(
                agent.base_model_id,
                model_manager,
                profile_registry,
                user=user,
            )
        except HTTPException:
            raise
        worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
        backend_model_id = state.backend_model_id
        vision_capable = bool(getattr(state.capabilities, "vision", False))
        return ProfileWorkerInvoker(
            worker_pool=worker_pool,
            worker_key=worker_key,
            backend_model_id=backend_model_id,
            vision_capable=vision_capable,
        )
    raise _openai_error(
        f"Agent '{agent.slug}' has no base model configured.",
        "invalid_request_error",
        param="model",
        code="agent_misconfigured",
        status_code=500,
    )


def extract_per_request_headers(request: Request) -> dict[str, str]:
    """Return all ``x-mcp-*`` headers (used to forward auth to MCP servers)."""
    out: dict[str, str] = {}
    for raw_key, value in request.headers.items():
        key_lower = raw_key.lower()
        if key_lower.startswith("x-mcp-"):
            out[key_lower] = value
    return out


def parse_allowed_tools_header(value: str | None) -> list[str] | None:
    """Parse ``x-ocabra-allowed-tools: t1,t2`` into a list (or ``None``)."""
    if value is None:
        return None
    items = [t.strip() for t in value.split(",")]
    cleaned = [t for t in items if t]
    return cleaned if cleaned else None


def build_subagent_runner(
    executor: AgentExecutor,
    *,
    model_manager: ModelManager,
    profile_registry: ProfileRegistry,
    user: UserContext,
    worker_pool,
):
    """Return a closure that resolves and runs child agents on demand."""

    async def _run_subagent(
        child_slug: str,
        args: dict[str, Any],
        user_ctx: UserContext,
        agent_chain: tuple[UUID, ...],
        per_request_headers: dict[str, str] | None,
        per_request_allowed_tools: list[str] | None,
    ) -> AgentExecutorResult:
        del user_ctx  # the outer request user is the authority for resolution
        from ocabra.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            child = await resolve_agent(f"agent/{child_slug}", session, user=user)
        if child is None:
            raise ValueError(f"subagent_not_found:{child_slug}")
        if child.id in agent_chain:
            raise ValueError(f"subagent_cycle:{child_slug}")
        child = replace(child, require_approval="never")
        invoker = await build_invoker_for_agent(
            child,
            model_manager=model_manager,
            profile_registry=profile_registry,
            user=user,
            worker_pool=worker_pool,
        )
        messages = [{"role": "user", "content": str(args.get("task") or "").strip()}]
        context = str(args.get("context") or "").strip()
        if context:
            messages[0]["content"] = f"{messages[0]['content']}\n\nContext:\n{context}"
        return await executor.run(
            child,
            messages,
            {},
            user,
            invoker,
            per_request_headers=per_request_headers,
            per_request_allowed_tools=per_request_allowed_tools,
            require_approval_override=REQUIRE_APPROVAL_NEVER,
            _agent_chain=agent_chain,
        )

    return _run_subagent
