"""How the api runs Docker operations: through the docker agent when there is one.

Every Docker call of the api goes through :func:`call` / :func:`stream` by
operation name (see :mod:`ocabra.docker_agent.ops` for the list and what each
one allows). With ``DOCKER_AGENT_URL`` set and the agent's token readable, the
request goes to the agent container, which holds the Docker socket; otherwise
the same operation runs here, with the same validation.

Phase 1 of SEG-03 (2026-11-17): the api still has the socket, so an agent that
does not answer falls back to running locally (logged). Phase 2 removes the
socket from the api and, with it, that fallback.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.config import settings
from ocabra.docker_agent import ops

logger = structlog.get_logger(__name__)

OpError = ops.OpError
EXIT_MARKER = ops.EXIT_MARKER


def _agent_token() -> str | None:
    if not settings.docker_agent_url:
        return None
    try:
        token = Path(settings.docker_agent_token_file).read_text().strip()
    except OSError:
        return None
    return token or None


def available() -> bool:
    """Whether this api can run Docker operations at all (agent or local socket)."""
    return bool(_agent_token()) or os.path.exists("/var/run/docker.sock")


def using_agent() -> bool:
    return _agent_token() is not None


async def call(op: str, **args: Any) -> dict[str, Any]:
    """Run ``op``; returns its result (``code``, ``stdout``, ``stderr``, extras).

    Raises :class:`OpError` when the operation rejects the arguments.
    """
    token = _agent_token()
    if token:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(
                    f"{settings.docker_agent_url.rstrip('/')}/v1/op",
                    json={"op": op, "args": args},
                    headers={"Authorization": f"Bearer {token}"},
                )
            if response.status_code == 400:
                raise OpError(response.json().get("detail") or "rejected by the docker agent")
            response.raise_for_status()
            return response.json()
        except OpError:
            raise
        except httpx.HTTPError as exc:
            logger.warning("docker_agent_unavailable_local_fallback", op=op, error=str(exc))
    return await ops.run(op, args)


async def stream(op: str, **args: Any) -> AsyncIterator[str]:
    """Run streaming ``op``, yielding output lines; the last is ``EXIT_MARKER<code>``."""
    token = _agent_token()
    if token:
        started = False
        try:
            async for line in _agent_stream(token, op, args):
                started = True
                yield line
            return
        except OpError:
            raise
        except httpx.HTTPError as exc:
            if started:
                # Cut halfway: the phase already ran in the agent; running it
                # again here would compile twice. Report it as failed.
                raise RuntimeError(f"docker agent stream for {op} was cut: {exc}") from exc
            logger.warning("docker_agent_unavailable_local_fallback", op=op, error=str(exc))
    async for line in ops.stream(op, args):
        yield line


async def _agent_stream(token: str, op: str, args: dict[str, Any]) -> AsyncIterator[str]:
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client,
        client.stream(
            "POST",
            f"{settings.docker_agent_url.rstrip('/')}/v1/stream",
            json={"op": op, "args": args},
            headers={"Authorization": f"Bearer {token}"},
        ) as response,
    ):
        if response.status_code == 400:
            detail = json.loads(await response.aread()).get("detail")
            raise OpError(detail or "rejected by the docker agent")
        response.raise_for_status()
        async for raw in response.aiter_lines():
            if not raw:
                continue
            item = json.loads(raw)
            if "exit" in item:
                yield f"{EXIT_MARKER}{int(item['exit'])}"
            else:
                yield str(item.get("line", ""))
