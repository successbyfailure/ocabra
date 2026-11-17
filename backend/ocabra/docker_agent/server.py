"""HTTP front of the docker agent: the operations of :mod:`.ops`, nothing more.

Listens only on the compose network (no published port). Every call needs the
token the agent generates at startup and writes to a volume shared only with
the api (``settings.docker_agent_token_file``), so there is no secret to
configure, and a restart of the agent rotates it.

- ``POST /v1/op``     ``{"op": ..., "args": {...}}`` → the operation's JSON result
- ``POST /v1/stream`` same body → NDJSON ``{"line": ...}`` lines, then ``{"exit": n}``
- ``GET  /health``    no token, no information
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ocabra.config import settings

from . import ops

logger = structlog.get_logger(__name__)

app = FastAPI(title="oCabra docker agent", docs_url=None, redoc_url=None, openapi_url=None)
_token = ""


def _write_token() -> str:
    token = secrets.token_urlsafe(32)
    path = Path(settings.docker_agent_token_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(token)
    # Readable by the api, which runs as another user. The volume holding it
    # is mounted only by the agent and the api.
    os.chmod(tmp, 0o644)
    tmp.replace(path)  # atomic: the api never reads a half-written token
    return token


@app.on_event("startup")
async def _startup() -> None:  # noqa: D401 - FastAPI hook
    global _token
    _token = _write_token()
    logger.info("docker_agent_ready", operations=sorted([*ops.OPERATIONS, *ops.STREAM_OPERATIONS]))


def _authorize(request: Request) -> None:
    given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not _token or not given or not hmac.compare_digest(given, _token):
        raise HTTPException(status_code=401, detail="invalid agent token")


async def _body(request: Request) -> tuple[str, dict[str, Any]]:
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    op, args = payload.get("op"), payload.get("args") or {}
    if not isinstance(op, str) or not isinstance(args, dict):
        raise HTTPException(status_code=400, detail="expected {op: str, args: object}")
    return op, args


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/op")
async def run_op(request: Request) -> JSONResponse:
    _authorize(request)
    op, args = await _body(request)
    try:
        result = await ops.run(op, args)
    except ops.OpError as exc:
        logger.warning("docker_agent_rejected", op=op, reason=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("docker_agent_op", op=op, code=result.get("code"))
    return JSONResponse(result)


@app.post("/v1/stream")
async def run_stream(request: Request) -> StreamingResponse:
    _authorize(request)
    op, args = await _body(request)
    try:
        lines = ops.stream(op, args)
        first = await anext(lines)  # validation runs here: reject before the 200
    except ops.OpError as exc:
        logger.warning("docker_agent_rejected", op=op, reason=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StopAsyncIteration:
        first = None
    logger.info("docker_agent_stream", op=op)

    async def _ndjson():
        pending = [first] if first is not None else []
        try:
            for line in pending:
                yield _encode(line)
            async for line in lines:
                yield _encode(line)
        finally:
            await lines.aclose()  # kills the process if the client went away

    return StreamingResponse(_ndjson(), media_type="application/x-ndjson")


def _encode(line: str) -> bytes:
    if line.startswith(ops.EXIT_MARKER):
        return (json.dumps({"exit": int(line[len(ops.EXIT_MARKER) :])}) + "\n").encode()
    return (json.dumps({"line": line}) + "\n").encode()
