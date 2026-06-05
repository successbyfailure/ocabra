"""
POST /v1/completions — legacy text completions endpoint.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ocabra.api._deps_auth import UserContext

from ._deps import (
    STREAMING_LOAD_RESPONSE_DOC,
    build_model_status_headers,
    check_capability,
    compute_worker_key,
    ensure_worker_loaded,
    keepalive_until_done,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    lookup_profile,
    merge_profile_defaults,
    raise_upstream_http_error,
    resolve_profile,
    sse_ocabra_event,
    to_backend_body,
)

router = APIRouter()


@router.post(
    "/completions",
    summary="Create text completion",
    responses=STREAMING_LOAD_RESPONSE_DOC,
)
async def completions(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Create a text completion. Proxies to the resolved model worker (backend-agnostic).
    """
    body = await request.json()
    model_id: str = body.get("model", "")
    stream: bool = body.get("stream", False)

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    # --- Federation hook ---
    federation_manager = get_federation_manager(request)
    if federation_manager is not None:
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            target, peer = await resolve_federated(
                model_id, model_manager, federation_manager, profile_registry
            )
            if target == "remote":
                request.state.federation_remote_node_id = peer.peer_id
                if stream:
                    return StreamingResponse(
                        federation_manager.proxy_stream(
                            peer=peer,
                            path=request.url.path,
                            body=body,
                            headers=dict(request.headers),
                        ),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                resp = await federation_manager.proxy_request(
                    peer=peer,
                    path=request.url.path,
                    body=body,
                    headers=dict(request.headers),
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"),
                )
    # --- End federation hook ---

    worker_pool = request.app.state.worker_pool

    # Streaming: pre-flush headers (incl. X-Ocabra-Expected-Wait-Seconds)
    # before triggering the load.
    if stream:
        try:
            profile = await lookup_profile(model_id, profile_registry, user=user)
        except HTTPException:
            profile = None

        if profile is not None:
            worker_key = compute_worker_key(
                profile.base_model_id, profile.load_overrides
            )
            headers = {
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
            headers.update(
                await build_model_status_headers(
                    model_manager, worker_key, profile.base_model_id
                )
            )
            return StreamingResponse(
                _stream_completions_with_load(
                    model_manager=model_manager,
                    worker_pool=worker_pool,
                    profile=profile,
                    worker_key=worker_key,
                    body=body,
                ),
                media_type="text/event-stream",
                headers=headers,
            )

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
    )
    check_capability(state, "completion", "text completions")

    merged_body = merge_profile_defaults(profile, body)
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)

    if stream:
        return StreamingResponse(
            _stream_completions(worker_pool, worker_key, to_backend_body(state, merged_body)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await worker_pool.forward_request(
            worker_key,
            "/v1/completions",
            to_backend_body(state, merged_body),
        )
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
    return result


def _sse_error(message: str, code: str) -> bytes:
    payload = json.dumps(
        {"error": {"message": message, "type": "server_error", "code": code}}
    )
    return f"data: {payload}\n\n".encode() + b"data: [DONE]\n\n"


async def _stream_completions(worker_pool, model_id: str, body: dict):
    try:
        async for chunk in worker_pool.forward_stream(model_id, "/v1/completions", body):
            yield chunk
    except Exception as e:
        error_payload = json.dumps(
            {"error": {"message": str(e), "type": "server_error", "code": "stream_error"}}
        )
        yield f"data: {error_payload}\n\n".encode()
        yield b"data: [DONE]\n\n"


async def _stream_completions_with_load(
    *,
    model_manager,
    worker_pool,
    profile,
    worker_key: str,
    body: dict,
):
    """SSE generator that triggers the model load lazily — see chat.py."""
    pre_state = await model_manager.get_state(worker_key)
    if pre_state is None and worker_key != profile.base_model_id:
        pre_state = await model_manager.get_state(profile.base_model_id)
    pre_status = pre_state.status.value if pre_state is not None else "unknown"
    expected_wait = None
    if pre_status != "loaded":
        expected_wait = await model_manager.get_expected_load_seconds(worker_key)
        if expected_wait is None and worker_key != profile.base_model_id:
            expected_wait = await model_manager.get_expected_load_seconds(
                profile.base_model_id
            )

    yield sse_ocabra_event(
        "model_loading",
        model_id=profile.base_model_id,
        worker_key=worker_key,
        status=pre_status,
        expected_wait_seconds=expected_wait,
    )

    load_started_at = time.monotonic()
    load_task = asyncio.ensure_future(
        ensure_worker_loaded(
            model_manager,
            profile.base_model_id,
            worker_key,
            profile.load_overrides,
        )
    )
    async for ka in keepalive_until_done(load_task):
        yield ka
    if load_task.exception() is not None:
        exc = load_task.exception()
        if isinstance(exc, HTTPException):
            detail = exc.detail.get("error", {}) if isinstance(exc.detail, dict) else {}
            yield _sse_error(
                detail.get("message") or str(exc.detail),
                detail.get("code") or "model_load_failed",
            )
        else:
            yield _sse_error(str(exc), "model_load_failed")
        return
    state = load_task.result()

    load_duration_ms = int((time.monotonic() - load_started_at) * 1000)
    yield sse_ocabra_event(
        "model_ready",
        model_id=state.model_id,
        worker_key=worker_key,
        load_duration_ms=load_duration_ms,
        was_cold_start=(pre_status != "loaded"),
    )

    try:
        check_capability(state, "completion", "text completions")
    except HTTPException as exc:
        detail = exc.detail.get("error", {}) if isinstance(exc.detail, dict) else {}
        yield _sse_error(
            detail.get("message") or str(exc.detail),
            detail.get("code") or "model_not_capable",
        )
        return

    merged_body = merge_profile_defaults(profile, body)
    backend_body = to_backend_body(state, merged_body)
    try:
        async for chunk in worker_pool.forward_stream(
            worker_key, "/v1/completions", backend_body
        ):
            yield chunk
    except Exception as exc:
        yield _sse_error(str(exc), "stream_error")
