"""
POST /v1/completions — legacy text completions endpoint.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ocabra.api._deps_auth import UserContext

from ._deps import (
    STREAMING_LOAD_RESPONSE_DOC,
    build_model_status_headers,
    check_capability,
    ensure_worker_loaded,
    fairness_slots,
    fairness_user_key,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    is_batch_origin_request,
    keepalive_until_done,
    lookup_profile,
    merge_profile_defaults,
    raise_upstream_http_error,
    resolve_profile,
    resolve_profile_runtime_plan,
    sse_ocabra_event,
    stream_error_details,
    to_backend_body,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


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

    federation_manager = get_federation_manager(request)
    if stream and federation_manager is not None:
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            execution_policy = "auto"
            fed_profile = await profile_registry.get(model_id)
            if fed_profile is None:
                alias_target = await profile_registry.resolve_alias(model_id)
                fed_profile = alias_target[0] if alias_target is not None else None
            if fed_profile is not None and getattr(fed_profile, "runtime_presets", []):
                try:
                    fed_plan = await resolve_profile_runtime_plan(
                        fed_profile, model_manager, body
                    )
                    selected = next(
                        preset
                        for preset in getattr(fed_profile, "runtime_presets", [])
                        if preset.preset_key == fed_plan.preset_key
                    )
                    execution_policy = selected.execution_policy
                except Exception:  # noqa: BLE001 -- federation stays best effort
                    pass
            if execution_policy == "local":
                target, peer = "local", None
            else:
                target, peer = await resolve_federated(
                    model_id, model_manager, federation_manager, profile_registry
                )
                if execution_policy == "remote" and fed_profile is not None:
                    forced_peer = federation_manager.find_best_peer(
                        fed_profile.base_model_id
                    )
                    if forced_peer is not None:
                        target, peer = "remote", forced_peer
            if target == "remote" and peer is not None:
                request.state.federation_remote_node_id = peer.peer_id
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

    if not stream:
        from ._federation import try_proxy_json

        fed_resp = await try_proxy_json(
            request,
            model_id=model_id,
            body=body,
            federation_manager=federation_manager,
            model_manager=model_manager,
            profile_registry=profile_registry,
            logger=logger,
        )
        if fed_resp is not None:
            return fed_resp

    worker_pool = request.app.state.worker_pool

    # Streaming: pre-flush headers (incl. X-Ocabra-Expected-Wait-Seconds)
    # before triggering the load.
    if stream:
        try:
            profile = await lookup_profile(model_id, profile_registry, user=user)
        except HTTPException:
            profile = None

        # Bloque 20 fix (2026-09-19): mirror the same router-in-streaming fix
        # from chat.py — without it a streaming completion call that hit a
        # router profile would load the router's base_model_id directly and
        # skip target selection entirely. See chat.py comment for the full
        # incident context (GLados-corrala 19/09).
        router_resolver = getattr(request.app.state, "router_resolver", None)
        if (
            profile is not None
            and router_resolver is not None
            and router_resolver.is_router(profile)
        ):
            resolved = await router_resolver.pick(profile, request_body=body)
            target = await profile_registry.get(resolved.target_profile_id)
            if target is not None and target.profile_id != profile.profile_id:
                try:
                    request.state.via_router_profile_id = profile.profile_id
                    request.state.resolved_model_id = target.profile_id
                except Exception:  # noqa: BLE001 — never break resolution
                    pass
                profile = target

        if profile is not None:
            runtime_plan = await resolve_profile_runtime_plan(
                profile,
                model_manager,
                body,
                request_state=request.state,
            )
            worker_key = runtime_plan.worker_key
            headers = {
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
            via_router = getattr(request.state, "via_router_profile_id", None)
            headers.update(
                await build_model_status_headers(
                    model_manager,
                    worker_key,
                    profile.base_model_id,
                    via_router_profile_id=via_router,
                    router_target_profile_id=profile.profile_id if via_router else None,
                )
            )
            return StreamingResponse(
                _stream_completions_with_load(
                    model_manager=model_manager,
                    worker_pool=worker_pool,
                    profile=profile,
                    worker_key=worker_key,
                    load_overrides=runtime_plan.load_overrides,
                    preferred_gpu=runtime_plan.preferred_gpu,
                    load_policy=runtime_plan.load_policy,
                    required_tokens=runtime_plan.required_tokens,
                    body=body,
                    user_key=fairness_user_key(user),
                ),
                media_type="text/event-stream",
                headers=headers,
            )

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
        request_body=body,
        router_resolver=getattr(request.app.state, "router_resolver", None),
        request_state=request.state,
    )
    check_capability(state, "completion", "text completions")

    merged_body = merge_profile_defaults(profile, body)
    worker_key = state.model_id
    user_key = fairness_user_key(user)
    slots = fairness_slots(state)
    is_batch = is_batch_origin_request(request)

    if stream:
        backend_body = to_backend_body(state, merged_body)
        stream_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        queue_state = worker_pool.peek_queue_state(worker_key) if user_key is not None else None
        if queue_state is not None and queue_state["saturated"]:
            return StreamingResponse(
                _stream_completions_queued(
                    worker_pool, worker_key, backend_body, user_key, slots, is_batch
                ),
                media_type="text/event-stream",
                headers=stream_headers,
            )
        gen = worker_pool.forward_stream(
            worker_key, "/v1/completions", backend_body,
            user_key=user_key, slots=slots, is_batch=is_batch,
        )
        first_chunk, first_exc = await _peek_first_chunk(gen)
        if first_exc is not None:
            if isinstance(first_exc, httpx.HTTPStatusError):
                raise_upstream_http_error(first_exc)
            message, code = stream_error_details(first_exc)
            raise HTTPException(
                status_code=502,
                detail={"error": {"message": message, "type": "server_error", "code": code}},
            )
        return StreamingResponse(
            _stream_completions_from(gen, first_chunk),
            media_type="text/event-stream",
            headers=stream_headers,
        )

    queue_state = worker_pool.peek_queue_state(worker_key) if not is_batch else None
    try:
        result = await worker_pool.forward_request(
            worker_key,
            "/v1/completions",
            to_backend_body(state, merged_body),
            user_key=user_key,
            slots=slots,
            is_batch=is_batch,
        )
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
    from fastapi.responses import JSONResponse

    from ocabra.api.openai._deps import ocabra_response_headers

    resp_headers = await ocabra_response_headers(
        request, model_manager, worker_key, profile.base_model_id
    )
    if queue_state is not None and queue_state["saturated"]:
        resp_headers["X-Ocabra-Slot-Queue-Saturated"] = "1"
        resp_headers["X-Ocabra-Slot-In-Flight"] = str(queue_state["in_flight"])
        resp_headers["X-Ocabra-Slot-Count"] = str(queue_state["slots"])
    return JSONResponse(content=result, headers=resp_headers)


def _sse_error(message: str, code: str) -> bytes:
    payload = json.dumps({"error": {"message": message, "type": "server_error", "code": code}})
    return f"data: {payload}\n\n".encode() + b"data: [DONE]\n\n"


async def _stream_completions_queued(
    worker_pool,
    model_id: str,
    body: dict,
    user_key: str | None,
    slots: int,
    is_batch: bool,
) -> AsyncIterator[bytes]:
    """Fallback for an already-saturated fair-share queue — see the
    identical rationale on ``ocabra.api.openai.chat._stream_chat_queued``.
    """
    try:
        queue_state = worker_pool.peek_queue_state(model_id)
        if queue_state is not None and queue_state["saturated"]:
            yield sse_ocabra_event(
                "slot_queued",
                model_id=model_id,
                in_flight=queue_state["in_flight"],
                slots=queue_state["slots"],
            )
        async for chunk in worker_pool.forward_stream(
            model_id, "/v1/completions", body, user_key=user_key, slots=slots, is_batch=is_batch
        ):
            yield chunk
    except Exception as e:
        message, code = stream_error_details(e)
        error_payload = json.dumps(
            {"error": {"message": message, "type": "server_error", "code": code}}
        )
        yield f"data: {error_payload}\n\n".encode()
        yield b"data: [DONE]\n\n"


async def _peek_first_chunk(
    gen: AsyncIterator[bytes],
) -> tuple[bytes | None, Exception | None]:
    """See ``ocabra.api.openai.chat._peek_first_chunk`` — same fix, same
    reason (2026-10-30): a rejected streaming request must not come back
    as HTTP 200 with an in-band SSE error when the rejection is already
    known before anything has been sent to the client.
    """
    try:
        first_chunk = await gen.__anext__()
        return first_chunk, None
    except StopAsyncIteration:
        return None, None
    except Exception as exc:  # noqa: BLE001 — any first-attempt failure is eligible for a real status
        return None, exc


async def _stream_completions_from(
    gen: AsyncIterator[bytes],
    first_chunk: bytes | None,
) -> AsyncIterator[bytes]:
    try:
        if first_chunk is not None:
            yield first_chunk
        async for chunk in gen:
            yield chunk
    except Exception as e:
        message, code = stream_error_details(e)
        error_payload = json.dumps(
            {"error": {"message": message, "type": "server_error", "code": code}}
        )
        yield f"data: {error_payload}\n\n".encode()
        yield b"data: [DONE]\n\n"


async def _stream_completions_with_load(
    *,
    model_manager,
    worker_pool,
    profile,
    worker_key: str,
    load_overrides: dict | None,
    preferred_gpu: int | None,
    load_policy: str | None,
    required_tokens: int | None,
    body: dict,
    user_key: str | None = None,
):
    """SSE generator that triggers the model load lazily — see chat.py.

    The fair-share slot is only acquired once the model is loaded and
    generation starts, same rationale as ``_stream_chat_with_load``.
    """
    pre_state = await model_manager.get_state(worker_key)
    if pre_state is None and worker_key != profile.base_model_id:
        pre_state = await model_manager.get_state(profile.base_model_id)
    pre_status = pre_state.status.value if pre_state is not None else "unknown"
    expected_wait = None
    if pre_status != "loaded":
        expected_wait = await model_manager.get_expected_load_seconds(worker_key)
        if expected_wait is None and worker_key != profile.base_model_id:
            expected_wait = await model_manager.get_expected_load_seconds(profile.base_model_id)

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
            load_overrides,
            preferred_gpu=preferred_gpu,
            load_policy=load_policy,
            required_tokens=required_tokens,
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
    slots = fairness_slots(state)
    try:
        if user_key is not None:
            queue_state = worker_pool.peek_queue_state(worker_key)
            if queue_state is not None and queue_state["saturated"]:
                yield sse_ocabra_event(
                    "slot_queued",
                    model_id=worker_key,
                    in_flight=queue_state["in_flight"],
                    slots=queue_state["slots"],
                )
        async for chunk in worker_pool.forward_stream(
            worker_key, "/v1/completions", backend_body, user_key=user_key, slots=slots
        ):
            yield chunk
    except Exception as exc:
        message, code = stream_error_details(exc)
        yield _sse_error(message, code)
