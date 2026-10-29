"""
POST /api/chat — Ollama chat compatibility endpoint.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from starlette.responses import StreamingResponse

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import (
    check_capability,
    ensure_loaded,
    fairness_slots,
    fairness_user_key,
    get_federation_manager,
    get_model_manager,
    get_profile_registry,
    merge_profile_defaults,
    resolve_profile,
)

from ._mapper import resolve_model  # noqa: I001
from ._shared import (
    apply_option_map,
    build_native_passthrough_body,
    get_ollama_user,
    iter_sse_payloads,
    now_iso_z,
)
from ocabra.agents.chat_glue import (
    build_invoker_for_agent,
    build_subagent_runner,
    extract_per_request_headers,
    parse_allowed_tools_header,
)
from ocabra.agents.executor import AgentExecutor
from ocabra.agents.mcp_registry import get_registry as get_mcp_registry
from ocabra.agents.resolver import is_agent_model, resolve_agent
from ocabra.database import AsyncSessionLocal

router = APIRouter()


def _native_backend_model_name(requested_model: str, backend_model_id: str | None) -> str:
    return backend_model_id or requested_model


@router.post("/chat", summary="Create chat completion")
async def chat(
    request: Request,
    user: UserContext = Depends(get_ollama_user),
):
    """
    Create a chat completion in Ollama format.

    Request fields include model, messages, stream, and options.
    Streaming responses are translated from OpenAI SSE to Ollama NDJSON.

    **Context window auto-sizing**: same mechanism as `/v1/chat/completions`
    — the runtime preset is picked from `prompt_tokens + output_reservation`,
    where `output_reservation` comes from `options.num_predict` if set,
    otherwise the target profile's own
    `runtime_policy.default_output_reservation_tokens` if configured, else a
    global 4096-token default. For
    reasoning models this can truncate the response (`done_reason: "length"`,
    empty `content`) if the model reasons longer than that default covers.
    Set `options.num_predict` generously, or force a preset with
    `"ocabra": {"context_preset": "<key>"}`. See docs/API.md.
    """
    body = await request.json()
    ollama_model = str(body.get("model", ""))
    stream = bool(body.get("stream", True))

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    # --- Federation hook ---
    # Agent invocations (model="agent/<slug>") are local; skip federation.
    federation_manager = get_federation_manager(request)
    if federation_manager is not None and not is_agent_model(ollama_model):
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            # Pass the profile_registry so resolve_federated can normalize
            # profile ids and router profiles to their base_model_id before
            # the local/remote lookup. Without it, router profiles like
            # ``recomendado-ha-chatbots`` fell through to the "not found
            # anywhere" branch and the caller received 404 model_not_found
            # from the federation layer, before resolve_profile ever ran.
            target, peer = await resolve_federated(
                ollama_model, model_manager, federation_manager, profile_registry
            )
            if target == "remote":
                import structlog as _structlog

                from ocabra.core.federation import should_fallback_to_local

                _fed_logger = _structlog.get_logger(__name__)
                request.state.federation_remote_node_id = peer.peer_id
                if stream:
                    return StreamingResponse(
                        federation_manager.proxy_stream(
                            peer=peer,
                            path=request.url.path,
                            body=body,
                            headers=dict(request.headers),
                        ),
                        media_type="application/x-ndjson",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                try:
                    resp = await federation_manager.proxy_request(
                        peer=peer,
                        path=request.url.path,
                        body=body,
                        headers=dict(request.headers),
                    )
                except Exception as exc:
                    _fed_logger.warning(
                        "federation_peer_network_error_fallback_local",
                        peer=peer.name,
                        model_id=ollama_model,
                        error=str(exc),
                    )
                    request.state.federation_remote_node_id = None
                    resp = None
                if resp is not None and (
                    resp.status_code < 400
                    or not should_fallback_to_local(resp.status_code)
                ):
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type"),
                    )
                if resp is not None:
                    _fed_logger.warning(
                        "federation_peer_rejected_fallback_local",
                        peer=peer.name,
                        model_id=ollama_model,
                        status=resp.status_code,
                        body_preview=resp.text[:200],
                    )
                    request.state.federation_remote_node_id = None
                # Fall through to local processing.
    # --- End federation hook ---

    # --- Agent dispatch: model="agent/<slug>" via Ollama-compat ---
    if is_agent_model(ollama_model):
        return await _dispatch_agent_ollama(
            request=request,
            user=user,
            body=body,
            ollama_model=ollama_model,
            stream=stream,
        )

    # Try profile resolution first
    try:
        # Pass the router resolver so router profiles (``recomendado-*``)
        # also work when called via the Ollama-compatible path, not only
        # the OpenAI one. Without it, resolve_profile treated the router
        # as "no base_model_id → routing_disabled" / 404 depending on
        # the branch, breaking Ollama clients that selected the router.
        profile, state = await resolve_profile(
            ollama_model,
            model_manager,
            profile_registry,
            user=user,
            request_body=body,
            router_resolver=getattr(request.app.state, "router_resolver", None),
            request_state=request.state,
        )
        merged_body = merge_profile_defaults(profile, body)
        worker_key = state.model_id
    except HTTPException as profile_exc:
        if profile_exc.status_code != 404:
            raise
        # Fallback to legacy Ollama resolution for native Ollama models
        model_id, resolved_state = await resolve_model(model_manager, ollama_model, user=user)
        if resolved_state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{ollama_model}' not found",
            ) from profile_exc
        state = await ensure_loaded(model_manager, model_id)
        merged_body = body
        worker_key = model_id
        profile = None

    check_capability(state, "chat", "chat")

    worker_pool = request.app.state.worker_pool
    if state.backend_type == "ollama":
        upstream_body = build_native_passthrough_body(
            merged_body,
            model=_native_backend_model_name(ollama_model, state.backend_model_id),
            stream=stream,
            content_keys=("messages",),
            passthrough_keys=("keep_alive", "format", "think", "tools"),
        )
        if stream:
            return StreamingResponse(
                worker_pool.forward_stream(worker_key, "/api/chat", upstream_body),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await worker_pool.forward_request(worker_key, "/api/chat", upstream_body)

    vllm_body = _build_vllm_chat_body(merged_body, state.backend_model_id, stream)
    user_key = fairness_user_key(user)
    slots = fairness_slots(state)

    if stream:
        return StreamingResponse(
            _stream_chat(worker_pool, worker_key, ollama_model, vllm_body, user_key, slots),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    started_ns = time.monotonic_ns()
    result = await worker_pool.forward_request(
        worker_key, "/v1/chat/completions", vllm_body, user_key=user_key, slots=slots
    )
    message = _extract_chat_ollama_message(result)
    total_duration = time.monotonic_ns() - started_ns
    usage = result.get("usage") or {}
    return {
        "model": ollama_model,
        "created_at": now_iso_z(),
        "message": message,
        "done": True,
        "done_reason": _extract_finish_reason(result),
        "total_duration": total_duration,
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "eval_count": int(usage.get("completion_tokens") or 0),
        "eval_duration": total_duration,
    }


def _normalize_messages_for_backend(messages: list) -> list:
    """Coerce ``content: null`` to ``content: ""`` on every message.

    OpenAI-style clients routinely send ``content: null`` for an assistant
    message that only carries ``tool_calls`` (that's the documented OpenAI
    shape). gemma4's llama.cpp chat template hard-crashes the worker process
    — not a clean error, the whole llama-server process dies — when it
    renders a historical assistant message shaped that way. Confirmed via
    direct reproduction against the worker on 2026-10-27: identical message
    with ``content: ""`` renders fine; ``content: null`` kills the process.
    Harmless for every other model, so apply unconditionally rather than
    gating on backend/model to avoid re-growing a backend-specific special
    case here.
    """
    out = []
    for raw in messages or []:
        if not isinstance(raw, dict):
            out.append(raw)
            continue
        msg = dict(raw)
        if msg.get("content") is None:
            msg["content"] = ""
        if isinstance(msg.get("tool_calls"), list):
            msg["tool_calls"] = [_normalize_tool_call_for_backend(tc) for tc in msg["tool_calls"]]
        out.append(msg)
    return out


def _normalize_tool_call_for_backend(tool_call: object) -> object:
    """Backfill the OpenAI-required ``type``/stringified-``arguments`` shape
    on a historical ``tool_calls`` entry before forwarding it upstream.

    We hand clients Ollama-native tool_calls (``{"function": {...}}``, no
    ``type``, ``arguments`` as a plain object — see
    ``_openai_tool_calls_to_ollama``). A client that echoes that exact
    message back as conversation history on the next turn (standard
    multi-turn tool-loop behavior) sends it straight through
    ``_build_vllm_chat_body`` unmodified — and llama-server's OpenAI-
    compatible endpoint rejects it outright: ``Failed to parse messages:
    Missing tool call type``. Confirmed live 2026-10-27 via a real Home
    Assistant conversation (``intent__HassTurnOn``) using the diagnostic
    logging added for the earlier content:null crash.
    """
    if not isinstance(tool_call, dict):
        return tool_call
    out = dict(tool_call)
    out.setdefault("type", "function")
    function = out.get("function")
    if isinstance(function, dict):
        function = dict(function)
        if isinstance(function.get("arguments"), dict):
            function["arguments"] = json.dumps(function["arguments"])
        out["function"] = function
    return out


def _build_vllm_chat_body(payload: dict, model_id: str, stream: bool) -> dict:
    body: dict = {
        "model": model_id,
        "messages": _normalize_messages_for_backend(payload.get("messages", [])),
        "stream": stream,
    }

    for key in (
        "tools",
        "tool_choice",
        "response_format",
        "temperature",
        "top_p",
        "stop",
        "seed",
        "max_tokens",
    ):
        if key in payload:
            body[key] = payload[key]

    options = payload.get("options") or {}
    apply_option_map(body, options)

    return body


async def _stream_chat(
    worker_pool,
    model_id: str,
    ollama_model: str,
    body: dict,
    user_key: str | None = None,
    slots: int = 1,
) -> AsyncIterator[bytes]:
    started_ns = time.monotonic_ns()
    prompt_eval_count = 0
    eval_count = 0
    finish_reason: str | None = None
    # OpenAI streams tool_calls incrementally (one delta per `index`, name
    # arriving once then `arguments` trickling in character by character) —
    # accumulate by index and only emit the assembled calls once, in the
    # final `done` chunk, matching how Ollama's own /api/chat reports them
    # (attached to the completed message, never streamed token-by-token
    # themselves).
    tool_calls_acc: dict[int, dict] = {}

    async for payload in iter_sse_payloads(
        worker_pool.forward_stream(
            model_id, "/v1/chat/completions", body, user_key=user_key, slots=slots
        )
    ):
        if payload == "[DONE]":
            total_duration = time.monotonic_ns() - started_ns
            done_message: dict = {"role": "assistant", "content": ""}
            if tool_calls_acc:
                done_message["tool_calls"] = _openai_tool_calls_to_ollama(
                    [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
                )
            done_payload = {
                "model": ollama_model,
                "created_at": now_iso_z(),
                "message": done_message,
                "done": True,
                "done_reason": finish_reason,
                "total_duration": total_duration,
                "load_duration": 0,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "eval_duration": total_duration,
            }
            yield (json.dumps(done_payload) + "\n").encode("utf-8")
            return

        if not isinstance(payload, dict):
            continue

        usage = payload.get("usage") or {}
        if usage:
            prompt_eval_count = int(usage.get("prompt_tokens") or prompt_eval_count)
            eval_count = int(usage.get("completion_tokens") or eval_count)

        choices = payload.get("choices") or []
        if not choices:
            continue

        if choices[0].get("finish_reason"):
            finish_reason = choices[0]["finish_reason"]

        delta = choices[0].get("delta") or {}

        for tc_delta in delta.get("tool_calls") or []:
            idx = tc_delta.get("index", 0)
            entry = tool_calls_acc.setdefault(idx, {"function": {"name": "", "arguments": ""}})
            fn_delta = tc_delta.get("function") or {}
            if fn_delta.get("name"):
                entry["function"]["name"] = fn_delta["name"]
            if fn_delta.get("arguments"):
                entry["function"]["arguments"] += fn_delta["arguments"]

        token = str(delta.get("content") or "")
        if token == "":
            continue

        chunk = {
            "model": ollama_model,
            "created_at": now_iso_z(),
            "message": {"role": "assistant", "content": token},
            "done": False,
        }
        yield (json.dumps(chunk) + "\n").encode("utf-8")


def _openai_tool_calls_to_ollama(tool_calls: list[dict] | None) -> list[dict]:
    """Convert OpenAI-shaped ``tool_calls`` to Ollama's ``/api/chat`` shape.

    OpenAI (and llama-server/vLLM's OpenAI-compatible endpoints) encode
    ``function.arguments`` as a JSON *string*; Ollama's native clients expect
    it as a plain object. Dropping this conversion is why tool calls appeared
    to "not work" for non-ollama backends: the HTTP request succeeded and the
    backend generated a correct tool call, but it never survived translation
    back into the Ollama response shape (see gpu-bench README, 2026-10-27).
    """
    out: list[dict] = []
    for call in tool_calls or []:
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        raw_args = function.get("arguments")
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {}
        out.append({"function": {"name": name, "arguments": arguments}})
    return out


def _extract_chat_ollama_message(payload: dict) -> dict:
    choices = payload.get("choices") or []
    if not choices:
        return {"role": "assistant", "content": ""}
    message = choices[0].get("message") or {}
    out: dict = {"role": "assistant", "content": str(message.get("content") or "")}
    if message.get("tool_calls"):
        out["tool_calls"] = _openai_tool_calls_to_ollama(message["tool_calls"])
    return out


def _extract_finish_reason(payload: dict) -> str | None:
    """OpenAI's ``finish_reason`` ("stop"/"length"/"tool_calls"/...) maps
    directly onto Ollama's ``done_reason`` — same vocabulary, different key.
    Without this, a response truncated by max_tokens or by running out of
    context looks identical to a normal completion to any Ollama client
    checking ``done_reason``, which is the one signal they have to tell
    "finished" from "got cut off, maybe ask for more"."""
    choices = payload.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


# ── Agent dispatch (Ollama adapter) ──────────────────────────────


def _ollama_messages_to_openai(messages: list[dict]) -> list[dict]:
    """Translate Ollama-style messages to OpenAI-style messages.

    Ollama supports: ``role``, ``content`` (string), ``images`` (list of b64),
    ``tool_calls``.  OpenAI expects ``content`` as string or content-parts list,
    ``role=tool`` with ``tool_call_id``.
    """
    out: list[dict] = []
    for raw in messages or []:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        # ``.get("content", "")`` only applies its default when the key is
        # absent — an explicit ``content: null`` (the OpenAI-standard shape
        # for a tool_calls-only assistant message) passes straight through
        # as None. See _normalize_messages_for_backend for why that crashes
        # gemma4's llama.cpp worker when replayed in history.
        content = raw.get("content") or ""
        images = raw.get("images") or []
        msg: dict = {"role": role}
        if images:
            parts: list[dict] = []
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            for img in images:
                if isinstance(img, str) and img:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )
            msg["content"] = parts
        else:
            msg["content"] = content
        if raw.get("tool_calls"):
            msg["tool_calls"] = [_normalize_tool_call_for_backend(tc) for tc in raw["tool_calls"]]
        if raw.get("tool_call_id"):
            msg["tool_call_id"] = raw["tool_call_id"]
        out.append(msg)
    return out


def _openai_response_to_ollama(payload: dict, ollama_model: str) -> dict:
    """Translate the executor's OpenAI response into Ollama's ``/api/chat`` shape."""
    msg_dict = (payload.get("choices") or [{}])[0].get("message") or {}
    out_msg: dict = {"role": "assistant", "content": msg_dict.get("content") or ""}
    if msg_dict.get("tool_calls"):
        out_msg["tool_calls"] = _openai_tool_calls_to_ollama(msg_dict["tool_calls"])
    usage = payload.get("usage") or {}
    return {
        "model": ollama_model,
        "created_at": now_iso_z(),
        "message": out_msg,
        "done": True,
        "done_reason": _extract_finish_reason(payload),
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "eval_count": int(usage.get("completion_tokens") or 0),
        "eval_duration": 0,
    }


async def _dispatch_agent_ollama(
    *,
    request: Request,
    user: UserContext,
    body: dict,
    ollama_model: str,
    stream: bool,
):
    """Run the AgentExecutor and return a response in Ollama format.

    Ollama clients send Ollama-shaped request bodies; we translate them to
    OpenAI format, run the executor, and translate the final assistant
    response back.  Streaming is exposed as NDJSON, mirroring the rest of
    ``/api/chat``.
    """
    async with AsyncSessionLocal() as session:
        agent = await resolve_agent(ollama_model, session, user=user)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{ollama_model}' not found",
        )

    registry = getattr(request.app.state, "mcp_registry", None) or get_mcp_registry()
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="MCP registry not available",
            headers={"Retry-After": "5"},
        )

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)
    invoker = await build_invoker_for_agent(
        agent,
        model_manager=model_manager,
        profile_registry=profile_registry,
        user=user,
        worker_pool=request.app.state.worker_pool,
    )

    executor = AgentExecutor(registry)
    executor._subagent_runner = build_subagent_runner(  # noqa: SLF001
        executor,
        model_manager=model_manager,
        profile_registry=profile_registry,
        user=user,
        worker_pool=request.app.state.worker_pool,
    )
    raw_messages = body.get("messages") or []
    messages = _ollama_messages_to_openai(raw_messages)

    # See the OpenAI dispatcher for the rationale behind the contextvar set.
    from ocabra.agents.executor import current_agent_id  # local import to avoid cycle

    current_agent_id.set(agent.id)

    request_options: dict = {}
    options = body.get("options") or {}
    if isinstance(options, dict):
        # Map the relevant Ollama options to OpenAI sampling args.
        for src, dst in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("seed", "seed"),
            ("num_predict", "max_tokens"),
            ("stop", "stop"),
        ):
            if src in options:
                request_options[dst] = options[src]
    if "tools" in body and isinstance(body["tools"], list):
        caller_tools = body["tools"]
    else:
        caller_tools = None

    per_request_headers = extract_per_request_headers(request)
    per_request_allowed = parse_allowed_tools_header(request.headers.get("x-ocabra-allowed-tools"))
    require_approval_override = request.headers.get("x-ocabra-require-approval")

    if stream:

        async def _gen():
            # Build the final response synchronously then emit a single done line.
            # The OpenAI streaming variant offers token-by-token deltas; for
            # Ollama we keep parity with the legacy implementation that drains
            # to a single chunk on tool-loops.
            result = await executor.run(
                agent,
                messages,
                request_options,
                user,
                invoker,
                per_request_headers=per_request_headers,
                per_request_allowed_tools=per_request_allowed,
                caller_tools=caller_tools,
                require_approval_override=require_approval_override,
            )
            translated = _openai_response_to_ollama(result.openai_response, ollama_model)
            yield (json.dumps(translated) + "\n").encode("utf-8")

        return StreamingResponse(
            _gen(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await executor.run(
        agent,
        messages,
        request_options,
        user,
        invoker,
        per_request_headers=per_request_headers,
        per_request_allowed_tools=per_request_allowed,
        caller_tools=caller_tools,
        require_approval_override=require_approval_override,
    )
    return _openai_response_to_ollama(result.openai_response, ollama_model)
