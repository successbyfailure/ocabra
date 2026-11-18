from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime

from fastapi import Depends, HTTPException

from ocabra.api._deps_auth import ROLE_HIERARCHY, UserContext, get_current_user


async def get_ollama_user(
    user: UserContext = Depends(get_current_user),
) -> UserContext:
    """Resolve auth for Ollama-compatible endpoints.

    Delegates to ``get_current_user`` which already handles:
    - Bearer API key resolution.
    - Cookie JWT resolution.
    - Anonymous access when ``require_api_key_ollama=False``.
    - HTTP 401 when ``require_api_key_ollama=True`` and no credentials provided.

    Returns:
        Resolved :class:`UserContext` for the caller.

    Raises:
        HTTPException 401: When authentication is required but missing or invalid.
    """
    return user


async def get_ollama_model_manager(
    user: UserContext = Depends(get_ollama_user),
) -> UserContext:
    """Like :func:`get_ollama_user`, but require the ``model_manager`` role.

    For the endpoints that change the model catalogue (``/api/pull``,
    ``/api/delete``): they act on every user's models, not on the caller's.
    The error keeps Ollama's ``{"error": "..."}`` shape.
    """
    if user.is_anonymous:
        raise HTTPException(status_code=401, detail={"error": "authentication required"})
    if ROLE_HIERARCHY.get(user.role, -1) < ROLE_HIERARCHY["model_manager"]:
        raise HTTPException(
            status_code=403, detail={"error": "role 'model_manager' or higher required"}
        )
    return user


OPTION_MAP: dict[str, str] = {
    "num_predict": "max_tokens",
    "num_ctx": "max_model_len",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop",
    "seed": "seed",
    "repeat_penalty": "repetition_penalty",
}

NATIVE_OPTION_ALIAS_MAP: dict[str, str] = {
    "max_tokens": "num_predict",
    "num_predict": "num_predict",
    "num_ctx": "num_ctx",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop",
    "seed": "seed",
    "repeat_penalty": "repeat_penalty",
}


def now_iso_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_native_passthrough_body(
    payload: Mapping[str, object],
    *,
    model: str,
    stream: bool,
    content_keys: tuple[str, ...],
    passthrough_keys: tuple[str, ...] = (),
) -> dict:
    body: dict = {
        "model": model,
        "stream": stream,
    }
    for key in (*content_keys, *passthrough_keys):
        if key in payload:
            body[key] = payload[key]

    options = payload.get("options")
    merged_options = dict(options) if isinstance(options, Mapping) else {}
    for source_key, target_key in NATIVE_OPTION_ALIAS_MAP.items():
        if source_key in payload and target_key not in merged_options:
            merged_options[target_key] = payload[source_key]
    body["options"] = merged_options
    return body


def apply_option_map(body: dict, options: object) -> None:
    if not isinstance(options, Mapping):
        return

    for key, value in options.items():
        mapped = OPTION_MAP.get(str(key))
        if mapped:
            body[mapped] = value


async def iter_sse_payloads(source: AsyncIterator[bytes]) -> AsyncIterator[dict | str]:
    """Parse `data: ...\\n\\n` SSE frames and yield decoded payloads."""
    buffer = ""
    async for chunk in source:
        if not chunk:
            continue
        buffer += chunk.decode("utf-8", errors="ignore")

        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in raw_event.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if not data_lines:
                continue

            data = "\n".join(data_lines).strip()
            if not data:
                continue
            if data == "[DONE]":
                yield "[DONE]"
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def request_stream_usage(body: dict) -> dict:
    """Ask an OpenAI-compatible upstream to report token usage on a stream.

    llama-server (and vLLM) only send the final ``usage`` chunk when the
    request carries ``stream_options.include_usage``. The /v1 routes get it
    from ``to_backend_body``; the Ollama translation built its own body
    without it, so every streamed /api/chat answered ``prompt_eval_count: 0``
    and ``eval_count: 0`` — to the client and to request_stats (2026-11-09:
    34 of 35 /api/chat requests on llamacpp_mks since 2026-11-06 recorded 0
    tokens).
    """
    if body.get("stream") is True:
        options = body.get("stream_options")
        body["stream_options"] = (
            {**options, "include_usage": True}
            if isinstance(options, dict)
            else {"include_usage": True}
        )
    return body


def usage_from_stream_payload(
    payload: dict, prompt_tokens: int, completion_tokens: int
) -> tuple[int, int]:
    """Token counts carried by one upstream stream chunk (``usage`` first,
    llama.cpp's ``timings.prompt_n``/``predicted_n`` as a fallback)."""
    usage = payload.get("usage") or {}
    if isinstance(usage, dict) and usage:
        prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
        completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
        return prompt_tokens, completion_tokens
    timings = payload.get("timings")
    if isinstance(timings, dict):
        try:
            if timings.get("prompt_n") is not None and not prompt_tokens:
                prompt_tokens = int(timings["prompt_n"]) + int(timings.get("cache_n") or 0)
            if timings.get("predicted_n") is not None and not completion_tokens:
                completion_tokens = int(timings["predicted_n"])
        except (TypeError, ValueError):
            pass
    return prompt_tokens, completion_tokens


# ── Dispatch errors (BUG-10) ─────────────────────────────────────────────────


def dispatch_error_types() -> tuple[type[Exception], ...]:
    """Worker errors that ``/v1`` already turns into proper status codes."""
    import httpx

    from ocabra.core.fair_slot_gate import KVCapacityError
    from ocabra.core.worker_pool import (
        InferenceTimeoutError,
        WorkerDegradedError,
        WorkerGoneError,
    )

    return (
        httpx.HTTPStatusError,
        InferenceTimeoutError,
        KVCapacityError,
        WorkerDegradedError,
        WorkerGoneError,
    )


def raise_dispatch_error(exc: Exception) -> None:
    """Same status, code and ``Retry-After`` as ``/v1`` for a worker error.

    ``/api/chat`` and ``/api/generate`` only caught the client leaving and the
    KV budget: a timeout, a degraded or unloaded worker or an upstream 4xx all
    reached the generic handler as a 500.
    """
    import httpx

    from ocabra.api.openai._deps import (
        raise_inference_dispatch_error,
        raise_upstream_http_error,
    )

    if isinstance(exc, httpx.HTTPStatusError):
        raise_upstream_http_error(exc)
    raise_inference_dispatch_error(exc)


async def open_stream(gen: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Start ``gen`` before answering, so a rejection is still a real status.

    The worker raises before its first byte when it refuses the request
    (``raise_for_status``), and at that point nothing has been sent: answer
    with the status instead of a 200 whose stream just ends.
    """
    from ocabra.api.openai.chat import _peek_first_chunk

    first, exc = await _peek_first_chunk(gen)
    if exc is not None:
        if isinstance(exc, dispatch_error_types()):
            raise_dispatch_error(exc)
        raise exc

    async def _replay() -> AsyncIterator[bytes]:
        if first is not None:
            yield first
        async for chunk in gen:
            yield chunk

    return _replay()


def stream_error_line(exc: Exception) -> bytes:
    """Ollama's in-stream error (``{"error": "..."}``) once the 200 is sent."""
    from ocabra.api.openai._deps import stream_error_details

    message, code = stream_error_details(exc)
    return (json.dumps({"error": message, "code": code}) + "\n").encode("utf-8")


# ── Reasoning (BUG-11) ───────────────────────────────────────────────────────

#: Keys a profile or client may set that the Ollama translation used to drop.
REASONING_KEYS = ("reasoning_effort", "reasoning_budget", "chat_template_kwargs")


def apply_think(payload: Mapping[str, object], body: dict) -> None:
    """Ollama's ``think`` as the ``reasoning_effort`` ``to_backend_body`` reads.

    ``false`` turns reasoning off; a level (``"low"``/``"medium"``/``"high"``,
    as Ollama accepts for some models) is used as is; ``true`` keeps the
    model's configured default, which is what "think" means for a model that
    reasons anyway. Without ``think`` the model default applies, as on ``/v1``.
    """
    think = payload.get("think")
    if think is False:
        body["reasoning_effort"] = "off"
    elif isinstance(think, str) and think.strip():
        body["reasoning_effort"] = think.strip().lower()


def reasoning_text(source: Mapping[str, object] | None) -> str:
    """The reasoning a backend returned, under either name it uses."""
    if not isinstance(source, Mapping):
        return ""
    value = source.get("reasoning_content") or source.get("reasoning") or ""
    return value if isinstance(value, str) else ""
