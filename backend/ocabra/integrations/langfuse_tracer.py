"""
Langfuse tracer -- optional LLM observability integration.

Disabled by default (LANGFUSE_ENABLED=false).
When enabled, records generation traces for all /v1/* and /api/* inference calls.
Content (messages + completions) is only sent if LANGFUSE_CAPTURE_CONTENT=true.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from ocabra.config import settings

logger = structlog.get_logger(__name__)

_client = None  # Langfuse singleton


def _get_client():  # noqa: ANN202
    global _client  # noqa: PLW0603
    if _client is not None:
        return _client
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("langfuse_disabled_missing_keys")
        return None
    try:
        import langfuse

        _client = langfuse.Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            flush_interval=settings.langfuse_flush_interval_s,
        )
        logger.info("langfuse_initialized", host=settings.langfuse_host)
    except Exception as exc:
        logger.warning("langfuse_init_failed", error=str(exc))
        return None
    return _client


def _should_sample() -> bool:
    return random.random() < settings.langfuse_sample_rate


def _extract_usage(response_body: dict) -> tuple[int | None, int | None]:
    usage = response_body.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def _extract_completion_text(response_body: dict) -> str | None:
    choices = response_body.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    message = choice.get("message") or {}
    if message.get("content"):
        return message["content"]
    if choice.get("text"):
        return choice["text"]
    return None


def _build_input(path: str, body: dict) -> Any:
    """Respects capture_content: only metadata when false, full messages when true."""
    if not settings.langfuse_capture_content:
        return {
            "model": body.get("model"),
            "temperature": body.get("temperature"),
            "max_tokens": body.get("max_tokens"),
            "stream": body.get("stream"),
        }
    if "messages" in body:
        return body["messages"]
    if "prompt" in body:
        return body["prompt"]
    return body


async def trace_generation(
    *,
    model_id: str,
    path: str,
    request_body: dict,
    response_body: dict,
    duration_ms: float,
    error: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Record a non-streaming generation trace. Fire-and-forget."""
    client = _get_client()
    if client is None or not _should_sample():
        return
    try:
        input_tokens, output_tokens = _extract_usage(response_body)
        trace = client.trace(
            name=f"ocabra:{path.split('/')[-1]}",
            user_id=user_id,
            session_id=session_id,
            metadata={"path": path},
        )
        trace.generation(
            name=model_id,
            model=model_id,
            input=_build_input(path, request_body),
            output=(
                _extract_completion_text(response_body)
                if settings.langfuse_capture_content
                else None
            ),
            usage={"input": input_tokens, "output": output_tokens, "unit": "TOKENS"},
            metadata={"duration_ms": duration_ms},
            level="ERROR" if error else "DEFAULT",
            status_message=error,
        )
    except Exception as exc:
        logger.warning("langfuse_trace_failed", error=str(exc))


async def wrap_stream(
    generator: AsyncIterator[bytes],
    *,
    model_id: str,
    path: str,
    request_body: dict,
    user_id: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[bytes]:
    """
    Transparent wrapper around a streaming SSE generator.

    Yields all chunks immediately (zero added latency to the client).
    Parses SSE in-flight to extract token counts from the final usage chunk.
    Sends the Langfuse trace in the finally block, after the last chunk is delivered.
    """
    client = _get_client()
    if client is None or not _should_sample():
        async for chunk in generator:
            yield chunk
        return

    start = time.monotonic()
    input_tokens_ref: list[int | None] = [None]
    output_tokens_ref: list[int | None] = [None]
    text_parts: list[str] = []
    error: str | None = None

    try:
        async for chunk in generator:
            yield chunk  # entrega inmediata -- sin buffering
            parse_sse_chunk(
                chunk,
                input_tokens_ref=input_tokens_ref,
                output_tokens_ref=output_tokens_ref,
                text_parts=text_parts if settings.langfuse_capture_content else None,
            )
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        input_tokens = input_tokens_ref[0]
        output_tokens = output_tokens_ref[0]
        try:
            trace = client.trace(
                name=f"ocabra:{path.split('/')[-1]}",
                user_id=user_id,
                session_id=session_id,
                metadata={"path": path, "stream": True},
            )
            trace.generation(
                name=model_id,
                model=model_id,
                input=_build_input(path, request_body),
                output="".join(text_parts) if settings.langfuse_capture_content else None,
                usage={"input": input_tokens, "output": output_tokens, "unit": "TOKENS"},
                metadata={"duration_ms": duration_ms},
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            )
        except Exception as exc:
            logger.warning("langfuse_stream_trace_failed", error=str(exc))


def parse_sse_chunk(
    chunk: bytes,
    *,
    input_tokens_ref: list[int | None],
    output_tokens_ref: list[int | None],
    text_parts: list[str] | None,
) -> None:
    """
    Parse a raw SSE chunk to extract usage data and optionally delta text.

    Mutates the mutable ref lists in-place.
    vLLM emits usage in the last data chunk when stream_options.include_usage=true.
    llama-server (BitNet) follows the same convention.
    """
    try:
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str in ("[DONE]", ""):
                continue
            data = json.loads(data_str)
            usage = data.get("usage") or {}
            if usage.get("prompt_tokens"):
                input_tokens_ref[0] = usage["prompt_tokens"]
            if usage.get("completion_tokens"):
                output_tokens_ref[0] = usage["completion_tokens"]
            if text_parts is not None:
                for choice in data.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text_parts.append(delta["content"])
    except Exception:
        pass  # chunk mal formado -- silencioso


async def shutdown() -> None:
    """Flush pending traces on process exit. Called from lifespan shutdown."""
    client = _get_client()
    if client:
        try:
            client.flush()
        except Exception:
            pass
