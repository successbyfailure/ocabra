import asyncio
import json as _json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from ocabra.backends.base import BackendInterface, ModalityType, WorkerInfo
from ocabra.config import settings

logger = structlog.get_logger(__name__)


def _ollama_native_messages(messages: Any) -> tuple[list[dict[str, Any]], bool]:
    """Convert OpenAI content blocks to Ollama native chat messages.

    Ollama's OpenAI endpoint accepts ``content`` arrays, while ``/api/chat``
    requires a string plus a separate base64 ``images`` list.  Remote image
    URLs and unknown block types stay on the OpenAI endpoint because fetching
    arbitrary URLs here would both change semantics and introduce SSRF risk.
    """

    if not isinstance(messages, list):
        return [], False

    converted: list[dict[str, Any]] = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            return [], False
        message = dict(raw_message)
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue

        text_parts: list[str] = []
        images = list(message.get("images") or [])
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                return [], False

            block_type = str(block.get("type") or "")
            text = block.get("text")
            if block_type in {"text", "input_text", "output_text"} and isinstance(
                text, str
            ):
                text_parts.append(text)
                continue
            if block_type == "image_url":
                image_url = block.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not isinstance(url, str) or not url.startswith("data:"):
                    return [], False
                marker = ";base64,"
                if marker not in url:
                    return [], False
                images.append(url.split(marker, 1)[1])
                continue
            return [], False

        message["content"] = "".join(text_parts)
        if images:
            message["images"] = images
        converted.append(message)
    return converted, True


def _ollama_native_body(path: str, body: dict) -> tuple[str, dict, str | None]:
    """Translate Ollama's OpenAI shim requests to its native API.

    The native endpoints are required for runtime presets because Ollama's
    OpenAI-compatible endpoints ignore ``options.num_ctx`` and silently reload
    the daemon default context.
    """

    if path not in {"/v1/chat/completions", "/v1/completions"}:
        return path, body, None
    kind = "chat" if path.endswith("chat/completions") else "completion"
    native: dict[str, Any] = {
        "model": body.get("model"),
        "stream": bool(body.get("stream", False)),
    }
    if kind == "chat":
        messages, compatible = _ollama_native_messages(body.get("messages", []))
        if not compatible:
            return path, body, None
        native["messages"] = messages
        for key in ("tools", "format", "think", "keep_alive"):
            if key in body:
                native[key] = body[key]
        native_path = "/api/chat"
    else:
        native["prompt"] = body.get("prompt", "")
        for key in ("suffix", "system", "template", "raw", "format", "think", "keep_alive"):
            if key in body:
                native[key] = body[key]
        native_path = "/api/generate"

    options = dict(body.get("options") or {})
    aliases = {
        "max_completion_tokens": "num_predict",
        "max_tokens": "num_predict",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "stop": "stop",
        "seed": "seed",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
    }
    for source, target in aliases.items():
        if source in body and target not in options:
            options[target] = body[source]
    native["options"] = options
    return native_path, native, kind


def _ollama_native_response(kind: str, result: dict, model: str) -> dict:
    usage = {
        "prompt_tokens": int(result.get("prompt_eval_count") or 0),
        "completion_tokens": int(result.get("eval_count") or 0),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    common = {
        "id": f"{kind}-{uuid.uuid4().hex}",
        "created": int(time.time()),
        "model": model,
        "usage": usage,
    }
    if kind == "chat":
        message = dict(result.get("message") or {})
        if "thinking" in message and "reasoning_content" not in message:
            message["reasoning_content"] = message.pop("thinking")
        return {
            **common,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": result.get("done_reason") or "stop",
                }
            ],
        }
    return {
        **common,
        "object": "text_completion",
        "choices": [
            {
                "index": 0,
                "text": str(result.get("response") or ""),
                "finish_reason": result.get("done_reason") or "stop",
            }
        ],
    }


class InferenceTimeoutError(TimeoutError):
    """An inference worker exceeded the configured upstream timeout."""

    def __init__(self, model_id: str, path: str, timeout_s: int) -> None:
        self.model_id = model_id
        self.path = path
        self.timeout_s = timeout_s
        super().__init__(f"Inference for model '{model_id}' timed out after {timeout_s}s")


class WorkerPool:
    def __init__(self) -> None:
        self._backends: dict[str, BackendInterface] = {}
        self._disabled_backends: dict[str, str] = {}
        self._workers: dict[str, WorkerInfo] = {}
        self._used_ports: set[int] = set()
        # Optional async callback (model_id) -> num_ctx cap | None, used to clamp
        # Ollama requests to a model's configured use-case context (wired in main).
        self._ollama_ctx_resolver: Any | None = None

    def set_ollama_ctx_resolver(self, resolver: Any) -> None:
        self._ollama_ctx_resolver = resolver

    async def _clamp_ollama_ctx(self, model_id: str, path: str, body: dict) -> dict:
        """Cap ``options.num_ctx`` on native Ollama requests to the model's
        configured use-case context. Ollama reserves KV = num_ctx × slots up front,
        so an unclamped client (OpenWebUI often sends 32k+) inflates VRAM for
        everyone. Applies to native ``/api/*`` and Ollama's OpenAI-compatible
        completion paths; it is a no-op without a resolver/cap.
        """
        context_aware_path = path.startswith("/api/") or path in {
            "/v1/chat/completions",
            "/v1/completions",
        }
        if not (self._ollama_ctx_resolver and context_aware_path):
            return body
        try:
            cap = await self._ollama_ctx_resolver(model_id)
        except Exception as exc:  # noqa: BLE001 — never block a request on the clamp
            logger.warning("ollama_ctx_clamp_failed", model_id=model_id, error=str(exc))
            return body
        if not cap:
            return body
        opts = dict(body.get("options") or {})
        current = opts.get("num_ctx")
        opts["num_ctx"] = (
            min(int(current), cap) if isinstance(current, int) and current > 0 else cap
        )
        return {**body, "options": opts}

    def register_backend(self, backend_type: str, backend: BackendInterface) -> None:
        self._backends[backend_type] = backend
        self._disabled_backends.pop(backend_type, None)
        logger.info("backend_registered", backend_type=backend_type)

    def register_disabled_backend(self, backend_type: str, reason: str) -> None:
        self._backends.pop(backend_type, None)
        self._disabled_backends[backend_type] = reason
        logger.info("backend_disabled", backend_type=backend_type, reason=reason)

    def registered_backends(self) -> dict[str, BackendInterface]:
        """Snapshot of currently registered (enabled) backends, keyed by backend_type."""
        return dict(self._backends)

    async def get_backend(self, backend_type: str) -> BackendInterface:
        if backend_type in self._disabled_backends:
            raise RuntimeError(
                f"Backend '{backend_type}' is disabled: {self._disabled_backends[backend_type]}"
            )
        if backend_type not in self._backends:
            raise KeyError(f"Backend '{backend_type}' not registered")
        return self._backends[backend_type]

    def get_worker(self, model_id: str) -> WorkerInfo | None:
        return self._workers.get(model_id)

    def set_worker(self, model_id: str, info: WorkerInfo) -> None:
        self._workers[model_id] = info
        self._used_ports.add(info.port)

    def remove_worker(self, model_id: str) -> None:
        info = self._workers.pop(model_id, None)
        if info:
            self._used_ports.discard(info.port)

    async def assign_port(self) -> int:
        for port in range(settings.worker_port_range_start, settings.worker_port_range_end):
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        raise RuntimeError("No available ports in worker port range")

    def release_port(self, port: int) -> None:
        self._used_ports.discard(port)

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"No worker found for model '{model_id}'")
        if worker.backend_type == "ollama":
            body = await self._clamp_ollama_ctx(model_id, path, body)
            requested_model = str(body.get("model") or worker.model_id)
            path, body, ollama_adapter = _ollama_native_body(path, body)
            base = (worker.endpoint_url or settings.ollama_base_url).rstrip("/")
            url = f"{base}{path}"
        else:
            ollama_adapter = None
            requested_model = worker.model_id
            url = f"http://127.0.0.1:{worker.port}{path}"
        start = time.monotonic()
        timeout_s = max(30, int(settings.inference_request_timeout_seconds))
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(float(timeout_s), connect=30.0)
            ) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                result = resp.json()
        except httpx.TimeoutException as exc:
            logger.warning(
                "inference_upstream_timeout",
                model_id=model_id,
                path=path,
                timeout_s=timeout_s,
            )
            raise InferenceTimeoutError(model_id, path, timeout_s) from exc
        if ollama_adapter is not None:
            result = _ollama_native_response(ollama_adapter, result, requested_model)
        # Normalize reasoning → reasoning_content for chat completions
        if "/chat/completions" in path:
            for choice in result.get("choices", []):
                msg = choice.get("message") or choice.get("delta")
                if msg and "reasoning" in msg:
                    msg["reasoning_content"] = msg.pop("reasoning")
        if settings.langfuse_enabled:
            from ocabra.integrations.langfuse_tracer import trace_generation

            asyncio.create_task(
                trace_generation(
                    model_id=model_id,
                    path=path,
                    request_body=body,
                    response_body=result,
                    duration_ms=(time.monotonic() - start) * 1000,
                    user_id=body.get("user"),
                )
            )
        return result

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"No worker found for model '{model_id}'")
        if worker.backend_type == "ollama":
            body = await self._clamp_ollama_ctx(model_id, path, body)
            requested_model = str(body.get("model") or worker.model_id)
            path, body, ollama_adapter = _ollama_native_body(path, body)
            base = (worker.endpoint_url or settings.ollama_base_url).rstrip("/")
            url = f"{base}{path}"
        else:
            ollama_adapter = None
            requested_model = worker.model_id
            url = f"http://127.0.0.1:{worker.port}{path}"

        async def _raw_stream() -> AsyncIterator[bytes]:
            # Cancellation contract: when the client disconnects, Starlette cancels
            # the streaming task; the CancelledError unwinds through this generator
            # and the `async with client.stream()` __aexit__ closes the upstream
            # connection, which makes vLLM / llama.cpp / Ollama abort the in-flight
            # generation and free the GPU. Keep the per-request client + `async with`
            # here — a shared/persistent client or manual iteration would break this
            # and leak orphaned generations on abandoned (e.g. agentic) requests.
            timeout_s = max(30, int(settings.inference_request_timeout_seconds))
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(float(timeout_s), connect=30.0)
                ) as client:
                    async with client.stream("POST", url, json=body) as resp:
                        # httpx does not buffer streaming responses. Read an
                        # error body while the response is still open so the
                        # API layer can preserve its validation detail.
                        if resp.is_error:
                            await resp.aread()
                        resp.raise_for_status()
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            except httpx.TimeoutException as exc:
                logger.warning(
                    "inference_upstream_stream_timeout",
                    model_id=model_id,
                    path=path,
                    timeout_s=timeout_s,
                )
                raise InferenceTimeoutError(model_id, path, timeout_s) from exc

        # Normalize reasoning → reasoning_content for chat completions
        is_chat = "/chat/completions" in path
        source: AsyncIterator[bytes] = _raw_stream()
        if ollama_adapter is not None:
            source = self._ollama_native_stream(
                source,
                kind=ollama_adapter,
                model=requested_model,
            )
        elif is_chat:
            source = self._normalize_reasoning_stream(source)

        if settings.langfuse_enabled:
            from ocabra.integrations.langfuse_tracer import wrap_stream

            async for chunk in wrap_stream(
                source,
                model_id=model_id,
                path=path,
                request_body=body,
                user_id=body.get("user"),
            ):
                yield chunk
        else:
            async for chunk in source:
                yield chunk

    # ------------------------------------------------------------------
    # SSE normalisation
    # ------------------------------------------------------------------

    @staticmethod
    async def _ollama_native_stream(
        raw: AsyncIterator[bytes],
        *,
        kind: str,
        model: str,
    ) -> AsyncIterator[bytes]:
        """Convert Ollama native NDJSON to OpenAI-compatible SSE."""

        buffer = ""
        response_id = f"{kind}-{uuid.uuid4().hex}"
        created = int(time.time())
        async for chunk in raw:
            buffer += chunk.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                try:
                    payload = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                done = bool(payload.get("done"))
                finish_reason = payload.get("done_reason") or ("stop" if done else None)
                if kind == "chat":
                    message = dict(payload.get("message") or {})
                    delta: dict[str, Any] = {}
                    if message.get("role"):
                        delta["role"] = message["role"]
                    if "content" in message:
                        delta["content"] = message.get("content") or ""
                    if message.get("thinking"):
                        delta["reasoning_content"] = message["thinking"]
                    if message.get("tool_calls"):
                        delta["tool_calls"] = message["tool_calls"]
                    event = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                else:
                    event = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "text": str(payload.get("response") or ""),
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                if done:
                    prompt_tokens = int(payload.get("prompt_eval_count") or 0)
                    completion_tokens = int(payload.get("eval_count") or 0)
                    event["usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                yield f"data: {_json.dumps(event)}\n\n".encode()
                if done:
                    yield b"data: [DONE]\n\n"
                    return

    @staticmethod
    async def _normalize_reasoning_stream(
        raw: AsyncIterator[bytes],
    ) -> AsyncIterator[bytes]:
        """Normalize ``reasoning`` → ``reasoning_content`` in SSE delta chunks.

        Both vLLM and Ollama emit a non-standard ``"reasoning"`` key inside
        ``choices[].delta``.  The de-facto standard (DeepSeek / OpenAI) is
        ``"reasoning_content"``.  This pass rewrites the field name so that
        downstream clients receive a consistent schema without touching the
        actual content.

        The function reassembles byte chunks into complete SSE lines, edits
        only the JSON payload of ``data:`` lines, and re-emits valid SSE.
        """
        buf = b""
        async for chunk in raw:
            buf += chunk
            # Process all complete lines in the buffer
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                line = line_bytes.rstrip(b"\r")

                if line.startswith(b"data: ") and line != b"data: [DONE]":
                    try:
                        payload = _json.loads(line[6:])
                        changed = False
                        for choice in payload.get("choices", []):
                            delta = choice.get("delta")
                            if delta and "reasoning" in delta:
                                delta["reasoning_content"] = delta.pop("reasoning")
                                changed = True
                        if changed:
                            line = (
                                b"data: "
                                + _json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                ).encode()
                            )
                    except (_json.JSONDecodeError, KeyError, TypeError):
                        pass  # forward as-is

                yield line + b"\n"

        # Flush any remaining bytes (no trailing newline)
        if buf:
            yield buf

    # ------------------------------------------------------------------
    # Modality helpers
    # ------------------------------------------------------------------

    def get_backends_for_modality(self, modality: ModalityType) -> list[str]:
        """Return names of registered backends that support the given modality.

        Excludes disabled backends.
        """
        return [
            name
            for name, backend in self._backends.items()
            if name not in self._disabled_backends and modality in backend.supported_modalities()
        ]

    def supports_modality(self, backend_name: str, modality: ModalityType) -> bool:
        """Quick check for API layer validation."""
        backend = self._backends.get(backend_name)
        return backend is not None and modality in backend.supported_modalities()
