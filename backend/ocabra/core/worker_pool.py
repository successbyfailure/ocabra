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
from ocabra.core.fair_slot_gate import WorkerPoolFairGates

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


def _coerce_tool_arguments_to_object(messages: list[dict[str, Any]]) -> None:
    """Normalize assistant tool_calls' ``arguments`` to Ollama's native format.

    OpenAI-standard clients send ``arguments`` as a JSON-encoded STRING
    (e.g. ``'{"city":"Madrid"}'``); Ollama's native ``/api/chat`` expects a
    dict. Without this coercion, the second turn of a tool loop fails with
    ``upstream_invalid_request: Value looks like object, but can't find
    closing '}' symbol`` because Ollama tries to parse the already-JSON
    string as a dict.
    """
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = _json.loads(arguments) if arguments else {}
                except _json.JSONDecodeError:
                    # Leave as-is; Ollama may still handle string arguments
                    # on malformed input, otherwise the upstream error
                    # bubbles up untouched.
                    pass


def _apply_openai_tool_choice(
    native: dict[str, Any], tool_choice: Any
) -> None:
    """Translate OpenAI's ``tool_choice`` into something Ollama can honour.

    Ollama 0.4+ does not expose a native ``tool_choice`` knob; the model
    decides. We approximate by filtering ``tools`` and injecting a system
    instruction so the model behaves as if the knob existed:

    - ``"auto"`` / unset: pass tools through, let the model decide.
    - ``"none"``: strip ``tools`` entirely; the model never sees them.
    - ``"required"``: keep tools and nudge the model to invoke one.
    - ``{"type":"function","function":{"name":"X"}}``: keep ONLY that tool
      and nudge the model to invoke it specifically.
    """
    if tool_choice in (None, "auto"):
        return
    tools = native.get("tools") or []
    messages = native.get("messages") or []
    if tool_choice == "none":
        native.pop("tools", None)
        return
    if tool_choice == "required":
        if tools:
            system_hint = (
                "You MUST call one of the provided tools to answer this "
                "turn. Do not reply with plain text."
            )
            native["messages"] = [
                {"role": "system", "content": system_hint},
                *messages,
            ]
        return
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function") or {}
        name = function.get("name")
        if isinstance(name, str) and name:
            filtered = [
                tool for tool in tools
                if isinstance(tool, dict)
                and (tool.get("function") or {}).get("name") == name
            ]
            if filtered:
                native["tools"] = filtered
            system_hint = (
                f"You MUST call the tool named '{name}' to answer this "
                "turn. Do not reply with plain text and do not call any "
                "other tool."
            )
            native["messages"] = [
                {"role": "system", "content": system_hint},
                *messages,
            ]


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
        _coerce_tool_arguments_to_object(messages)
        native["messages"] = messages
        for key in ("tools", "format", "think", "keep_alive"):
            if key in body:
                native[key] = body[key]
        _apply_openai_tool_choice(native, body.get("tool_choice"))
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


def _normalize_tool_calls_from_ollama(
    tool_calls: Any,
) -> list[dict[str, Any]]:
    """Reshape Ollama's tool_calls into the OpenAI wire format.

    Ollama emits each call as ``{"function": {"index": N, "name": X,
    "arguments": {...}}}``. OpenAI clients expect
    ``{"index": N, "id": "call_...", "type": "function",
    "function": {"name": X, "arguments": "<json-string>"}}`` and the AI SDK
    treats ``arguments`` as a plain string. Mismatched shapes break every
    standard client that parses on the OpenAI schema (including Vercel's
    AI SDK and openai-python).

    Returns a NEW list; the input is left untouched for the stream path
    that reuses this helper per chunk.
    """
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for position, raw in enumerate(tool_calls):
        if not isinstance(raw, dict):
            continue
        source = dict(raw)
        function = dict(source.get("function") or {})
        inner_index = function.pop("index", None)
        index = source.get("index")
        if index is None:
            index = inner_index if isinstance(inner_index, int) else position
        name = function.get("name") or ""
        arguments = function.get("arguments")
        if isinstance(arguments, (dict, list)):
            function["arguments"] = _json.dumps(arguments, ensure_ascii=False)
        elif arguments is None:
            function["arguments"] = ""
        elif not isinstance(arguments, str):
            function["arguments"] = _json.dumps(arguments, ensure_ascii=False)
        call_id = source.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{uuid.uuid4().hex[:12]}"
        normalized.append(
            {
                "index": int(index) if isinstance(index, (int, float)) else position,
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": function.get("arguments") or "",
                },
            }
        )
    return normalized


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
        raw_tool_calls = message.get("tool_calls")
        normalized_tool_calls = _normalize_tool_calls_from_ollama(raw_tool_calls)
        if normalized_tool_calls:
            message["tool_calls"] = normalized_tool_calls
        elif "tool_calls" in message:
            message.pop("tool_calls")
        # OpenAI contract: finish_reason MUST be "tool_calls" whenever the
        # assistant turn emits any tool_calls, regardless of Ollama's own
        # done_reason (Ollama frequently returns "stop" even when it also
        # emits a tool_call, which breaks downstream dispatchers that key
        # off this field).
        if normalized_tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = result.get("done_reason") or "stop"
        return {
            **common,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
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


class WorkerDegradedError(RuntimeError):
    """A worker has failed repeatedly in a way that indicates its backend
    process died (not a one-off error) — raised instead of forwarding another
    request so callers fail fast rather than queueing behind a dead worker."""

    def __init__(self, model_id: str, failure_count: int) -> None:
        self.model_id = model_id
        self.failure_count = failure_count
        super().__init__(
            f"Worker for model '{model_id}' failed {failure_count} consecutive "
            "times and looks like its backend process died; rejecting new "
            "requests until it recovers"
        )


# Substrings of a 5xx body that indicate the backend's own process died
# rather than a request-specific error (bad input, OOM on this one prompt,
# etc). Matching on these keeps the breaker from tripping on ordinary 5xx
# noise — only patterns that mean "the runner itself is gone" count.
_DEAD_BACKEND_MARKERS = (
    "no longer running",
    "connection refused",
    "connect call failed",
)


def _summarize_messages_for_log(body: dict) -> list[dict] | None:
    """Structural (not content) summary of a chat ``messages`` list for error
    logs — role, content type/length, tool_calls count, presence of
    reasoning_content — without dumping user/model text into logs. Added
    after a gemma4 worker crash (2026-10-27) that cost a debugging cycle
    purely because the triggering message shape wasn't recoverable after
    the fact; see gpu-bench README.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append({"type": type(m).__name__})
            continue
        content = m.get("content")
        out.append(
            {
                "role": m.get("role"),
                "content_type": type(content).__name__ if content is not None else "None",
                "content_len": len(content) if isinstance(content, (str, list)) else None,
                "tool_calls": len(m["tool_calls"]) if isinstance(m.get("tool_calls"), list) else 0,
                "has_reasoning_content": bool(m.get("reasoning_content")),
            }
        )
    return out


def _log_upstream_http_error(logger_, event: str, *, model_id: str, path: str, body: dict, exc: httpx.HTTPStatusError) -> None:
    try:
        error_body = exc.response.text[:2000]
    except Exception:  # noqa: BLE001 — best-effort diagnostics only
        error_body = None
    logger_.warning(
        event,
        model_id=model_id,
        path=path,
        status_code=exc.response.status_code,
        error_body=error_body,
        messages=_summarize_messages_for_log(body),
    )


def _looks_like_dead_backend(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code < 500:
        return False
    try:
        body = exc.response.text.lower()
    except Exception:  # noqa: BLE001 — body read failures just mean "no match"
        return False
    return any(marker in body for marker in _DEAD_BACKEND_MARKERS)


class WorkerPool:
    def __init__(self) -> None:
        self._backends: dict[str, BackendInterface] = {}
        self._disabled_backends: dict[str, str] = {}
        self._workers: dict[str, WorkerInfo] = {}
        self._used_ports: set[int] = set()
        # Optional async callback (model_id) -> num_ctx cap | None, used to clamp
        # Ollama requests to a model's configured use-case context (wired in main).
        self._ollama_ctx_resolver: Any | None = None
        # Circuit breaker bookkeeping — see WorkerDegradedError.
        self._consecutive_failures: dict[str, int] = {}
        self._last_failure_at: dict[str, float] = {}
        # Per-worker fair-share admission — see fair_slot_gate.py. Only
        # engaged when a caller passes user_key/slots to forward_request /
        # forward_stream; callers that don't (embeddings, images, internal
        # proxying, ...) dispatch exactly as before.
        self._fair_gates = WorkerPoolFairGates()
        # Batch-vs-interactive priority (2026-10-30) — optional, set by
        # main.py at startup. None means the feature is simply inert: the
        # fair gate still deprioritizes is_batch=True on new admissions, but
        # nothing ever preempts an already-running batch request.
        self._batch_priority: Any | None = None
        self._duration_estimator: Any | None = None

    def set_batch_priority(self, coordinator: Any, duration_estimator: Any | None = None) -> None:
        self._batch_priority = coordinator
        self._duration_estimator = duration_estimator

    def _check_circuit_breaker(self, model_id: str) -> None:
        failures = self._consecutive_failures.get(model_id, 0)
        if failures < settings.worker_circuit_breaker_threshold:
            return
        elapsed = time.monotonic() - self._last_failure_at.get(model_id, 0.0)
        if elapsed < settings.worker_circuit_breaker_cooldown_s:
            raise WorkerDegradedError(model_id, failures)
        # Cooldown elapsed: let one probe request through (half-open) instead
        # of tripping forever on a worker that may have already recovered.

    def peek_queue_state(self, model_id: str) -> dict[str, Any] | None:
        """Read-only snapshot of a worker's fair-share gate — ``None`` when
        no gate exists yet (nothing has used fairness on this worker).

        Used to tell a streaming client it's about to wait for a slot
        before blocking on it, same spirit as the existing model-load
        ``X-Ocabra-Expected-Wait-Seconds``/``model_loading`` SSE signal —
        that one only covers a cold model load; this covers the fair-share
        admission queue on an already-loaded model, which previously had
        zero client-visible signal (confirmed 2026-10-30: the connection
        just blocks silently, even when the wait is ~20s+).
        """
        gate = self._fair_gates.peek(model_id)
        if gate is None:
            return None
        return {
            "in_flight": gate._in_flight,
            "slots": gate.slots,
            "saturated": gate._in_flight >= gate.slots,
        }

    def peek_gate_priority(
        self, model_id: str, user_key: str, *, is_batch: bool = False
    ) -> tuple[int, int, int] | None:
        """Read-only priority peek for diagnostics — returns ``None`` when
        no gate exists for ``model_id`` yet (nothing has used fairness on
        it), rather than creating one just to answer a stats query."""
        gate = self._fair_gates.peek(model_id)
        if gate is None:
            return None
        return gate.peek_priority(user_key, is_batch=is_batch)

    def _record_forward_failure(self, model_id: str) -> None:
        self._consecutive_failures[model_id] = self._consecutive_failures.get(model_id, 0) + 1
        self._last_failure_at[model_id] = time.monotonic()

    def _record_forward_success(self, model_id: str) -> None:
        self._consecutive_failures.pop(model_id, None)
        self._last_failure_at.pop(model_id, None)

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
        self._record_forward_success(model_id)

    def remove_worker(self, model_id: str) -> None:
        info = self._workers.pop(model_id, None)
        if info:
            self._used_ports.discard(info.port)
        self._record_forward_success(model_id)

    async def assign_port(self) -> int:
        for port in range(settings.worker_port_range_start, settings.worker_port_range_end):
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        raise RuntimeError("No available ports in worker port range")

    def release_port(self, port: int) -> None:
        self._used_ports.discard(port)

    async def _preempt_batch_if_saturated(self, model_id: str, gate, is_batch: bool) -> None:
        """An interactive caller is about to wait on a gate every slot of
        which is held by batch-origin work — nothing will free up on its
        own soon. Ask the coordinator to cancel enough of it per policy
        before we start waiting, instead of just queueing behind it (see
        ``batch_priority.py`` for the measured motivation: with all slots
        saturated, a new request waits almost the full remaining duration
        of whatever finishes first).
        """
        if is_batch or self._batch_priority is None:
            return
        saturated = gate.batch_holds_all_slots()
        logger.info(
            "batch_priority_check",
            model_id=model_id,
            in_flight=gate._in_flight,
            in_flight_batch=gate._in_flight_batch,
            slots=gate.slots,
            saturated=saturated,
        )
        if not saturated:
            return
        try:
            n = await self._batch_priority.preempt_now(model_id, estimator=self._duration_estimator)
            logger.info("batch_priority_preempt_result", model_id=model_id, cancelled=n)
        except Exception:  # noqa: BLE001 — never let priority logic break a real request
            logger.exception("batch_priority_preempt_failed", model_id=model_id)

    async def forward_request(
        self,
        model_id: str,
        path: str,
        body: dict,
        *,
        user_key: str | None = None,
        slots: int = 1,
        is_batch: bool = False,
    ) -> Any:
        if user_key is not None:
            gate = self._fair_gates.get(model_id, slots)
            await self._preempt_batch_if_saturated(model_id, gate, is_batch)
            async with gate.slot(user_key, is_batch=is_batch):
                return await self._forward_request_unfair(model_id, path, body)
        return await self._forward_request_unfair(model_id, path, body)

    async def _forward_request_unfair(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"No worker found for model '{model_id}'")
        self._check_circuit_breaker(model_id)
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
        except httpx.HTTPStatusError as exc:
            _log_upstream_http_error(
                logger, "inference_upstream_http_error", model_id=model_id, path=path, body=body, exc=exc
            )
            if _looks_like_dead_backend(exc):
                self._record_forward_failure(model_id)
                logger.warning(
                    "worker_looks_dead",
                    model_id=model_id,
                    path=path,
                    consecutive_failures=self._consecutive_failures[model_id],
                )
            raise
        self._record_forward_success(model_id)
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

    async def forward_stream(
        self,
        model_id: str,
        path: str,
        body: dict,
        *,
        user_key: str | None = None,
        slots: int = 1,
        is_batch: bool = False,
    ) -> AsyncIterator[bytes]:
        if user_key is None:
            async for chunk in self._forward_stream_unfair(model_id, path, body):
                yield chunk
            return
        gate = self._fair_gates.get(model_id, slots)
        await self._preempt_batch_if_saturated(model_id, gate, is_batch)
        async with gate.slot(user_key, is_batch=is_batch):
            async for chunk in self._forward_stream_unfair(model_id, path, body):
                yield chunk

    async def _forward_stream_unfair(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"No worker found for model '{model_id}'")
        self._check_circuit_breaker(model_id)
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
                self._record_forward_success(model_id)
            except httpx.TimeoutException as exc:
                logger.warning(
                    "inference_upstream_stream_timeout",
                    model_id=model_id,
                    path=path,
                    timeout_s=timeout_s,
                )
                raise InferenceTimeoutError(model_id, path, timeout_s) from exc
            except httpx.HTTPStatusError as exc:
                _log_upstream_http_error(
                    logger, "inference_upstream_stream_http_error", model_id=model_id, path=path, body=body, exc=exc
                )
                if _looks_like_dead_backend(exc):
                    self._record_forward_failure(model_id)
                    logger.warning(
                        "worker_looks_dead",
                        model_id=model_id,
                        path=path,
                        consecutive_failures=self._consecutive_failures[model_id],
                    )
                raise

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
                    normalized_tool_calls = _normalize_tool_calls_from_ollama(
                        message.get("tool_calls")
                    )
                    if normalized_tool_calls:
                        delta["tool_calls"] = normalized_tool_calls
                        # Stream finish_reason must match the final shape
                        # (OpenAI emits ``"tool_calls"`` on the terminating
                        # chunk whenever the assistant produced any call).
                        if done:
                            finish_reason = "tool_calls"
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
