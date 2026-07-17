import asyncio
import json as _json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from ocabra.backends.base import BackendInterface, ModalityType, WorkerInfo
from ocabra.config import settings

logger = structlog.get_logger(__name__)


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
        everyone. Only touches ``/api/*`` bodies; a no-op without a resolver/cap.
        """
        if not (self._ollama_ctx_resolver and path.startswith("/api/")):
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
        opts["num_ctx"] = min(int(current), cap) if isinstance(current, int) and current > 0 else cap
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
            base = settings.ollama_base_url.rstrip("/")
            url = f"{base}{path}"
        else:
            url = f"http://127.0.0.1:{worker.port}{path}"
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            result = resp.json()
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
            base = settings.ollama_base_url.rstrip("/")
            url = f"{base}{path}"
        else:
            url = f"http://127.0.0.1:{worker.port}{path}"

        async def _raw_stream() -> AsyncIterator[bytes]:
            # Cancellation contract: when the client disconnects, Starlette cancels
            # the streaming task; the CancelledError unwinds through this generator
            # and the `async with client.stream()` __aexit__ closes the upstream
            # connection, which makes vLLM / llama.cpp / Ollama abort the in-flight
            # generation and free the GPU. Keep the per-request client + `async with`
            # here — a shared/persistent client or manual iteration would break this
            # and leak orphaned generations on abandoned (e.g. agentic) requests.
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", url, json=body) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        # Normalize reasoning → reasoning_content for chat completions
        is_chat = "/chat/completions" in path
        source: AsyncIterator[bytes] = _raw_stream()
        if is_chat:
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
                            line = b"data: " + _json.dumps(
                                payload, ensure_ascii=False,
                            ).encode()
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
            if name not in self._disabled_backends
            and modality in backend.supported_modalities()
        ]

    def supports_modality(self, backend_name: str, modality: ModalityType) -> bool:
        """Quick check for API layer validation."""
        backend = self._backends.get(backend_name)
        return backend is not None and modality in backend.supported_modalities()
