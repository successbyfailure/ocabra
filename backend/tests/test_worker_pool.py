import asyncio
from unittest.mock import patch

import httpx
import pytest

from ocabra import config
from ocabra.backends._mock import MockBackend
from ocabra.core.worker_pool import (
    WorkerPool,
    _ollama_native_body,
    _ollama_native_response,
)


@pytest.fixture
def pool():
    p = WorkerPool()
    p.register_backend("mock", MockBackend())
    return p


@pytest.mark.asyncio
async def test_assign_ports_no_collision(pool):
    p1 = await pool.assign_port()
    p2 = await pool.assign_port()
    p3 = await pool.assign_port()
    assert len({p1, p2, p3}) == 3


@pytest.mark.asyncio
async def test_release_port_makes_it_available(pool):
    port = await pool.assign_port()
    pool.release_port(port)
    reused = await pool.assign_port()
    assert reused == port


@pytest.mark.asyncio
async def test_get_backend_not_registered(pool):
    with pytest.raises(KeyError):
        await pool.get_backend("nonexistent")


@pytest.mark.asyncio
async def test_set_get_remove_worker(pool):
    from ocabra.backends.base import WorkerInfo

    info = WorkerInfo(
        backend_type="mock",
        model_id="test/model",
        gpu_indices=[1],
        port=18050,
        pid=999,
        vram_used_mb=4096,
    )
    pool.set_worker("test/model", info)
    assert pool.get_worker("test/model") == info

    pool.remove_worker("test/model")
    assert pool.get_worker("test/model") is None


@pytest.mark.asyncio
async def test_assign_port_range_exhausted():
    pool = WorkerPool()
    # Use a tiny range (only 2 ports)
    with patch.object(config.settings, "worker_port_range_start", 19000), \
         patch.object(config.settings, "worker_port_range_end", 19002):
        await pool.assign_port()
        await pool.assign_port()
        with pytest.raises(RuntimeError, match="No available ports"):
            await pool.assign_port()


@pytest.mark.asyncio
async def test_get_backend_disabled_reason():
    pool = WorkerPool()
    pool.register_disabled_backend("tensorrt_llm", "feature flag disabled")

    with pytest.raises(RuntimeError, match="feature flag disabled"):
        await pool.get_backend("tensorrt_llm")


@pytest.mark.asyncio
async def test_ollama_context_cap_applies_to_openai_compat_path():
    pool = WorkerPool()

    async def resolve_cap(_model_id: str) -> int:
        return 32_768

    pool.set_ollama_ctx_resolver(resolve_cap)
    body = await pool._clamp_ollama_ctx(
        "ollama/gemma4:12b::runtime",
        "/v1/chat/completions",
        {"model": "gemma4:12b", "messages": []},
    )

    assert body["options"]["num_ctx"] == 32_768


def test_ollama_openai_chat_uses_native_endpoint_and_normalizes_response():
    path, body, kind = _ollama_native_body(
        "/v1/chat/completions",
        {
            "model": "gemma4:12b",
            "messages": [{"role": "user", "content": "hola"}],
            "max_tokens": 12,
            "options": {"num_ctx": 32_768},
        },
    )

    assert path == "/api/chat"
    assert kind == "chat"
    assert body["options"] == {"num_ctx": 32_768, "num_predict": 12}
    normalized = _ollama_native_response(
        "chat",
        {
            "message": {"role": "assistant", "content": "hola"},
            "prompt_eval_count": 4,
            "eval_count": 2,
            "done_reason": "stop",
        },
        "gemma4:12b",
    )
    assert normalized["choices"][0]["message"]["content"] == "hola"
    assert normalized["usage"]["total_tokens"] == 6


def test_ollama_native_chat_converts_openai_content_blocks():
    path, body, kind = _ollama_native_body(
        "/v1/chat/completions",
        {
            "model": "gemma4:12b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe "},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                        },
                        {"type": "text", "text": "this image"},
                    ],
                }
            ],
        },
    )

    assert path == "/api/chat"
    assert kind == "chat"
    assert body["messages"] == [
        {
            "role": "user",
            "content": "Describe this image",
            "images": ["aGVsbG8="],
        }
    ]


def test_ollama_native_chat_keeps_remote_images_on_openai_endpoint():
    original = {
        "model": "gemma4:12b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/image.png"},
                    }
                ],
            }
        ],
    }

    path, body, kind = _ollama_native_body("/v1/chat/completions", original)

    assert path == "/v1/chat/completions"
    assert body is original
    assert kind is None


@pytest.mark.asyncio
async def test_forward_stream_raises_on_http_error(pool, monkeypatch):
    class _ErrorResponse:
        status_code = 503
        is_error = True

        def __init__(self, url: str) -> None:
            self.request = httpx.Request("POST", url)
            self.was_read = False

        async def aread(self) -> bytes:
            self.was_read = True
            return b'{"error":{"message":"Service unavailable"}}'

        def raise_for_status(self) -> None:
            assert self.was_read
            raise httpx.HTTPStatusError(
                "Service unavailable",
                request=self.request,
                response=httpx.Response(
                    self.status_code,
                    request=self.request,
                    content=b'{"error":{"message":"Service unavailable"}}',
                ),
            )

        async def aiter_bytes(self):
            if False:
                yield b""

    class _StreamContext:
        def __init__(self, response: _ErrorResponse) -> None:
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, json: dict):
            return _StreamContext(_ErrorResponse(url))

    monkeypatch.setattr("ocabra.core.worker_pool.httpx.AsyncClient", _FakeAsyncClient)
    from ocabra.backends.base import WorkerInfo

    pool.set_worker(
        "test/model",
        WorkerInfo(
            backend_type="mock",
            model_id="test/model",
            gpu_indices=[0],
            port=18042,
            pid=123,
            vram_used_mb=0,
        ),
    )

    with pytest.raises(httpx.HTTPStatusError):
        async for _chunk in pool.forward_stream("test/model", "/v1/chat/completions", {}):
            pass


def _make_post_client(status_code: int, body: bytes):
    class _FakeResponse:
        def __init__(self) -> None:
            self.status_code = status_code
            self._request = httpx.Request("POST", "http://x/")

        def raise_for_status(self) -> None:
            if 400 <= self.status_code:
                raise httpx.HTTPStatusError(
                    "error",
                    request=self._request,
                    response=httpx.Response(
                        self.status_code, request=self._request, content=body
                    ),
                )

        def json(self):
            return {"ok": True}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, json: dict):
            return _FakeResponse()

    return _FakeAsyncClient


def _register_worker(pool, model_id: str = "test/model") -> None:
    from ocabra.backends.base import WorkerInfo

    pool.set_worker(
        model_id,
        WorkerInfo(
            backend_type="mock",
            model_id=model_id,
            gpu_indices=[0],
            port=18043,
            pid=321,
            vram_used_mb=0,
        ),
    )


@pytest.mark.asyncio
async def test_forward_request_trips_circuit_breaker_on_repeated_dead_backend_errors(
    pool, monkeypatch
):
    """A worker that fails the same "process died" way several times in a
    row should get fast-failed instead of accepting (and timing out on)
    another request against a process that's already gone."""
    monkeypatch.setattr(config.settings, "worker_circuit_breaker_threshold", 2)
    monkeypatch.setattr(
        "ocabra.core.worker_pool.httpx.AsyncClient",
        _make_post_client(500, b"llama-server process no longer running: unknown"),
    )
    _register_worker(pool)

    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            await pool.forward_request("test/model", "/v1/chat/completions", {})

    from ocabra.core.worker_pool import WorkerDegradedError

    with pytest.raises(WorkerDegradedError):
        await pool.forward_request("test/model", "/v1/chat/completions", {})


@pytest.mark.asyncio
async def test_forward_request_does_not_trip_breaker_on_unrelated_errors(pool, monkeypatch):
    """An ordinary request-specific 5xx (bad prompt, transient OOM on one
    call) must not be treated as evidence the worker process died — only
    the known "dead runner" error patterns should count."""
    monkeypatch.setattr(config.settings, "worker_circuit_breaker_threshold", 2)
    monkeypatch.setattr(
        "ocabra.core.worker_pool.httpx.AsyncClient",
        _make_post_client(500, b'{"error": "context length exceeded"}'),
    )
    _register_worker(pool)

    for _ in range(5):
        with pytest.raises(httpx.HTTPStatusError):
            await pool.forward_request("test/model", "/v1/chat/completions", {})
    # Still raises the real HTTPStatusError, never the breaker.


@pytest.mark.asyncio
async def test_forward_request_breaker_resets_after_worker_replaced(pool, monkeypatch):
    """Reloading a model installs a fresh worker; a prior worker's failure
    history must not haunt the new process."""
    monkeypatch.setattr(config.settings, "worker_circuit_breaker_threshold", 1)
    monkeypatch.setattr(
        "ocabra.core.worker_pool.httpx.AsyncClient",
        _make_post_client(500, b"llama-server process no longer running: unknown"),
    )
    _register_worker(pool)

    with pytest.raises(httpx.HTTPStatusError):
        await pool.forward_request("test/model", "/v1/chat/completions", {})

    from ocabra.core.worker_pool import WorkerDegradedError

    with pytest.raises(WorkerDegradedError):
        await pool.forward_request("test/model", "/v1/chat/completions", {})

    _register_worker(pool)  # simulates a reload installing a fresh worker
    with pytest.raises(httpx.HTTPStatusError):
        await pool.forward_request("test/model", "/v1/chat/completions", {})


def _make_controllable_post_client(dispatch_log: list[str]):
    """Fake httpx.AsyncClient whose .post() blocks until the test explicitly
    releases it (via the returned per-call asyncio.Event), so a test can
    orchestrate overlapping in-flight requests deterministically instead of
    racing real async scheduling."""

    release_events: dict[int, asyncio.Event] = {}
    call_index = {"n": 0}

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"ok": True}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, json: dict):
            idx = call_index["n"]
            call_index["n"] += 1
            event = asyncio.Event()
            release_events[idx] = event
            dispatch_log.append(f"start:{json.get('_tag')}")
            await event.wait()
            dispatch_log.append(f"end:{json.get('_tag')}")
            return _FakeResponse()

    return _FakeAsyncClient, release_events


@pytest.mark.asyncio
async def test_forward_request_fair_gate_interleaves_two_users_on_one_slot(pool, monkeypatch):
    """End-to-end regression test for the fair-share admission gate
    (2026-10-28 design discussion): with slots=1, if alice has a backlog
    queued and bob arrives after alice's first request already started,
    bob must be served on the round right after whatever of alice's work
    was already queued ahead of him — not after her entire backlog."""
    dispatch_log: list[str] = []
    fake_client_cls, release_events = _make_controllable_post_client(dispatch_log)
    monkeypatch.setattr("ocabra.core.worker_pool.httpx.AsyncClient", fake_client_cls)
    _register_worker(pool)

    async def call(tag: str, user_key: str):
        await pool.forward_request(
            "test/model",
            "/v1/chat/completions",
            {"_tag": tag},
            user_key=user_key,
            slots=1,
        )

    alice1 = asyncio.ensure_future(call("alice-1", "alice"))
    await asyncio.sleep(0.01)  # let alice-1 actually dispatch (start:alice-1 logged)
    assert dispatch_log == ["start:alice-1"]

    alice2 = asyncio.ensure_future(call("alice-2", "alice"))
    bob1 = asyncio.ensure_future(call("bob-1", "bob"))
    await asyncio.sleep(0.01)  # both queued behind alice-1; neither dispatched yet
    assert dispatch_log == ["start:alice-1"]

    release_events[0].set()  # finish alice-1
    await alice1
    await asyncio.sleep(0.01)
    assert dispatch_log == ["start:alice-1", "end:alice-1", "start:alice-2"]

    release_events[1].set()  # finish alice-2
    await alice2
    await asyncio.sleep(0.01)
    assert dispatch_log[-1] == "start:bob-1"

    release_events[2].set()
    await bob1
