"""Admission (in-flight cap, maintenance, circuit breaker) through the whole
request stack — REF-08.

Everything here goes through real middlewares and a real ``ModelManager``
(its in-flight counters are what the cap reads), not mocks of them. The first
half characterises what admission already did before it left the stats
middleware: status codes, bodies, ``Retry-After``, which key a request is
counted under, when it is released. The second half covers what was broken:
uploads admitted without being counted (a burst overshot the cap) and
endpoints that counted a request a second time.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from ocabra.core.request_telemetry import current_inflight_request_id
from ocabra.core.worker_pool import InferenceTimeoutError, WorkerDegradedError
from ocabra.stats.collector import StatsMiddleware
from tests._request_stack import install_request_stack as _install
from tests.test_vram_eviction import _add_loaded_model, _make_mm

# The handlers' tests patch ``httpx.AsyncClient`` (the worker call) on the
# module itself; the test client must be the real one.
_HttpClient = httpx.AsyncClient


def _profile(pid: str, base: str, **extra) -> SimpleNamespace:
    return SimpleNamespace(
        profile_id=pid,
        base_model_id=base,
        load_overrides=None,
        enabled=True,
        is_default=True,
        profile_kind="execution",
        request_defaults={},
        assets=None,
        **extra,
    )


def _registry(*profiles: SimpleNamespace) -> SimpleNamespace:
    by_id = {p.profile_id: p for p in profiles}

    async def get(pid):
        return by_id.get(pid)

    async def list_by_model(_mid):
        return []

    async def resolve_alias(_pid):
        return None

    return SimpleNamespace(get=get, list_by_model=list_by_model, resolve_alias=resolve_alias)


def _app(mm=None, *profiles: SimpleNamespace) -> FastAPI:
    app = FastAPI()
    app.state.model_manager = mm if mm is not None else _make_mm()
    app.state.profile_registry = _registry(*(profiles or (_profile("chat-public", "llama/chat"),)))
    return app


@pytest.fixture
def record_stat():
    with patch("ocabra.stats.collector._record_stat", new=AsyncMock()) as mock:
        yield mock


def _chat(model: str = "chat-public", **extra) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hola que tal"}], **extra}


# ── characterisation: JSON requests ──────────────────────────────────────


def test_json_request_is_counted_under_its_worker_while_it_runs(record_stat):
    app = _app()
    mm = app.state.model_manager
    seen: dict = {}

    @app.post("/v1/chat/completions")
    async def _chat_endpoint(request: Request):
        seen["count"] = mm.inflight_count("llama/chat")
        seen["rid"] = current_inflight_request_id.get()
        seen["state_rid"] = getattr(request.state, "inflight_request_id", None)
        (active,) = mm.active_requests_snapshot()
        seen["active"] = active
        return JSONResponse({"usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    _install(app)
    response = TestClient(app).post("/v1/chat/completions", json=_chat(max_tokens=50))

    assert response.status_code == 200
    assert seen["count"] == 1
    assert seen["rid"] and seen["rid"] == seen["state_rid"] == seen["active"].request_id
    active = seen["active"]
    assert active.profile_id == "chat-public"
    assert active.request_kind == "chat"
    assert active.streaming is False
    assert active.max_output_tokens == 50
    assert active.input_tokens_estimate == 3
    assert mm.inflight_count("llama/chat") == 0
    assert mm.active_requests_snapshot() == []
    kwargs = record_stat.await_args.kwargs
    assert kwargs["model_id"] == "chat-public"
    assert kwargs["status_code"] == 200
    assert kwargs["output_tokens"] == 2


def test_maintenance_gives_503_before_the_endpoint(record_stat):
    app = _app()
    mm = app.state.model_manager
    mm.set_maintenance("llama/chat", True)

    @app.post("/v1/chat/completions")
    async def _unreachable():
        raise AssertionError("maintenance must stop the request first")

    _install(app)
    response = TestClient(app).post("/v1/chat/completions", json=_chat())

    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert response.json() == {
        "error": {
            "message": "Model 'llama/chat' is under maintenance. Please retry shortly.",
            "type": "server_error",
            "code": "model_maintenance",
        }
    }
    assert mm.active_requests_snapshot() == []
    kwargs = record_stat.await_args.kwargs
    assert kwargs["model_id"] == "llama/chat"
    assert kwargs["status_code"] == 503
    assert kwargs["error_message"] == "model_maintenance"
    assert kwargs["request_kind"] == "chat"
    assert kwargs["input_tokens"] is None


@pytest.mark.parametrize(("slots", "cap", "retry"), [(1, 4, "10"), (2, 8, "2")])
def test_at_capacity_gives_429_with_retry_after_by_cap(record_stat, slots, cap, retry):
    app = _app()
    mm = app.state.model_manager
    _add_loaded_model(mm, "llama/chat", in_flight=cap)
    with patch("ocabra.core.model_manager.resolve_worker_slots", return_value=slots):

        @app.post("/v1/chat/completions")
        async def _unreachable():
            raise AssertionError("over the cap")

        _install(app)
        response = TestClient(app).post("/v1/chat/completions", json=_chat())

    assert response.status_code == 429
    assert response.headers["retry-after"] == retry
    assert response.json()["error"] == {
        "message": (
            f"Model 'llama/chat' is at capacity ({cap} concurrent requests for this "
            "worker). Please retry shortly."
        ),
        "type": "rate_limit_error",
        "code": "model_at_capacity",
    }
    assert mm.inflight_count("llama/chat") == cap  # nothing leaked, nothing added
    kwargs = record_stat.await_args.kwargs
    assert (kwargs["status_code"], kwargs["error_message"]) == (429, "model_at_capacity")
    assert kwargs["model_id"] == "llama/chat"


def test_encoder_kinds_use_the_flat_cap_not_the_slot_one(record_stat):
    """Embeddings have no KV slots: the 1-slot fallback (4) must not apply."""
    app = _app(None, _profile("emb", "llama/emb"))
    mm = app.state.model_manager
    mm._in_flight["llama/emb"] = 10

    @app.post("/v1/embeddings")
    async def _emb():
        return JSONResponse({"data": [], "usage": {"prompt_tokens": 1}})

    @app.post("/v1/chat/completions")
    async def _chat_endpoint():
        return JSONResponse({})

    _install(app)
    client = TestClient(app)
    assert client.post("/v1/embeddings", json={"model": "emb", "input": "x"}).status_code == 200
    assert client.post("/v1/chat/completions", json=_chat("emb")).status_code == 429


def test_requests_bound_for_a_peer_skip_the_local_slot_cap(record_stat):
    app = _app(None, _profile("remote", "llama/remote"))
    mm = app.state.model_manager
    mm._in_flight["llama/remote"] = 10
    app.state.federation_manager = SimpleNamespace(
        get_remote_models=lambda: {"llama/remote": ["peer"]},
        remote_model_for=lambda _id: None,
        rules_for_user=lambda _uid: None,
    )

    @app.post("/v1/chat/completions")
    async def _chat_endpoint():
        return JSONResponse({})

    _install(app)
    with patch("ocabra.config.settings.federation_enabled", True):
        assert TestClient(app).post("/v1/chat/completions", json=_chat("remote")).status_code == 200
    with patch("ocabra.config.settings.federation_enabled", False):
        assert TestClient(app).post("/v1/chat/completions", json=_chat("remote")).status_code == 429


def test_router_request_is_counted_against_the_warm_target(record_stat):
    from ocabra.core.model_manager import ModelStatus

    router = SimpleNamespace(
        profile_id="router",
        enabled=True,
        profile_kind="router",
        routing_targets=["cold", "warm"],
    )
    app = _app(None, router, _profile("cold", "llama/cold"), _profile("warm", "llama/warm"))
    mm = app.state.model_manager
    seen: dict = {}

    async def get_state(key):
        if key == "llama/warm":
            return SimpleNamespace(status=ModelStatus.LOADED, model_id=key)
        return None

    @app.post("/v1/chat/completions")
    async def _chat_endpoint():
        seen.update(mm.inflight_snapshot())
        return JSONResponse({})

    _install(app)
    with patch.object(mm, "get_state", get_state):
        TestClient(app).post("/v1/chat/completions", json=_chat("router"))
    assert seen == {"llama/warm": 1}


def test_batch_origin_requests_are_admitted_like_any_other(record_stat):
    """The batch processor retries 429s itself: admission has no exemption."""
    from ocabra.api._deps_auth import INTERNAL_TOKEN, INTERNAL_TOKEN_HEADER

    app = _app()
    mm = app.state.model_manager
    mm._in_flight["llama/chat"] = 4

    @app.post("/v1/chat/completions")
    async def _chat_endpoint():
        return JSONResponse({})

    _install(app)
    response = TestClient(app).post(
        "/v1/chat/completions",
        json=_chat(),
        headers={INTERNAL_TOKEN_HEADER: INTERNAL_TOKEN, "X-Internal-Request-Origin": "batch"},
    )
    assert response.status_code == 429


def test_untracked_paths_are_left_alone(record_stat):
    app = _app()
    mm = app.state.model_manager
    mm.set_maintenance("llama/chat", True)

    @app.post("/ocabra/whatever")
    async def _internal():
        return JSONResponse({"ok": True})

    _install(app)
    response = TestClient(app).post("/ocabra/whatever", json=_chat())
    assert response.status_code == 200
    record_stat.assert_not_awaited()


# ── characterisation: errors and the circuit breaker ──────────────────────


@pytest.mark.parametrize(
    ("path", "exc", "status", "body", "retry"),
    [
        (
            "/v1/chat/completions",
            InferenceTimeoutError("llama/chat", "/v1/chat/completions", 900),
            504,
            "generation_timeout",
            None,
        ),
        ("/api/chat", InferenceTimeoutError("llama/chat", "/api/chat", 900), 504, None, None),
        (
            "/v1/chat/completions",
            WorkerDegradedError("llama/chat", 3),
            503,
            "worker_degraded",
            "30",
        ),
        ("/api/chat", WorkerDegradedError("llama/chat", 3), 503, None, "30"),
    ],
)
def test_dispatch_errors_map_to_their_status_and_release(
    record_stat, path, exc, status, body, retry
):
    app = _app()
    mm = app.state.model_manager

    @app.post(path)
    async def _fails():
        raise exc

    _install(app)
    response = TestClient(app, raise_server_exceptions=False).post(path, json=_chat())

    assert response.status_code == status
    if body is None:
        assert response.json() == {"error": str(exc)}
    else:
        assert response.json() == {
            "error": {"message": str(exc), "type": "server_error", "code": body}
        }
    assert response.headers.get("retry-after") == retry
    assert mm.inflight_count("llama/chat") == 0
    kwargs = record_stat.await_args.kwargs
    assert (kwargs["status_code"], kwargs["error_message"]) == (status, str(exc))
    assert kwargs["model_id"] == "chat-public"


def test_unexpected_error_is_a_500_and_releases(record_stat):
    app = _app()
    mm = app.state.model_manager

    @app.post("/v1/chat/completions")
    async def _boom():
        raise RuntimeError("kaput")

    _install(app)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/chat/completions", json=_chat()
    )
    assert response.status_code == 500
    assert mm.inflight_count("llama/chat") == 0
    kwargs = record_stat.await_args.kwargs
    assert (kwargs["status_code"], kwargs["error_message"]) == (500, "kaput")


# ── characterisation: streams, disconnects, cancellation ──────────────────


def _http_scope(app: FastAPI, path: str, body: bytes, content_type: str) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", content_type.encode()), (b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": app,
    }


async def _stream_with_disconnect(app, chunks_before_disconnect: int, release: asyncio.Event):
    """Run one streaming request on the raw ASGI app; the client hangs up
    after ``chunks_before_disconnect`` body chunks."""
    body = json.dumps(_chat(stream=True)).encode()
    sent_body = False
    disconnected = asyncio.Event()
    received: list[dict] = []

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        received.append(message)
        chunks = [m for m in received if m["type"] == "http.response.body" and m.get("body")]
        if len(chunks) >= chunks_before_disconnect:
            disconnected.set()

    await asyncio.wait_for(
        app(_http_scope(app, "/v1/chat/completions", body, "application/json"), receive, send),
        timeout=5,
    )
    return received


def _streaming_app(gate: asyncio.Event, observed: list):
    app = _app()
    mm = app.state.model_manager

    @app.post("/v1/chat/completions")
    async def _stream():
        async def gen():
            yield b'data: {"choices":[{"delta":{"content":"hola"}}]}\n\n'
            observed.append(mm.inflight_count("llama/chat"))
            await gate.wait()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    _install(app)
    return app, mm


@pytest.mark.asyncio
async def test_stream_holds_its_slot_until_it_ends(record_stat):
    gate = asyncio.Event()
    observed: list = []
    app, mm = _streaming_app(gate, observed)
    task = asyncio.create_task(_stream_with_disconnect(app, 99, gate))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if observed:
            break
    assert observed == [1]
    assert mm.inflight_count("llama/chat") == 1  # headers sent, still running
    gate.set()
    await task
    assert mm.inflight_count("llama/chat") == 0
    await asyncio.sleep(0)
    assert record_stat.await_args.kwargs["status_code"] == 200


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_releases_the_slot(record_stat):
    gate = asyncio.Event()  # never set: the generator would wait forever
    observed: list = []
    app, mm = _streaming_app(gate, observed)
    await _stream_with_disconnect(app, 1, gate)
    await asyncio.sleep(0)
    assert mm.inflight_count("llama/chat") == 0
    assert mm.active_requests_snapshot() == []
    # Recorded as 499 when the stats tee sees the hang-up first, as a short
    # 200 when the handler's own StreamingResponse does (a race inside
    # BaseHTTPMiddleware, unchanged by REF-08). The slot is released either way.
    assert record_stat.await_args.kwargs["status_code"] in (200, 499)


@pytest.mark.asyncio
async def test_admin_cancel_releases_and_records_499(record_stat):
    app = _app()
    mm = app.state.model_manager
    started = asyncio.Event()

    @app.post("/v1/chat/completions")
    async def _slow():
        started.set()
        await asyncio.sleep(30)

    _install(app)
    transport = httpx.ASGITransport(app=app)
    async with _HttpClient(transport=transport, base_url="http://t") as client:
        task = asyncio.create_task(client.post("/v1/chat/completions", json=_chat()))
        await asyncio.wait_for(started.wait(), 5)
        (active,) = mm.active_requests_snapshot()
        assert mm.cancel_request(active.request_id) is not None
        with pytest.raises(asyncio.CancelledError):
            await task
    assert mm.inflight_count("llama/chat") == 0
    kwargs = record_stat.await_args.kwargs
    assert (kwargs["status_code"], kwargs["error_message"]) == (499, "cancelled")


# ── characterisation: federation peers under rules ────────────────────────


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_peer_rules_reach_the_handler_before_it_loads(record_stat):
    """«Me first» and «don't evict» come from the rules admission attaches to
    the request (state + contextvar), in the handler's own context."""
    from ocabra.api.openai._deps import is_batch_origin_request
    from ocabra.core import federation_rules as fr

    rules = fr.PeerRules(local_first=True, allow_evict=False)
    app = _app()
    app.state.federation_manager = SimpleNamespace(
        rules_for_user=lambda uid: rules if uid == "peer-user" else None,
        get_remote_models=dict,
        remote_model_for=lambda _id: None,
    )
    seen: dict = {}

    @app.post("/v1/chat/completions")
    async def _chat_endpoint(request: Request):
        seen["batch"] = is_batch_origin_request(request)
        seen["ctx"] = fr.current_peer_rules.get()
        (active,) = app.state.model_manager.active_requests_snapshot()
        seen["user"] = (active.user_id, active.username, active.api_key_name)
        return JSONResponse({})

    ctx = SimpleNamespace(
        user_id="peer-user", username="peer", api_key_name="fed", is_anonymous=False
    )
    _install(app)
    with (
        patch("ocabra.api._deps_auth._resolve_api_key", new=AsyncMock(return_value=ctx)),
        patch("ocabra.database.AsyncSessionLocal", new=lambda: _Session()),
    ):
        response = TestClient(app).post(
            "/v1/chat/completions", json=_chat(), headers={"Authorization": "Bearer sk-x"}
        )
    assert response.status_code == 200
    assert seen == {"batch": True, "ctx": rules, "user": ("peer-user", "peer", "fed")}


# ── multipart uploads ─────────────────────────────────────────────────────

_WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


def _multipart(model: str, audio: bytes, boundary: str = "XyZ") -> bytes:
    return (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"{model}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + audio
        + f"\r\n--{boundary}--\r\n".encode()
    )


def _transcription_app(monkeypatch, worker_gate: asyncio.Event, in_worker: list):
    """The real /v1/audio/transcriptions handler, a real ModelManager, and a
    whisper worker that answers when ``worker_gate`` opens."""
    from ocabra.api._deps_auth import UserContext, get_current_user
    from ocabra.api.openai import audio as audio_mod
    from ocabra.api.openai import router as openai_router
    from ocabra.api.openai._deps import get_openai_user
    from ocabra.backends.base import BackendCapabilities

    profile = _profile("whisper-public", "whisper/large")
    state = SimpleNamespace(
        model_id="whisper/large",
        backend_model_id="large",
        backend_type="faster_whisper_test",
        capabilities=BackendCapabilities(audio_transcription=True),
    )
    monkeypatch.setattr(audio_mod, "resolve_profile", AsyncMock(return_value=(profile, state)))
    mm = _make_mm()

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **k):
            in_worker.append(mm.inflight_count("whisper/large"))
            await worker_gate.wait()
            return httpx.Response(
                200, json={"text": "hola"}, request=httpx.Request("POST", "http://w")
            )

    monkeypatch.setattr(audio_mod.httpx, "AsyncClient", _Client)
    admin = UserContext(
        user_id=None,
        username=None,
        role="system_admin",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )

    async def _user():
        return admin

    app = _app(mm, profile)
    app.include_router(openai_router, prefix="/v1")
    for dep in (get_current_user, get_openai_user):
        app.dependency_overrides[dep] = _user
    backend = SimpleNamespace(health_check=AsyncMock(return_value=True))
    app.state.worker_pool = SimpleNamespace(
        get_worker=MagicMock(return_value=SimpleNamespace(port=18004)),
        get_backend=AsyncMock(return_value=backend),
    )
    _install(app)
    return app, mm


async def _slow_upload(body: bytes, pieces: int = 8):
    """An upload that arrives over several event-loop turns, like a real one."""
    step = max(1, len(body) // pieces)
    for i in range(0, len(body), step):
        yield body[i : i + step]
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_upload_is_counted_once_while_the_worker_transcribes(monkeypatch, record_stat):
    gate = asyncio.Event()
    in_worker: list = []
    app, mm = _transcription_app(monkeypatch, gate, in_worker)
    transport = httpx.ASGITransport(app=app)
    async with _HttpClient(transport=transport, base_url="http://t") as client:
        task = asyncio.create_task(
            client.post(
                "/v1/audio/transcriptions",
                content=_multipart("whisper-public", _WAV * 100),
                headers={"content-type": "multipart/form-data; boundary=XyZ"},
            )
        )
        for _ in range(100):
            await asyncio.sleep(0.01)
            if in_worker:
                break
        assert in_worker == [1]
        (active,) = mm.active_requests_snapshot()
        assert active.model_id == "whisper/large"
        gate.set()
        response = await task
    assert response.status_code == 200
    assert mm.inflight_count("whisper/large") == 0
    assert mm.active_requests_snapshot() == []


@pytest.mark.asyncio
async def test_burst_of_uploads_never_overshoots_the_cap(monkeypatch, record_stat):
    """Eight uploads at once against a cap of 4 (one slot x 4): exactly four
    are admitted. Before, the middleware only *checked* the cap and the
    handler counted the request after reading the file, so every upload that
    arrived while the others were still uploading passed."""
    gate = asyncio.Event()
    in_worker: list = []
    app, mm = _transcription_app(monkeypatch, gate, in_worker)
    peak = 0

    async def watch():
        nonlocal peak
        while not gate.is_set():
            peak = max(peak, mm.inflight_count("whisper/large"))
            await asyncio.sleep(0.002)

    transport = httpx.ASGITransport(app=app)
    async with _HttpClient(transport=transport, base_url="http://t") as client:
        watcher = asyncio.create_task(watch())
        tasks = [
            asyncio.create_task(
                client.post(
                    "/v1/audio/transcriptions",
                    content=_slow_upload(_multipart("whisper-public", _WAV * 200)),
                    headers={"content-type": "multipart/form-data; boundary=XyZ"},
                )
            )
            for _ in range(8)
        ]
        for _ in range(200):
            await asyncio.sleep(0.01)
            if sum(t.done() for t in tasks) + len(in_worker) >= 8:
                break
        gate.set()
        responses = await asyncio.gather(*tasks)
        await watcher
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200] * 4 + [429] * 4, statuses
    assert peak <= 4
    assert mm.inflight_count("whisper/large") == 0


# ── no double counting in endpoints that used to count themselves ─────────


def _image_app(monkeypatch, in_worker: list):
    from ocabra.api._deps_auth import UserContext, get_current_user
    from ocabra.api.openai import images as images_mod
    from ocabra.api.openai import router as openai_router
    from ocabra.api.openai._deps import get_openai_user
    from ocabra.backends.base import BackendCapabilities

    profile = _profile("sdxl", "diffusers/sdxl")
    state = SimpleNamespace(
        model_id="diffusers/sdxl",
        backend_model_id="sdxl",
        backend_type="diffusers",
        status=SimpleNamespace(value="loaded"),
        capabilities=BackendCapabilities(image_generation=True, image_editing=True),
    )
    monkeypatch.setattr(images_mod, "resolve_profile", AsyncMock(return_value=(profile, state)))
    monkeypatch.setattr(images_mod, "build_model_status_headers", AsyncMock(return_value={}))
    mm = _make_mm()

    async def forward_request(worker_key, _path, _body):
        in_worker.append(mm.inflight_snapshot())
        return {"images": [{"b64_json": "aGk="}]}

    admin = UserContext(
        user_id=None,
        username=None,
        role="system_admin",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )

    async def _user():
        return admin

    app = _app(mm, profile)
    app.include_router(openai_router, prefix="/v1")
    for dep in (get_current_user, get_openai_user):
        app.dependency_overrides[dep] = _user
    app.state.worker_pool = SimpleNamespace(forward_request=forward_request)
    _install(app)
    return app, mm


def test_image_generation_is_counted_once(monkeypatch, record_stat):
    in_worker: list = []
    app, mm = _image_app(monkeypatch, in_worker)
    response = TestClient(app).post(
        "/v1/images/generations", json={"model": "sdxl", "prompt": "un gato"}
    )
    assert response.status_code == 200, response.text
    assert in_worker == [{"diffusers/sdxl": 1}]
    assert mm.inflight_snapshot() == {}


def test_image_edit_upload_is_counted_once(monkeypatch, record_stat):
    in_worker: list = []
    app, mm = _image_app(monkeypatch, in_worker)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    response = TestClient(app).post(
        "/v1/images/edits",
        data={"model": "sdxl", "prompt": "con sombrero"},
        files={"image": ("a.png", png, "image/png")},
    )
    assert response.status_code == 200, response.text
    assert in_worker == [{"diffusers/sdxl": 1}]
    assert mm.inflight_snapshot() == {}


def test_main_registers_admission_between_the_federation_hop_and_the_stats():
    """Outermost first. Admission must see ``federation_info`` (set by the
    hop marker) and wrap the stats, so its slot spans the whole response."""
    from ocabra.core.admission import AdmissionMiddleware
    from ocabra.core.federation import FederationHopMiddleware
    from ocabra.main import app
    from ocabra.stats.collector import record_admission_rejection

    order = [m.cls for m in app.user_middleware]
    hop, adm, stats = (
        order.index(FederationHopMiddleware),
        order.index(AdmissionMiddleware),
        order.index(StatsMiddleware),
    )
    assert hop < adm < stats
    assert adm + 1 == stats
    (entry,) = [m for m in app.user_middleware if m.cls is AdmissionMiddleware]
    assert entry.kwargs == {"on_reject": record_admission_rejection}


def test_endpoint_inflight_takes_the_admitted_slot_and_moves_it_to_the_runtime():
    from ocabra.core.admission import EndpointInflight

    mm = _make_mm()
    rid = mm.try_begin_request("gpt-image-1", 0)
    request = SimpleNamespace(
        state=SimpleNamespace(inflight_request_id=rid, inflight_model_id="gpt-image-1")
    )
    inflight = EndpointInflight(request, mm, "diffusers/sdxl")
    assert inflight.acquire() == rid
    assert mm.inflight_snapshot() == {"diffusers/sdxl": 1}
    assert request.state.inflight_model_id == "diffusers/sdxl"
    inflight.release()  # admission's to release, not the endpoint's
    assert mm.inflight_snapshot() == {"diffusers/sdxl": 1}


def test_endpoint_inflight_without_admission_opens_and_closes_its_own():
    from ocabra.core.admission import EndpointInflight

    mm = _make_mm()
    inflight = EndpointInflight(SimpleNamespace(state=SimpleNamespace()), mm, "w/x")
    rid = inflight.acquire()
    assert mm.inflight_snapshot() == {"w/x": 1}
    assert [a.request_id for a in mm.active_requests_snapshot()] == [rid]
    inflight.release()
    inflight.release()  # idempotent: never ends a request twice
    assert mm.inflight_snapshot() == {}
