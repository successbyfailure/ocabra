"""Progreso en vivo de peticiones para el dashboard (2026-11-09).

Cubre las tres piezas que atribuyen a una peticion concreta lo que solo ve la
capa de transporte:

* ``return_progress`` de llama.cpp: se pide, se lee y se quita del stream sin
  que el cliente ni el gate noten diferencia;
* ``FairSlotGate.requests_snapshot``: leases y cola por peticion, en el orden
  real de turno (menos servido primero, no por llegada);
* ``ModelManager``: primer token y progreso de prefill en ``ActiveRequest``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.backends.base import WorkerInfo
from ocabra.core.fair_slot_gate import FairSlotGate
from ocabra.core.request_telemetry import current_inflight_request_id
from ocabra.core.worker_pool import WorkerPool


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _progress_chunk(processed: int, total: int = 1000, cache: int = 100) -> dict:
    return {
        "choices": [
            {"finish_reason": None, "index": 0, "delta": {"role": "assistant", "content": None}}
        ],
        "object": "chat.completion.chunk",
        "prompt_progress": {
            "total": total,
            "cache": cache,
            "processed": processed,
            "time_ms": processed,
        },
    }


def _content_chunk(text: str) -> dict:
    return {
        "choices": [{"finish_reason": None, "index": 0, "delta": {"content": text}}],
        "object": "chat.completion.chunk",
    }


UPSTREAM = (
    _sse(_progress_chunk(100))
    + _sse(_progress_chunk(600))
    + _sse(_progress_chunk(1000))
    + _sse(_content_chunk("Hola"))
    + _sse(_content_chunk(" mundo"))
    + b"data: [DONE]\n\n"
)


def _fake_stream_client(upstream: bytes, seen_bodies: list[dict], *, piece: int = 37):
    """Cliente httpx falso que devuelve ``upstream`` troceado a ``piece`` bytes,
    para que un evento SSE quede partido entre chunks."""

    class _Resp:
        status_code = 200
        is_error = False

        def raise_for_status(self) -> None:
            return None

        async def aread(self) -> bytes:
            return b""

        async def aiter_bytes(self):
            for i in range(0, len(upstream), piece):
                yield upstream[i : i + piece]

    class _StreamCtx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *exc):
            return None

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        def stream(self, method: str, url: str, json: dict):
            seen_bodies.append(json)
            return _StreamCtx()

    return _Client


def _pool_with_worker(backend_type: str) -> WorkerPool:
    pool = WorkerPool()
    pool.register_backend(backend_type, MockBackend())
    pool.set_worker(
        "m",
        WorkerInfo(
            backend_type=backend_type,
            model_id="m",
            gpu_indices=[1],
            port=18999,
            pid=1,
            vram_used_mb=0,
        ),
    )
    return pool


async def _collect(pool: WorkerPool, body: dict, **kwargs) -> bytes:
    out = b""
    async for chunk in pool.forward_stream("m", "/v1/chat/completions", body, **kwargs):
        out += chunk
    return out


@pytest.mark.asyncio
async def test_llama_cpp_progress_is_requested_consumed_and_stripped(monkeypatch):
    bodies: list[dict] = []
    monkeypatch.setattr(
        "ocabra.core.worker_pool.httpx.AsyncClient", _fake_stream_client(UPSTREAM, bodies)
    )
    pool = _pool_with_worker("llamacpp_mks")
    calls: list[tuple] = []
    pool.set_prompt_progress_sink(lambda rid, **kw: calls.append((rid, kw)))

    token = current_inflight_request_id.set("req-1")
    try:
        out = await _collect(pool, {"stream": True, "messages": []})
    finally:
        current_inflight_request_id.reset(token)

    assert bodies[0]["return_progress"] is True
    assert b"prompt_progress" not in out
    assert b"Hola" in out and b" mundo" in out and out.endswith(b"data: [DONE]\n\n")
    # La primera llamada solo avisa de que se ha pedido progreso.
    assert calls[0][1]["total"] is None
    calls = calls[1:]
    assert [kw["processed"] for _rid, kw in calls] == [100, 600, 1000]
    assert all(rid == "req-1" for rid, _kw in calls)
    assert calls[-1][1] == {"total": 1000, "cached": 100, "processed": 1000, "elapsed_ms": 1000}


@pytest.mark.asyncio
async def test_client_that_asked_for_progress_still_gets_it(monkeypatch):
    bodies: list[dict] = []
    monkeypatch.setattr(
        "ocabra.core.worker_pool.httpx.AsyncClient", _fake_stream_client(UPSTREAM, bodies)
    )
    pool = _pool_with_worker("llama_cpp")
    calls: list = []
    pool.set_prompt_progress_sink(lambda rid, **kw: calls.append(kw))

    token = current_inflight_request_id.set("req-2")
    try:
        out = await _collect(pool, {"stream": True, "return_progress": True})
    finally:
        current_inflight_request_id.reset(token)

    assert out == UPSTREAM
    assert len(calls) == 4  # aviso inicial + 3 eventos


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_type", "body", "request_id"),
    [
        ("vllm", {"stream": True}, "req-3"),  # otro backend: no sabe de return_progress
        ("llamacpp_mks", {"stream": True}, None),  # sin peticion en vuelo a la que atribuirlo
        ("llamacpp_mks", {"stream": False}, "req-4"),
    ],
)
async def test_progress_not_requested_outside_llama_cpp_streams(
    monkeypatch, backend_type, body, request_id
):
    plain = _sse(_content_chunk("x")) + b"data: [DONE]\n\n"
    bodies: list[dict] = []
    monkeypatch.setattr(
        "ocabra.core.worker_pool.httpx.AsyncClient", _fake_stream_client(plain, bodies)
    )
    pool = _pool_with_worker(backend_type)
    pool.set_prompt_progress_sink(lambda rid, **kw: None)

    token = current_inflight_request_id.set(request_id)
    try:
        await _collect(pool, dict(body))
    finally:
        current_inflight_request_id.reset(token)

    assert "return_progress" not in bodies[0]


@pytest.mark.asyncio
async def test_gate_first_chunk_is_still_the_first_token(monkeypatch):
    """El gate marca fin de prefill con el primer chunk que ve. Los chunks de
    progreso llegan DURANTE el prefill: si se colaran, liberarian la
    admision de prefills grandes antes de tiempo."""
    bodies: list[dict] = []
    monkeypatch.setattr(
        "ocabra.core.worker_pool.httpx.AsyncClient", _fake_stream_client(UPSTREAM, bodies)
    )
    pool = _pool_with_worker("llamacpp_mks")
    pool.set_prompt_progress_sink(lambda rid, **kw: None)

    token = current_inflight_request_id.set("req-5")
    try:
        gen = pool.forward_stream(
            "m", "/v1/chat/completions", {"stream": True}, user_key="u", slots=2
        )
        first = await gen.__anext__()
        await gen.aclose()
    finally:
        current_inflight_request_id.reset(token)

    assert b"prompt_progress" not in first
    assert b"Hola" in first


def test_gate_snapshot_orders_queue_by_least_served_not_arrival():
    """alice ya fue servida y tiene dos en cola; bob llega despues sin haber
    sido servido nunca: bob va primero, luego alterna (alice, alice)."""

    async def scenario():
        gate = FairSlotGate(slots=1, kv_capacity_tokens=1000)
        holders = []

        async def hold(user: str, rid: str, kv: int):
            token = current_inflight_request_id.set(rid)
            try:
                async with gate.slot(user, kv_tokens=kv):
                    await asyncio.Event().wait()
            finally:
                current_inflight_request_id.reset(token)

        holders.append(asyncio.create_task(hold("alice", "a1", 300)))
        await asyncio.sleep(0)
        holders.append(asyncio.create_task(hold("alice", "a2", 100)))
        holders.append(asyncio.create_task(hold("alice", "a3", 100)))
        await asyncio.sleep(0)
        holders.append(asyncio.create_task(hold("bob", "b1", 200)))
        await asyncio.sleep(0.01)

        snap = gate.requests_snapshot()
        for task in holders:
            task.cancel()
        await asyncio.gather(*holders, return_exceptions=True)
        return snap

    snap = asyncio.run(scenario())
    assert [lease["requestId"] for lease in snap["leases"]] == ["a1"]
    assert snap["leases"][0]["kvTokens"] == 300
    assert [q["requestId"] for q in snap["queued"]] == ["b1", "a2", "a3"]
    assert [q["position"] for q in snap["queued"]] == [0, 1, 2]


def test_gate_snapshot_does_not_mutate_dispatch_state():
    async def scenario():
        gate = FairSlotGate(slots=1)
        tasks = []

        async def hold(user: str):
            async with gate.slot(user):
                await asyncio.Event().wait()

        for user in ("a", "a", "b"):
            tasks.append(asyncio.create_task(hold(user)))
            await asyncio.sleep(0)
        before = (dict(gate._last_served), gate._seq)  # noqa: SLF001
        gate.requests_snapshot()
        gate.requests_snapshot()
        after = (dict(gate._last_served), gate._seq)  # noqa: SLF001
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return before, after

    before, after = asyncio.run(scenario())
    assert before == after


def test_model_manager_tracks_first_token_and_prefill_progress():
    from ocabra.core.model_manager import ModelManager

    mm = ModelManager(worker_pool=WorkerPool())
    rid = mm.try_begin_request("m", 0, request_kind="chat", input_tokens_estimate=900)
    assert rid is not None
    req = mm.active_requests_snapshot()[0]
    assert req.request_kind == "chat"
    assert req.first_token_at is None

    mm.update_prompt_progress(rid, total=1000, cached=100, processed=500, elapsed_ms=250)
    assert (req.prompt_tokens_total, req.prompt_tokens_cached, req.prompt_tokens_processed) == (
        1000,
        100,
        500,
    )
    assert req.first_token_at is None

    mm.mark_request_first_token(rid)
    first = req.first_token_at
    assert first is not None
    mm.mark_request_first_token(rid)
    assert req.first_token_at == first

    mm.end_request("m", rid)
    # Peticion ya terminada: no-op, no excepcion.
    mm.update_prompt_progress(rid, total=1, cached=0, processed=1, elapsed_ms=1)
    mm.mark_request_first_token(rid)


@pytest.mark.asyncio
async def test_contextvar_set_in_middleware_reaches_streaming_body():
    """La suposicion de la que depende todo lo anterior: el id que fija el
    middleware de stats antes de ``call_next`` se ve dentro del generador de
    un ``StreamingResponse`` (donde corre ``forward_stream``)."""
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    app = FastAPI()

    class _SetId(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            current_inflight_request_id.set("from-middleware")
            return await call_next(request)

    app.add_middleware(_SetId)

    @app.get("/s")
    async def stream():
        async def body():
            await asyncio.sleep(0)
            yield (current_inflight_request_id.get() or "none").encode()

        return StreamingResponse(body())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.get("/s")
    assert resp.text == "from-middleware"
