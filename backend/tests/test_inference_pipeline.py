"""El motor de flujo único de petición (REF-01, fase A): ``core/inference/pipeline``.

Comprueba que :func:`run_request` ejecuta los pasos en orden delegando en los
helpers existentes y, sobre todo, que el paso ``admit`` es consciente de la
clase de petición: embeddings/pooling NO se enganchan a la reja de reparto
justo (``forward_request`` sin ``user_key``), de modo que su concurrencia la
sigue gobernando solo ``AdmissionMiddleware`` (ver
``project_embeddings_inflight_cap``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from ocabra.core.inference.body import ClientGoneError
from ocabra.core.inference.pipeline import (
    RequestDescriptor,
    RequestOutcome,
    resolve_admission_kwargs,
    run_request,
)
from ocabra.core.worker_pool import WorkerGoneError

PIPELINE = "ocabra.core.inference.pipeline"


async def _agen(chunks, raise_after=None):
    """Generador asíncrono de ``forward_stream`` falso."""
    for chunk in chunks:
        yield chunk
    if raise_after is not None:
        raise raise_after


async def _reject_gen(exc):
    """Generador que rechaza en el primer ``__anext__`` (nada emitido aún)."""
    raise exc
    yield b""  # pragma: no cover — lo hace generador


CHAT_STREAM_DESCRIPTOR = RequestDescriptor(
    request_kind="chat",
    capability="chat",
    dispatch_path="/v1/chat/completions",
    streaming=True,
    applies_fair_share=True,
)

EMBEDDINGS_DESCRIPTOR = RequestDescriptor(
    request_kind="embeddings",
    capability="embeddings",
    dispatch_path="/v1/embeddings",
    honors_disconnect=True,
)

CHAT_DESCRIPTOR = RequestDescriptor(
    request_kind="chat",
    capability="chat",
    dispatch_path="/v1/chat/completions",
    streaming=False,
    honors_disconnect=True,
    applies_fair_share=True,
)


def _make_request(router_resolver=None):
    app = SimpleNamespace(state=SimpleNamespace(router_resolver=router_resolver))
    return SimpleNamespace(state=SimpleNamespace(), app=app, headers={})


def _make_state(model_id="be::preset", backend_type="llamacpp_mks"):
    return SimpleNamespace(
        model_id=model_id,
        backend_type=backend_type,
        backend_model_id="e",
        base_model_id="be",
        capabilities=SimpleNamespace(embeddings=True),
        extra_config={},
        live_context=None,
    )


def _make_profile():
    return SimpleNamespace(
        profile_id="e",
        base_model_id="be",
        load_overrides=None,
        request_defaults=None,
        assets=None,
    )


# ── admit consciente de la clase de petición ──────────────────────────────


def test_admit_devuelve_vacio_para_embeddings():
    """El hueco ``Fair`` de embeddings se rellena como NO-OP: sin reja."""
    kwargs = resolve_admission_kwargs(
        EMBEDDINGS_DESCRIPTOR,
        _make_request(),
        _make_state(),
        SimpleNamespace(user_id="u1", username="x"),
    )
    assert kwargs == {}


def test_admit_arma_la_reja_para_clases_generativas():
    req = _make_request()
    req.state.runtime_required_tokens = None
    kwargs = resolve_admission_kwargs(
        CHAT_DESCRIPTOR,
        req,
        _make_state(backend_type="ollama"),
        SimpleNamespace(user_id="u1", username="x"),
    )
    assert kwargs["user_key"] == "u1"
    assert "slots" in kwargs and "is_batch" in kwargs
    # Sin KV unificado (backend ollama) no se reserva presupuesto de KV.
    assert "kv_capacity_tokens" not in kwargs


def test_admit_sin_usuario_no_pone_reja():
    kwargs = resolve_admission_kwargs(CHAT_DESCRIPTOR, _make_request(), _make_state(), None)
    assert kwargs == {}


# ── run_request: camino feliz y pasos en orden ────────────────────────────


@pytest.mark.asyncio
async def test_run_request_despacha_sin_reja_para_embeddings():
    profile, state = _make_profile(), _make_state()
    pool = MagicMock()
    pool.forward_request = AsyncMock(return_value={"data": [], "model": "e"})
    request = _make_request()

    with (
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{PIPELINE}.check_capability") as cap,
        patch(f"{PIPELINE}.merge_profile_defaults", lambda _p, body: body),
        patch(f"{PIPELINE}.to_backend_body", lambda _s, body: body),
        patch(f"{PIPELINE}.ocabra_response_headers", AsyncMock(return_value={"X": "1"})),
    ):
        outcome = await run_request(
            request,
            descriptor=EMBEDDINGS_DESCRIPTOR,
            body={"model": "e", "input": ["hola"]},
            user=SimpleNamespace(user_id="u1", username="x"),
            model_manager=MagicMock(),
            profile_registry=MagicMock(),
            worker_pool=pool,
            federate=AsyncMock(return_value=None),
        )

    assert isinstance(outcome, RequestOutcome)
    assert outcome.result == {"data": [], "model": "e"}
    assert outcome.worker_key == "be::preset"
    assert outcome.response_headers == {"X": "1"}
    cap.assert_called_once_with(state, "embeddings", "embeddings")
    # Clave del worker (con preset) y SIN kwargs de reja de reparto justo.
    call = pool.forward_request.await_args
    assert call.args[0] == "be::preset"
    assert call.args[1] == "/v1/embeddings"
    assert "user_key" not in call.kwargs


@pytest.mark.asyncio
async def test_run_request_corto_circuita_federacion():
    peer_response = Response(content=b"{}", status_code=200, media_type="application/json")
    pool = MagicMock()
    pool.forward_request = AsyncMock()

    outcome = await run_request(
        _make_request(),
        descriptor=EMBEDDINGS_DESCRIPTOR,
        body={"model": "e", "input": ["x"]},
        user=None,
        model_manager=MagicMock(),
        profile_registry=MagicMock(),
        worker_pool=pool,
        federate=AsyncMock(return_value=peer_response),
    )
    assert outcome is peer_response
    pool.forward_request.assert_not_called()


@pytest.mark.asyncio
async def test_run_request_desconexion_da_499():
    profile, state = _make_profile(), _make_state()
    pool = MagicMock()
    pool.forward_request = AsyncMock(return_value={"data": []})
    with (
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{PIPELINE}.check_capability"),
        patch(f"{PIPELINE}.merge_profile_defaults", lambda _p, body: body),
        patch(f"{PIPELINE}.to_backend_body", lambda _s, body: body),
        patch(
            f"{PIPELINE}.await_unless_client_gone",
            AsyncMock(side_effect=ClientGoneError("gone")),
        ),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await run_request(
                _make_request(),
                descriptor=EMBEDDINGS_DESCRIPTOR,
                body={"model": "e", "input": ["x"]},
                user=None,
                model_manager=MagicMock(),
                profile_registry=MagicMock(),
                worker_pool=pool,
                federate=AsyncMock(return_value=None),
            )
    assert excinfo.value.status_code == 499
    assert excinfo.value.detail["error"]["code"] == "client_disconnected"


@pytest.mark.asyncio
async def test_run_request_mapea_error_upstream():
    profile, state = _make_profile(), _make_state()
    http_exc = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("POST", "http://w/v1/embeddings"),
        response=httpx.Response(400, json={"error": {"message": "bad"}}),
    )
    pool = MagicMock()
    pool.forward_request = AsyncMock(side_effect=http_exc)
    with (
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{PIPELINE}.check_capability"),
        patch(f"{PIPELINE}.merge_profile_defaults", lambda _p, body: body),
        patch(f"{PIPELINE}.to_backend_body", lambda _s, body: body),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await run_request(
                _make_request(),
                descriptor=RequestDescriptor(
                    request_kind="embeddings",
                    capability="embeddings",
                    dispatch_path="/v1/embeddings",
                    honors_disconnect=False,
                ),
                body={"model": "e", "input": ["x"]},
                user=None,
                model_manager=MagicMock(),
                profile_registry=MagicMock(),
                worker_pool=pool,
                federate=AsyncMock(return_value=None),
            )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_run_request_desconexion_cancela_el_despacho_en_curso():
    """La desconexión real cancela la corrutina de despacho (no solo mapea 499)."""
    import asyncio

    profile, state = _make_profile(), _make_state()
    cancelled = {"value": False}

    async def _slow(*_a, **_k):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled["value"] = True
            raise

    pool = MagicMock()
    pool.forward_request = _slow  # callable → corrutina al invocarse

    request = _make_request()
    request.is_disconnected = AsyncMock(return_value=True)

    with (
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{PIPELINE}.check_capability"),
        patch(f"{PIPELINE}.merge_profile_defaults", lambda _p, body: body),
        patch(f"{PIPELINE}.to_backend_body", lambda _s, body: body),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await run_request(
                request,
                descriptor=EMBEDDINGS_DESCRIPTOR,
                body={"model": "e", "input": ["x"]},
                user=None,
                model_manager=MagicMock(),
                profile_registry=MagicMock(),
                worker_pool=pool,
                federate=AsyncMock(return_value=None),
            )
    assert excinfo.value.status_code == 499
    assert cancelled["value"] is True


# ── run_request: camino de streaming (fase B) ─────────────────────────────


def _streaming_patches(profile, state):
    return (
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{PIPELINE}.check_capability"),
        patch(f"{PIPELINE}.merge_profile_defaults", lambda _p, body: body),
        patch(f"{PIPELINE}.to_backend_body", lambda _s, body: body),
    )


@pytest.mark.asyncio
async def test_run_request_streaming_relay_byte_identico():
    """El relay devuelve los chunks del worker byte a byte (peek + continuación)."""
    profile, state = _make_profile(), _make_state()
    chunks = [b'data: {"id":1}\n\n', b'data: {"id":2}\n\n', b"data: [DONE]\n\n"]
    pool = MagicMock()
    pool.forward_stream = MagicMock(return_value=_agen(chunks))
    p1, p2, p3, p4 = _streaming_patches(profile, state)
    with p1, p2, p3, p4:
        resp = await run_request(
            _make_request(),
            descriptor=CHAT_STREAM_DESCRIPTOR,
            body={"model": "e", "stream": True},
            user=None,
            model_manager=MagicMock(),
            profile_registry=MagicMock(),
            worker_pool=pool,
            federate=AsyncMock(return_value=None),
        )
    assert isinstance(resp, StreamingResponse)
    out = [chunk async for chunk in resp.body_iterator]
    assert out == chunks
    assert resp.media_type == "text/event-stream"
    assert resp.headers["X-Accel-Buffering"] == "no"


@pytest.mark.asyncio
async def test_run_request_streaming_rechazo_primer_chunk_da_estado_real():
    """Un rechazo ANTES del primer byte sale como estado HTTP real, no 200 (BUG-17)."""
    profile, state = _make_profile(), _make_state()
    http_exc = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("POST", "http://w/v1/chat/completions"),
        response=httpx.Response(400, json={"error": {"message": "bad tool call"}}),
    )
    pool = MagicMock()
    pool.forward_stream = MagicMock(return_value=_reject_gen(http_exc))
    p1, p2, p3, p4 = _streaming_patches(profile, state)
    with p1, p2, p3, p4, pytest.raises(HTTPException) as excinfo:
        await run_request(
            _make_request(),
            descriptor=CHAT_STREAM_DESCRIPTOR,
            body={"model": "e", "stream": True},
            user=None,
            model_manager=MagicMock(),
            profile_registry=MagicMock(),
            worker_pool=pool,
            federate=AsyncMock(return_value=None),
        )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_run_request_streaming_error_a_mitad_va_en_banda():
    """Un fallo tras el primer chunk se serializa como error SSE + [DONE]."""
    profile, state = _make_profile(), _make_state()
    mid_exc = httpx.HTTPStatusError(
        "late",
        request=httpx.Request("POST", "http://w/v1/chat/completions"),
        response=httpx.Response(500, text="kaboom"),
    )
    pool = MagicMock()
    pool.forward_stream = MagicMock(
        return_value=_agen([b'data: {"id":1}\n\n'], raise_after=mid_exc)
    )
    p1, p2, p3, p4 = _streaming_patches(profile, state)
    with p1, p2, p3, p4:
        resp = await run_request(
            _make_request(),
            descriptor=CHAT_STREAM_DESCRIPTOR,
            body={"model": "e", "stream": True},
            user=None,
            model_manager=MagicMock(),
            profile_registry=MagicMock(),
            worker_pool=pool,
            federate=AsyncMock(return_value=None),
        )
    out = [chunk async for chunk in resp.body_iterator]
    assert out[0] == b'data: {"id":1}\n\n'
    assert b'"error"' in out[1]
    assert out[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_run_request_streaming_engancha_reja_para_chat():
    """Streaming + usuario: ``forward_stream`` recibe la reja de reparto justo."""
    profile, state = _make_profile(), _make_state(backend_type="ollama")
    pool = MagicMock()
    pool.forward_stream = MagicMock(return_value=_agen([b"data: [DONE]\n\n"]))
    request = _make_request()
    request.state.runtime_required_tokens = None
    p1, p2, p3, p4 = _streaming_patches(profile, state)
    with p1, p2, p3, p4:
        resp = await run_request(
            request,
            descriptor=CHAT_STREAM_DESCRIPTOR,
            body={"model": "e", "stream": True},
            user=SimpleNamespace(user_id="u1", username="x"),
            model_manager=MagicMock(),
            profile_registry=MagicMock(),
            worker_pool=pool,
            federate=AsyncMock(return_value=None),
        )
    # Consumir para que se ejecute el peek y forward_stream reciba su llamada.
    _ = [chunk async for chunk in resp.body_iterator]
    call = pool.forward_stream.call_args
    assert call.args[1] == "/v1/chat/completions"
    assert call.kwargs.get("user_key") == "u1"
    assert "slots" in call.kwargs and "is_batch" in call.kwargs


# ── run_request: extras de la fase B (chat) ───────────────────────────────


@pytest.mark.asyncio
async def test_run_request_cuela_queue_state_de_reparto_justo():
    """La respuesta no-streaming generativa lleva el peek de cola (heads-up)."""
    profile, state = _make_profile(), _make_state()
    queue = {"saturated": True, "slot_saturated": True, "in_flight": 3, "slots": 1}
    pool = MagicMock()
    pool.forward_request = AsyncMock(return_value={"ok": 1})
    pool.peek_queue_state = MagicMock(return_value=queue)
    request = _make_request()
    request.state.runtime_required_tokens = None
    with (
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{PIPELINE}.check_capability"),
        patch(f"{PIPELINE}.merge_profile_defaults", lambda _p, body: body),
        patch(f"{PIPELINE}.to_backend_body", lambda _s, body: body),
        patch(f"{PIPELINE}.ocabra_response_headers", AsyncMock(return_value={})),
    ):
        outcome = await run_request(
            request,
            descriptor=CHAT_DESCRIPTOR,
            body={"model": "e"},
            user=SimpleNamespace(user_id="u1", username="x"),
            model_manager=MagicMock(),
            profile_registry=MagicMock(),
            worker_pool=pool,
            federate=AsyncMock(return_value=None),
        )
    assert isinstance(outcome, RequestOutcome)
    assert outcome.queue_state == queue
    pool.peek_queue_state.assert_called_once()


@pytest.mark.asyncio
async def test_run_request_on_dispatch_error_reenruta():
    """El hook de error de despacho puede devolver una Response (reenrutado)."""
    profile, state = _make_profile(), _make_state()
    reroute = Response(content=b"{}", status_code=200, media_type="application/json")
    pool = MagicMock()
    pool.forward_request = AsyncMock(side_effect=WorkerGoneError("unloaded"))
    pool.peek_queue_state = MagicMock(return_value=None)
    request = _make_request()
    request.state.runtime_required_tokens = None

    async def _hook(exc, ctx):
        assert isinstance(exc, WorkerGoneError)
        assert ctx.worker_key == "be::preset"
        return reroute

    with (
        patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{PIPELINE}.check_capability"),
        patch(f"{PIPELINE}.merge_profile_defaults", lambda _p, body: body),
        patch(f"{PIPELINE}.to_backend_body", lambda _s, body: body),
    ):
        result = await run_request(
            request,
            descriptor=CHAT_DESCRIPTOR,
            body={"model": "e"},
            user=SimpleNamespace(user_id="u1", username="x"),
            model_manager=MagicMock(),
            profile_registry=MagicMock(),
            worker_pool=pool,
            federate=AsyncMock(return_value=None),
            on_dispatch_error=_hook,
        )
    assert result is reroute


@pytest.mark.asyncio
async def test_run_request_capability_label_en_el_mensaje():
    """La etiqueta humana del check de capacidad se conserva (byte a byte)."""
    profile = _make_profile()
    state = _make_state()
    state.capabilities = SimpleNamespace(chat=False)
    pool = MagicMock()
    pool.forward_request = AsyncMock()
    descriptor = RequestDescriptor(
        request_kind="chat",
        capability="chat",
        dispatch_path="/v1/chat/completions",
        capability_label="chat completions",
    )
    with patch(f"{PIPELINE}.resolve_profile", AsyncMock(return_value=(profile, state))):
        with pytest.raises(HTTPException) as excinfo:
            await run_request(
                _make_request(),
                descriptor=descriptor,
                body={"model": "e"},
                user=None,
                model_manager=MagicMock(),
                profile_registry=MagicMock(),
                worker_pool=pool,
                federate=AsyncMock(return_value=None),
            )
    assert "does not support chat completions" in excinfo.value.detail["error"]["message"]
