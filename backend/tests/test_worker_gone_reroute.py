"""Un worker que desaparece mientras la peticion espera su turno (2026-11-13).

El router elige destino al entrar la peticion; si ese worker se descarga
mientras espera en la cola de reparto (descarga manual, caida, un servicio que
reclama la GPU), antes la peticion fallaba con un 500 (sin streaming) o un 502
(con streaming). Ahora una peticion de router vuelve a elegir destino una vez;
cualquier otra recibe un 503 reintentable.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from ocabra.core.worker_pool import WorkerGoneError, WorkerPool


def test_worker_gone_is_still_a_key_error():
    assert issubclass(WorkerGoneError, KeyError)
    assert "No worker found for model 'a/b'" in str(WorkerGoneError("a/b"))


def test_forwarding_to_a_missing_worker_raises_worker_gone():
    pool = WorkerPool()
    with pytest.raises(WorkerGoneError):
        asyncio.run(pool.forward_request("a/b", "/v1/chat/completions", {}))


def test_worker_gone_maps_to_a_retryable_503():
    from ocabra.api.openai._deps import raise_inference_dispatch_error, stream_error_details

    with pytest.raises(HTTPException) as info:
        raise_inference_dispatch_error(WorkerGoneError("a/b"))
    assert info.value.status_code == 503
    assert info.value.detail["error"]["code"] == "worker_unloaded"
    assert info.value.headers["Retry-After"]
    assert stream_error_details(WorkerGoneError("a/b"))[1] == "worker_unloaded"


def _loaded(model_id):
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    return ModelState(
        model_id=model_id,
        backend_model_id=model_id.split("/")[-1],
        display_name=model_id,
        backend_type="llamacpp_mks",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(chat=True, streaming=True),
        extra_config={"_runtime": {"parallelism": 4}},
    )


def _gen_raising(exc):
    async def _g():
        raise exc
        yield b""  # pragma: no cover

    return _g()


def _gen(chunks):
    async def _g():
        for c in chunks:
            yield c

    return _g()


def _drain(gen):
    async def _run():
        return [c async for c in gen]

    return asyncio.run(_run())


def _stream(reroute):
    from ocabra.api.openai.chat import _stream_chat_with_load
    from tests.test_openai_api import _make_test_profile

    mm = MagicMock()
    mm.get_state = AsyncMock(return_value=None)
    mm.get_expected_load_seconds = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.peek_queue_state = MagicMock(return_value=None)
    pool.forward_stream = MagicMock(return_value=_gen_raising(WorkerGoneError("a/gemma")))
    profile = _make_test_profile(profile_id="gemma", base_model_id="a/gemma")
    with patch(
        "ocabra.api.openai.chat.ensure_worker_loaded",
        side_effect=AsyncMock(return_value=_loaded("a/gemma")),
    ):
        return _drain(
            _stream_chat_with_load(
                model_manager=mm,
                worker_pool=pool,
                profile=profile,
                worker_key="a/gemma",
                load_overrides=None,
                preferred_gpu=None,
                load_policy=None,
                required_tokens=None,
                kv_reservation_tokens=None,
                body={"messages": [{"role": "user", "content": "hola"}]},
                content_kinds=set(),
                user_key="u",
                reroute=reroute,
            )
        )


def test_a_router_stream_goes_to_another_target_when_its_worker_is_gone():
    calls = []

    def reroute():
        calls.append(1)
        return _gen([b'data: {"choices":[{"delta":{"content":"hola"}}]}\n\n', b"data: [DONE]\n\n"])

    chunks = b"".join(_stream(reroute))

    assert calls == [1]
    assert b"rerouted" in chunks
    assert b'"content":"hola"' in chunks
    assert b"worker_unloaded" not in chunks


def test_a_plain_stream_reports_worker_unloaded():
    chunks = b"".join(_stream(None))
    assert b"worker_unloaded" in chunks


def _app_with_router(monkeypatch, *, via_router: bool, outcomes):
    from fastapi.testclient import TestClient

    from ocabra.api.openai import chat as chat_mod
    from tests.test_openai_api import _make_app

    app = _make_app(worker_result={"choices": [{"message": {"content": "hola"}}]})
    resolved: list[str] = []
    real_resolve = chat_mod.resolve_profile

    async def fake_resolve(model_id, *a, request_state=None, **k):
        resolved.append(model_id)
        if via_router and request_state is not None:
            request_state.via_router_profile_id = "recomendado-x"
        return await real_resolve(model_id, *a, request_state=request_state, **k)

    monkeypatch.setattr(chat_mod, "resolve_profile", fake_resolve)
    app.state.worker_pool.forward_request = AsyncMock(side_effect=outcomes)
    return TestClient(app), resolved


def test_a_router_request_without_stream_is_resolved_again_once(monkeypatch):
    client, resolved = _app_with_router(
        monkeypatch,
        via_router=True,
        outcomes=[WorkerGoneError("test-model"), {"choices": [{"message": {"content": "ok"}}]}],
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hola"}]},
    )
    assert resp.status_code == 200
    assert len(resolved) == 2


def test_the_reroute_happens_only_once(monkeypatch):
    client, resolved = _app_with_router(
        monkeypatch,
        via_router=True,
        outcomes=[WorkerGoneError("test-model"), WorkerGoneError("test-model")],
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hola"}]},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "worker_unloaded"
    assert len(resolved) == 2


def test_a_plain_request_gets_503_not_500(monkeypatch):
    client, resolved = _app_with_router(
        monkeypatch, via_router=False, outcomes=[WorkerGoneError("test-model")]
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hola"}]},
    )
    assert resp.status_code == 503
    assert resp.headers["Retry-After"]
    assert len(resolved) == 1
