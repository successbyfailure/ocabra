"""BUG-17: streaming requests sent to a federation peer.

The three streaming paths (``/v1/chat/completions``, ``/v1/completions``,
``/api/chat``) returned the peer's stream straight away: a peer that refused
gave the client a 200 whose stream broke, and nothing fell back to local.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import structlog

from ocabra.api.openai._federation import try_proxy_stream


def _request():
    return SimpleNamespace(
        url=SimpleNamespace(path="/v1/chat/completions"), headers={}, state=SimpleNamespace()
    )


def _peer():
    return SimpleNamespace(peer_id="p1", name="Mapache", mark_fallback=lambda *a, **k: None)


class _FM:
    def __init__(self, gen_factory):
        self._gen_factory = gen_factory
        self.bodies = []

    def proxy_stream(self, *, peer, path, body, headers):
        self.bodies.append(body)
        return self._gen_factory()


def _refusing(status: int, body: bytes = b'{"error": "nope"}'):
    async def gen():
        req = httpx.Request("POST", "http://peer/v1/chat/completions")
        resp = httpx.Response(
            status, content=body, request=req, headers={"content-type": "application/json"}
        )
        raise httpx.HTTPStatusError("peer", request=req, response=resp)
        yield b""  # pragma: no cover

    return gen


async def _body(response) -> bytes:
    out = b""
    async for chunk in response.body_iterator:
        out += chunk if isinstance(chunk, bytes) else chunk.encode()
    return out


@pytest.mark.asyncio
async def test_a_streaming_peer_is_relayed_whole():
    async def gen():
        yield b"data: 1\n\n"
        yield b"data: [DONE]\n\n"

    fm = _FM(gen)
    resp = await try_proxy_stream(
        _request(), peer=_peer(), model_id="m", body={"model": "target"},
        federation_manager=fm, logger=structlog.get_logger(),
    )  # fmt: skip
    assert resp.status_code == 200
    assert await _body(resp) == b"data: 1\n\ndata: [DONE]\n\n"
    assert fm.bodies == [{"model": "target"}]


@pytest.mark.asyncio
async def test_a_peer_that_refuses_with_a_fallback_status_is_served_locally():
    request = _request()
    resp = await try_proxy_stream(
        request, peer=_peer(), model_id="m", body={},
        federation_manager=_FM(_refusing(404)), logger=structlog.get_logger(),
    )  # fmt: skip
    assert resp is None
    assert request.state.federation_remote_node_id is None


@pytest.mark.asyncio
async def test_a_peer_refusal_that_is_the_answer_keeps_its_status_not_200():
    from ocabra.core.federation import PEER_FALLBACK_STATUSES

    status = next(s for s in (400, 413, 422) if s not in PEER_FALLBACK_STATUSES)
    resp = await try_proxy_stream(
        _request(), peer=_peer(), model_id="m", body={},
        federation_manager=_FM(_refusing(status, b'{"error": "too long"}')),
        logger=structlog.get_logger(),
    )  # fmt: skip
    assert resp.status_code == status
    assert b"too long" in resp.body


@pytest.mark.asyncio
async def test_an_unreachable_peer_is_served_locally():
    def gen():
        async def _g():
            raise httpx.ConnectError("down")
            yield b""  # pragma: no cover

        return _g()

    resp = await try_proxy_stream(
        _request(), peer=_peer(), model_id="m", body={},
        federation_manager=_FM(gen), logger=structlog.get_logger(),
    )  # fmt: skip
    assert resp is None


def test_the_three_streaming_paths_use_it():
    import inspect

    from ocabra.api.ollama import chat as ollama_chat
    from ocabra.api.openai import audio, chat, completions

    for module in (chat, completions, ollama_chat, audio):
        source = inspect.getsource(module)
        assert "try_proxy_stream(" in source, module.__name__
        assert "federation_manager.proxy_stream(" not in source, module.__name__
