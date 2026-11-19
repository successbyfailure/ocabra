"""REF-10: one error format per API family, keeping ``detail`` for compatibility.

``/v1/*`` answers OpenAI's top-level ``error`` object, ``/api/*`` Ollama's
``error`` string, both next to the ``detail`` FastAPI always sent; every other
path is left as it was.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from ocabra.api._errors import error_content, error_family, ollama_error, openai_error

OPENAI_DETAIL = {
    "error": {
        "message": "The model 'x' does not exist.",
        "type": "invalid_request_error",
        "param": "model",
        "code": "model_not_found",
    }
}
PREFERRED_DETAIL = {
    "error": {
        "message": "None of the nodes preferred for 'm' (pod-1) can serve it right now",
        "type": "service_unavailable",
        "code": "preferred_nodes_unavailable",
    }
}


class _Body(BaseModel):
    model: str
    n: int


def _app() -> FastAPI:
    import ocabra.main as main_mod
    from ocabra.core.worker_pool import VisionNoRoomError

    app = FastAPI()
    app.add_exception_handler(StarletteHTTPException, main_mod._peer_fallback_not_found_handler)
    app.add_exception_handler(RequestValidationError, main_mod._request_validation_handler)
    app.add_exception_handler(VisionNoRoomError, main_mod._vision_no_room_handler)
    app.add_exception_handler(Exception, main_mod._log_unhandled_exception)

    for prefix in ("/v1", "/api", "/ocabra"):

        @app.get(f"{prefix}/string")
        async def _string():
            raise HTTPException(status_code=404, detail="Image expired or not found")

        @app.get(f"{prefix}/openai")
        async def _openai():
            raise HTTPException(status_code=404, detail=OPENAI_DETAIL)

        @app.get(f"{prefix}/ollama")
        async def _ollama():
            raise HTTPException(status_code=404, detail={"error": "model 'x' not found"})

        @app.get(f"{prefix}/retry")
        async def _retry():
            raise HTTPException(
                status_code=503, detail=PREFERRED_DETAIL, headers={"Retry-After": "30"}
            )

        @app.post(f"{prefix}/validate")
        async def _validate(body: _Body):
            return {"ok": True}

        @app.get(f"{prefix}/vision")
        async def _vision():
            raise VisionNoRoomError("no room for the projector")

        @app.get(f"{prefix}/boom")
        async def _boom():
            raise RuntimeError("boom")

    return app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(_app(), raise_server_exceptions=False)


# ── /v1: OpenAI's top-level error, detail kept ──────────────────────────────


def test_v1_string_detail_gets_an_openai_error_object(client):
    resp = client.get("/v1/string")
    assert resp.status_code == 404
    body = resp.json()
    assert body["detail"] == "Image expired or not found"
    assert body["error"] == {
        "message": "Image expired or not found",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }


def test_v1_nested_detail_is_lifted_without_double_wrapping(client):
    body = client.get("/v1/openai").json()
    assert body["detail"] == OPENAI_DETAIL
    assert body["error"] == OPENAI_DETAIL["error"]
    assert "error" not in body["error"]


def test_v1_federation_refusal_keeps_status_code_and_retry_after(client):
    resp = client.get("/v1/retry")
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"
    body = resp.json()
    assert body["detail"] == PREFERRED_DETAIL
    assert body["error"]["code"] == "preferred_nodes_unavailable"
    assert body["error"]["type"] == "service_unavailable"
    assert body["error"]["param"] is None


def test_v1_validation_error_keeps_422_and_names_the_field(client):
    resp = client.post("/v1/validate", json={"model": "m"})
    assert resp.status_code == 422
    body = resp.json()
    assert isinstance(body["detail"], list)
    assert body["detail"][0]["loc"] == ["body", "n"]
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "n"
    assert "n:" in body["error"]["message"]


def test_v1_vision_no_room_and_unhandled_get_the_error_object(client):
    vision = client.get("/v1/vision")
    assert vision.status_code == 503
    assert vision.json()["detail"]["error"]["code"] == "vision_no_room"
    assert vision.json()["error"]["code"] == "vision_no_room"

    boom = client.get("/v1/boom")
    assert boom.status_code == 500
    assert boom.json()["detail"]["error"]["code"] == "RuntimeError"
    assert boom.json()["error"]["type"] == "server_error"


def test_v1_ollama_style_detail_becomes_an_object(client):
    body = client.get("/v1/ollama").json()
    assert body["error"]["message"] == "model 'x' not found"
    assert body["error"]["type"] == "invalid_request_error"


# ── /api: Ollama's error string, detail kept ────────────────────────────────


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/api/string", "Image expired or not found"),
        ("/api/openai", "The model 'x' does not exist."),
        ("/api/ollama", "model 'x' not found"),
        ("/api/vision", "no room for the projector"),
        ("/api/boom", "Internal Server Error"),
    ],
)
def test_api_gets_an_ollama_error_string(client, path, message):
    body = client.get(path).json()
    assert body["error"] == message
    assert "detail" in body


def test_api_validation_error_is_a_string_and_keeps_422(client):
    resp = client.post("/api/validate", json={})
    assert resp.status_code == 422
    assert isinstance(resp.json()["error"], str)
    assert "model" in resp.json()["error"]


def test_api_keeps_retry_after(client):
    resp = client.get("/api/retry")
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"
    assert resp.json()["error"].startswith("None of the nodes preferred")


# ── Everything else: unchanged ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "detail"),
    [
        ("/ocabra/string", "Image expired or not found"),
        ("/ocabra/openai", OPENAI_DETAIL),
        ("/ocabra/ollama", {"error": "model 'x' not found"}),
    ],
)
def test_ocabra_paths_are_untouched(client, path, detail):
    assert client.get(path).json() == {"detail": detail}


def test_ocabra_validation_error_is_fastapis_default(client):
    resp = client.post("/ocabra/validate", json={})
    assert resp.status_code == 422
    assert set(resp.json()) == {"detail"}


def test_ocabra_unhandled_and_vision_keep_the_detail_envelope(client):
    assert set(client.get("/ocabra/boom").json()) == {"detail"}
    assert set(client.get("/ocabra/vision").json()) == {"detail"}


# ── Federation ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_peer_unavailable_on_v1_carries_both_shapes():
    from ocabra.main import _peer_fallback_not_found_handler

    request = SimpleNamespace(
        url=SimpleNamespace(path="/v1/chat/completions"),
        state=SimpleNamespace(federation_info={"fallback": True, "fallback_peer": "pod-1"}),
        headers={},
    )
    resp = await _peer_fallback_not_found_handler(
        request, StarletteHTTPException(status_code=404, detail="nope")
    )
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "15"
    assert body["detail"]["error"]["code"] == "peer_unavailable"
    assert body["error"]["code"] == "peer_unavailable"
    assert "pod-1" in body["error"]["message"]


@pytest.mark.parametrize(
    "code", ["outside_hours", "read_only", "model_not_offered", "model_not_warm"]
)
def test_federation_rule_refusals_still_trigger_the_local_fallback(code):
    """The rule refusals go over the tunnel as they are (no handler), and the
    receiving side decides on the status code alone; a rule body put through
    the /v1 formatter is lifted as is, not wrapped again."""
    from ocabra.core.federation import should_fallback_to_local
    from ocabra.core.federation_rules import _refuse

    for status in (503, 404):
        status_code, body = _refuse(status, code, "refused")
        assert should_fallback_to_local(status_code)
        content = error_content("/v1/chat/completions", status_code, body)
        assert content["error"]["code"] == code
        assert content["error"]["type"] == "federation_rules"
        assert content["detail"] == body


def test_preferred_nodes_unavailable_status_still_falls_back():
    from ocabra.core.federation import should_fallback_to_local

    assert should_fallback_to_local(503)


# ── Unit: normalisation ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "family"),
    [
        ("/v1/chat/completions", "openai"),
        ("/v1", "openai"),
        ("/v1beta/x", None),
        ("/api/chat", "ollama"),
        ("/apix", None),
        ("/ocabra/models", None),
        ("/health", None),
        ("", None),
    ],
)
def test_error_family(path, family):
    assert error_family(path) == family


def test_openai_error_defaults_type_by_status():
    assert openai_error("x", 401)["type"] == "authentication_error"
    assert openai_error("x", 403)["type"] == "permission_error"
    assert openai_error("x", 429)["type"] == "rate_limit_error"
    assert openai_error("x", 502)["type"] == "server_error"
    assert openai_error(None, 404)["message"] == "Not Found"


def test_openai_error_keeps_extra_keys_and_flat_message_dicts():
    nested = openai_error({"error": {"message": "m", "code": "c", "retry_after": 5}}, 429)
    assert nested == {
        "message": "m",
        "code": "c",
        "retry_after": 5,
        "type": "rate_limit_error",
        "param": None,
    }
    flat = openai_error({"message": "m", "type": "t", "other": 1}, 400)
    assert flat == {"message": "m", "type": "t", "param": None, "code": None}
    other = openai_error({"reason": "x"}, 400)
    assert other["message"] == '{"reason": "x"}'


def test_ollama_error_from_every_shape():
    assert ollama_error({"error": {"message": "m"}}, 500) == "m"
    assert ollama_error({"error": "e"}, 500) == "e"
    assert ollama_error({"message": "m"}, 500) == "m"
    assert ollama_error("s", 500) == "s"
    assert ollama_error(None, 503) == "Service Unavailable"
