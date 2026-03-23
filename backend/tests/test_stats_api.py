from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app() -> FastAPI:
    from ocabra.api.internal.stats import router as stats_router

    app = FastAPI()
    app.include_router(stats_router, prefix="/ocabra")
    return app


def test_tokens_endpoint_exists():
    app = _make_app()
    client = TestClient(app)

    fake_payload = {
        "totalInputTokens": 10,
        "totalOutputTokens": 20,
        "byBackend": [],
        "series": [],
    }

    with patch(
        "ocabra.stats.aggregator.get_token_stats", new=AsyncMock(return_value=fake_payload)
    ):
        resp = client.get("/ocabra/stats/tokens")

    assert resp.status_code == 200
    assert resp.json() == fake_payload


def test_requests_endpoint_accepts_model_id_snake_case():
    app = _make_app()
    client = TestClient(app)

    with patch(
        "ocabra.stats.aggregator.get_request_stats",
        new=AsyncMock(return_value={"totalRequests": 0, "series": []}),
    ) as mock_get:
        resp = client.get("/ocabra/stats/requests?model_id=my-model")

    assert resp.status_code == 200
    assert mock_get.await_args.args[2] == "my-model"


def test_requests_endpoint_accepts_model_id_camel_case():
    app = _make_app()
    client = TestClient(app)

    with patch(
        "ocabra.stats.aggregator.get_request_stats",
        new=AsyncMock(return_value={"totalRequests": 0, "series": []}),
    ) as mock_get:
        resp = client.get("/ocabra/stats/requests?modelId=my-model")

    assert resp.status_code == 200
    assert mock_get.await_args.args[2] == "my-model"


def test_overview_endpoint_exists_and_filters_model() -> None:
    app = _make_app()
    client = TestClient(app)

    fake_payload = {
        "totalRequests": 3,
        "totalErrors": 1,
        "avgDurationMs": 123,
        "tokenizedRequests": 2,
        "totalInputTokens": 10,
        "totalOutputTokens": 5,
        "byBackend": [],
        "byRequestKind": [],
    }

    with patch(
        "ocabra.stats.aggregator.get_overview_stats",
        new=AsyncMock(return_value=fake_payload),
    ) as mock_get:
        resp = client.get("/ocabra/stats/overview?modelId=test-model")

    assert resp.status_code == 200
    assert resp.json() == fake_payload
    assert mock_get.await_args.args[2] == "test-model"
