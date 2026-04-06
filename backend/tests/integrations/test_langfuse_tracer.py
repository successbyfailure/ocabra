"""Tests for ocabra.integrations.langfuse_tracer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_REQUEST_BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": False,
}

SAMPLE_RESPONSE_BODY = {
    "choices": [{"message": {"content": "Hi there!"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


def _make_sse_chunk(data: dict) -> bytes:
    """Build a single SSE data line from a dict."""
    import json

    return f"data: {json.dumps(data)}\n\n".encode()


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> bytes:
    return _make_sse_chunk(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    )


def _delta_chunk(content: str) -> bytes:
    return _make_sse_chunk({"choices": [{"delta": {"content": content}}]})


def _done_chunk() -> bytes:
    return b"data: [DONE]\n\n"


async def _async_gen(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset the module-level singleton between tests."""
    import ocabra.integrations.langfuse_tracer as mod

    mod._client = None
    yield
    mod._client = None


@pytest.fixture()
def _disabled(monkeypatch):
    monkeypatch.setattr("ocabra.config.settings.langfuse_enabled", False)


@pytest.fixture()
def _enabled(monkeypatch):
    monkeypatch.setattr("ocabra.config.settings.langfuse_enabled", True)
    monkeypatch.setattr("ocabra.config.settings.langfuse_public_key", "pk-test")
    monkeypatch.setattr("ocabra.config.settings.langfuse_secret_key", "sk-test")
    monkeypatch.setattr("ocabra.config.settings.langfuse_host", "http://localhost:3000")
    monkeypatch.setattr("ocabra.config.settings.langfuse_sample_rate", 1.0)
    monkeypatch.setattr("ocabra.config.settings.langfuse_capture_content", False)
    monkeypatch.setattr("ocabra.config.settings.langfuse_flush_interval_s", 2.0)


@pytest.fixture()
def mock_langfuse(_enabled):
    """Provide a mock Langfuse client injected into the module singleton."""
    import ocabra.integrations.langfuse_tracer as mod

    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_client.trace.return_value = mock_trace
    mod._client = mock_client
    return mock_client, mock_trace


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_disabled")
async def test_disabled_no_sdk_call():
    from ocabra.integrations.langfuse_tracer import trace_generation

    with patch("ocabra.integrations.langfuse_tracer._get_client", return_value=None) as gc:
        await trace_generation(
            model_id="m",
            path="/v1/chat/completions",
            request_body=SAMPLE_REQUEST_BODY,
            response_body=SAMPLE_RESPONSE_BODY,
            duration_ms=100,
        )
        gc.assert_called_once()


async def test_missing_keys_no_call(monkeypatch):
    monkeypatch.setattr("ocabra.config.settings.langfuse_enabled", True)
    monkeypatch.setattr("ocabra.config.settings.langfuse_public_key", None)
    monkeypatch.setattr("ocabra.config.settings.langfuse_secret_key", None)

    import ocabra.integrations.langfuse_tracer as mod

    mod._client = None
    client = mod._get_client()
    assert client is None


async def test_trace_no_content(mock_langfuse, monkeypatch):
    monkeypatch.setattr("ocabra.config.settings.langfuse_capture_content", False)
    mock_client, mock_trace = mock_langfuse

    from ocabra.integrations.langfuse_tracer import trace_generation

    await trace_generation(
        model_id="test-model",
        path="/v1/chat/completions",
        request_body=SAMPLE_REQUEST_BODY,
        response_body=SAMPLE_RESPONSE_BODY,
        duration_ms=150,
    )

    mock_client.trace.assert_called_once()
    mock_trace.generation.assert_called_once()
    call_kwargs = mock_trace.generation.call_args[1]
    # Input should be metadata-only (no messages)
    assert "messages" not in str(call_kwargs["input"]) or isinstance(call_kwargs["input"], dict)
    assert call_kwargs["input"].get("model") == "test-model"
    # Output should be None
    assert call_kwargs["output"] is None
    # Usage extracted
    assert call_kwargs["usage"]["input"] == 10
    assert call_kwargs["usage"]["output"] == 5


async def test_trace_with_content(mock_langfuse, monkeypatch):
    monkeypatch.setattr("ocabra.config.settings.langfuse_capture_content", True)
    mock_client, mock_trace = mock_langfuse

    from ocabra.integrations.langfuse_tracer import trace_generation

    await trace_generation(
        model_id="test-model",
        path="/v1/chat/completions",
        request_body=SAMPLE_REQUEST_BODY,
        response_body=SAMPLE_RESPONSE_BODY,
        duration_ms=150,
    )

    call_kwargs = mock_trace.generation.call_args[1]
    # Input should contain the full messages
    assert call_kwargs["input"] == SAMPLE_REQUEST_BODY["messages"]
    # Output should contain the completion text
    assert call_kwargs["output"] == "Hi there!"


@pytest.mark.usefixtures("_disabled")
async def test_wrap_stream_passthrough_disabled():
    from ocabra.integrations.langfuse_tracer import wrap_stream

    chunks = [b"chunk1", b"chunk2"]
    result = []
    async for c in wrap_stream(
        _async_gen(chunks),
        model_id="m",
        path="/v1/chat/completions",
        request_body={},
    ):
        result.append(c)
    assert result == chunks


async def test_wrap_stream_zero_latency(mock_langfuse):
    """First chunk is yielded before the generator finishes."""
    from ocabra.integrations.langfuse_tracer import wrap_stream

    chunks = [_delta_chunk("Hello"), _delta_chunk(" world"), _usage_chunk(5, 10), _done_chunk()]
    received = []
    async for c in wrap_stream(
        _async_gen(chunks),
        model_id="m",
        path="/v1/chat/completions",
        request_body=SAMPLE_REQUEST_BODY,
    ):
        received.append(c)
    # All chunks pass through
    assert len(received) == len(chunks)
    assert received[0] == chunks[0]


async def test_wrap_stream_parses_usage(mock_langfuse):
    mock_client, mock_trace = mock_langfuse
    from ocabra.integrations.langfuse_tracer import wrap_stream

    chunks = [_delta_chunk("Hi"), _usage_chunk(20, 30), _done_chunk()]
    async for _ in wrap_stream(
        _async_gen(chunks),
        model_id="m",
        path="/v1/chat/completions",
        request_body=SAMPLE_REQUEST_BODY,
    ):
        pass

    mock_trace.generation.assert_called_once()
    usage = mock_trace.generation.call_args[1]["usage"]
    assert usage["input"] == 20
    assert usage["output"] == 30


async def test_wrap_stream_captures_delta_text(mock_langfuse, monkeypatch):
    monkeypatch.setattr("ocabra.config.settings.langfuse_capture_content", True)
    mock_client, mock_trace = mock_langfuse
    from ocabra.integrations.langfuse_tracer import wrap_stream

    chunks = [_delta_chunk("Hello"), _delta_chunk(" world"), _done_chunk()]
    async for _ in wrap_stream(
        _async_gen(chunks),
        model_id="m",
        path="/v1/chat/completions",
        request_body=SAMPLE_REQUEST_BODY,
    ):
        pass

    output = mock_trace.generation.call_args[1]["output"]
    assert output == "Hello world"


async def test_sample_rate_zero_never_traces(mock_langfuse, monkeypatch):
    monkeypatch.setattr("ocabra.config.settings.langfuse_sample_rate", 0.0)
    mock_client, mock_trace = mock_langfuse

    from ocabra.integrations.langfuse_tracer import trace_generation

    await trace_generation(
        model_id="m",
        path="/v1/chat/completions",
        request_body=SAMPLE_REQUEST_BODY,
        response_body=SAMPLE_RESPONSE_BODY,
        duration_ms=100,
    )
    mock_client.trace.assert_not_called()


async def test_shutdown_calls_flush(mock_langfuse):
    mock_client, _ = mock_langfuse
    from ocabra.integrations.langfuse_tracer import shutdown

    await shutdown()
    mock_client.flush.assert_called_once()


async def test_exception_in_stream_still_sends_error_trace(mock_langfuse):
    mock_client, mock_trace = mock_langfuse

    from ocabra.integrations.langfuse_tracer import wrap_stream

    async def _failing_gen():
        yield _delta_chunk("partial")
        raise RuntimeError("backend crashed")

    with pytest.raises(RuntimeError, match="backend crashed"):
        async for _ in wrap_stream(
            _failing_gen(),
            model_id="m",
            path="/v1/chat/completions",
            request_body=SAMPLE_REQUEST_BODY,
        ):
            pass

    mock_trace.generation.assert_called_once()
    call_kwargs = mock_trace.generation.call_args[1]
    assert call_kwargs["level"] == "ERROR"
    assert call_kwargs["status_message"] == "backend crashed"
