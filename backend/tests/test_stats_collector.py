import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from ocabra.stats.collector import (
    _classify_request_kind,
    _extract_last_payload_from_stream,
    _extract_response_payload_and_rebuild,
    _extract_usage_tokens,
)


def test_classify_request_kind_known_paths() -> None:
    assert _classify_request_kind("/v1/chat/completions") == "chat"
    assert _classify_request_kind("/v1/audio/transcriptions") == "audio_transcription"
    assert _classify_request_kind("/api/generate") == "ollama_generate"


def test_extract_usage_tokens_openai_payload() -> None:
    input_tokens, output_tokens = _extract_usage_tokens(
        {"usage": {"prompt_tokens": 12, "completion_tokens": 34}}
    )
    assert input_tokens == 12
    assert output_tokens == 34


def test_extract_usage_tokens_ollama_payload() -> None:
    input_tokens, output_tokens = _extract_usage_tokens(
        {"prompt_eval_count": 7, "eval_count": 11}
    )
    assert input_tokens == 7
    assert output_tokens == 11


def test_extract_usage_tokens_audio_transcription_payload() -> None:
    input_tokens, output_tokens = _extract_usage_tokens(
        {"text": "hola mundo desde whisper"},
        request_kind="audio_transcription",
    )
    assert input_tokens is None
    assert output_tokens == 4


def test_extract_last_payload_from_sse_stream_uses_usage_chunk() -> None:
    body = b"""data: {\"choices\":[{\"delta\":{\"content\":\"hola\"}}]}\n\ndata: {\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n"""

    payload = _extract_last_payload_from_stream(body, "text/event-stream")

    assert payload == {"usage": {"prompt_tokens": 3, "completion_tokens": 5}}


def test_extract_last_payload_from_ndjson_stream_uses_done_chunk() -> None:
    body = b"{\"message\":{\"content\":\"hola\"},\"done\":false}\n{\"prompt_eval_count\":7,\"eval_count\":11,\"done\":true}\n"

    payload = _extract_last_payload_from_stream(body, "application/x-ndjson")

    assert payload == {"prompt_eval_count": 7, "eval_count": 11, "done": True}


@pytest.mark.asyncio
async def test_extract_response_payload_json_only() -> None:
    json_resp = JSONResponse({"usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    payload, rebuilt = await _extract_response_payload_and_rebuild(json_resp)
    assert payload == {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    assert rebuilt.status_code == 200

    streaming_resp = StreamingResponse(iter([b"chunk"]))
    payload, rebuilt = await _extract_response_payload_and_rebuild(streaming_resp)
    assert payload is None
    assert isinstance(rebuilt, StreamingResponse)
