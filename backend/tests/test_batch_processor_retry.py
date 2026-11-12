import pytest

from ocabra.core.batch_processor import _is_transient_batch_error


def test_insufficient_vram_409_is_transient() -> None:
    assert _is_transient_batch_error(409, {"code": "insufficient_vram"}) is True


def test_other_409_codes_stay_permanent() -> None:
    assert _is_transient_batch_error(409, {"code": "model_at_capacity"}) is False


def test_insufficient_vram_at_other_status_codes_stays_permanent() -> None:
    # Same code, but not the 409 this specific condition came from — don't
    # widen retry eligibility beyond what's been reproduced/verified.
    assert _is_transient_batch_error(500, {"code": "insufficient_vram"}) is False


def test_missing_code_stays_permanent() -> None:
    assert _is_transient_batch_error(409, {"message": "no code field"}) is False


def test_whisper_out_of_memory_503_is_transient() -> None:
    """2026-11-12: un batch diarizado en la 3060 llena fallaba linea a linea."""
    assert _is_transient_batch_error(503, {"code": "transcription_out_of_memory"}) is True
    assert _is_transient_batch_error(503, {"code": "diarization_out_of_memory"}) is True
    assert _is_transient_batch_error(503, {"code": "model_maintenance"}) is False
    assert _is_transient_batch_error(500, {"code": "transcription_out_of_memory"}) is False


@pytest.mark.asyncio
async def test_dispatch_reads_the_code_inside_fastapi_s_detail_envelope() -> None:
    """Los endpoints responden {"detail": {"error": {...}}}. Sin desenvolverlo
    todo codigo salia como "http_<status>" y ningun error transitorio
    (insufficient_vram, la falta de memoria de whisper) se reencolaba."""
    from fastapi import FastAPI

    from ocabra.api.openai._deps import _openai_error
    from ocabra.core.batch_processor import BatchProcessor

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _boom():
        raise _openai_error("no cabe", "server_error", code="insufficient_vram", status_code=409)

    processor = BatchProcessor(app)
    _body, status, error = await processor._dispatch(
        endpoint="/v1/chat/completions", body={"model": "x"}, user_id="u", api_key_name=""
    )

    assert status == 409
    assert error["code"] == "insufficient_vram"
    assert _is_transient_batch_error(status, error) is True
