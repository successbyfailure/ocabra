from ocabra.api.openai.batches import (
    ImageEditsBatchLineBody,
    TranscriptionBatchLineBody,
    _line_body_shape_errors,
)


def test_transcription_line_skips_file_id_but_catches_missing_model() -> None:
    errors = _line_body_shape_errors(
        TranscriptionBatchLineBody, {}, skip_fields={"file_id"}
    )
    codes_fields = [e["message"] for e in errors]
    assert any("model" in msg for msg in codes_fields)
    assert not any("file_id" in msg for msg in codes_fields)


def test_transcription_line_accepts_full_valid_body() -> None:
    body = {
        "file_id": "file-abc",
        "model": "whisper-large-v3-turbo",
        "language": "es",
        "timestamp_granularities": ["segment"],
    }
    assert _line_body_shape_errors(TranscriptionBatchLineBody, body, skip_fields={"file_id"}) == []


def test_image_edits_line_skips_image_file_id_but_catches_missing_prompt() -> None:
    errors = _line_body_shape_errors(
        ImageEditsBatchLineBody,
        {"image_file_id": "file-abc", "model": "qwen-image-edit"},
        skip_fields={"image_file_id"},
    )
    assert any("prompt" in e["message"] for e in errors)
    assert not any("image_file_id" in e["message"] for e in errors)


def test_image_edits_line_accepts_full_valid_body() -> None:
    body = {
        "image_file_id": "file-abc",
        "mask_file_id": "file-def",
        "model": "qwen-image-edit",
        "prompt": "añade un sombrero",
    }
    assert _line_body_shape_errors(ImageEditsBatchLineBody, body, skip_fields={"image_file_id"}) == []
