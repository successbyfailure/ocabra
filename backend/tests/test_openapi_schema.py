from ocabra.main import app


def _schema() -> dict:
    app.openapi_schema = None
    return app.openapi()


def test_batches_endpoint_and_purpose_are_discoverable_enums() -> None:
    schema = _schema()
    comp = schema["components"]["schemas"]
    assert set(comp["BatchCreateBody"]["properties"]["endpoint"]["enum"]) == {
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/completions",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/audio/generate",
        "/v1/pooling",
        "/v1/score",
        "/v1/rerank",
        "/v1/classify",
    }
    body_schema = comp["Body_upload_file_v1_files_post"]
    assert set(body_schema["properties"]["purpose"]["enum"]) == {
        "batch",
        "batch_output",
        "user_data",
        "batch_input_audio",
        "batch_input_image",
    }


def test_batch_line_shapes_and_previously_undocumented_request_bodies_are_injected() -> None:
    schema = _schema()
    comp = schema["components"]["schemas"]
    for name in (
        "TranscriptionBatchLineBody",
        "ImageEditsBatchLineBody",
        "PoolingRequest",
        "ScoreRequest",
        "ClassifyRequest",
    ):
        assert name in comp

    for path in ("/v1/pooling", "/v1/score", "/v1/classify", "/v1/images/edits"):
        assert "requestBody" in schema["paths"][path]["post"]
