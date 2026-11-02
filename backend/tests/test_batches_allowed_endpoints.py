from ocabra.api.openai.batches import ALLOWED_ENDPOINTS


def test_all_batchable_endpoints_are_allowed() -> None:
    assert ALLOWED_ENDPOINTS == {
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/completions",
        "/v1/images/generations",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/pooling",
        "/v1/score",
        "/v1/rerank",
        "/v1/classify",
        "/v1/audio/generate",
        "/v1/images/edits",
    }
