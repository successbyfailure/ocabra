from ocabra.core.batch_priority import BatchPriorityCoordinator


def test_runtime_canonicalization_preserves_gguf_artifact_separator() -> None:
    model_id = "llama_cpp/deadbydawn/chaos-gguf::Chaos-Q4_K_M"

    assert BatchPriorityCoordinator._canonical(model_id) == model_id


def test_runtime_canonicalization_strips_only_worker_hash() -> None:
    model_id = "llama_cpp/deadbydawn/chaos-gguf::Chaos-Q4_K_M"

    assert BatchPriorityCoordinator._canonical(f"{model_id}::0123456789ab") == model_id
