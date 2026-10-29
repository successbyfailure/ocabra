"""_build_update_request() — resolves an installed model to a re-pull
request for the generic DownloadManager (huggingface/ollama/bitnet).

2026-10-29: /ocabra/models/update-all surfaced real bugs for several
backends when run live — see each test's docstring for the reproduced
failure.
"""

from __future__ import annotations

from types import SimpleNamespace

from ocabra.api.internal.models_update import _build_update_request


def _state(model_id: str, backend_type: str, extra_config: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(model_id=model_id, backend_type=backend_type, extra_config=extra_config)


def test_standard_piper_voice_is_skipped_not_misresolved_as_an_hf_repo() -> None:
    """Reproduced live: the voice name ('es_ES-davefx-medium') was passed as
    an HF repo_id, 404ing against https://huggingface.co/api/models/<voice>.
    Piper voices never go through DownloadManager — the worker caches them
    itself via hf_hub_download() on load, with its own change-check."""
    state = _state("piper/es_ES-davefx-medium", "piper", extra_config=None)

    req, reason = _build_update_request(state)

    assert req is None
    assert "not updatable via the download queue" in reason


def test_custom_piper_voice_is_skipped_as_local_only() -> None:
    """A fine-tuned voice (e.g. glados-es) has no remote source at all —
    extra_config.piper.onnx_path points at a local file."""
    state = _state(
        "piper/glados-es",
        "piper",
        extra_config={"piper": {"onnx_path": "/data/models/piper/glados-es/glados-es.onnx"}},
    )

    req, reason = _build_update_request(state)

    assert req is None
    assert "local-only Piper voice" in reason


def test_vibeasr_un_flattens_the_repo_id_before_using_it() -> None:
    """Reproduced live: model_id stores '/' flattened to '--' for local
    directory naming (vibeasr_backend.py:flat_id) — passing the flattened
    string straight to hf_hub_download fails with 'Cannot have -- or .. in
    repo_id'. Must be un-flattened back to the real HF repo_id."""
    state = _state("vibeasr/microsoft--VibeVoice-ASR-BitNet", "vibeasr", extra_config=None)

    req, reason = _build_update_request(state)

    assert reason is None
    assert req is not None
    assert req.model_ref == "microsoft/VibeVoice-ASR-BitNet"
    assert req.source == "huggingface"


def test_custom_assembled_diffusers_pipeline_is_skipped() -> None:
    """Reproduced live: qwen-image-2.1's model_id is a local alias, not a
    real HF repo — the actual weights come from gguf_transformer_path
    (a different repo entirely) plus a separate text encoder. No single
    coherent remote source to re-pull from."""
    state = _state(
        "diffusers/qwen-image-2.1",
        "diffusers",
        extra_config={
            "diffusers": {
                "gguf_transformer_path": (
                    "/data/hf_cache/hub/models--Abiray--Qwen-Image-2.1-GGUF/"
                    "snapshots/abc/qwen_image_2.1_Q5_K_M.gguf"
                ),
            }
        },
    )

    req, reason = _build_update_request(state)

    assert req is None
    assert "custom-assembled" in reason


def test_regular_diffusers_model_without_gguf_assembly_still_resolves() -> None:
    """A normal diffusers model (model_id IS a real HF repo) must still go
    through the generic huggingface branch unaffected by the new check."""
    state = _state("diffusers/stabilityai/stable-diffusion-xl-base-1.0", "diffusers", extra_config=None)

    req, reason = _build_update_request(state)

    assert reason is None
    assert req is not None
    assert req.model_ref == "stabilityai/stable-diffusion-xl-base-1.0"


def test_bitnet_and_llama_cpp_are_unaffected_by_the_new_branches() -> None:
    state = _state(
        "llama_cpp/org/repo",
        "llama_cpp",
        extra_config={"model_path": "/data/models/huggingface/org--repo/model.Q4_K_M.gguf"},
    )

    req, reason = _build_update_request(state)

    assert reason is None
    assert req is not None
    assert req.source == "huggingface"
    assert req.artifact == "model.Q4_K_M.gguf"
