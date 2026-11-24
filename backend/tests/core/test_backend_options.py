"""Characterization tests for ocabra.core.backend_options (REF-05 step 1).

Lock the camelCase/snake_case option reading (BUG-28) and the vLLM
gpu_memory_utilization clamp unified out of the scheduler and model manager
(DUP-11).
"""

from __future__ import annotations

from types import SimpleNamespace

from ocabra.core import backend_options as bo
from ocabra.core import model_manager_helpers as helpers


def _state(extra: dict) -> SimpleNamespace:
    return SimpleNamespace(extra_config=extra, backend_type="vllm", model_id="m", backend_model_id="m")


def test_snake_to_camel() -> None:
    assert bo.snake_to_camel("gpu_layers") == "gpuLayers"
    assert bo.snake_to_camel("cache_type_k") == "cacheTypeK"
    assert bo.snake_to_camel("ctx_size") == "ctxSize"


def test_resolve_llama_cpp_option_accepts_both_spellings() -> None:
    assert bo.resolve_llama_cpp_option(_state({"llama_cpp": {"gpu_layers": 99}}), "gpu_layers", 0) == 99
    assert bo.resolve_llama_cpp_option(_state({"llama_cpp": {"gpuLayers": 99}}), "gpu_layers", 0) == 99
    assert bo.resolve_llama_cpp_option(_state({"gpuLayers": 42}), "gpu_layers", 0) == 42
    assert bo.resolve_llama_cpp_option(_state({}), "gpu_layers", 7) == 7


def test_resolve_vllm_option_accepts_camelcase() -> None:
    # BUG-28: the estimator must see maxModelLen saved by the UI.
    assert bo.resolve_vllm_option(_state({"vllm": {"maxModelLen": 8192}}), "max_model_len") == 8192
    assert bo.resolve_vllm_option(_state({"vllm": {"max_model_len": 8192}}), "max_model_len") == 8192
    assert bo.resolve_vllm_option(_state({"maxModelLen": 4096}), "max_model_len") == 4096
    assert bo.resolve_vllm_option(_state({}), "max_model_len", 0) == 0


def test_resolve_vllm_gpu_memory_utilization_clamps() -> None:
    assert bo.resolve_vllm_gpu_memory_utilization(0.9) == 0.9
    assert bo.resolve_vllm_gpu_memory_utilization(1.5) == 1.0
    assert bo.resolve_vllm_gpu_memory_utilization(-0.5) == 0.0
    # Unparseable / None fall back to the configured default (in [0, 1]).
    fallback = bo.resolve_vllm_gpu_memory_utilization(None)
    assert 0.0 <= fallback <= 1.0
    assert bo.resolve_vllm_gpu_memory_utilization("nope") == fallback


def test_split_policy() -> None:
    assert bo.resolve_llama_cpp_split_policy(_state({"llama_cpp": {"split_policy": "always"}})) == "always"
    assert bo.resolve_llama_cpp_split_policy(_state({"llama_cpp": {"split_mode": "none"}})) == "single"
    assert bo.resolve_llama_cpp_split_policy(_state({})) == "when_needed"


def test_helpers_reexports_are_the_same_objects() -> None:
    assert helpers.snake_to_camel is bo.snake_to_camel
    assert helpers.resolve_llama_cpp_option is bo.resolve_llama_cpp_option
    assert helpers.resolve_bitnet_option is bo.resolve_bitnet_option
    assert helpers.resolve_llama_cpp_split_policy is bo.resolve_llama_cpp_split_policy
    assert helpers.resolve_vllm_gpu_memory_utilization is bo.resolve_vllm_gpu_memory_utilization
