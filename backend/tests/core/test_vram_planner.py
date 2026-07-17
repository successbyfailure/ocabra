"""Tests for the VRAM planner primitive and per-backend translators.

The KV formula is validated against a measured Ollama footprint (qwen3:4b):
weights+overhead ≈ 2454 MiB, so 32k ctx should land within a couple % of the
measured 7197 MiB.
"""

from __future__ import annotations

from ocabra.core import vram_planner as vp

# qwen3:4b architecture (from Ollama /api/show model_info)
QWEN3_4B = vp.ModelArch(
    layers=36, n_kv_heads=8, key_length=128, value_length=128,
    hidden_size=2560, context_length=262144,
)


def test_kv_bytes_per_token_exact():
    # 2 (K+V) · 36 · 8 · 128 · 2 bytes = 147456 = 144 KB/token
    assert QWEN3_4B.kv_bytes_per_token(2.0) == 147456


def test_kv_dtype_scales_linearly():
    fp16 = QWEN3_4B.kv_bytes_per_token(2.0)
    fp8 = QWEN3_4B.kv_bytes_per_token(1.0)
    assert fp8 == fp16 // 2


def test_ollama_translator_matches_measured_footprint():
    # Anchor weights+overhead on the measured 4k point, then predict 32k.
    kv4 = vp.kv_vram_mb(QWEN3_4B, 4096)
    weights_and_overhead = 3030 - kv4  # measured size_vram at num_ctx=4096
    predicted_32k = weights_and_overhead + vp.kv_vram_mb(QWEN3_4B, 32768)
    measured_32k = 7197
    assert abs(predicted_32k - measured_32k) / measured_32k < 0.03  # within 3%


def test_ollama_translator_multiplies_by_parallel():
    one = vp.plan_ollama_vram_mb(QWEN3_4B, 2400, 8192, num_parallel=1)
    two = vp.plan_ollama_vram_mb(QWEN3_4B, 2400, 8192, num_parallel=2)
    # the KV portion doubles; weights+overhead stays put
    kv = vp.kv_vram_mb(QWEN3_4B, 8192)
    assert abs((two - one) - kv) < 2


def test_max_context_inverse_of_estimate():
    weights, free = 2400, 22000
    c = vp.max_context_tokens(free, weights, QWEN3_4B, slots=1)
    # feeding c back must not exceed the budget
    assert vp.estimate_total_vram_mb(weights, QWEN3_4B, c) <= free
    # one token more would overflow (within rounding)
    assert vp.estimate_total_vram_mb(weights, QWEN3_4B, c + 1024) > free


def test_max_context_halves_with_double_slots():
    one = vp.max_context_tokens(22000, 2400, QWEN3_4B, slots=1)
    two = vp.max_context_tokens(22000, 2400, QWEN3_4B, slots=2)
    assert abs(two - one // 2) <= 1


def test_llama_cpp_partial_offload_scales_weights_and_kv():
    arch = vp.ModelArch(layers=40, n_kv_heads=2, key_length=256, value_length=256)
    full = vp.plan_llama_cpp_vram_mb(arch, 20000, 4096, gpu_layers=40)
    half = vp.plan_llama_cpp_vram_mb(arch, 20000, 4096, gpu_layers=20)
    assert half < full  # fewer offloaded layers → less weight + KV on GPU


def test_llama_cpp_ctx_size_not_multiplied_by_slots():
    # llama.cpp --ctx-size is the TOTAL budget; the planner takes ctx_size directly
    arch = vp.ModelArch(layers=40, n_kv_heads=2, key_length=256, value_length=256)
    a = vp.plan_llama_cpp_vram_mb(arch, 20000, 8192)
    b = vp.plan_llama_cpp_vram_mb(arch, 20000, 16384)
    kv_delta = vp.kv_vram_mb(arch, 16384) - vp.kv_vram_mb(arch, 8192)
    assert abs((b - a) - kv_delta) < 2


def test_vllm_footprint_is_fixed_fraction():
    assert vp.vllm_footprint_mb(24576, 0.85) == int(24576 * 0.85)


def test_vllm_max_model_len_shrinks_with_concurrency():
    arch = vp.ModelArch(layers=36, n_kv_heads=8, key_length=128, value_length=128)
    one = vp.vllm_max_model_len(arch, 16000, 24576, concurrency=1)
    four = vp.vllm_max_model_len(arch, 16000, 24576, concurrency=4)
    assert one > four > 0
    assert abs(four - one // 4) <= 1


def test_arch_from_hf_config():
    cfg = {
        "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "hidden_size": 4096, "head_dim": 128,
        "max_position_embeddings": 40960, "torch_dtype": "bfloat16",
    }
    arch = vp.arch_from_hf_config(cfg)
    assert arch is not None
    assert arch.layers == 36 and arch.n_kv_heads == 8 and arch.key_length == 128


def test_arch_from_hf_config_head_dim_fallback():
    # no explicit head_dim → hidden_size / num_attention_heads
    cfg = {
        "num_hidden_layers": 22, "num_attention_heads": 32,
        "num_key_value_heads": 4, "hidden_size": 2048,
        "torch_dtype": "float16",
    }
    arch = vp.arch_from_hf_config(cfg)
    assert arch is not None and arch.key_length == 2048 // 32


def test_arch_from_ollama_model_info():
    mi = {
        "qwen3.block_count": 36,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": 128,
        "qwen3.attention.value_length": 128,
        "qwen3.embedding_length": 2560,
        "qwen3.context_length": 262144,
    }
    arch = vp.arch_from_ollama_model_info(mi, "qwen3")
    assert arch == QWEN3_4B


def test_arch_extractors_return_none_on_garbage():
    assert vp.arch_from_hf_config({}) is None
    assert vp.arch_from_ollama_model_info({}, "qwen3") is None
    assert vp.arch_from_ollama_model_info({"x.block_count": 1}, "") is None
