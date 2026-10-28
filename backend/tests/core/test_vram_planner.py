"""Tests for the VRAM planner primitive and per-backend translators.

The KV formula is validated against a measured Ollama footprint (qwen3:4b):
weights+overhead ≈ 2454 MiB, so 32k ctx should land within a couple % of the
measured 7197 MiB.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from ocabra.core import vram_planner as vp

# Real GGUF metadata for qwen3.8:27b (arch "qwen35"), read from the production
# blob with vp.read_gguf_metadata — see docs/tasks/ (rs_cache investigation).
QWEN35_REAL_GGUF_KV = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 65,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.full_attention_interval": 4,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.inner_size": 6144,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.group_count": 16,
    "qwen35.embedding_length": 5120,
    "qwen35.context_length": 262144,
}
QWEN35_FILE_SIZE_MB = 16810714464 / (1024 * 1024)  # real blob size on disk


def _gguf_bytes(kv: dict) -> bytes:
    """Minimal synthetic GGUF binary: just the magic/version/counts + a KV
    section, matching what vp.read_gguf_metadata parses. Strings use GGUF
    type 8; lists of ints/bools are written as a uint32 array (type 9,
    element type 7/bool — matches real exports like gemma4's per-layer
    head_count_kv / sliding_window_pattern); everything else is written as
    an unsigned 32-bit int (type 4) — enough for every key arch_from_gguf
    reads."""
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 0)  # tensor_count
    out += struct.pack("<Q", len(kv))  # kv_count
    for key, value in kv.items():
        key_bytes = key.encode("utf-8")
        out += struct.pack("<Q", len(key_bytes)) + key_bytes
        if isinstance(value, str):
            out += struct.pack("<I", 8)
            val_bytes = value.encode("utf-8")
            out += struct.pack("<Q", len(val_bytes)) + val_bytes
        elif isinstance(value, list):
            out += struct.pack("<I", 9)  # array
            out += struct.pack("<I", 7)  # element type: bool/uint8
            out += struct.pack("<Q", len(value))
            for item in value:
                out += struct.pack("<B", int(item))
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", int(value))
    return bytes(out)


def _write_gguf(tmp_path: Path, kv: dict) -> str:
    p = tmp_path / "model.gguf"
    p.write_bytes(_gguf_bytes(kv))
    return str(p)

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


@pytest.mark.parametrize(
    ("cache_type", "expected"),
    [
        (None, 2.0),
        ("", 2.0),
        ("f16", 2.0),
        ("F16", 2.0),
        ("bf16", 2.0),
        ("q8_0", 1.0),
        ("q8", 1.0),
        ("q5_1", 0.65),
        ("q4_0", 0.5),
        ("totally-unknown", 2.0),
    ],
)
def test_kv_dtype_bytes_for_cache_type(cache_type, expected):
    # The single source of truth both the scheduler's pre-load estimate
    # (model_manager_helpers.estimate_llama_cpp_vram_from_config) and the
    # /memory-estimate UI panel now share — previously each kept its own
    # copy of this mapping, and a third (llama_cpp_estimator.py, now
    # unused for this) used yet another one keyed differently.
    assert vp.kv_dtype_bytes_for_cache_type(cache_type) == expected


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


def test_rs_cache_mb_per_slot_matches_qwen35_real_params():
    # recurrent_layers = 65 - ceil(65/4) = 65 - 17 = 48
    # n_embd_r = (4-1) * (6144 + 2*16*128) = 3 * 10240 = 30720
    # n_embd_s = 128 * 6144 = 786432
    # bytes/slot = 48 * (30720 + 786432) * 4 (F32) = 156893184 B = 149.625 MiB exactly
    mb = vp._hybrid_rs_cache_mb_per_slot(
        recurrent_layers=48, ssm_d_conv=4, ssm_d_inner=6144, ssm_d_state=128, ssm_n_group=16,
    )
    assert mb == 149.625


def test_rs_cache_vram_mb_scales_with_slots_and_rollback():
    arch = vp.ModelArch(layers=65, n_kv_heads=4, key_length=256, value_length=256,
                         rs_cache_mb_per_slot=149.625)
    assert vp.rs_cache_vram_mb(arch, slots=1) == 149.625
    assert vp.rs_cache_vram_mb(arch, slots=2) == 149.625 * 2
    # self-speculative rollback (n_rs_seq) widens every slot's rows by (1 + n_rs_seq)
    assert vp.rs_cache_vram_mb(arch, slots=1, rollback_snapshots=3) == 149.625 * 4
    assert vp.rs_cache_vram_mb(arch, slots=2, rollback_snapshots=3) == 149.625 * 2 * 4


def test_rs_cache_zero_for_non_hybrid_arch():
    arch = vp.ModelArch(layers=36, n_kv_heads=8, key_length=128, value_length=128)
    assert vp.rs_cache_vram_mb(arch, slots=4, rollback_snapshots=3) == 0.0


def test_max_context_tokens_deducts_rs_cache_budget():
    arch_dense = vp.ModelArch(layers=65, n_kv_heads=4, key_length=256, value_length=256)
    arch_hybrid = vp.ModelArch(layers=65, n_kv_heads=4, key_length=256, value_length=256,
                                kv_elements_per_token_override=17 * 4 * (256 + 256),
                                rs_cache_mb_per_slot=149.625)
    dense_ctx = vp.max_context_tokens(24000, 16000, arch_dense, slots=1)
    hybrid_ctx = vp.max_context_tokens(24000, 16000, arch_hybrid, slots=1)
    # same free budget, hybrid has fewer KV-bearing layers (17 vs 65) → more
    # context fits even after paying the fixed rs_cache tax
    assert hybrid_ctx > dense_ctx > 0


def test_arch_from_gguf_hybrid_counts_only_full_attention_layers(tmp_path):
    path = _write_gguf(tmp_path, QWEN35_REAL_GGUF_KV)
    arch = vp.arch_from_gguf(path)
    assert arch is not None
    # Same reduction the Ollama-path test already validates: only the 17
    # full-attention layers (ceil(65/4)) contribute to the KV-per-token cost.
    assert arch.kv_bytes_per_token(1.0) == 17 * 4 * (256 + 256)


def test_arch_from_gguf_computes_rs_cache_for_qwen35(tmp_path):
    path = _write_gguf(tmp_path, QWEN35_REAL_GGUF_KV)
    arch = vp.arch_from_gguf(path)
    assert arch is not None
    assert arch.rs_cache_mb_per_slot == 149.625


def test_arch_from_gguf_no_full_attention_interval_means_no_hybrid_fields(tmp_path):
    # A plain dense model (no full_attention_interval key at all) must not
    # pick up a spurious KV override or rs_cache.
    kv = dict(QWEN35_REAL_GGUF_KV)
    del kv["qwen35.full_attention_interval"]
    path = _write_gguf(tmp_path, kv)
    arch = vp.arch_from_gguf(path)
    assert arch is not None
    assert arch.kv_elements_per_token_override == 0


# Real metadata for gemma4:26b (arch "gemma4"), read from the production
# blob with vp.read_gguf_metadata after _gguf_consume_value started
# capturing small arrays — see arch_from_gguf's gemma4 branch (2026-10-27).
GEMMA4_REAL_GGUF_KV = {
    "general.architecture": "gemma4",
    "gemma4.block_count": 30,
    "gemma4.attention.head_count": 16,
    "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 2] * 5,
    "gemma4.attention.key_length": 512,
    "gemma4.attention.value_length": 512,
    "gemma4.attention.sliding_window_pattern": [1, 1, 1, 1, 1, 0] * 5,
    "gemma4.attention.sliding_window": 1024,
    "gemma4.embedding_length": 2816,
    "gemma4.context_length": 262144,
}


def test_arch_from_gguf_handles_per_layer_head_count_kv_array(tmp_path):
    """2026-10-27 bug: a model whose GGUF stores attention.head_count_kv as
    a per-layer array (confirmed real for gemma4, not a synthetic edge
    case) crashed arch_from_gguf with TypeError('int() argument ... not
    list') once array-typed metadata started being captured at all —
    previously silently invisible since all arrays were discarded."""
    path = _write_gguf(tmp_path, GEMMA4_REAL_GGUF_KV)
    arch = vp.arch_from_gguf(path)
    assert arch is not None
    assert arch.n_kv_heads == 8  # max() of the per-layer values


def test_arch_from_gguf_sliding_pattern_uses_per_layer_n_kv(tmp_path):
    """Global (non-sliding) layers in gemma4 use FEWER kv heads (2) than
    local/SWA layers (8) — the sliding_pattern branch must use each layer's
    own head_count_kv, not a single max()-derived value, or it overcounts
    the 5 global layers' contribution by 4x."""
    path = _write_gguf(tmp_path, GEMMA4_REAL_GGUF_KV)
    arch = vp.arch_from_gguf(path)
    assert arch is not None
    # 5 global layers (indices 5,11,17,23,29) * n_kv=2 * (512+512), PLUS the
    # gemma4-specific calibrated residual (+13396, see arch_from_gguf).
    assert arch.kv_elements_per_token_override == 5 * 2 * 1024 + 13396


def test_arch_from_gguf_gemma4_matches_real_measurements(tmp_path):
    """Exact reproduction of 2 real llama_cpp/gemma4-26b loads (1 slot,
    --spec-type draft-mtp, q8_0 KV, with mmproj — 2026-10-27): ctx=32768 ->
    19009 MiB real, ctx=131072 -> 21225 MiB real. The calibrated
    kv_elements_override + fixed_load_overhead_mb in arch_from_gguf's
    gemma4 branch were solved to reproduce these exactly."""
    path = _write_gguf(tmp_path, GEMMA4_REAL_GGUF_KV)
    arch = vp.arch_from_gguf(path)
    assert arch is not None
    weights_mb = 16283
    draft_overhead = vp.mtp_draft_overhead_mb(1)
    for ctx, real_mb in [(32768, 19009), (131072, 21225)]:
        predicted = vp.estimate_total_vram_mb(
            weights_mb, arch, ctx,
            kv_dtype_bytes=1.0,
            overhead_mb=vp.DEFAULT_OVERHEAD_MB + draft_overhead,
            slots=1, rs_rollback_snapshots=3,
        )
        assert predicted == real_mb


def test_arch_from_gguf_falls_back_to_head_count_when_head_count_kv_missing(tmp_path):
    """gemma4's real GGUF has no attention.head_count_kv key at all (full
    MHA, no GQA) — must fall back to head_count instead of returning None
    (the real symptom: /capacity reported applicable=false, 'no se pudo
    leer el header del GGUF', for a GGUF that parses everything else fine)."""
    kv = {
        "general.architecture": "gemma4",
        "gemma4.block_count": 30,
        "gemma4.attention.head_count": 16,
        "gemma4.attention.key_length": 512,
        "gemma4.attention.value_length": 512,
        "gemma4.embedding_length": 2816,
        "gemma4.context_length": 262144,
    }
    path = _write_gguf(tmp_path, kv)
    arch = vp.arch_from_gguf(path)
    assert arch is not None
    assert arch.n_kv_heads == 16


_QWEN35_REAL_ARCH = vp.ModelArch(
    layers=65, n_kv_heads=4, key_length=256, value_length=256,
    hidden_size=5120, context_length=262144,
    kv_elements_per_token_override=17 * 4 * (256 + 256),
    rs_cache_mb_per_slot=149.625,
)


def test_mtp_draft_overhead_mb_matches_measured_lookup_table():
    assert vp.mtp_draft_overhead_mb(1) == 321.77
    assert vp.mtp_draft_overhead_mb(2) == 1098.27
    assert vp.mtp_draft_overhead_mb(3) == 834.29
    assert vp.mtp_draft_overhead_mb(0) == 0.0


def test_mtp_draft_overhead_mb_falls_back_to_conservative_rate_beyond_table():
    # Not a real measurement — 4 slots was never loaded and verified. Must
    # use the highest per-slot rate seen (2-slot point) rather than silently
    # assuming the apparent 3-slot dip continues.
    assert vp.mtp_draft_overhead_mb(4) == pytest.approx((1098.27 / 2) * 4)


def test_plan_llama_cpp_vram_mb_matches_real_measurement_1_slot_32k():
    """Ground truth, 2026-10-27: a real qwen3.8:27b load (q8_0 KV,
    --spec-type draft-mtp, ubatch=128, no --spec-draft-n-max override, so
    n_rs_seq defaults to 3) at 1 slot / 32768 ctx, pinned to one idle GPU
    via CUDA_VISIBLE_DEVICES, measured 17665 MiB via nvidia-smi before/after
    diff — see the rs_cache investigation report and gpu-bench/README.md.
    The reverted dense/no-rs_cache estimate for this exact config was
    20591 MiB (over by ~2.9 GiB — safe, but so conservative it capped real
    production capacity at 1 slot where 5+ would fit). The fixed estimate
    must stay >= real (safety) without drifting back to that old padding."""
    predicted = vp.plan_llama_cpp_vram_mb(
        _QWEN35_REAL_ARCH, QWEN35_FILE_SIZE_MB, 32768, gpu_layers=99,
        kv_dtype_bytes=1.0, parallel_slots=1, rs_rollback_snapshots=3,
    )
    real = 17665
    assert predicted >= real
    assert predicted - real < 1500  # safety margin, not the old blind padding


def test_plan_llama_cpp_vram_mb_matches_real_measurement_2_slots_128k_total():
    """Same real worker, 2 slots / 131072 ctx total (65536/slot) — the exact
    production configuration running the night this was investigated.
    Measured 22303 MiB. This is the config that OOM'd when the rs_cache fix
    shipped without the draft-overhead term (predicted 21980 < real 22303)."""
    predicted = vp.plan_llama_cpp_vram_mb(
        _QWEN35_REAL_ARCH, QWEN35_FILE_SIZE_MB, 131072, gpu_layers=99,
        kv_dtype_bytes=1.0, parallel_slots=2, rs_rollback_snapshots=3,
    )
    real = 22303
    assert predicted >= real
    assert predicted - real < 1500


def test_plan_llama_cpp_vram_mb_matches_real_measurement_3_slots_32768_each():
    """Same real worker, 3 slots / 32768 ctx each (98304 total) — measured
    21549 MiB via nvidia-smi. This is the config the original slots^2 draft-
    overhead formula wrongly predicted as NOT fitting on the 3090 alone
    (requested 32768 > its claimed max of 30932); the real measurement says
    it fits with ~2.6 GiB to spare, which is why the formula was replaced
    with a lookup table of real measurements instead of an extrapolated
    curve."""
    predicted = vp.plan_llama_cpp_vram_mb(
        _QWEN35_REAL_ARCH, QWEN35_FILE_SIZE_MB, 98304, gpu_layers=99,
        kv_dtype_bytes=1.0, parallel_slots=3, rs_rollback_snapshots=3,
    )
    real = 21549
    assert predicted >= real
    assert predicted - real < 1500


def test_plan_use_case_lets_3x32k_fit_on_the_3090_alone():
    """Regression for the over-conservative slots^2 formula: with the real
    3-slot measurement in the lookup table, 3 slots of 32768 ctx each must
    resolve as fitting on a single 3090 (24576 MiB) — it does in reality,
    with ~2.6 GiB of real headroom to spare."""
    plan = vp.plan_use_case(
        "llama_cpp", _QWEN35_REAL_ARCH, QWEN35_FILE_SIZE_MB,
        gpu_total_mb=24576, requested_context=32768, slots=3,
        kv_dtype_bytes=1.0, rs_rollback_snapshots=3,
    )
    assert plan["effective_context"] == 32768
    assert plan["warnings"] == []


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


def test_arch_from_ollama_gemma4_per_layer_kv_and_sliding_window():
    mi = {
        "gemma4.block_count": 6,
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 1],
        "gemma4.attention.key_length": 512,
        "gemma4.attention.value_length": 512,
        "gemma4.attention.sliding_window_pattern": [True, True, True, True, True, False],
        "gemma4.embedding_length": 3840,
        "gemma4.context_length": 262144,
    }

    arch = vp.arch_from_ollama_model_info(mi, "gemma4")

    assert arch is not None
    assert arch.kv_bytes_per_token(1.0) == 1024


def test_arch_from_ollama_hybrid_counts_only_full_attention_layers():
    mi = {
        "qwen35.block_count": 65,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.full_attention_interval": 4,
        "qwen35.embedding_length": 5120,
        "qwen35.context_length": 262144,
    }

    arch = vp.arch_from_ollama_model_info(mi, "qwen35")

    assert arch is not None
    assert arch.kv_bytes_per_token(1.0) == 17 * 4 * (256 + 256)


def test_capacity_rows_scales_with_slots_and_caps_native():
    arch = vp.ModelArch(36, 8, 128, 128, context_length=40000)
    rows = vp.capacity_rows("ollama", arch, 2400, gpu_total_mb=24000, slots=(1, 2, 4))
    assert [r["slots"] for r in rows] == [1, 2, 4]
    # more slots → less context per slot
    assert rows[0]["max_context"] > rows[1]["max_context"] > rows[2]["max_context"]
    # capped at native context
    assert all(r["max_context_capped"] <= 40000 for r in rows)


def test_capacity_rows_vllm_uses_concurrency():
    arch = vp.ModelArch(36, 8, 128, 128, context_length=131072)
    rows = vp.capacity_rows("vllm", arch, 16000, gpu_total_mb=24576, slots=(1, 4))
    assert rows[0]["max_context"] > rows[1]["max_context"] > 0


def test_plan_use_case_auto_picks_max():
    arch = vp.ModelArch(36, 8, 128, 128, context_length=262144)
    plan = vp.plan_use_case("ollama", arch, 2400, gpu_total_mb=24000, requested_context="max", slots=1)
    assert plan["auto"] is True
    assert plan["effective_context"] == plan["max_context"] > 0
    assert plan["warnings"] == []


def test_plan_use_case_clamps_to_fit():
    arch = vp.ModelArch(36, 8, 128, 128, context_length=262144)
    # ask for far more than fits → clamped to max with a warning
    plan = vp.plan_use_case("ollama", arch, 2400, gpu_total_mb=24000, requested_context=500000, slots=1)
    assert plan["effective_context"] == plan["max_context"]
    assert plan["warnings"]


def test_plan_use_case_caps_at_native():
    arch = vp.ModelArch(36, 8, 128, 128, context_length=8192)
    plan = vp.plan_use_case("ollama", arch, 2400, gpu_total_mb=24000, requested_context=32768, slots=1)
    assert plan["effective_context"] == 8192
    assert any("nativo" in w for w in plan["warnings"])


def test_plan_use_case_fits_exactly_no_warning():
    arch = vp.ModelArch(36, 8, 128, 128, context_length=262144)
    plan = vp.plan_use_case("ollama", arch, 2400, gpu_total_mb=24000, requested_context=8192, slots=1)
    assert plan["effective_context"] == 8192
    assert plan["warnings"] == []


def test_plan_use_case_rollback_shrinks_max_context_for_hybrid_mtp_model():
    """The capacity-display path (runtime_preset_capacity.py's
    /runtime-presets/resolved and /preview endpoints) must apply the same
    rs_cache-rollback + MTP-draft-overhead penalty the real load-time
    estimator does — otherwise it advertises more context than the real
    process can hold (the exact gap that caused the 2026-10-26 OOM, just in
    the "what fits" display instead of the admission check)."""
    without_rollback = vp.plan_use_case(
        "llama_cpp", _QWEN35_REAL_ARCH, QWEN35_FILE_SIZE_MB,
        gpu_total_mb=24576, requested_context="max", slots=1, kv_dtype_bytes=1.0,
    )
    with_rollback = vp.plan_use_case(
        "llama_cpp", _QWEN35_REAL_ARCH, QWEN35_FILE_SIZE_MB,
        gpu_total_mb=24576, requested_context="max", slots=1, kv_dtype_bytes=1.0,
        rs_rollback_snapshots=3,
    )
    assert with_rollback["max_context"] < without_rollback["max_context"]


def test_vram_curve_monotonic_and_capped():
    arch = vp.ModelArch(36, 8, 128, 128, context_length=32768)
    curve = vp.vram_curve(arch, 2400)
    ctxs = [p["context"] for p in curve]
    assert max(ctxs) <= 32768  # capped at native
    vrams = [p["vram_mb"] for p in curve]
    assert vrams == sorted(vrams)  # VRAM grows with context


def test_arch_extractors_return_none_on_garbage():
    assert vp.arch_from_hf_config({}) is None
    assert vp.arch_from_ollama_model_info({}, "qwen3") is None
    assert vp.arch_from_ollama_model_info({"x.block_count": 1}, "") is None
