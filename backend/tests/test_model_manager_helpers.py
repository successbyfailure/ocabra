"""Unit tests for VRAM estimation helpers in ``model_manager_helpers``.

Focus: ``estimate_vllm_vram_from_config`` — the piece the scheduler consults
before deciding whether to evict to make room for a vLLM load. Regression
target is the 2026-08-11 incident on ``vllm/mattbucci/gemma-4-12B-AWQ``
where the previous weights-only heuristic passed the eviction check but
vLLM crashed inside asking for KV cache that no eviction was going to free.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocabra.core.model_manager_helpers import estimate_vllm_vram_from_config
from ocabra.core.vram_planner import DEFAULT_OVERHEAD_MB


@dataclass
class _FakeState:
    """Minimal ModelState-shaped object for the helper.

    The helper only reads ``backend_model_id`` and ``extra_config``, so a
    plain dataclass keeps the tests free of ModelManager wiring.
    """

    backend_model_id: str
    extra_config: dict[str, Any]
    model_id: str = ""


def _write_dummy_safetensors(path: Path, size_bytes: int) -> None:
    """Write a file that is ``size_bytes`` on disk.

    The helper only reads ``stat().st_size``, so the contents don't matter —
    but making a valid-ish header keeps things debuggable if a future test
    ever wants to actually parse it.
    """
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    header_len = len(header)
    prefix = struct.pack("<Q", header_len) + header
    with open(path, "wb") as f:
        f.write(prefix)
        # Sparse extension: tests only inspect stat().st_size. Writing gigabytes
        # of zeroes filled /tmp and made unrelated tests fail with ENOSPC.
        f.truncate(size_bytes)


def _make_hf_snapshot(
    tmp_path: Path,
    repo: str,
    *,
    config: dict[str, Any],
    weights_mb: int,
) -> Path:
    """Materialize a HuggingFace snapshot in ``models_dir/huggingface/<flat>/``.

    That's the simpler "flat" HF layout supported by the helper's lookup.
    """
    flat = repo.replace("/", "--")
    snapshot = tmp_path / "huggingface" / flat
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(config))
    _write_dummy_safetensors(snapshot / "model.safetensors", weights_mb * 1024 * 1024)
    return snapshot


def _make_hf_cache_snapshot(
    hf_cache_dir: Path,
    repo: str,
    *,
    config: dict[str, Any],
    weights_mb: int,
    commit: str = "abc123deadbeef",
) -> Path:
    """Materialize a snapshot under the canonical HF cache layout.

    Layout: ``<hf_cache_dir>/hub/models--<flat>/{snapshots/<commit>/,refs/main}``.
    That's what ``snapshot_download`` writes and what the vLLM backend loads
    from — the estimator must find it there too, otherwise it returns 0 and
    the caller keeps the weights-only heuristic (which was the regression).
    """
    flat = repo.replace("/", "--")
    cache_root = hf_cache_dir / "hub" / f"models--{flat}"
    snapshot = cache_root / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(config))
    _write_dummy_safetensors(snapshot / "model.safetensors", weights_mb * 1024 * 1024)
    refs = cache_root / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(commit)
    return snapshot


# A Gemma4-12B-shaped config with numbers close to what mattbucci/gemma-4-12B
# actually ships. Exact params don't matter — the tests only care that KV
# scales linearly with max_model_len and with the kv_cache_dtype byte width.
_GEMMA12_CONFIG = {
    "architectures": ["Gemma4ForCausalLM"],
    "num_hidden_layers": 42,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "hidden_size": 3584,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16",
}


def test_estimate_includes_kv_cache_for_long_context(tmp_path: Path) -> None:
    """A 256k config must reserve substantially more than weights alone.

    Concrete regression: on 2026-08-11 the old estimate (~weights × 1.2)
    let this exact configuration pass a weights-only eviction check with
    only ~8 GB free, and vLLM then crashed asking for ~9 GB of KV. The
    new estimate should include that KV so the caller either evicts more
    or refuses the load up front.
    """
    weights_mb = 7300  # matches the on-disk AWQ footprint
    _make_hf_snapshot(
        tmp_path,
        "mattbucci/gemma-4-12B-AWQ",
        config=_GEMMA12_CONFIG,
        weights_mb=weights_mb,
    )
    state = _FakeState(
        backend_model_id="mattbucci/gemma-4-12B-AWQ",
        extra_config={"vllm": {"max_model_len": 262144}},
    )

    est = estimate_vllm_vram_from_config(
        state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
    )

    # 42 layers × 8 kv_heads × (256+256) × 2 bytes/token = 344064 B/tok
    # × 262144 tok = ~86 GiB → obviously way more than weights.
    assert est > weights_mb + 10_000, (
        f"Expected KV to dominate at 256k; got {est} MB "
        f"(weights alone {weights_mb} MB)"
    )
    # And it should include the overhead constant so the reservation
    # matches what actually gets grabbed.
    assert est > weights_mb + DEFAULT_OVERHEAD_MB


def test_estimate_scales_linearly_with_context(tmp_path: Path) -> None:
    """Doubling max_model_len must double the KV portion of the estimate.

    That's how the planner ties reservations back to the operator's
    context choice — if you halve max_model_len you should see the
    reservation drop by roughly the KV(context) delta.
    """
    weights_mb = 7300
    _make_hf_snapshot(
        tmp_path,
        "mattbucci/gemma-4-12B-AWQ",
        config=_GEMMA12_CONFIG,
        weights_mb=weights_mb,
    )
    base_extra = {"vllm": {}}

    def _est(max_len: int) -> int:
        state = _FakeState(
            backend_model_id="mattbucci/gemma-4-12B-AWQ",
            extra_config={"vllm": {**base_extra["vllm"], "max_model_len": max_len}},
        )
        return estimate_vllm_vram_from_config(
            state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
        )

    est_32k = _est(32768)
    est_64k = _est(65536)
    est_128k = _est(131072)

    # KV portion at 64k should be ~2× KV portion at 32k. Since weights + overhead
    # are constant, (est_64k - est_32k) ≈ (est_128k - est_64k) / 2.
    kv_delta_64 = est_64k - est_32k
    kv_delta_128 = est_128k - est_64k
    assert abs(kv_delta_128 - 2 * kv_delta_64) < max(200, kv_delta_64 * 0.03), (
        f"KV should scale linearly with context; got deltas "
        f"32→64={kv_delta_64} MB, 64→128={kv_delta_128} MB"
    )


def test_fp8_kv_cache_halves_kv_reservation(tmp_path: Path) -> None:
    """fp8 KV cache should reserve half the KV of bf16 (same weights)."""
    weights_mb = 7300
    _make_hf_snapshot(
        tmp_path,
        "mattbucci/gemma-4-12B-AWQ",
        config=_GEMMA12_CONFIG,
        weights_mb=weights_mb,
    )

    def _est(kv_dtype: str) -> int:
        state = _FakeState(
            backend_model_id="mattbucci/gemma-4-12B-AWQ",
            extra_config={
                "vllm": {"max_model_len": 32768, "kv_cache_dtype": kv_dtype}
            },
        )
        return estimate_vllm_vram_from_config(
            state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
        )

    est_bf16 = _est("auto")  # vLLM default
    est_fp8 = _est("fp8")

    # (est_bf16 - est_fp8) should equal roughly half the bf16 KV portion.
    kv_bf16_portion = est_bf16 - weights_mb - DEFAULT_OVERHEAD_MB
    kv_fp8_portion = est_fp8 - weights_mb - DEFAULT_OVERHEAD_MB
    assert abs(kv_fp8_portion * 2 - kv_bf16_portion) < 10, (
        f"fp8 KV should be half of bf16; bf16={kv_bf16_portion} MB, "
        f"fp8={kv_fp8_portion} MB"
    )


def test_returns_zero_when_max_model_len_missing(tmp_path: Path) -> None:
    """No ``max_model_len`` = no honest KV number, fall back to caller's heuristic."""
    _make_hf_snapshot(
        tmp_path,
        "mattbucci/gemma-4-12B-AWQ",
        config=_GEMMA12_CONFIG,
        weights_mb=100,
    )
    state = _FakeState(
        backend_model_id="mattbucci/gemma-4-12B-AWQ",
        extra_config={"vllm": {}},
    )
    assert (
        estimate_vllm_vram_from_config(
            state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
        )
        == 0
    )


def test_returns_zero_when_arch_unreadable(tmp_path: Path) -> None:
    """If ``config.json`` is missing the arch fields the helper can't compute
    KV, so it must decline (return 0) rather than guess and mislead the
    scheduler.
    """
    _make_hf_snapshot(
        tmp_path,
        "some/model",
        config={"architectures": ["MysteryModel"]},  # no layers/heads/head_dim
        weights_mb=100,
    )
    state = _FakeState(
        backend_model_id="some/model",
        extra_config={"vllm": {"max_model_len": 32768}},
    )
    assert (
        estimate_vllm_vram_from_config(
            state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
        )
        == 0
    )


def test_returns_zero_when_snapshot_absent(tmp_path: Path) -> None:
    """Model not downloaded yet → helper declines rather than making things up."""
    state = _FakeState(
        backend_model_id="never/downloaded",
        extra_config={"vllm": {"max_model_len": 32768}},
    )
    assert (
        estimate_vllm_vram_from_config(
            state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
        )
        == 0
    )


def test_finds_snapshot_in_hf_cache_layout(tmp_path: Path) -> None:
    """Regression: on 2026-08-12 the estimator returned 0 for
    ``mattbucci/gemma-4-12B-AWQ`` because the model lives under
    ``$HF_CACHE_DIR/hub/models--…/snapshots/<commit>/`` and the helper only
    checked ``$MODELS_DIR/huggingface/…``. That silent 0 made the scheduler
    fall back to weights-only and vLLM crashed at startup with the exact
    "KV cache needed 9.32 GiB, available 2.31 GiB" the fix was supposed to
    prevent. Test both HF cache paths: refs/main-pointed commit and
    latest-mtime fallback.
    """
    models_dir = tmp_path / "models"
    hf_cache = tmp_path / "hf_cache"
    weights_mb = 7300
    _make_hf_cache_snapshot(
        hf_cache,
        "mattbucci/gemma-4-12B-AWQ",
        config=_GEMMA12_CONFIG,
        weights_mb=weights_mb,
    )
    state = _FakeState(
        backend_model_id="mattbucci/gemma-4-12B-AWQ",
        extra_config={"vllm": {"max_model_len": 262144}},
    )

    est = estimate_vllm_vram_from_config(
        state,
        models_dir=models_dir,
        default_gpu_memory_utilization=0.94,
        hf_cache_dir=hf_cache,
    )
    assert est > weights_mb + 10_000, (
        f"Expected KV to dominate at 256k with HF cache layout; got {est} MB"
    )


def test_returns_zero_without_hf_cache_dir_when_only_in_cache(tmp_path: Path) -> None:
    """If the caller doesn't pass ``hf_cache_dir`` and the model isn't in
    ``models_dir``, the estimator must decline — not silently guess against
    the wrong tree. Guards against a caller forgetting to plumb the new arg.
    """
    hf_cache = tmp_path / "hf_cache"
    _make_hf_cache_snapshot(
        hf_cache,
        "some/model",
        config=_GEMMA12_CONFIG,
        weights_mb=1000,
    )
    state = _FakeState(
        backend_model_id="some/model",
        extra_config={"vllm": {"max_model_len": 32768}},
    )
    assert (
        estimate_vllm_vram_from_config(
            state,
            models_dir=tmp_path / "models",
            default_gpu_memory_utilization=0.94,
            # hf_cache_dir intentionally omitted
        )
        == 0
    )


def test_reads_arch_from_text_config_sub_block(tmp_path: Path) -> None:
    """Multimodal configs put transformer params under ``text_config``.

    Regression: Gemma 4 12B (mattbucci/gemma-4-12B-AWQ) ships a top-level
    config with only ``architectures`` + the multimodal glue; every arch
    field the estimator needs lives under ``text_config``. Without the
    flatten step ``arch_from_hf_config`` reads ``None`` for every field
    and the helper returns 0, silently dropping back to the weights-only
    heuristic — which is exactly the crash we deployed the fix to prevent.
    """
    weights_mb = 7300
    _make_hf_snapshot(
        tmp_path,
        "some/multimodal-model",
        config={
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "text_config": _GEMMA12_CONFIG,
        },
        weights_mb=weights_mb,
    )
    state = _FakeState(
        backend_model_id="some/multimodal-model",
        extra_config={"vllm": {"max_model_len": 32768}},
    )
    est = estimate_vllm_vram_from_config(
        state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
    )
    assert est > weights_mb + DEFAULT_OVERHEAD_MB, (
        "text_config values should let the estimator include KV; "
        f"got {est} MB (weights alone {weights_mb} MB)"
    )


def test_sliding_window_layers_shrink_effective_kv(tmp_path: Path) -> None:
    """Hybrid sliding+full-attention models must not be sized as pure full-attn.

    Regression on Gemma 4 12B @ 256k: naive KV = 48 layers × max_model_len
    landed the estimate around ~100 GiB and would reject a load that
    vLLM itself accepts (its own math says ~9 GiB). With the layer schedule
    honoured the sliding layers only reserve ``sliding_window`` tokens each,
    so the estimate drops into the same ballpark as vLLM's own accounting.

    Test shape: same weights, same context, only difference is the layer
    schedule — expect the hybrid estimate strictly below the pure full-attn
    one, and by a factor consistent with 8 full + 40 sliding layers at
    max_model_len=262144 and sliding_window=1024.
    """
    weights_mb = 7300
    pure_full_config = {**_GEMMA12_CONFIG}
    hybrid_config = {
        **_GEMMA12_CONFIG,
        "sliding_window": 1024,
        # 40 sliding + 8 full, matching real Gemma 4 12B.
        "layer_types": (["sliding_attention"] * 40) + (["full_attention"] * 8),
    }

    def _est(repo: str, cfg: dict) -> int:
        _make_hf_snapshot(tmp_path, repo, config=cfg, weights_mb=weights_mb)
        state = _FakeState(
            backend_model_id=repo,
            extra_config={"vllm": {"max_model_len": 262144}},
        )
        return estimate_vllm_vram_from_config(
            state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
        )

    est_full = _est("test/pure-full", pure_full_config)
    est_hybrid = _est("test/hybrid", hybrid_config)

    assert est_hybrid < est_full, (
        f"Hybrid must be smaller than pure full-attn; "
        f"hybrid={est_hybrid} MB vs full={est_full} MB"
    )
    # 8 of 48 layers full-attn ⇒ hybrid KV should be roughly 8/48 = 16.7% of
    # pure full-attn KV (sliding layers add negligibly at 1024 tokens each).
    # Allow a wide margin — weights and overhead dominate at small contexts.
    kv_full = est_full - weights_mb - DEFAULT_OVERHEAD_MB
    kv_hybrid = est_hybrid - weights_mb - DEFAULT_OVERHEAD_MB
    ratio = kv_hybrid / kv_full
    assert 0.10 < ratio < 0.25, (
        f"Hybrid KV should be ~8/48 of full; got ratio={ratio:.2f} "
        f"(hybrid_kv={kv_hybrid} MB, full_kv={kv_full} MB)"
    )


def test_missing_sliding_window_falls_back_to_full(tmp_path: Path) -> None:
    """Without ``sliding_window`` set, layer_types alone shouldn't shrink KV
    (we don't know how much to shrink by). Ensures we don't silently
    undersell on configs that use ``layer_types`` for a different purpose.
    """
    weights_mb = 7300
    _make_hf_snapshot(
        tmp_path,
        "test/no-sw",
        config={
            **_GEMMA12_CONFIG,
            "layer_types": ["sliding_attention"] * 48,  # but no sliding_window
        },
        weights_mb=weights_mb,
    )
    state = _FakeState(
        backend_model_id="test/no-sw",
        extra_config={"vllm": {"max_model_len": 8192}},
    )
    est_no_sw = estimate_vllm_vram_from_config(
        state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
    )

    _make_hf_snapshot(
        tmp_path,
        "test/plain",
        config=_GEMMA12_CONFIG,
        weights_mb=weights_mb,
    )
    plain_state = _FakeState(
        backend_model_id="test/plain",
        extra_config={"vllm": {"max_model_len": 8192}},
    )
    est_plain = estimate_vllm_vram_from_config(
        plain_state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
    )
    assert est_no_sw == est_plain


def test_strips_worker_key_suffix_before_snapshot_lookup(tmp_path: Path) -> None:
    """Profile worker_keys are ``base::<hash>`` — the estimator must strip
    the suffix or every snapshot lookup silently misses.

    Regression: 2026-08-12 — the ctx256k profile called into the estimator
    with backend_model_id ``mattbucci/gemma-4-12B-AWQ::ca327621814f``;
    without stripping, ``_find_local_hf_dir`` returned ``None`` → helper
    returned 0 → scheduler fell back to the weights-only heuristic exactly
    where we needed the KV-aware one.
    """
    weights_mb = 7300
    _make_hf_snapshot(
        tmp_path,
        "mattbucci/gemma-4-12B-AWQ",
        config=_GEMMA12_CONFIG,
        weights_mb=weights_mb,
    )
    state = _FakeState(
        backend_model_id="mattbucci/gemma-4-12B-AWQ::ca327621814f",
        extra_config={"vllm": {"max_model_len": 65536}},
    )
    est = estimate_vllm_vram_from_config(
        state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
    )
    assert est > weights_mb + DEFAULT_OVERHEAD_MB, (
        f"Profile worker_key form should resolve the same as base repo id; "
        f"got {est} MB"
    )


def test_explicit_override_bypasses_helper(tmp_path: Path) -> None:
    """When the operator has pinned ``vram_estimate_mb`` we must not second-guess
    it — that override exists precisely for cases where the heuristic is wrong
    (custom quants, MTP, etc.). Returning 0 lets the existing override path in
    the vLLM backend run.
    """
    _make_hf_snapshot(
        tmp_path,
        "mattbucci/gemma-4-12B-AWQ",
        config=_GEMMA12_CONFIG,
        weights_mb=7300,
    )
    state = _FakeState(
        backend_model_id="mattbucci/gemma-4-12B-AWQ",
        extra_config={
            "vllm": {"max_model_len": 262144, "vram_estimate_mb": 5000}
        },
    )
    assert (
        estimate_vllm_vram_from_config(
            state, models_dir=tmp_path, default_gpu_memory_utilization=0.94
        )
        == 0
    )
