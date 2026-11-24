"""Pure helper functions for model manager lifecycle concerns."""

from __future__ import annotations

import json
from pathlib import Path

from ocabra.core.backend_options import (
    LLAMA_CPP_SPLIT_POLICIES,  # noqa: F401 - re-exported for backward compatibility
    PRISMML_DEFAULT_GPU_LAYERS,  # noqa: F401 - re-exported for backward compatibility
    _bitnet_parallel,
    is_prismml_bitnet_state,  # noqa: F401 - re-exported for backward compatibility
    resolve_bitnet_gpu_layers,
    resolve_bitnet_option,
    resolve_llama_cpp_allowed_gpus,  # noqa: F401 - re-exported for backward compatibility
    resolve_llama_cpp_gpu_layers,
    resolve_llama_cpp_option,
    resolve_llama_cpp_split_policy,  # noqa: F401 - re-exported for backward compatibility
    resolve_vllm_gpu_memory_utilization,  # noqa: F401 - re-exported for backward compatibility
    resolve_vllm_option,
    snake_to_camel,  # noqa: F401 - re-exported for backward compatibility
)
from ocabra.core.model_files import find_local_hf_dir, sum_hf_weight_files_mb
from ocabra.core.model_ids import (  # noqa: F401 - re-exported for backward compatibility
    build_diarized_extra_config,
    compute_worker_key,
    is_derived_worker_key,
    is_worker_suffix,
    overrides_enable_diarization,
    strip_diarization_overrides,
    strip_worker_suffix,
)
from ocabra.core.vram_planner import (
    arch_from_gguf,
    kv_dtype_bytes_for_cache_type,
    plan_llama_cpp_vram_mb,
)

# Per-backend ``extra_config`` section that may hold a ``vram_estimate_mb``
# override. Mirrors ``config_schema._VRAM_OVERRIDE_SECTIONS`` (kept in sync by a
# test): llama.cpp variants share the ``llama_cpp`` section, vLLM its own.
_VRAM_OVERRIDE_SECTIONS = {"llama_cpp": "llama_cpp", "llamacpp_mks": "llama_cpp", "vllm": "vllm"}

# Sections the cold-start context fallback scans, in order. Mirrors
# ``capabilities.resolve_context_length_fallback``.
_CONTEXT_SECTIONS = ("vllm", "sglang", "strata", "llama_cpp", "bitnet", "tensorrt_llm", "whisper")
_CONTEXT_KEYS = ("context_length", "max_model_len", "ctx_size")


def _context_length_from_extra(extra_config: dict) -> int | None:
    """Cold-start context length snapshot from ``extra_config`` (top level then
    per-backend section, snake or camelCase). Mirrors
    ``capabilities.resolve_context_length_fallback`` but returns ``None`` when
    nothing is configured, for a nullable column."""
    for key in _CONTEXT_KEYS:
        value = extra_config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for section_name in _CONTEXT_SECTIONS:
        section = extra_config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in _CONTEXT_KEYS:
            for candidate in (key, snake_to_camel(key)):
                value = section.get(candidate)
                if isinstance(value, int) and value > 0:
                    return value
    return None


def _vram_estimate_override_from_extra(extra_config: dict, backend_type: str) -> int | None:
    """Operator ``vram_estimate_mb`` override snapshot from the backend's
    section (``llama_cpp``/``vllm``) or top level, snake or camelCase."""
    candidates: list = []
    section_name = _VRAM_OVERRIDE_SECTIONS.get(backend_type)
    if section_name:
        section = extra_config.get(section_name)
        if isinstance(section, dict):
            candidates += [section.get("vram_estimate_mb"), section.get("vramEstimateMb")]
    candidates += [extra_config.get("vram_estimate_mb"), extra_config.get("vramEstimateMb")]
    for value in candidates:
        if isinstance(value, bool):
            continue
        try:
            as_int = int(value)
        except (TypeError, ValueError):
            continue
        if as_int > 0:
            return as_int
    return None


def promoted_columns_from_extra_config(
    extra_config: dict | None, backend_type: str
) -> dict[str, object]:
    """Derive the ``ModelConfig`` columns promoted out of ``extra_config`` (REF-20).

    Single bridge between the ``extra_config`` grab-bag and the real columns
    ``runtime_state``, ``default_reasoning_effort``, ``context_length`` and
    ``vram_estimate_mb``. Read-only: never mutates ``extra_config``. Returns a
    dict keyed by column name, with ``None`` where nothing is configured, so the
    columns can be kept in sync wherever the row is written.

    The ``extra_config`` keys are intentionally left in place: ~20 read sites
    across ``api/`` and ``backends/`` still consume them. See
    docs/REVISION-2026-11-16.md REF-20 for the deferred reader cutover.
    """
    extra = extra_config if isinstance(extra_config, dict) else {}
    runtime = extra.get("_runtime")
    runtime_state = runtime if isinstance(runtime, dict) else None
    raw_effort = extra.get("default_reasoning_effort") or extra.get("defaultReasoningEffort")
    default_reasoning_effort = str(raw_effort).strip().lower() if raw_effort else None
    return {
        "runtime_state": runtime_state,
        "default_reasoning_effort": default_reasoning_effort,
        "context_length": _context_length_from_extra(extra),
        "vram_estimate_mb": _vram_estimate_override_from_extra(extra, backend_type),
    }


def estimate_bitnet_vram_from_config(
    state,
    *,
    default_gpu_layers: int,
    default_total_layers: int = 32,
    default_model_vram_mb: int = 400,
    default_ctx_size: int = 4096,
    models_dir: str | Path | None = None,
) -> int:
    gpu_layers = resolve_bitnet_gpu_layers(state, default_gpu_layers)
    if gpu_layers <= 0:
        return 0
    total_layers = max(1, resolve_bitnet_option(state, "total_layers", default_total_layers))
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("bitnet") if isinstance(extra.get("bitnet"), dict) else {}
    has_explicit_estimate = "model_vram_mb" in extra or "model_vram_mb" in nested
    model_vram_mb = max(1, resolve_bitnet_option(state, "model_vram_mb", default_model_vram_mb))
    model_file: Path | None = None
    file_size_mb = 0.0
    if not has_explicit_estimate:
        configured_path = nested.get("model_path") or extra.get("model_path")
        candidates = [Path(str(configured_path))] if configured_path else []
        backend_id = str(getattr(state, "backend_model_id", "") or "")
        if models_dir and backend_id:
            root = Path(models_dir)
            candidates.extend(
                [
                    root / backend_id,
                    root / f"{backend_id}.gguf",
                    root / "huggingface" / backend_id.replace("/", "--"),
                ]
            )
        model_file = next((path for path in candidates if path.is_file()), None)
        if model_file is not None:
            file_size_mb = model_file.stat().st_size / (1024 * 1024)
            model_vram_mb = max(
                model_vram_mb,
                int(file_size_mb * 1.08),
            )
    if model_file is not None and file_size_mb > 0:
        arch = arch_from_gguf(model_file)
        if arch is not None:
            try:
                ctx_size = int(nested.get("ctx_size", extra.get("ctx_size", default_ctx_size)))
            except (TypeError, ValueError):
                ctx_size = default_ctx_size
            cache_type = str(
                nested.get("cache_type_k")
                or extra.get("cache_type_k")
                or nested.get("cache_type_v")
                or extra.get("cache_type_v")
                or "f16"
            ).lower()
            kv_dtype_bytes = kv_dtype_bytes_for_cache_type(cache_type)
            estimate = plan_llama_cpp_vram_mb(
                arch,
                file_size_mb,
                ctx_size,
                gpu_layers=gpu_layers,
                kv_dtype_bytes=kv_dtype_bytes,
            )
            if estimate > 0:
                return estimate
    return int(model_vram_mb * min(gpu_layers, total_layers) / total_layers)


def resolve_worker_slots(state) -> int | None:
    """How many requests this model's worker genuinely serves in parallel.

    Single source of truth, because there used to be two and they drifted.
    ``fairness_slots`` (the fair-share gate) already fell back to the
    configured llama.cpp ``parallel_slots``, while
    ``ModelManager.effective_inflight_cap`` (admission) only ever read
    ``_runtime.parallelism`` and otherwise assumed 1. Since
    ``_runtime.parallelism`` is written only by the runtime-preset coordinator,
    every llama.cpp model without a preset ladder ended up with a 4-slot gate
    and an admission cap of 1 x ``max_queued_per_slot`` = 4 — so the 5th
    concurrent request got a 429 even though the worker had three idle slots
    and room to queue twelve more (measured 2026-11-06: cap 4 where it should
    be 16, on every llamacpp_mks model, all of which set parallel_slots=4).

    Returns ``None`` when the backend has no real slot pool. Callers that need
    a number use ``or 1``; callers that report to the UI keep the ``None`` and
    render an honest "N en curso" rather than a fabricated ratio — see the
    whisper "1/32" note in ``api/internal/models.py``.
    """
    from ocabra.config import settings
    from ocabra.core.model_ref import is_llama_cpp_family

    # A context changed in place reports its real slots; extra_config keeps
    # the ones it was loaded with (see ModelManager.resize_context).
    live = getattr(state, "live_context", None)
    if isinstance(live, dict):
        try:
            live_parallel = int(live.get("n_parallel") or 0)
        except (TypeError, ValueError):
            live_parallel = 0
        if live_parallel > 0:
            return live_parallel

    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    runtime_meta = extra.get("_runtime")
    if isinstance(runtime_meta, dict):
        try:
            parallelism = int(runtime_meta.get("parallelism") or 0)
        except (TypeError, ValueError):
            parallelism = 0
        if parallelism > 0:
            return parallelism

    backend_type = getattr(state, "backend_type", "")
    if is_llama_cpp_family(backend_type) or backend_type == "bitnet":
        configured = (
            _bitnet_parallel(extra)
            if backend_type == "bitnet"
            else resolve_llama_cpp_option(state, "parallel_slots", None)
        )
        if configured is None:
            use_case = extra.get("use_case")
            if isinstance(use_case, dict):
                configured = use_case.get("slots")
        try:
            resolved = int(configured or 0)
        except (TypeError, ValueError):
            resolved = 0
        if resolved > 0:
            return resolved

    # Piper has no preset ladder but does declare a real concurrency ceiling.
    if backend_type == "piper":
        return max(1, int(settings.piper_parallelism))

    return None



# llama.cpp CLI default for the draft length (common_params_speculative_draft::n_max
# in common/common.h). When a self-speculative load (--spec-type draft-mtp /
# draft-eagle3 / draft-dflash / draft-dspark) doesn't override it with
# --spec-draft-n-max, this is also the number of recurrent-state rollback
# snapshots (n_rs_seq) llama.cpp reserves for hybrid SSM models — widening the
# rs_cache tensors to (1 + n_rs_seq) rows per sequence. See common.h
# need_n_rs_seq() / llama-context.cpp cparams.n_rs_seq.
_LLAMA_CPP_DEFAULT_DRAFT_N_MAX = 3
# --spec-type values that require recurrent-state rollback (need_n_rs_seq()).
_LLAMA_CPP_SPEC_TYPES_NEEDING_RS_ROLLBACK = {
    "draft-mtp", "draft-eagle3", "draft-dflash", "draft-dspark",
}


def _resolve_rs_rollback_snapshots(state) -> int:
    """How many extra recurrent-state rollback rows (llama.cpp's n_rs_seq) this
    load needs, from the configured ``speculative`` block. 0 unless the model
    self-speculates with a type that needs rollback (draft-mtp and friends)."""
    spec_cfg = resolve_llama_cpp_option(state, "speculative", None)
    if not isinstance(spec_cfg, dict):
        return 0
    spec_type = str(spec_cfg.get("spec_type") or "").lower()
    if spec_type not in _LLAMA_CPP_SPEC_TYPES_NEEDING_RS_ROLLBACK:
        return 0
    try:
        draft_n = spec_cfg.get("draft_n")
        return int(draft_n) if draft_n is not None else _LLAMA_CPP_DEFAULT_DRAFT_N_MAX
    except (TypeError, ValueError):
        return _LLAMA_CPP_DEFAULT_DRAFT_N_MAX


def _resolve_n_cpu_moe(state, total_layers: int = 0) -> int | None:
    """Layers whose MoE experts stay in RAM, or None.

    ``llama_cpp.n_cpu_moe`` when set; otherwise an ``override_tensor`` that
    sends the experts of the first N layers to CPU (``exps=CPU`` = all of
    them, the older spelling of ``--cpu-moe``). Without the second case such a
    model was estimated with every expert on the GPU.
    """
    try:
        raw = resolve_llama_cpp_option(state, "n_cpu_moe", None)
        if raw is not None:
            return int(raw)
    except (TypeError, ValueError):
        return None
    from ocabra.core.vram_planner import n_cpu_moe_from_override_tensor

    return n_cpu_moe_from_override_tensor(
        resolve_llama_cpp_option(state, "override_tensor", None), total_layers
    )


def estimate_llama_cpp_vram_from_config(
    state,
    *,
    default_gpu_layers: int,
    default_ctx_size: int = 4096,
    default_total_layers: int = 32,
) -> int:
    """Pre-load VRAM estimate for a llama.cpp GGUF model.

    The backend itself can only estimate *after* load (its option cache is
    populated in ``load()``), so the scheduler would otherwise see 0 MB, never
    check VRAM and never evict to make room.

    When the GGUF architecture is readable we compute weights + an *exact* KV
    cache for the configured ctx_size (KV scales with context, which a flat
    file-size multiplier misses entirely — e.g. a low-KV-head model over-reserves
    and a large-context one under-reserves). Hybrid SSM/attention models (e.g.
    qwen3.8:27b / arch "qwen35") additionally reserve a fixed-per-sequence
    "rs cache" for their recurrent layers — see ``plan_llama_cpp_vram_mb``.
    Otherwise we fall back to the on-disk GGUF size scaled by the
    offloaded-layer fraction plus a small overhead.
    Returns 0 for CPU-only (gpu_layers <= 0).
    """
    explicit_estimate = resolve_llama_cpp_option(state, "vram_estimate_mb", None)
    if explicit_estimate is not None:
        try:
            estimate_mb = int(explicit_estimate)
        except (TypeError, ValueError):
            estimate_mb = 0
        if estimate_mb > 0:
            return estimate_mb

    gpu_layers = resolve_llama_cpp_gpu_layers(state, default_gpu_layers)
    if gpu_layers <= 0:
        return 0

    model_path = resolve_llama_cpp_option(state, "model_path", None) or resolve_llama_cpp_option(
        state, "model_file", None
    )
    size_mb = 0
    if model_path:
        try:
            size_mb = int(Path(str(model_path)).stat().st_size / (1024 * 1024))
        except OSError:
            size_mb = 0
    if size_mb <= 0:
        return 0  # unknown size → let the backend/scheduler proceed as before

    try:
        ctx_size = int(resolve_llama_cpp_option(state, "ctx_size", default_ctx_size))
    except (TypeError, ValueError):
        ctx_size = default_ctx_size

    # KV-aware path: read the architecture straight from the GGUF header.
    arch = arch_from_gguf(str(model_path)) if model_path else None
    if arch is not None:
        # Size the KV cache to the configured cache_type (quantized KV halves/
        # quarters the footprint). Without this the estimate always assumes f16
        # and over-reserves ~2x, wrongly rejecting q8 long-context loads.
        cache_type = str(
            resolve_llama_cpp_option(state, "cache_type_k", None)
            or resolve_llama_cpp_option(state, "cache_type_v", None)
            or "f16"
        ).lower()
        kv_dtype_bytes = kv_dtype_bytes_for_cache_type(cache_type)
        try:
            parallel_slots = max(1, int(resolve_llama_cpp_option(state, "parallel_slots", 1)))
        except (TypeError, ValueError):
            parallel_slots = 1
        rs_rollback_snapshots = _resolve_rs_rollback_snapshots(state)
        est = plan_llama_cpp_vram_mb(
            arch,
            size_mb,
            ctx_size,
            gpu_layers=gpu_layers,
            kv_dtype_bytes=kv_dtype_bytes,
            parallel_slots=parallel_slots,
            rs_rollback_snapshots=rs_rollback_snapshots,
            n_cpu_moe=_resolve_n_cpu_moe(state, arch.layers),
        )
        if est > 0:
            return est

    # Fallback: weights-only heuristic (arch unreadable). ~1.08x leaves headroom
    # to trigger eviction without over-reserving so much that a model that really
    # fits gets bumped to tensor-parallel across a too-small GPU.
    try:
        total_layers = max(
            1, int(resolve_llama_cpp_option(state, "total_layers", default_total_layers))
        )
    except (TypeError, ValueError):
        total_layers = default_total_layers
    fraction = min(gpu_layers, total_layers) / total_layers
    return int(size_mb * fraction * 1.08)


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

from ocabra.core.vram_planner import (  # noqa: E402 - optional estimator section
    DEFAULT_OVERHEAD_MB,
    arch_from_hf_config,
    kv_vram_mb,
)

# vLLM's paged KV pool assertion at startup only checks whether one sequence at
# max_model_len fits — max_num_seqs just gates concurrent scheduling, it does
# NOT gate startup. So the reservation that keeps parity with what vLLM will
# refuse to launch without is weights + KV(max_model_len × 1) + overhead.
_VLLM_KV_DTYPE_BYTES: dict[str, float] = {
    "auto": 2.0,
    "": 2.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "fp8_e5m2": 1.0,
    "fp8_e4m3": 1.0,
    "int8": 1.0,
}


def _flatten_hf_config_for_arch(cfg: dict) -> dict:
    """Merge ``text_config`` into the top level so arch fields resolve.

    Multimodal HF configs (Gemma 3/4, Llama-3 vision, Qwen-VL, PaliGemma …)
    keep the transformer sub-block under ``text_config`` — the top level only
    has ``architectures`` and the multimodal glue. ``arch_from_hf_config``
    only reads top-level keys, so without this merge our arch resolution
    silently returns ``None`` for exactly the models with the biggest KV
    footprints. text_config wins on conflicts; the top level still lets
    callers override e.g. ``torch_dtype`` at the model root.
    """
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None
    if not text:
        return cfg
    merged = {**cfg, **text}
    # ``torch_dtype`` on the model root is authoritative when present.
    if cfg.get("torch_dtype") is not None:
        merged["torch_dtype"] = cfg["torch_dtype"]
    return merged


def _kv_tokens_for_max_model_len(cfg_for_arch: dict, layers: int, max_model_len: int) -> int:
    """Return the effective total KV tokens across all layers for a single
    sequence at ``max_model_len``.

    For hybrid sliding+full attention models (Gemma 3/4, Mistral 7B v0.2, …)
    the sliding layers only ever hold ``sliding_window`` tokens of KV each, so
    scaling every layer to ``max_model_len`` overestimates the KV footprint
    by an order of magnitude on 200k+ contexts — which would flip the fix
    from "reserve the honest KV" to "refuse loads that actually fit".

    Uses ``layer_types`` when present (the authoritative per-layer schedule
    published by HuggingFace for hybrid models) and falls back to counting
    every layer at max_model_len when it isn't. Both branches sum layers,
    so the caller stays free of per-layer bookkeeping.
    """
    layer_types = cfg_for_arch.get("layer_types")
    sliding_window = cfg_for_arch.get("sliding_window")
    try:
        sliding_window_int = int(sliding_window) if sliding_window else 0
    except (TypeError, ValueError):
        sliding_window_int = 0
    if not isinstance(layer_types, list) or sliding_window_int <= 0:
        return max(0, layers) * max(0, max_model_len)

    sliding_tokens = min(max_model_len, sliding_window_int)
    total = 0
    counted_full = 0
    counted_sliding = 0
    for kind in layer_types:
        name = str(kind).lower()
        if "sliding" in name:
            total += sliding_tokens
            counted_sliding += 1
        else:
            total += max_model_len
            counted_full += 1
    # ``layer_types`` occasionally omits a trailing layer or two; make sure
    # the missing ones count as full-attention so we don't undersell KV.
    remainder = max(0, layers - (counted_full + counted_sliding))
    total += remainder * max_model_len
    return total


def estimate_vllm_vram_from_config(
    state,
    *,
    models_dir: str | Path,
    default_gpu_memory_utilization: float,
    hf_cache_dir: str | Path | None = None,
) -> int:
    """Pre-load VRAM estimate for a vLLM model that accounts for the KV cache.

    The vLLM backend's ``get_vram_estimate_mb`` heuristic (``sum(safetensors)
    × 1.2``) undersells long-context configs because vLLM reserves KV up front
    proportionally to ``max_model_len``. That undersell means the scheduler
    frees weight-sized VRAM before load, then vLLM crashes at startup asking
    for KV that no eviction was ever going to make room for.

    This helper reads the HF ``config.json`` locally and computes weights +
    an effective-KV reserve + overhead. Effective KV honours hybrid
    sliding+full attention schedules (``layer_types`` + ``sliding_window``,
    published by HF for Gemma 3/4, Mistral, …): sliding layers only ever
    hold ``sliding_window`` tokens each, so scaling every layer to
    ``max_model_len`` overshoots by ~10× on 256k-context Gemmas and would
    flip the fix into "reject loads that actually fit".

    Returns 0 when the arch can't be read or ``max_model_len`` isn't
    configured — the caller then falls back to the existing heuristic.
    Also returns 0 when ``vram_estimate_mb`` is set in ``extra_config``,
    since that override signals the operator wants exact control.
    """
    if resolve_vllm_option(state, "vram_estimate_mb", None):
        return 0

    try:
        max_model_len = int(resolve_vllm_option(state, "max_model_len", 0) or 0)
    except (TypeError, ValueError):
        max_model_len = 0
    if max_model_len <= 0:
        return 0

    repo = getattr(state, "backend_model_id", None) or getattr(state, "model_id", None)
    if not repo:
        return 0
    model_dir = find_local_hf_dir(repo, models_dir, hf_cache_dir)
    if model_dir is None:
        return 0
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return 0
    flat_cfg = _flatten_hf_config_for_arch(cfg)
    arch = arch_from_hf_config(flat_cfg)
    if arch is None:
        return 0
    weights_mb = sum_hf_weight_files_mb(model_dir)
    if weights_mb <= 0:
        return 0

    kv_label = str(
        resolve_vllm_option(state, "kv_cache_dtype", "") or ""
    ).lower()
    kv_dtype_bytes = _VLLM_KV_DTYPE_BYTES.get(kv_label, 2.0)

    # ``gpu_memory_utilization`` doesn't enter the estimate directly — vLLM
    # will grab that fraction regardless — but keeping it in the signature
    # lets callers evolve toward reserving ``max(gmu × total, weights + KV)``
    # if they later want to over-reserve on GPUs where headroom is tight.
    _ = default_gpu_memory_utilization

    # Compute effective KV tokens (accounts for sliding-attention layers),
    # then convert to MB using the per-layer KV formula divided by layer
    # count so we can apply the schedule. ``kv_bytes_per_token`` treats all
    # layers as full-attention, so scale it by ``kv_tokens / (layers ×
    # max_model_len)`` — that fraction is 1.0 for pure full-attn models
    # and drops proportionally as sliding layers dominate.
    kv_tokens = _kv_tokens_for_max_model_len(flat_cfg, arch.layers, max_model_len)
    full_tokens = arch.layers * max_model_len
    if full_tokens <= 0:
        return 0
    full_kv_mb = kv_vram_mb(arch, max_model_len, kv_dtype_bytes)
    effective_kv_mb = full_kv_mb * (kv_tokens / full_tokens)

    return int(weights_mb + effective_kv_mb + DEFAULT_OVERHEAD_MB)
