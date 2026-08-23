"""Pure helper functions for model manager lifecycle concerns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ocabra.core.vram_planner import arch_from_gguf, plan_llama_cpp_vram_mb

PRISMML_DEFAULT_GPU_LAYERS = 99


def compute_worker_key(base_model_id: str, load_overrides: dict | None) -> str:
    """Derive the stable worker key for a model profile."""
    if not load_overrides:
        return base_model_id
    canonical = json.dumps(load_overrides, sort_keys=True, separators=(",", ":"))
    short_hash = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"{base_model_id}::{short_hash}"


def build_diarized_extra_config(base_extra_config: dict | None) -> dict:
    """Build extra_config with diarization enabled, used by profile creation."""
    merged = dict(base_extra_config or {})
    merged["diarization_enabled"] = True
    whisper_cfg = merged.get("whisper") if isinstance(merged.get("whisper"), dict) else {}
    merged["whisper"] = {**whisper_cfg, "diarizationEnabled": True}
    return merged


def resolve_bitnet_option(state, key: str, default: int) -> int:
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("bitnet") if isinstance(extra.get("bitnet"), dict) else None
    if nested and key in nested:
        return int(nested[key])
    if key in extra:
        return int(extra[key])
    return int(default)


def is_prismml_bitnet_state(state) -> bool:
    """Detect Bonsai/PrismML before load from ids or configured GGUF path."""
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("bitnet") if isinstance(extra.get("bitnet"), dict) else {}
    markers = (
        getattr(state, "model_id", ""),
        getattr(state, "backend_model_id", ""),
        extra.get("model_path", ""),
        nested.get("model_path", ""),
    )
    text = " ".join(str(value).lower() for value in markers if value)
    return "bonsai" in text or "q1_0" in text or "prismml" in text


def resolve_bitnet_gpu_layers(state, default_gpu_layers: int) -> int:
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("bitnet") if isinstance(extra.get("bitnet"), dict) else None
    if nested and "gpu_layers" in nested:
        return int(nested["gpu_layers"])
    if "gpu_layers" in extra:
        return int(extra["gpu_layers"])
    if is_prismml_bitnet_state(state):
        return PRISMML_DEFAULT_GPU_LAYERS
    return int(default_gpu_layers)


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
            kv_dtype_bytes = {
                "q8_0": 1.0,
                "q8": 1.0,
                "q4_0": 0.5,
                "q4_1": 0.5,
                "q4": 0.5,
            }.get(cache_type, 2.0)
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


def resolve_llama_cpp_option(state, key: str, default):
    """Read a llama.cpp option from ``extra_config['llama_cpp'][key]`` or the
    top level (accepting snake_case), mirroring the backend's own resolution."""
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("llama_cpp") if isinstance(extra.get("llama_cpp"), dict) else None
    if nested and key in nested:
        return nested[key]
    if key in extra:
        return extra[key]
    return default


def resolve_llama_cpp_gpu_layers(state, default_gpu_layers: int) -> int:
    try:
        return int(resolve_llama_cpp_option(state, "gpu_layers", default_gpu_layers))
    except (TypeError, ValueError):
        return int(default_gpu_layers)


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
    and a large-context one under-reserves). Otherwise we fall back to the on-disk
    GGUF size scaled by the offloaded-layer fraction plus a small overhead.
    Returns 0 for CPU-only (gpu_layers <= 0).
    """
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
        kv_dtype_bytes = {
            "f16": 2.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "q8_0": 1.0,
            "q8": 1.0,
            "q5_0": 0.65,
            "q5_1": 0.65,
            "q4_0": 0.5,
            "q4_1": 0.5,
            "q4": 0.5,
            "iq4_nl": 0.5,
        }.get(cache_type, 2.0)
        est = plan_llama_cpp_vram_mb(
            arch, size_mb, ctx_size, gpu_layers=gpu_layers, kv_dtype_bytes=kv_dtype_bytes
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


def _resolve_vllm_option(state, key: str, default=None):
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    nested = extra.get("vllm") if isinstance(extra.get("vllm"), dict) else None
    if nested and key in nested:
        return nested[key]
    if key in extra:
        return extra[key]
    return default


def _find_local_hf_dir(
    repo: str,
    models_dir: str | Path,
    hf_cache_dir: str | Path | None = None,
) -> Path | None:
    """Best-effort lookup of a local HuggingFace snapshot dir for ``repo``.

    Must stay in sync with ``VLLMBackend._resolve_local_model_dir`` — otherwise
    a model the backend can load fine looks "unknown" to the estimator, which
    returns 0, and the scheduler falls back to the weights-only heuristic we
    are trying to replace. Checks (same priority as the vLLM backend):

      1. ``<models_dir>/<repo>`` — direct legacy layout
      2. ``<models_dir>/huggingface/<flat>`` — flat HF layout
      3. Snapshots under ``<models_dir>/huggingface/models--<flat>/snapshots/*``
      4. ``<hf_cache_dir>/hub/models--<flat>/snapshots/<commit>`` — canonical
         HuggingFace cache layout populated by ``snapshot_download``.

    For the HF cache we prefer the commit ``refs/main`` points at (matching the
    snapshot the backend will actually load); otherwise the newest snapshot dir.
    """
    if not repo:
        return None
    # Strip the ``::<hash>`` suffix that profile worker_keys carry — the base
    # model on disk is the same regardless of load_overrides, and any lookup
    # that keeps the suffix misses the snapshot and forces the caller back to
    # the weights-only heuristic we're trying to replace.
    if "::" in repo:
        repo = repo.split("::", 1)[0]
    flat = repo.replace("/", "--")

    base = Path(models_dir)
    direct = base / repo
    if (direct / "config.json").is_file():
        return direct
    hf_base = base / "huggingface"
    for c in (hf_base / flat, hf_base / repo):
        if (c / "config.json").is_file():
            return c
    for snaps in list(hf_base.glob(f"models--{flat}/snapshots/*")) + list(
        hf_base.glob(f"*{flat}*/snapshots/*")
    ):
        if (snaps / "config.json").is_file():
            return snaps

    if hf_cache_dir:
        cache_root = Path(hf_cache_dir) / "hub" / f"models--{flat}"
        snapshots = cache_root / "snapshots"
        if snapshots.is_dir():
            commit = ""
            ref_main = cache_root / "refs" / "main"
            if ref_main.is_file():
                try:
                    commit = ref_main.read_text(encoding="utf-8").strip()
                except OSError:
                    commit = ""
            if commit:
                snap = snapshots / commit
                if (snap / "config.json").is_file():
                    return snap
            candidates = [p for p in snapshots.iterdir() if p.is_dir()]
            if candidates:
                newest = max(candidates, key=lambda p: p.stat().st_mtime)
                if (newest / "config.json").is_file():
                    return newest
    return None


def _sum_hf_weight_files_mb(model_dir: Path) -> float:
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for f in model_dir.glob(pattern):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


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
    if _resolve_vllm_option(state, "vram_estimate_mb", None):
        return 0

    try:
        max_model_len = int(_resolve_vllm_option(state, "max_model_len", 0) or 0)
    except (TypeError, ValueError):
        max_model_len = 0
    if max_model_len <= 0:
        return 0

    repo = getattr(state, "backend_model_id", None) or getattr(state, "model_id", None)
    if not repo:
        return 0
    model_dir = _find_local_hf_dir(repo, models_dir, hf_cache_dir)
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
    weights_mb = _sum_hf_weight_files_mb(model_dir)
    if weights_mb <= 0:
        return 0

    kv_label = str(
        _resolve_vllm_option(state, "kv_cache_dtype", "") or ""
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
