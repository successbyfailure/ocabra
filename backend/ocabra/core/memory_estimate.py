"""Model memory estimation and the model-file/config introspection it needs.

Extracted from ``api/internal/models.py`` (REF-07). ``build_model_memory_estimate``
and ``estimate_llama_cpp_memory`` back ``POST /ocabra/models/{id}/memory-estimate``
and the model-editor preview; the smaller helpers (path resolution, context
length and tensor-parallel introspection) are shared with the capabilities
serialization that stays in ``api/internal/models.py``.

Lives in ``core/`` because the estimate math must not depend on the API layer,
and because the path/config helpers are consumed from both the endpoint and the
internal disk-size helpers (``api/internal/models_files.py``).

Note (REF-07 overlap for reconciliation): ``_resolve_local_model_path`` and
``_compute_path_size_bytes`` overlap conceptually with ``core/model_files.py``
(``find_local_hf_dir`` / ``sum_hf_weight_files_mb``). They were kept here rather
than merged into ``model_files.py`` because that module is owned by a parallel
extraction; a later pass can unify the HF-layout resolution.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import structlog

from ocabra.config import settings
from ocabra.core import fs_scan_cache
from ocabra.core import vram_planner as vp
from ocabra.core.backend_options import resolve_vllm_gpu_memory_utilization, snake_to_camel
from ocabra.core.model_ref import is_llama_cpp_family

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Model-file path resolution
# ---------------------------------------------------------------------------


def _resolve_local_model_path(model_id: str) -> Path | None:
    base = Path(settings.models_dir)
    direct = base / model_id
    if direct.exists():
        return direct

    if "::" in model_id:
        base_model_id, _variant = model_id.split("::", 1)
        base_direct = base / base_model_id
        if base_direct.exists():
            return base_direct

    # Hugging Face local layout: /data/models/huggingface/org--repo[--artifact-stem]
    if "::" in model_id:
        repo_id, variant_stem = model_id.split("::", 1)
        hf_dir_name = f"{repo_id.replace('/', '--')}--{variant_stem}"
    else:
        hf_dir_name = model_id.replace("/", "--")
    hf_layout = base / "huggingface" / hf_dir_name
    if hf_layout.exists():
        return hf_layout

    if "::" in model_id:
        repo_id, _variant_stem = model_id.split("::", 1)
        base_hf_layout = base / "huggingface" / repo_id.replace("/", "--")
        if base_hf_layout.exists():
            return base_hf_layout

    # GGUF / local file fallback for llama.cpp and bitnet style registrations.
    base_model_id = model_id.split("::", 1)[0]
    leaf = Path(base_model_id).name
    variant_stem = model_id.split("::", 1)[1] if "::" in model_id else ""
    candidate_stems = {leaf, Path(leaf).stem}
    if variant_stem:
        candidate_stems.add(variant_stem)
        candidate_stems.add(Path(variant_stem).stem)
    # Cached walk: this runs for every model in a listing, and the walk covers
    # the whole models directory (see fs_scan_cache for the invalidation).
    gguf_candidates = [
        path
        for path in fs_scan_cache.glob_files(base, "*.gguf")
        if (
            path.name == leaf
            or any(path.stem == stem or path.stem.endswith(stem) for stem in candidate_stems)
        )
        and path.exists()
    ]
    if gguf_candidates:
        return max(gguf_candidates, key=lambda path: path.stat().st_mtime)

    # Optional HF cache layout fallback.
    hf_cache_dir = (settings.hf_cache_dir or "").strip()
    if hf_cache_dir:
        cache_root = (
            Path(hf_cache_dir) / "hub" / f"models--{model_id.split('::', 1)[0].replace('/', '--')}"
        )
        snapshots_dir = cache_root / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            refs_main = cache_root / "refs" / "main"
            if refs_main.exists():
                try:
                    commit = refs_main.read_text(encoding="utf-8").strip()
                except Exception:
                    commit = ""
                if commit:
                    candidate = snapshots_dir / commit
                    if candidate.exists():
                        return candidate
            candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)

    return None


def _resolve_extra_config_path(extra_config: dict | None, *keys: str) -> Path | None:
    if not isinstance(extra_config, dict):
        return None
    allowed_bases = [Path(settings.models_dir or "/data/models")]
    if settings.ollama_shared_models_dir:
        allowed_bases.append(Path(settings.ollama_shared_models_dir))
    # llama.cpp stores its load options — including model_path — nested
    # under extra_config["llama_cpp"], not at the top level like other
    # backends' extra_config. Check both so GGUF weights reused from
    # Ollama's blob store (hence the extra allowed base above, outside
    # models_dir) still resolve for size/memory-estimate lookups.
    nested = extra_config.get("llama_cpp")
    nested = nested if isinstance(nested, dict) else {}
    for key in keys:
        raw = extra_config.get(key) or nested.get(key)
        if not raw or not isinstance(raw, str):
            continue
        path = Path(raw)
        if path.exists() and any(_is_path_within_base(path, base) for base in allowed_bases):
            return path
    return None


def _resolve_tensorrt_engine_path(engine_name: str) -> Path | None:
    if settings.tensorrt_llm_engines_dir:
        base = Path(settings.tensorrt_llm_engines_dir)
    else:
        models_container = settings.tensorrt_llm_docker_models_mount_container or "/data/models"
        base = Path(models_container) / "tensorrt_llm"

    candidate = base / engine_name
    if candidate.exists():
        return candidate
    engine_subdir = candidate / "engine"
    if engine_subdir.exists():
        return engine_subdir
    return None


def _compute_path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _is_path_within_base(path: Path, base: Path) -> bool:
    try:
        return path.resolve(strict=False).is_relative_to(base.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False


def _read_first_positive_int(path: Path, keys: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    for key in keys:
        value = payload.get(key)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return numeric
    return 0


# ---------------------------------------------------------------------------
# Config introspection (context length / tensor-parallel / vLLM utilization)
# ---------------------------------------------------------------------------


def _resolve_context_length_fallback(state) -> int:
    extra_config = state.extra_config if isinstance(state.extra_config, dict) else {}

    for key in ("context_length", "max_model_len", "ctx_size"):
        value = extra_config.get(key)
        if isinstance(value, int) and value > 0:
            return value

    for section_name in ("vllm", "strata"):
        section = extra_config.get(section_name)
        if isinstance(section, dict):
            for key in ("max_model_len", "context_length"):
                value = section.get(key)
                if isinstance(value, int) and value > 0:
                    return value

    backend_model_id = getattr(state, "backend_model_id", "")
    model_path = _resolve_extra_config_path(
        extra_config, "model_path", "base_model_id"
    ) or _resolve_local_model_path(backend_model_id)
    if model_path is None:
        return 0

    for file_name, keys in (
        (
            "config.json",
            (
                "max_position_embeddings",
                "max_seq_len",
                "model_max_length",
                "max_sequence_length",
                "seq_length",
                "n_positions",
                "context_length",
                "max_context_length",
            ),
        ),
        ("tokenizer_config.json", ("model_max_length",)),
    ):
        value = _read_first_positive_int(model_path / file_name, keys)
        if value > 0:
            return value

    return 0


def _resolve_requested_context_length(state, extra_config: dict[str, Any]) -> int | None:
    for key in ("context_length", "max_model_len", "ctx_size"):
        for candidate in (key, snake_to_camel(key)):
            value = extra_config.get(candidate)
            if isinstance(value, int) and value > 0:
                return value
    for section in ("vllm", "sglang", "strata", "llama_cpp", "bitnet", "tensorrt_llm"):
        nested = extra_config.get(section)
        if isinstance(nested, dict):
            for key in ("context_length", "max_model_len", "ctx_size"):
                for candidate in (key, snake_to_camel(key)):
                    value = nested.get(candidate)
                    if isinstance(value, int) and value > 0:
                        return value
    fallback = _resolve_context_length_fallback(state)
    return fallback if fallback > 0 else None


def _resolve_tensor_parallel_size(state, extra_config: dict[str, Any]) -> int | None:
    for section in ("vllm", "sglang"):
        nested = extra_config.get(section)
        if isinstance(nested, dict):
            value = nested.get("tensor_parallel_size") or nested.get("tensorParallelSize")
            if isinstance(value, int) and value > 0:
                return value
    if state.backend_type == "tensorrt_llm":
        engine_path = _resolve_extra_config_path(
            extra_config, "engine_dir"
        ) or _resolve_tensorrt_engine_path(state.backend_model_id)
        if engine_path is not None and engine_path.exists():
            scan_dir = engine_path / "engine" if (engine_path / "engine").is_dir() else engine_path
            try:
                cfg = json.loads((scan_dir / "config.json").read_text())
                mapping = cfg.get("pretrained_config", {}).get("mapping", {})
                return int(mapping.get("tp_size") or mapping.get("world_size") or 1)
            except Exception:  # noqa: BLE001 — unreadable engine config → assume tp=1
                logger.debug(
                    "tensorrt_tp_size_probe_failed",
                    model_id=getattr(state, "model_id", None),
                    exc_info=True,
                )
                return 1
    return 1


def _vllm_gpu_memory_utilization(extra_config: dict[str, Any]) -> float:
    """vLLM ``gpu_memory_utilization`` read from ``extra_config`` (nested or top
    level, snake_case or camelCase), clamped with the shared settings fallback.

    DUP-11: the clamp and settings fallback are delegated to
    ``core.backend_options.resolve_vllm_gpu_memory_utilization`` (single source),
    replacing a third local copy of that logic that lived in this module.
    """
    candidates: list[Any] = []
    vllm_config = extra_config.get("vllm")
    if isinstance(vllm_config, dict):
        candidates.append(vllm_config.get("gpu_memory_utilization"))
        candidates.append(vllm_config.get("gpuMemoryUtilization"))
    candidates.append(extra_config.get("gpu_memory_utilization"))
    candidates.append(extra_config.get("gpuMemoryUtilization"))
    value = next((candidate for candidate in candidates if candidate is not None), None)
    return resolve_vllm_gpu_memory_utilization(value)


# ---------------------------------------------------------------------------
# Memory estimate
# ---------------------------------------------------------------------------


async def _build_model_memory_estimate(
    request: Any,
    state,
    extra_config: dict[str, Any],
    preferred_gpu: int | None,
    run_probe: bool,
) -> dict[str, Any]:
    worker_pool = getattr(request.app.state, "worker_pool", None)
    gpu_manager = getattr(request.app.state, "gpu_manager", None)
    backend = await worker_pool.get_backend(state.backend_type) if worker_pool else None

    gpu_index = (
        preferred_gpu
        if preferred_gpu is not None
        else state.preferred_gpu
        if getattr(state, "preferred_gpu", None) is not None
        else settings.default_gpu_index
    )
    gpu_state = None
    if gpu_manager is not None:
        try:
            gpu_state = await gpu_manager.get_state(gpu_index)
        except (KeyError, IndexError):
            pass
    total_vram_mb = int(getattr(gpu_state, "total_vram_mb", 0) or 0) or None
    free_vram_mb = int(getattr(gpu_state, "free_vram_mb", 0) or 0) or None

    estimate = {
        "backend_type": state.backend_type,
        "gpu_index": gpu_index,
        "total_vram_mb": total_vram_mb,
        "free_vram_mb": free_vram_mb,
        "budget_vram_mb": None,
        # Can fall back to finding the model on disk and reading its config.json.
        "requested_context_length": await asyncio.to_thread(
            _resolve_requested_context_length, state, extra_config
        ),
        "estimated_weights_mb": None,
        "estimated_engine_mb_per_gpu": None,
        "estimated_kv_cache_mb": None,
        "estimated_max_context_length": None,
        "model_loading_memory_mb": None,
        "maximum_concurrency": None,
        "tensor_parallel_size": _resolve_tensor_parallel_size(state, extra_config),
        "fits_current_gpu": None,
        "engine_present": None,
        "source": "heuristic",
        "status": "ok",
        "warning": None,
        "notes": [],
    }

    if backend is None:
        estimate["status"] = "warning"
        estimate["warning"] = "No backend runtime available to calculate memory estimates."
        return estimate

    heuristic_mb = int(
        await backend.get_vram_estimate_mb(state.backend_model_id, extra_config=extra_config) or 0
    )

    if state.backend_type == "vllm":
        gpu_util = _vllm_gpu_memory_utilization(extra_config)
        budget_vram_mb = int((total_vram_mb or 0) * gpu_util) if total_vram_mb is not None else None
        estimate.update(
            {
                "budget_vram_mb": budget_vram_mb,
                "estimated_weights_mb": heuristic_mb or None,
                "estimated_kv_cache_mb": None,
                "fits_current_gpu": None,
            }
        )
        estimate["notes"] = [
            "La estimación rápida de vLLM aproxima el peso cargado y la reserva objetivo de VRAM.",
            "La memoria de KV cache y el contexto máximo real dependen del engine; usa el probe para una validación fiable.",
        ]
        if run_probe and hasattr(backend, "estimate_memory_profile"):
            profile = await backend.estimate_memory_profile(
                state.backend_model_id,
                gpu_index=gpu_index,
                extra_config=extra_config,
            )
            estimate.update(
                {
                    "source": "runtime_probe",
                    "model_loading_memory_mb": profile.get("model_loading_memory_mb"),
                    "estimated_kv_cache_mb": profile.get("available_kv_cache_mb")
                    or estimate["estimated_kv_cache_mb"],
                    "estimated_max_context_length": profile.get("estimated_max_model_len")
                    or profile.get("gpu_kv_cache_tokens"),
                    "maximum_concurrency": profile.get("maximum_concurrency"),
                }
            )
            if profile.get("requested_context_length") and not estimate["requested_context_length"]:
                estimate["requested_context_length"] = profile.get("requested_context_length")
            if profile.get("status") != "ok":
                estimate["status"] = "error"
                estimate["warning"] = profile.get("error") or "vLLM runtime probe failed."
                if profile.get("estimated_max_model_len"):
                    estimate["notes"].append(
                        f"El engine estima un máximo de {profile['estimated_max_model_len']} tokens con esta configuración."
                    )
            else:
                estimate["notes"].append(
                    "Probe real de vLLM completado; las cifras de KV cache y contexto vienen del engine."
                )
        return estimate

    if is_llama_cpp_family(state.backend_type):
        # Sprint 17.2 — deterministic GGUF estimator. In a thread: it locates
        # the GGUF on disk and, the first time per file, parses its header
        # (~0.1 s on a 20 GB model).
        return await asyncio.to_thread(
            _estimate_llama_cpp_memory,
            estimate=estimate,
            state=state,
            extra_config=extra_config,
            heuristic_mb=heuristic_mb,
            total_vram_mb=total_vram_mb,
        )

    if state.backend_type == "tensorrt_llm":
        engine_path = _resolve_extra_config_path(
            extra_config, "engine_dir"
        ) or _resolve_tensorrt_engine_path(state.backend_model_id)
        tp_size = _resolve_tensor_parallel_size(state, extra_config)
        estimate.update(
            {
                "estimated_engine_mb_per_gpu": heuristic_mb or None,
                "engine_present": bool(engine_path and engine_path.exists()),
                "fits_current_gpu": (
                    heuristic_mb <= total_vram_mb
                    if total_vram_mb is not None and heuristic_mb > 0
                    else None
                ),
                "tensor_parallel_size": tp_size,
            }
        )
        if not estimate["engine_present"]:
            estimate["status"] = "error"
            estimate["warning"] = "Engine directory not found for this TensorRT-LLM model."
        estimate["notes"] = [
            "La estimación de TensorRT-LLM usa el tamaño real del engine por GPU y el tp_size detectado en config.json.",
        ]
        return estimate

    estimate.update(
        {
            "estimated_weights_mb": heuristic_mb or None,
            "fits_current_gpu": (
                heuristic_mb <= total_vram_mb
                if total_vram_mb is not None and heuristic_mb > 0
                else None
            ),
            "notes": [
                "Estimación heurística basada en el tamaño de los artefactos locales del modelo."
            ],
        }
    )
    return estimate


def _estimate_llama_cpp_memory(
    *,
    estimate: dict[str, Any],
    state,
    extra_config: dict[str, Any],
    heuristic_mb: int,
    total_vram_mb: int | None,
) -> dict[str, Any]:
    """Deterministic GGUF VRAM estimator for llama.cpp models.

    Shares its architecture/KV/rs_cache/MTP-draft-overhead math with
    ``vram_planner.py`` — the same module the scheduler's pre-load admission
    check and the runtime-presets panel use — so this "Memoria y contexto"
    panel and the rest of the UI never disagree about the same model. Falls
    back to the legacy heuristic when the GGUF file cannot be located or
    parsed (e.g. only-on-disk via Ollama, or remote model) or when the
    configured ``gpu_layers`` is 0 (CPU-only, nothing to size on the GPU).
    """
    from ocabra.core.model_manager_helpers import (
        _LLAMA_CPP_DEFAULT_DRAFT_N_MAX,
        _LLAMA_CPP_SPEC_TYPES_NEEDING_RS_ROLLBACK,
    )
    from ocabra.schemas.backend_load import LlamaCppLoadConfig

    def _fallback_to_heuristic(
        notes: list[str], *, status: str = "ok", warning: str | None = None
    ) -> dict[str, Any]:
        estimate.update(
            {
                "estimated_weights_mb": heuristic_mb or None,
                "fits_current_gpu": (
                    heuristic_mb <= total_vram_mb
                    if total_vram_mb is not None and heuristic_mb > 0
                    else None
                ),
                "status": status,
                "warning": warning,
                "notes": notes,
            }
        )
        return estimate

    nested = extra_config.get("llama_cpp") if isinstance(extra_config, dict) else None
    payload = nested if isinstance(nested, dict) else {}
    try:
        load_config = LlamaCppLoadConfig.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — surface as warning, keep estimate
        return _fallback_to_heuristic(
            ["Estimación heurística (config inválida)."],
            status="warning",
            warning=f"Configuración llama.cpp inválida: {exc}",
        )

    gguf_path = _resolve_extra_config_path(extra_config, "model_path") or _resolve_local_model_path(
        state.backend_model_id
    )
    gguf_file: Path | None = None
    if gguf_path is not None:
        if gguf_path.is_file():
            # model_path points straight at the weights file — trust it
            # regardless of extension. Ollama's blob store names files by
            # content hash ("sha256-<hash>", no ".gguf" suffix) when
            # llama_cpp reuses weights pulled through Ollama.
            gguf_file = gguf_path
        elif gguf_path.is_dir():
            ggufs = sorted(gguf_path.rglob("*.gguf"))
            if ggufs:
                gguf_file = max(ggufs, key=lambda p: p.stat().st_size)

    arch = vp.arch_from_gguf(str(gguf_file)) if gguf_file is not None else None
    if arch is None or gguf_file is None:
        return _fallback_to_heuristic(["Estimación heurística: GGUF no encontrado o no parseable."])

    gpu_layers = (
        load_config.gpu_layers
        if load_config.gpu_layers is not None
        else settings.llama_cpp_gpu_layers
    )
    if gpu_layers <= 0:
        return _fallback_to_heuristic(["Estimación heurística: gpu_layers=0 (modelo en CPU)."])

    ctx_size = int(load_config.ctx_size or settings.llama_cpp_ctx_size)
    parallel_slots = max(1, load_config.parallel_slots or settings.llama_cpp_parallel_slots)
    kv_dtype_bytes = vp.kv_dtype_bytes_for_cache_type(
        load_config.cache_type_k or load_config.cache_type_v
    )

    rs_rollback_snapshots = 0
    if load_config.speculative is not None:
        spec_type = (load_config.speculative.spec_type or "").lower()
        if spec_type in _LLAMA_CPP_SPEC_TYPES_NEEDING_RS_ROLLBACK:
            rs_rollback_snapshots = (
                load_config.speculative.draft_n
                if load_config.speculative.draft_n is not None
                else _LLAMA_CPP_DEFAULT_DRAFT_N_MAX
            )

    file_size_mb = gguf_file.stat().st_size / (1024 * 1024)
    total_layers = max(1, arch.layers)
    offloaded = max(0, min(gpu_layers, total_layers))
    frac = offloaded / total_layers
    n_cpu_moe = load_config.n_cpu_moe
    if n_cpu_moe is None:
        n_cpu_moe = vp.n_cpu_moe_from_override_tensor(load_config.override_tensor, total_layers)
    moe_cpu_mb = vp.moe_cpu_weights_mb(
        file_size_mb,
        total_layers,
        offloaded,
        n_cpu_moe,
        expert_mb_by_layer=arch.expert_mb_by_layer,
    )
    weights_mb = max(0.0, file_size_mb * frac - moe_cpu_mb)
    kv_arch = (
        arch
        if offloaded == total_layers
        else vp.ModelArch(
            offloaded,
            arch.n_kv_heads,
            arch.key_length,
            arch.value_length,
            arch.hidden_size,
            arch.context_length,
            arch.weight_dtype_bytes,
            int(arch.kv_elements_per_token_override * frac)
            if arch.kv_elements_per_token_override
            else 0,
            arch.rs_cache_mb_per_slot * frac,
            arch.fixed_load_overhead_mb * frac,
        )
    )
    draft_overhead_mb = (
        vp.mtp_draft_overhead_mb(parallel_slots) if rs_rollback_snapshots > 0 else 0.0
    )
    rs_mb = vp.rs_cache_vram_mb(
        kv_arch, slots=parallel_slots, rollback_snapshots=rs_rollback_snapshots
    )
    fixed_mb = vp.fixed_overhead_vram_mb(kv_arch)
    kv_mb = vp.kv_vram_mb(kv_arch, max(1, ctx_size), kv_dtype_bytes)
    overhead_mb = vp.DEFAULT_OVERHEAD_MB + draft_overhead_mb
    total_mb = int(weights_mb + kv_mb + rs_mb + fixed_mb + overhead_mb)

    estimated_max_context_length = None
    if total_vram_mb is not None:
        estimated_max_context_length = (
            vp.max_context_tokens(
                total_vram_mb,
                weights_mb,
                kv_arch,
                slots=parallel_slots,
                kv_dtype_bytes=kv_dtype_bytes,
                overhead_mb=overhead_mb,
                rs_rollback_snapshots=rs_rollback_snapshots,
            )
            or None
        )

    notes = [
        "Estimación determinística desde el header GGUF (arquitectura, KV cache, "
        "rs_cache híbrido y overhead de draft MTP, cuando aplica).",
        f"Desglose en MB — pesos: {int(weights_mb)} · KV: {int(kv_mb)}"
        + (f" · rs_cache: {int(rs_mb)}" if rs_mb else "")
        + (f" · overhead fijo: {int(fixed_mb)}" if fixed_mb else "")
        + (f" · overhead MTP: {int(draft_overhead_mb)}" if draft_overhead_mb else "")
        + f" · overhead base: {vp.DEFAULT_OVERHEAD_MB} · total: {total_mb}.",
    ]
    if moe_cpu_mb:
        notes.append(
            f"Expertos MoE de {n_cpu_moe} capas en RAM (~{int(moe_cpu_mb)} MB fuera "
            + (
                "de la GPU), medidos en el GGUF."
                if arch.expert_mb_by_layer
                else "de la GPU): estimación aproximada."
            )
        )
    if load_config.cache_type_k or load_config.cache_type_v:
        notes.append(
            "KV cache cuantizado: K="
            f"{load_config.cache_type_k or 'f16'} · V={load_config.cache_type_v or 'f16'}."
        )
    if rs_rollback_snapshots > 0:
        notes.append(
            f"Decodificación especulativa autoreferente ({load_config.speculative.spec_type}): "
            f"reserva adicional de rs_cache y buffer de draft para {rs_rollback_snapshots} tokens."
        )

    estimate.update(
        {
            "estimated_weights_mb": int(weights_mb),
            "estimated_kv_cache_mb": int(kv_mb),
            "breakdown_mb": {
                k: int(v)
                for k, v in {
                    "weights": weights_mb,
                    "kv": kv_mb,
                    "rs_cache": rs_mb,
                    "fixed": fixed_mb,
                    "draft": draft_overhead_mb,
                    "overhead": vp.DEFAULT_OVERHEAD_MB,
                }.items()
                if v
            },
            "estimated_max_context_length": estimated_max_context_length,
            # ctx_size is llama.cpp's TOTAL KV (split across slots, or the
            # shared pool): compare it with this, not with the per-slot max.
            "estimated_max_total_context": (
                estimated_max_context_length * parallel_slots
                if estimated_max_context_length
                else None
            ),
            "model_loading_memory_mb": total_mb,
            # CPU side of a MoE split (experts left in RAM), so the preview can
            # weigh it against the host's RAM like a llama-fit projection.
            "host_ram_mb": int(moe_cpu_mb) or None,
            "fits_current_gpu": (total_mb <= total_vram_mb if total_vram_mb is not None else None),
            "source": "gguf_header",
            "notes": notes,
        }
    )
    return estimate
