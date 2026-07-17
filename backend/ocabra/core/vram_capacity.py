"""Resolve a model's architecture + weight footprint from whatever metadata its
backend exposes, so the capacity planner can size KV against the real model.

I/O lives here (HTTP to Ollama, reading GGUF/config.json/safetensors off disk);
the pure math is in :mod:`ocabra.core.vram_planner`.
"""

from __future__ import annotations

import json
from pathlib import Path

import copy

import httpx

from ocabra.config import settings
from ocabra.core import vram_planner as vp
from ocabra.core.vram_planner import (
    ModelArch,
    arch_from_gguf,
    arch_from_hf_config,
    arch_from_ollama_model_info,
)

_MB = 1024 * 1024

# kv_dtype label -> (bytes per KV element, vLLM kv_cache_dtype or None to leave default)
KV_DTYPES: dict[str, tuple[float, str | None]] = {
    "fp16": (2.0, None),
    "bf16": (2.0, None),
    "fp8": (1.0, "fp8"),
    "q8": (1.0, None),
    "q4": (0.5, None),
}

# Backends that don't have a context-scaling KV cache (encoders, diffusion, TTS).
_NON_KV_BACKENDS = {"whisper", "diffusers", "tts", "chatterbox", "comfyui", "a1111"}


class ArchResolution:
    __slots__ = ("arch", "weights_mb", "note")

    def __init__(self, arch: ModelArch | None, weights_mb: float, note: str):
        self.arch = arch
        self.weights_mb = weights_mb
        self.note = note


def _find_hf_model_dir(repo: str) -> Path | None:
    """Locate a local HF model dir (config.json) for a repo id like ``Qwen/Qwen3-8B``."""
    if not repo:
        return None
    base = Path(settings.models_dir) / "huggingface"
    flat = repo.replace("/", "--")
    candidates = [base / flat, base / repo]
    for c in candidates:
        if (c / "config.json").is_file():
            return c
    # HF cache layout: models--Qwen--Qwen3-8B/snapshots/<hash>/config.json
    for snaps in list(base.glob(f"models--{flat}/snapshots/*")) + list(
        base.glob(f"*{flat}*/snapshots/*")
    ):
        if (snaps / "config.json").is_file():
            return snaps
    return None


def _sum_weight_files_mb(model_dir: Path) -> float:
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for f in model_dir.glob(pattern):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / _MB


async def _resolve_ollama(name: str) -> ArchResolution:
    base = settings.ollama_base_url.rstrip("/")
    arch: ModelArch | None = None
    weights_mb = 0.0
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            show = (await client.post(f"{base}/api/show", json={"model": name})).json()
            family = (show.get("details") or {}).get("family", "")
            arch = arch_from_ollama_model_info(show.get("model_info") or {}, family)
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        try:
            tags = (await client.get(f"{base}/api/tags")).json()
            for m in tags.get("models", []):
                if m.get("name") == name or m.get("model") == name:
                    weights_mb = int(m.get("size", 0)) / _MB
                    break
        except (httpx.HTTPError, ValueError):
            pass
    note = "" if arch else "no se pudo leer la arquitectura de Ollama /api/show"
    return ArchResolution(arch, weights_mb, note)


def _resolve_llama_cpp(state) -> ArchResolution:
    cfg = (state.extra_config or {}).get("llama_cpp", {}) if isinstance(state.extra_config, dict) else {}
    model_path = cfg.get("model_path") or cfg.get("model_file")
    if not model_path:
        return ArchResolution(None, 0.0, "sin model_path del GGUF")
    p = Path(str(model_path))
    weights_mb = 0.0
    try:
        weights_mb = p.stat().st_size / _MB
    except OSError:
        pass
    arch = arch_from_gguf(str(p))
    note = "" if arch else "no se pudo leer el header del GGUF"
    return ArchResolution(arch, weights_mb, note)


def _resolve_hf(state) -> ArchResolution:
    repo = state.backend_model_id or state.model_id
    model_dir = _find_hf_model_dir(repo)
    if model_dir is None:
        return ArchResolution(None, 0.0, f"no se encontró config.json local para {repo}")
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return ArchResolution(None, 0.0, "config.json ilegible")
    weights_mb = _sum_weight_files_mb(model_dir)
    # Encoders (rerankers, embeddings, classifiers) process the whole sequence in
    # one pass — no per-step KV cache that scales with context — so the planner
    # doesn't apply to them even though they have transformer layers.
    archs = cfg.get("architectures") or []
    generative = any(
        any(m in a for m in ("CausalLM", "ConditionalGeneration", "LMHeadModel"))
        for a in archs
    )
    if archs and not generative:
        return ArchResolution(None, weights_mb, "modelo encoder/embedding/rerank: sin KV generativo que escale con el contexto")
    arch = arch_from_hf_config(cfg)
    note = "" if arch else "config.json sin campos de arquitectura reconocibles"
    return ArchResolution(arch, weights_mb, note)


def _use_case_config(state) -> dict | None:
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    uc = extra.get("use_case")
    return uc if isinstance(uc, dict) else None


async def resolve_use_case(state, gpu_total_mb: float) -> dict | None:
    """Resolve the ``use_case`` block (target context + slots + kv_dtype) against
    the model's capacity on ``gpu_total_mb``. Returns the plan (effective context,
    warnings, computed backend flags) or None when there's no use_case to apply.
    """
    uc = _use_case_config(state)
    if uc is None:
        return None
    res = await resolve_arch_and_weights(state)
    if res.arch is None:
        return {"applied": False, "note": res.note or "arquitectura no disponible"}

    backend = (state.backend_type or "").lower()
    kv_label = str(uc.get("kv_dtype", "fp16")).lower()
    kv_bytes, vllm_kv = KV_DTYPES.get(kv_label, (2.0, None))
    slots = max(1, int(uc.get("slots", 1) or 1))
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    gpu_mem_util = float(
        (extra.get("vllm") or {}).get("gpu_memory_utilization")
        or settings.vllm_gpu_memory_utilization
    )

    plan = vp.plan_use_case(
        backend, res.arch, res.weights_mb,
        gpu_total_mb=gpu_total_mb,
        requested_context=uc.get("context"),
        slots=slots,
        gpu_memory_utilization=gpu_mem_util,
        kv_dtype_bytes=kv_bytes,
    )
    plan["applied"] = True
    plan["kv_dtype"] = kv_label
    plan["vllm_kv_cache_dtype"] = vllm_kv
    return plan


def apply_use_case_flags(extra_config: dict, backend_type: str, plan: dict) -> dict:
    """Return a copy of ``extra_config`` with the resolved use-case translated into
    the concrete backend load flags. Only touches keys the use case owns; runtime
    use (not persisted), so an explicit config in the DB is unchanged on disk.
    """
    backend = (backend_type or "").lower()
    ctx = plan.get("effective_context")
    slots = plan.get("slots", 1)
    if not ctx or not plan.get("applied"):
        return extra_config
    merged = copy.deepcopy(extra_config) if isinstance(extra_config, dict) else {}

    if backend in ("llama_cpp", "bitnet"):
        sect = merged.setdefault("llama_cpp", {})
        sect["ctx_size"] = ctx
        sect["parallel_slots"] = slots
    elif backend in ("vllm", "sglang"):
        sect = merged.setdefault("vllm", {})
        sect["max_model_len"] = ctx
        if plan.get("vllm_kv_cache_dtype"):
            sect["kv_cache_dtype"] = plan["vllm_kv_cache_dtype"]
    elif backend == "ollama":
        # Ollama takes num_ctx per request, not at load; stash for the request path.
        sect = merged.setdefault("ollama", {})
        sect["num_ctx"] = ctx
        sect["num_parallel"] = slots
    return merged


async def resolve_arch_and_weights(state) -> ArchResolution:
    """Best-effort (arch, weights_mb) for a model state. arch is None when the
    backend has no context-scaling KV cache or metadata can't be read."""
    backend = (state.backend_type or "").lower()
    if backend in _NON_KV_BACKENDS:
        return ArchResolution(None, 0.0, f"backend '{backend}' sin KV que escale con el contexto")
    if backend == "ollama":
        return await _resolve_ollama(state.backend_model_id)
    if backend in ("llama_cpp", "bitnet"):
        return _resolve_llama_cpp(state)
    if backend in ("vllm", "sglang"):
        return _resolve_hf(state)
    return ArchResolution(None, 0.0, f"backend '{backend}' no soportado por el planner")
