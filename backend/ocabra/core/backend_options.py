"""Per-backend option readers over ``ModelState.extra_config``.

Pure, stateless helpers extracted from ``model_manager_helpers`` (REF-05 step 1).
Every reader accepts both snake_case and camelCase spellings of an option,
because the frontend's PATCH write path saves camelCase while the code defaults
to snake_case — reading only one spelling is what made the vLLM estimator return
0 for ``maxModelLen`` models (BUG-28) and the llama.cpp gate treat a fully
offloaded model as CPU-only (``snake_to_camel`` docstring). Keeping them in one
place is what stops the accepted spellings from drifting apart again.
"""

from __future__ import annotations

from ocabra.config import settings

PRISMML_DEFAULT_GPU_LAYERS = 99

LLAMA_CPP_SPLIT_POLICIES = ("single", "when_needed", "always")


def snake_to_camel(key: str) -> str:
    """``gpu_layers`` -> ``gpuLayers``. Shared by every llama.cpp option
    reader so the accepted key spellings can never drift apart again.

    2026-10-29: ``resolve_llama_cpp_option`` only checked the literal
    snake_case key while ``LlamaCppBackend._get_option`` (which actually
    builds the subprocess CLI args) also accepted camelCase — the frontend's
    PATCH write path saves camelCase. Any model configured through the UI
    (e.g. qwen3.8-27b: ``gpuLayers``, not ``gpu_layers``) silently resolved
    to ``default_gpu_layers`` here, which defaults to 0 — so the admission
    gate treated a fully GPU-offloaded model as CPU-only and skipped VRAM
    checks/eviction entirely, while the real process still requested full
    GPU offload. Reproduced in production: qwen3.8-27b's llama-server
    started with no VRAM admitted for it, fought silently for whatever was
    actually free, and got stuck with no CUDA context at all when there
    wasn't enough.
    """
    parts = key.split("_")
    return "".join(part.capitalize() if i else part for i, part in enumerate(parts))


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------


def resolve_llama_cpp_option(state, key: str, default):
    """Read a llama.cpp option from ``extra_config['llama_cpp'][key]`` or the
    top level, accepting both snake_case and camelCase — mirrors
    ``LlamaCppBackend._get_option`` exactly (see ``snake_to_camel``)."""
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    camel_key = snake_to_camel(key)
    nested = extra.get("llama_cpp") if isinstance(extra.get("llama_cpp"), dict) else None
    if isinstance(nested, dict):
        if key in nested:
            return nested[key]
        if camel_key in nested:
            return nested[camel_key]
    if key in extra:
        return extra[key]
    if camel_key in extra:
        return extra[camel_key]
    return default


def resolve_llama_cpp_split_policy(state) -> str:
    """What a llama.cpp model without presets does when one GPU is not enough.

    ``llama_cpp.split_policy`` when set; otherwise the behaviour every model
    had before the option existed: ``split_mode: "none"`` never spreads
    ("single"), anything else spreads only when no single GPU can hold it
    ("when_needed").
    """
    raw = resolve_llama_cpp_option(state, "split_policy", None)
    if isinstance(raw, str) and raw.strip().lower() in LLAMA_CPP_SPLIT_POLICIES:
        return raw.strip().lower()
    split_mode = resolve_llama_cpp_option(state, "split_mode", None)
    if isinstance(split_mode, str) and split_mode.strip().lower() == "none":
        return "single"
    return "when_needed"


def resolve_llama_cpp_allowed_gpus(state, gpu_indices: list[int]) -> list[int]:
    """``gpu_indices`` minus the model's ``llama_cpp.disabled_gpus``."""
    raw = resolve_llama_cpp_option(state, "disabled_gpus", None) or []
    try:
        disabled = {int(index) for index in raw}
    except (TypeError, ValueError):
        disabled = set()
    return [int(index) for index in gpu_indices if int(index) not in disabled]


def resolve_llama_cpp_gpu_layers(state, default_gpu_layers: int) -> int:
    try:
        return int(resolve_llama_cpp_option(state, "gpu_layers", default_gpu_layers))
    except (TypeError, ValueError):
        return int(default_gpu_layers)


# ---------------------------------------------------------------------------
# bitnet
# ---------------------------------------------------------------------------


def _bitnet_parallel(extra: dict) -> object:
    """bitnet's slot count: ``bitnet.parallel`` (or top level), what its
    worker launches with ``--parallel``. Reading llama.cpp's
    ``parallel_slots`` instead sized the fair gate with a key bitnet ignores."""
    section = extra.get("bitnet")
    if isinstance(section, dict) and section.get("parallel") is not None:
        return section.get("parallel")
    return extra.get("parallel")


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


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


def resolve_vllm_option(state, key: str, default=None):
    """Read a vLLM option the way the backend does: ``extra_config["vllm"]``
    first, then the top level, each in snake_case or camelCase (the UI saves
    camelCase). Reading only snake_case made the estimator return 0 for
    ``maxModelLen`` models and fall back to the weights-only heuristic (BUG-28)."""
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    camel_key = snake_to_camel(key)
    nested = extra.get("vllm") if isinstance(extra.get("vllm"), dict) else None
    for source in (nested or {}, extra):
        for candidate in (key, camel_key):
            if candidate in source:
                return source[candidate]
    return default


def resolve_vllm_gpu_memory_utilization(value: float | None = None) -> float:
    """Clamp a vLLM ``gpu_memory_utilization`` into ``[0, 1]``.

    ``None`` (or an unparseable value) falls back to
    ``settings.vllm_gpu_memory_utilization``. Single source for the clamp that
    used to live in both the scheduler and the model manager (DUP-11)."""
    if value is None:
        value = settings.vllm_gpu_memory_utilization
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(settings.vllm_gpu_memory_utilization)
    return max(0.0, min(1.0, resolved))
