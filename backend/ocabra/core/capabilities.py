"""Cold-model capability heuristics.

Pure, stateless capability computation extracted from ``model_manager`` (REF-05
step 1). Models only report their *real* capabilities once loaded; for cold
models the model list falls back to a heuristic derived from backend type and
model name (``declared``). The display-time resolution that layers the GGUF
chat template on top (``resolve_display_capabilities``) stays in ``model_manager``
because its template cache is patched there by tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocabra.backends.base import BackendCapabilities
from ocabra.core.backend_options import snake_to_camel
from ocabra.core.model_ref import is_llama_cpp_family

if TYPE_CHECKING:
    from ocabra.core.model_manager import ModelState

# Backend families whose models are text LLMs (chat/completion) by default.
_TEXT_LLM_BACKENDS = {
    "vllm",
    "sglang",
    "strata",
    "llama_cpp",
    "llamacpp_mks",
    "bitnet",
    "tensorrt_llm",
    "ollama",
    "transformers",
}


def resolve_context_length_fallback(state: ModelState) -> int:
    """Best-effort context length from ``extra_config`` when the worker is cold.

    Reads the top level then the per-backend section, each in snake_case or
    camelCase, for ``context_length``/``max_model_len``/``ctx_size``."""
    extra_config = state.extra_config if isinstance(state.extra_config, dict) else {}

    for key in ("context_length", "max_model_len", "ctx_size"):
        value = extra_config.get(key)
        if isinstance(value, int) and value > 0:
            return value

    for section_name in (
        "vllm",
        "sglang",
        "strata",
        "llama_cpp",
        "bitnet",
        "tensorrt_llm",
        "whisper",
    ):
        section = extra_config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in ("context_length", "max_model_len", "ctx_size"):
            for candidate in (key, snake_to_camel(key)):
                value = section.get(candidate)
                if isinstance(value, int) and value > 0:
                    return value

    return 0


def capabilities_populated(capabilities: BackendCapabilities) -> bool:
    """True when *capabilities* carries any real signal (a flag or context len)."""
    data = capabilities.to_dict()
    if int(data.get("context_length", 0) or 0) > 0:
        return True
    return any(value is True for value in data.values())


def configured_reasoning_default(state: ModelState) -> str | None:
    """Return the explicit llama.cpp launch default, if configured.

    This is intentionally separate from the reasoning capability: a worker
    launched with thinking off can still enable it for an individual request.
    """

    if not is_llama_cpp_family(state.backend_type):
        return None
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    llama_cpp = extra.get("llama_cpp")
    llama_cpp = llama_cpp if isinstance(llama_cpp, dict) else {}
    mode = llama_cpp.get("reasoning")
    budget = llama_cpp.get("reasoning_budget", llama_cpp.get("reasoningBudget"))
    normalized = str(mode).strip().lower()
    if normalized == "off" or budget == 0:
        return "off"
    if normalized in {"on", "auto"}:
        return normalized
    return None


def declared_capabilities(state: ModelState) -> BackendCapabilities:
    """Best-effort capabilities for a never-loaded model, from backend + name.

    Heuristic only — meant to give clients a usable hint while a model is cold,
    not an authoritative answer (that arrives once the model is loaded).
    """
    caps = BackendCapabilities()
    backend_type = (state.backend_type or "").strip().lower()
    name = f"{state.backend_model_id or ''} {state.model_id or ''}".lower()
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}

    context_length = resolve_context_length_fallback(state)
    if context_length > 0:
        caps.context_length = context_length

    if backend_type in _TEXT_LLM_BACKENDS:
        if "embed" in name:
            caps.embeddings = True
            caps.pooling = True
        elif "rerank" in name or "re-rank" in name or "cross-encoder" in name:
            caps.rerank = True
            caps.score = True
        else:
            caps.chat = True
            caps.completion = True
            caps.streaming = True
            if any(hint in name for hint in ("-vl", "vl-", "vision", "llava", "omni")):
                caps.vision = True
            if any(
                hint in name
                for hint in (
                    "deepseek-r1",
                    "deepseek-v3",
                    "qwen3",
                    "qwq",
                    "-r1",
                    "reason",
                    "cyberagent",
                    "nemotron",
                    "magistral",
                    "cogito",
                    "thinking",
                )
            ):
                caps.reasoning = True
            if any(
                hint in name
                for hint in (
                    "llama",
                    "mistral",
                    "mixtral",
                    "qwen",
                    "gemma",
                    "nemotron",
                    "command-r",
                    "devstral",
                    "ministral",
                    "phi",
                    "tool",
                )
            ):
                caps.tools = True
    elif backend_type in {"whisper", "whisper_cpp", "vibeasr"}:
        caps.audio_transcription = True
    elif backend_type in {"chatterbox", "tts", "voxtral", "piper"}:
        caps.tts = True
        caps.streaming = True
    elif backend_type in {"diffusers", "mage"}:
        caps.image_generation = True
    elif backend_type == "acestep":
        caps.music_generation = True
    elif backend_type == "upscaler":
        caps.video_upscaling = caps.image_upscaling = True
    elif backend_type == "flashvsr":
        caps.video_upscaling = True
    elif backend_type == "comfyui_video":
        caps.video_generation = True

    if backend_type == "strata":
        section = extra.get("strata")
        section = section if isinstance(section, dict) else {}
        runtime = extra.get("_runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        caps.completion = False
        caps.chat = caps.tools = caps.reasoning = caps.streaming = True
        try:
            caps.native_context_length = max(0, int(section.get("native_context_length") or 0))
            caps.parallel_slots = max(
                1,
                int(runtime.get("parallelism") or section.get("parallel_slots") or 1),
            )
        except (TypeError, ValueError):
            caps.parallel_slots = 1
        caps.context_per_request = caps.context_length

    return caps
