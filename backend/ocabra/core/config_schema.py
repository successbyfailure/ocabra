"""Editable configuration of a model deployment, described per backend.

The model editor renders its forms from this description instead of hard-coding
fields, and the PATCH endpoint validates incoming ``extra_config`` against it.
Each field says where it lives (``path`` inside ``extra_config``), its type and
limits, its default (literal or a ``settings`` attribute, resolved when the
schema is served), which page section it belongs to, whether changing it needs
a reload and whether it changes the memory footprint (so the editor knows when
to refresh its preview).

``tests/core/test_config_schema.py`` scans the backends for the option keys
they actually read, so a new option without a field here fails the suite
instead of drifting silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ocabra.config import settings
from ocabra.core.model_ref import LLAMA_CPP_CONFIG_SECTION, is_llama_cpp_family

# Page sections, in display order.
GROUPS: list[dict[str, str]] = [
    {"id": "load", "label": "Carga"},
    {"id": "gpus", "label": "GPUs"},
    {"id": "context", "label": "Contexto"},
    {"id": "response", "label": "Respuesta"},
    {"id": "acceleration", "label": "Aceleración"},
    {"id": "audio", "label": "Audio"},
    {"id": "image", "label": "Imagen"},
    {"id": "voice", "label": "Voz"},
    {"id": "runtime", "label": "Arranque"},
    {"id": "advanced", "label": "Avanzado"},
]

_UNSET = object()


@dataclass(frozen=True)
class Field:
    path: tuple[str, ...]
    type: str  # int | float | bool | enum | string | text | int_list | float_list | json | string_list
    label: str
    group: str
    help: str = ""
    default: Any = _UNSET
    default_setting: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None
    options: tuple[tuple[Any, str], ...] = ()
    nullable: bool = True
    reload: bool = True
    memory: bool = False
    widget: str | None = None
    requires: dict[str, Any] = field(default_factory=dict)
    advanced: bool = False
    # Accepted by the backend but deliberately not offered as a control.
    hidden: bool = False

    @property
    def key(self) -> str:
        return ".".join(self.path)

    def resolved_default(self) -> Any:
        if self.default_setting:
            return getattr(settings, self.default_setting, None)
        return None if self.default is _UNSET else self.default

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "path": list(self.path),
            "type": self.type,
            "label": self.label,
            "group": self.group,
            "help": self.help,
            "default": self.resolved_default(),
            "nullable": self.nullable,
            "reload": self.reload,
            "memory": self.memory,
            "advanced": self.advanced,
        }
        for name in ("min", "max", "step", "unit", "widget"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.options:
            out["options"] = [{"value": v, "label": label} for v, label in self.options]
        if self.requires:
            out["requires"] = dict(self.requires)
        return out


def _f(section: str | None, key: str, type_: str, label: str, group: str, **kw: Any) -> Field:
    path = (section, *key.split(".")) if section else tuple(key.split("."))
    return Field(path=path, type=type_, label=label, group=group, **kw)


_ON_OFF_AUTO = (("on", "Sí"), ("off", "No"), ("auto", "Automático"))
_KV_TYPES = (
    ("f16", "f16 (sin cuantizar)"),
    ("q8_0", "q8_0 (mitad de memoria)"),
    ("q5_1", "q5_1"),
    ("q5_0", "q5_0"),
    ("q4_1", "q4_1"),
    ("q4_0", "q4_0 (un cuarto)"),
)


# ── Universal (any backend) ─────────────────────────────────────────────────

_KEEP_ALIVE = _f(
    None,
    "keep_alive_seconds",
    "int",
    "Descargar tras inactividad",
    "load",
    help="Solo en modelos bajo demanda. 0 = no descargar nunca por inactividad. "
    "Vacío = el valor global.",
    default_setting="idle_timeout_seconds",
    min=0,
    unit="s",
    reload=False,
    widget="idle_timeout",
)

_REASONING_EFFORT = _f(
    None,
    "default_reasoning_effort",
    "enum",
    "Esfuerzo de razonamiento por defecto",
    "response",
    help="Cuando el cliente no lo pide. Solo surte efecto si la plantilla del "
    "modelo lo lee.",
    default_setting="default_reasoning_effort",
    options=(("off", "Apagado"), ("low", "Bajo"), ("medium", "Medio"), ("high", "Alto")),
    reload=False,
)

_MAX_PARALLELISM = _f(
    None,
    "runtime_capabilities.max_parallelism",
    "int",
    "Máximo de peticiones simultáneas",
    "context",
    help="Tope para el paralelismo de los presets de contexto.",
    min=1,
    max=64,
    advanced=True,
)


# ── llama.cpp family (llama_cpp, llamacpp_mks) ──────────────────────────────

_L = LLAMA_CPP_CONFIG_SECTION

_LLAMA_FIELDS: list[Field] = [
    _f(_L, "gpu_layers", "int", "Capas en GPU", "gpus",
       help="0 = todo en CPU. 999 = todas las capas en GPU.",
       default_setting="llama_cpp_gpu_layers", min=0, max=999, memory=True, widget="gpu_layers"),
    _f(_L, "main_gpu", "int", "GPU principal", "gpus",
       help="Índice entre las GPUs visibles (las no excluidas).", min=0, memory=True,
       widget="main_gpu"),
    _f(_L, "disabled_gpus", "int_list", "GPUs excluidas", "gpus", memory=True,
       widget="gpu_exclude"),
    _f(_L, "split_mode", "enum", "Modo de reparto entre GPUs", "gpus",
       options=(("layer", "Por capas"), ("row", "Por filas"), ("tensor", "Por tensores"),
                ("none", "Una sola GPU")),
       memory=True),
    _f(_L, "split_strategy", "enum", "Estrategia de reparto", "gpus",
       help="Sin reparto explícito: a partes iguales o cargando más la principal.",
       options=(("evenly", "A partes iguales"), ("favor_main", "Favorecer la principal")),
       memory=True),
    _f(_L, "tensor_split", "float_list", "Reparto entre GPUs", "gpus",
       help="Proporción de capas por GPU visible.", memory=True, widget="tensor_split"),
    _f(_L, "n_cpu_moe", "int", "Expertos MoE en CPU", "gpus",
       help="Capas de expertos que se quedan en RAM.", min=0, memory=True, advanced=True),
    _f(_L, "override_tensor", "string", "Override de tensores", "gpus",
       help="Patrón --override-tensor (regex=dispositivo).", memory=True, advanced=True),
    _f(_L, "mmproj_gpu", "int", "GPU del proyector de visión", "gpus", min=0, memory=True,
       widget="gpu"),
    _f(_L, "ctx_size", "int", "Contexto", "context",
       help="Con KV compartida es el total del pool; si no, se reparte entre los slots.",
       default_setting="llama_cpp_ctx_size", min=256, max=1_048_576, unit="tokens",
       memory=True, widget="context"),
    _f(_L, "parallel_slots", "int", "Peticiones simultáneas (slots)", "context",
       default_setting="llama_cpp_parallel_slots", min=1, max=64, memory=True),
    _f(_L, "kv_unified", "bool", "KV compartida (pool)", "context",
       help="Todos los slots comparten un único depósito de KV.", memory=True,
       widget="kv_mode"),
    _f(_L, "kv_unified_per_slot", "int", "Reserva por slot en el pool", "context",
       min=1, unit="tokens", memory=True, advanced=True, requires={"llama_cpp.kv_unified": True}),
    _f(_L, "cache_type_k", "enum", "Precisión de la KV (K)", "context",
       options=_KV_TYPES, memory=True, widget="kv_dtype"),
    _f(_L, "cache_type_v", "enum", "Precisión de la KV (V)", "context",
       help="Distinta de f16 requiere flash attention.", options=_KV_TYPES, memory=True,
       widget="kv_dtype", requires={"llama_cpp.flash_attn": True}),
    _f(_L, "no_kv_offload", "bool", "KV en RAM (no en GPU)", "context", memory=True,
       advanced=True),
    _f(_L, "cache_ram_mib", "int", "Cache de prompts en RAM", "context",
       help="-1 = sin límite, 0 = desactivada.", min=-1, unit="MiB", advanced=True),
    _f(_L, "cache_idle_slots", "bool", "Guardar la KV de slots ociosos", "context",
       advanced=True),
    _f(_L, "rope_freq_base", "float", "RoPE: frecuencia base", "context", min=0,
       advanced=True),
    _f(_L, "rope_freq_scale", "float", "RoPE: escala", "context", min=0, advanced=True),
    _f(_L, "reasoning", "enum", "Razonamiento", "response", options=_ON_OFF_AUTO),
    _f(_L, "reasoning_format", "enum", "Formato del razonamiento", "response",
       options=(("none", "Ninguno"), ("deepseek", "deepseek"), ("deepseek-legacy",
                "deepseek-legacy"), ("auto", "Automático"))),
    _f(_L, "reasoning_budget", "int", "Presupuesto de razonamiento", "response",
       help="-1 = sin límite, 0 = sin razonamiento.", min=-1, unit="tokens"),
    _f(_L, "jinja", "bool", "Plantilla Jinja", "response"),
    _f(_L, "chat_template", "text", "Plantilla de chat", "response",
       help="Nombre de una plantilla integrada o Jinja completo. Vacío = la del GGUF.",
       advanced=True),
    _f(_L, "embedding", "bool", "Modo embeddings", "response",
       default_setting="llama_cpp_embeddings"),
    _f(_L, "seed", "int", "Semilla", "response", advanced=True),
    _f(_L, "speculative", "json", "Decodificación especulativa", "acceleration",
       help="spec_type, draft_model_id, draft_n, draft_min, draft_p_min.", memory=True,
       widget="speculative"),
    _f(_L, "vision_mode", "enum", "Visión", "acceleration",
       options=(("enabled", "Siempre cargada"), ("on_demand", "Bajo demanda"),
                ("disabled", "Apagada")),
       memory=True, widget="vision_mode"),
    _f(_L, "flash_attn", "bool", "Flash attention", "acceleration",
       default_setting="llama_cpp_flash_attn"),
    _f(_L, "batch_size", "int", "Batch", "acceleration", default_setting="llama_cpp_batch_size",
       min=1, max=65_536, memory=True),
    _f(_L, "ubatch_size", "int", "Micro-batch", "acceleration",
       default_setting="llama_cpp_ubatch_size", min=1, max=8192, memory=True),
    _f(_L, "cont_batching", "bool", "Batching continuo", "acceleration",
       default_setting="llama_cpp_cont_batching"),
    _f(_L, "threads", "int", "Hilos de CPU", "acceleration", min=1, max=256),
    _f(_L, "mlock", "bool", "Bloquear en RAM (mlock)", "acceleration",
       default_setting="llama_cpp_mlock", advanced=True),
    _f(_L, "mmap", "bool", "Mapear el fichero (mmap)", "acceleration", advanced=True),
    _f(_L, "vram_estimate_mb", "int", "Reserva de VRAM fija", "advanced",
       help="Sustituye la estimación automática en la admisión.", min=1, unit="MB",
       memory=True),
    _f(_L, "runtime", "enum", "Runtime", "runtime",
       help="Variante del binario llama-server.", options=(("cuda", "CUDA"),)),
    _f(_L, "startup_timeout_s", "int", "Tiempo máximo de arranque", "runtime",
       default_setting="llama_cpp_startup_timeout_s", min=5, unit="s", reload=False),
    _f(_L, "keep_alive_seconds", "int", "Descargar tras inactividad", "load",
       min=0, unit="s", reload=False, hidden=True),
    _f(_L, "total_layers", "int", "Capas totales (estimador)", "advanced", min=1, hidden=True),
    _f(_L, "mmproj_model_id", "string", "Proyector de visión", "advanced", hidden=True),
    _f(_L, "model_path", "string", "Fichero del modelo", "advanced", hidden=True),
]


# ── bitnet ──────────────────────────────────────────────────────────────────

_B = "bitnet"
_BITNET_FIELDS: list[Field] = [
    _f(_B, "gpu_layers", "int", "Capas en GPU", "gpus", default_setting="bitnet_gpu_layers",
       min=0, max=999, memory=True, widget="gpu_layers"),
    _f(_B, "ctx_size", "int", "Contexto", "context", default_setting="bitnet_ctx_size",
       min=256, unit="tokens", memory=True, widget="context"),
    _f(_B, "parallel", "int", "Peticiones simultáneas (slots)", "context",
       default_setting="bitnet_parallel", min=1, max=64, memory=True),
    _f(_B, "kv_unified", "bool", "KV compartida (pool)", "context",
       help="Solo builds PrismML.", memory=True, widget="kv_mode"),
    _f(_B, "cache_type_k", "enum", "Precisión de la KV (K)", "context", options=_KV_TYPES,
       memory=True, widget="kv_dtype"),
    _f(_B, "cache_type_v", "enum", "Precisión de la KV (V)", "context", options=_KV_TYPES,
       memory=True, widget="kv_dtype"),
    _f(_B, "flash_attn", "bool", "Flash attention", "acceleration",
       default_setting="bitnet_flash_attn"),
    _f(_B, "batch_size", "int", "Batch", "acceleration", default_setting="bitnet_batch_size",
       min=1),
    _f(_B, "ubatch_size", "int", "Micro-batch", "acceleration",
       default_setting="bitnet_ubatch_size", min=1),
    _f(_B, "threads", "int", "Hilos de CPU", "acceleration", min=1, max=256),
    _f(_B, "mlock", "bool", "Bloquear en RAM (mlock)", "acceleration",
       default_setting="bitnet_mlock", advanced=True),
    _f(_B, "startup_timeout_s", "int", "Tiempo máximo de arranque", "runtime",
       default_setting="bitnet_startup_timeout_s", min=5, unit="s", reload=False),
    _f(_B, "model_vram_mb", "int", "VRAM de los pesos (estimador)", "advanced", min=1,
       unit="MB", memory=True, advanced=True),
    _f(_B, "total_layers", "int", "Capas totales (estimador)", "advanced", min=1,
       advanced=True),
    _f(_B, "model_path", "string", "Fichero del modelo", "advanced", hidden=True),
]


# ── Ollama ──────────────────────────────────────────────────────────────────

_OLLAMA_FIELDS: list[Field] = [
    _f("ollama", "num_ctx", "int", "Contexto", "context",
       help="Se envía a Ollama al cargar el modelo.", min=256, unit="tokens", memory=True,
       widget="context"),
]


# ── vLLM ────────────────────────────────────────────────────────────────────

_V = "vllm"
_VLLM_FIELDS: list[Field] = [
    _f(_V, "tensor_parallel_size", "int", "Tensor parallel (GPUs)", "gpus",
       default_setting="vllm_tensor_parallel_size", min=1, max=8, memory=True),
    _f(_V, "pipeline_parallel_size", "int", "Pipeline parallel", "gpus",
       default_setting="vllm_pipeline_parallel_size", min=1, max=8, memory=True,
       advanced=True),
    _f(_V, "gpu_memory_utilization", "float", "Fracción de VRAM para el motor", "gpus",
       help="vLLM reserva esta parte de la GPU al arrancar.",
       default_setting="vllm_gpu_memory_utilization", min=0.05, max=0.99, step=0.01,
       memory=True, widget="fraction"),
    _f(_V, "max_model_len", "int", "Contexto máximo", "context",
       default_setting="vllm_max_model_len", min=256, unit="tokens", memory=True,
       widget="context"),
    _f(_V, "max_num_seqs", "int", "Secuencias simultáneas", "context",
       default_setting="vllm_max_num_seqs", min=1, max=1024, memory=True),
    _f(_V, "kv_cache_dtype", "enum", "Precisión de la KV", "context",
       default_setting="vllm_kv_cache_dtype",
       options=(("auto", "Automática"), ("fp8", "fp8"), ("fp8_e5m2", "fp8_e5m2"),
                ("fp8_e4m3", "fp8_e4m3")),
       memory=True, widget="kv_dtype"),
    _f(_V, "max_num_batched_tokens", "int", "Tokens por paso", "acceleration",
       default_setting="vllm_max_num_batched_tokens", min=1, memory=True),
    _f(_V, "enable_prefix_caching", "bool", "Cache de prefijos", "acceleration",
       default_setting="vllm_enable_prefix_caching"),
    _f(_V, "enable_chunked_prefill", "bool", "Prefill por trozos", "acceleration",
       default_setting="vllm_enable_chunked_prefill"),
    _f(_V, "enforce_eager", "bool", "Sin CUDA graphs (eager)", "acceleration",
       default_setting="vllm_enforce_eager"),
    _f(_V, "attention_backend", "string", "Backend de atención", "acceleration",
       default_setting="vllm_attention_backend", advanced=True),
    _f(_V, "use_flashinfer_sampler", "bool", "Sampler FlashInfer", "acceleration",
       default_setting="vllm_use_flashinfer_sampler", advanced=True),
    _f(_V, "speculative_config", "json", "Decodificación especulativa", "acceleration",
       memory=True, advanced=True),
    _f(_V, "tool_call_parser", "string", "Parser de tool calls", "response",
       default_setting="vllm_tool_call_parser"),
    _f(_V, "tool_parser_plugin", "string", "Plugin de parser de tools", "response",
       default_setting="vllm_tool_parser_plugin", advanced=True),
    _f(_V, "reasoning_parser", "string", "Parser de razonamiento", "response",
       default_setting="vllm_reasoning_parser"),
    _f(_V, "chat_template", "text", "Plantilla de chat", "response",
       default_setting="vllm_chat_template", advanced=True),
    _f(_V, "chat_template_content_format", "enum", "Formato del contenido", "response",
       default_setting="vllm_chat_template_content_format",
       options=(("auto", "Automático"), ("string", "Texto"), ("openai", "OpenAI")),
       advanced=True),
    _f(_V, "generation_config", "string", "generation_config", "response",
       default_setting="vllm_generation_config", advanced=True),
    _f(_V, "override_generation_config", "json", "Sobrescribir generation_config",
       "response", advanced=True),
    _f(_V, "limit_mm_per_prompt", "json", "Límite multimodal por prompt", "response",
       help='Por ejemplo {"image": 4}.', memory=True, advanced=True),
    _f(_V, "language_model_only", "bool", "Solo texto (sin torre de visión)", "acceleration",
       default_setting="vllm_language_model_only", memory=True),
    _f(_V, "trust_remote_code", "bool", "Confiar en código remoto", "runtime",
       default_setting="vllm_trust_remote_code"),
    _f(_V, "model_impl", "string", "Implementación del modelo", "runtime",
       default_setting="vllm_model_impl", advanced=True),
    _f(_V, "runner", "string", "Runner", "runtime", default_setting="vllm_runner",
       advanced=True),
    _f(_V, "hf_overrides", "json", "hf_overrides", "advanced"),
    _f(_V, "extra_args", "string_list", "Argumentos extra", "advanced",
       help="Se añaden tal cual al comando de vLLM."),
    _f(_V, "disable_log_requests", "bool", "No registrar peticiones", "advanced",
       default_setting="vllm_disable_log_requests", advanced=True),
    _f(_V, "vram_estimate_mb", "int", "Reserva de VRAM fija", "advanced",
       help="Sustituye la estimación automática en la admisión.", min=1, unit="MB",
       memory=True),
    _f(_V, "swap_space", "int", "swap_space", "advanced", hidden=True),
    # Written by Explore (vllm_recipes) and by the engine probe; the editor
    # shows them as the recipe/probe card, not as fields.
    *[
        _f(_V, key, "json", key, "advanced", hidden=True)
        for key in (
            "recipe_id", "recipe_notes", "recipe_model_impl", "recipe_runner",
            "suggested_config", "suggested_tuning", "probe_status", "probe_reason",
            "probe_observed_at", "probe_recommended_model_impl", "probe_recommended_runner",
        )
    ],
]


# ── SGLang ──────────────────────────────────────────────────────────────────

_S = "sglang"
_SGLANG_FIELDS: list[Field] = [
    _f(_S, "tensor_parallel_size", "int", "Tensor parallel (GPUs)", "gpus",
       default_setting="sglang_tensor_parallel_size", min=1, max=8, memory=True),
    _f(_S, "mem_fraction_static", "float", "Fracción de VRAM para el motor", "gpus",
       default_setting="sglang_mem_fraction_static", min=0.05, max=0.99, step=0.01,
       memory=True, widget="fraction"),
    _f(_S, "context_length", "int", "Contexto máximo", "context",
       default_setting="sglang_context_length", min=256, unit="tokens", memory=True,
       widget="context"),
    _f(_S, "disable_radix_cache", "bool", "Sin cache radix", "acceleration",
       default_setting="sglang_disable_radix_cache"),
    _f(_S, "disable_cuda_graph", "bool", "Sin CUDA graphs", "acceleration"),
    _f(_S, "attention_backend", "string", "Backend de atención", "acceleration",
       advanced=True),
    _f(_S, "prefill_attention_backend", "string", "Atención en prefill", "acceleration",
       advanced=True),
    _f(_S, "decode_attention_backend", "string", "Atención en decode", "acceleration",
       advanced=True),
    _f(_S, "sampling_backend", "string", "Backend de muestreo", "acceleration",
       advanced=True),
    _f(_S, "trust_remote_code", "bool", "Confiar en código remoto", "runtime",
       default_setting="sglang_trust_remote_code"),
    _f(_S, "startup_timeout_s", "int", "Tiempo máximo de arranque", "runtime",
       default_setting="sglang_startup_timeout_s", min=5, unit="s", reload=False),
]


# ── TensorRT-LLM ────────────────────────────────────────────────────────────

_T = "tensorrt_llm"
_TRT_FIELDS: list[Field] = [
    _f(_T, "context_length", "int", "Contexto máximo", "context",
       default_setting="tensorrt_llm_context_length", min=256, unit="tokens",
       widget="context"),
    _f(_T, "max_batch_size", "int", "Batch máximo", "context",
       default_setting="tensorrt_llm_max_batch_size", min=1),
    _f(_T, "max_num_tokens", "int", "Tokens por paso", "acceleration", min=1),
    _f(_T, "engine_dir", "string", "Directorio del engine", "runtime"),
    _f(_T, "tokenizer_path", "string", "Tokenizer", "runtime",
       default_setting="tensorrt_llm_tokenizer_path"),
    _f(_T, "backend", "enum", "Backend", "runtime", default_setting="tensorrt_llm_backend",
       options=(("trt", "TensorRT"), ("pytorch", "PyTorch"))),
    _f(_T, "launch_mode", "enum", "Modo de arranque", "runtime",
       default_setting="tensorrt_llm_launch_mode",
       options=(("binary", "Binario"), ("module", "Módulo Python"), ("docker", "Docker"))),
    _f(_T, "trust_remote_code", "bool", "Confiar en código remoto", "runtime",
       default_setting="tensorrt_llm_trust_remote_code"),
    _f(_T, "startup_timeout_s", "int", "Tiempo máximo de arranque", "runtime",
       default_setting="tensorrt_llm_startup_timeout_s", min=5, unit="s", reload=False),
    _f(_T, "python_bin", "string", "Python", "runtime", advanced=True,
       default_setting="tensorrt_llm_python_bin"),
    _f(_T, "serve_bin", "string", "trtllm-serve", "runtime", advanced=True,
       default_setting="tensorrt_llm_serve_bin"),
    _f(_T, "serve_module", "string", "Módulo de serve", "runtime", advanced=True,
       default_setting="tensorrt_llm_serve_module"),
    _f(_T, "docker_bin", "string", "Docker", "runtime", advanced=True,
       default_setting="tensorrt_llm_docker_bin"),
    _f(_T, "docker_image", "string", "Imagen Docker", "runtime", advanced=True,
       default_setting="tensorrt_llm_docker_image"),
    _f(_T, "docker_models_mount_host", "string", "Modelos (host)", "runtime", advanced=True,
       default_setting="tensorrt_llm_docker_models_mount_host"),
    _f(_T, "docker_models_mount_container", "string", "Modelos (contenedor)", "runtime",
       advanced=True, default_setting="tensorrt_llm_docker_models_mount_container"),
    _f(_T, "docker_hf_cache_mount_host", "string", "Cache HF (host)", "runtime",
       advanced=True, default_setting="tensorrt_llm_docker_hf_cache_mount_host"),
    _f(_T, "docker_hf_cache_mount_container", "string", "Cache HF (contenedor)", "runtime",
       advanced=True, default_setting="tensorrt_llm_docker_hf_cache_mount_container"),
]


# ── Audio, voice, image, others ─────────────────────────────────────────────

_WHISPER_FIELDS: list[Field] = [
    _f("whisper", "num_workers", "int", "Transcripciones simultáneas", "audio",
       help="Hilos del worker; cada uno añade memoria por petición.", default=5, min=1,
       max=16, memory=True),
    _f("whisper", "diarization_model_id", "string", "Modelo de diarización", "audio",
       default="pyannote/speaker-diarization-3.1"),
    _f("whisper", "base_model_id", "string", "Modelo base", "advanced", advanced=True),
    _f("whisper", "diarization_enabled", "bool", "Diarizar por defecto", "audio",
       help="Ahora se decide por petición; los perfiles -diarized lo activan.", hidden=True),
]

_WHISPERCPP_FIELDS: list[Field] = [
    _f("whisper_cpp", "threads", "int", "Hilos de CPU", "audio",
       default_setting="whispercpp_threads", min=1, max=256),
    _f("whisper_cpp", "diarization_model_id", "string", "Modelo de diarización", "audio",
       default="pyannote/speaker-diarization-3.1"),
    _f("whisper_cpp", "startup_timeout_s", "int", "Tiempo máximo de arranque", "runtime",
       default_setting="whispercpp_startup_timeout_s", min=5, unit="s", reload=False),
    _f("whisper_cpp", "model_path", "string", "Fichero del modelo", "advanced", hidden=True),
]

_VIBEASR_FIELDS: list[Field] = [
    _f("vibeasr", "threads", "int", "Hilos de CPU", "audio", default_setting="vibeasr_threads",
       min=1, max=256),
    _f("vibeasr", "vae_model", "string", "Modelo VAE", "advanced", advanced=True),
    _f("vibeasr", "lm_model", "string", "Modelo de lenguaje", "advanced", advanced=True),
]

_PIPER_FIELDS: list[Field] = [
    _f("piper", "voice", "string", "Voz", "voice"),
    _f("piper", "onnx_path", "string", "Fichero ONNX", "advanced", advanced=True),
    _f("piper", "config_path", "string", "Configuración de la voz", "advanced", advanced=True),
]

_DIFFUSERS_FIELDS: list[Field] = [
    _f("diffusers", "offload_mode", "enum", "Descarga a CPU", "image",
       help="Menos VRAM a cambio de velocidad.", default_setting="diffusers_offload_mode",
       options=(("none", "Todo en GPU (rápido)"), ("model", "Por modelo"),
                ("sequential", "Secuencial (mínima VRAM)")),
       memory=True, widget="offload"),
    _f("diffusers", "torch_dtype", "enum", "Precisión", "image",
       default_setting="diffusers_torch_dtype",
       options=(("bfloat16", "bf16"), ("float16", "fp16"), ("float32", "fp32"),
                ("auto", "Automática")),
       memory=True),
    _f("diffusers", "text_encoder_quant", "enum", "Cuantización del text encoder", "image",
       default="nf4", options=(("nf4", "nf4"), ("int8", "int8"), ("none", "Sin cuantizar")),
       memory=True),
    _f("diffusers", "pipeline_class", "string", "Clase del pipeline", "advanced",
       advanced=True),
    _f("diffusers", "transformer_class", "string", "Clase del transformer", "advanced",
       advanced=True),
    _f("diffusers", "text_encoder_class", "string", "Clase del text encoder", "advanced",
       advanced=True),
    _f("diffusers", "gguf_transformer_path", "string", "Transformer GGUF", "advanced",
       memory=True, advanced=True),
]

_STRATA_FIELDS: list[Field] = [
    _f("strata", "config_path", "string", "Configuración de ejecución", "runtime",
       help="JSON con args, gpu, cwd, model_name, parallel y vision.", nullable=False),
    _f("strata", "native_context_length", "int", "Contexto nativo", "context", default=262_144,
       min=256, unit="tokens", widget="context"),
    _f("strata", "min_available_ram_mb", "int", "RAM libre mínima para arrancar", "load",
       min=0, unit="MB"),
    _f("strata", "vram_total_mb", "int", "VRAM total que usa", "gpus", min=1, unit="MB",
       memory=True),
    _f("strata", "startup_timeout_s", "int", "Tiempo máximo de arranque", "runtime",
       default=300, min=5, unit="s", reload=False),
    _f("strata", "request_timeout_s", "int", "Tiempo máximo por petición", "runtime",
       default=900, min=5, unit="s", reload=False),
    _f("strata", "runtime_dir", "string", "Directorio del runtime", "runtime", advanced=True),
    _f("strata", "python_bin", "string", "Python", "runtime", advanced=True),
    _f("strata", "server_path", "string", "Servidor", "runtime", advanced=True),
    _f("strata", "xdg_data_home", "string", "XDG_DATA_HOME", "runtime", advanced=True),
    _f("strata", "xdg_state_home", "string", "XDG_STATE_HOME", "runtime", advanced=True),
]


_LLM_BACKENDS = {"llama_cpp", "llamacpp_mks", "bitnet", "ollama", "vllm", "sglang",
                 "tensorrt_llm", "strata"}

_BACKEND_FIELDS: dict[str, list[Field]] = {
    "llama_cpp": _LLAMA_FIELDS,
    "llamacpp_mks": _LLAMA_FIELDS,
    "bitnet": _BITNET_FIELDS,
    "ollama": _OLLAMA_FIELDS,
    "vllm": _VLLM_FIELDS,
    "sglang": _SGLANG_FIELDS,
    "tensorrt_llm": _TRT_FIELDS,
    "whisper": _WHISPER_FIELDS,
    "whisper_cpp": _WHISPERCPP_FIELDS,
    "vibeasr": _VIBEASR_FIELDS,
    "piper": _PIPER_FIELDS,
    "diffusers": _DIFFUSERS_FIELDS,
    "strata": _STRATA_FIELDS,
}

# Keys of extra_config that the system writes and the editor must not offer.
INTERNAL_KEYS = {"_runtime", "registry_metadata", "update_source", "use_case"}

# Backends whose admission honours a fixed ``vram_estimate_mb``.
_VRAM_OVERRIDE_SECTIONS = {"llama_cpp": _L, "llamacpp_mks": _L, "vllm": "vllm"}

# The section each backend reads its options from (llamacpp_mks shares llama.cpp's).
def section_for(backend_type: str) -> str:
    return _L if is_llama_cpp_family(backend_type) else backend_type


def vram_override_section(backend_type: str) -> str | None:
    """extra_config section holding ``vram_estimate_mb`` for this backend, or
    None when its admission has no such override."""
    return _VRAM_OVERRIDE_SECTIONS.get(backend_type)


def fields_for(backend_type: str) -> list[Field]:
    fields = [_KEEP_ALIVE]
    if backend_type in _LLM_BACKENDS:
        fields += [_REASONING_EFFORT, _MAX_PARALLELISM]
    return fields + list(_BACKEND_FIELDS.get(backend_type, []))


def _available_llama_runtimes() -> list[tuple[str, str]]:
    from pathlib import Path

    options = [("cuda", "CUDA")]
    labels = {"rocm": "ROCm", "vulkan": "Vulkan", "cpu": "CPU"}
    root = Path(settings.backends_dir)
    for runtime, label in labels.items():
        if (root / f"llama_cpp_{runtime}").is_dir():
            options.append((runtime, label))
    return options


def schema_for(backend_type: str) -> dict[str, Any]:
    """JSON description of the editable options of ``backend_type``."""
    out: list[dict[str, Any]] = []
    for f in fields_for(backend_type):
        if f.hidden:
            continue
        item = f.to_json()
        if f.path[-1] == "runtime" and f.path[0] == _L:
            # Only the runtimes with an installed binary are choosable.
            item["options"] = [
                {"value": v, "label": label} for v, label in _available_llama_runtimes()
            ]
            if len(item["options"]) == 1:
                continue
        out.append(item)
    return {
        "backend": backend_type,
        "section": section_for(backend_type),
        "groups": GROUPS,
        "fields": out,
        "vramOverride": vram_override_section(backend_type) is not None,
        "isLlm": backend_type in _LLM_BACKENDS,
    }


# ── Validation ──────────────────────────────────────────────────────────────


def _get_path(data: dict, path: tuple[str, ...]) -> tuple[bool, Any]:
    node: Any = data
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _type_error(f: Field, value: Any) -> str | None:
    if value is None:
        return None if f.nullable else "no puede quedar vacío"
    t = f.type
    if t == "bool":
        return None if isinstance(value, bool) else "debe ser sí/no"
    if t in ("int", "float"):
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        if t == "int":
            ok = ok and float(value).is_integer()
        if not ok:
            return "debe ser un número entero" if t == "int" else "debe ser un número"
        if f.min is not None and value < f.min:
            return f"mínimo {f.min:g}"
        if f.max is not None and value > f.max:
            return f"máximo {f.max:g}"
        return None
    if t == "enum":
        allowed = [v for v, _ in f.options]
        if f.path[-1] == "runtime" and f.path[0] == _L:
            allowed = [v for v, _ in _available_llama_runtimes()]
        return None if value in allowed else "valor no válido: " + ", ".join(map(str, allowed))
    if t in ("string", "text"):
        return None if isinstance(value, str) else "debe ser texto"
    if t in ("int_list", "float_list"):
        if not isinstance(value, list):
            return "debe ser una lista"
        for item in value:
            if not isinstance(item, int | float) or isinstance(item, bool) or item < 0:
                return "la lista solo admite números no negativos"
            if t == "int_list" and not float(item).is_integer():
                return "la lista solo admite enteros"
        return None
    if t == "string_list":
        if not isinstance(value, list) or not all(isinstance(i, str) for i in value):
            return "debe ser una lista de textos"
        return None
    if t == "json":
        return None if isinstance(value, dict | list | str) else "debe ser un objeto JSON"
    return None


def validate_extra_config(backend_type: str, extra: dict | None) -> list[dict[str, str]]:
    """Errors in the known options of ``extra`` (a PATCH payload or a draft).

    Unknown keys are not errors (stored configs carry history); see
    :func:`unknown_keys` for the warnings the editor shows about them.
    """
    if not isinstance(extra, dict):
        return []
    errors: list[dict[str, str]] = []
    for f in fields_for(backend_type):
        present, value = _get_path(extra, f.path)
        if not present:
            continue
        message = _type_error(f, value)
        if message:
            errors.append({"key": f.key, "label": f.label, "message": message})
    if backend_type in ("llama_cpp", "llamacpp_mks"):
        section = extra.get(_L) if isinstance(extra.get(_L), dict) else {}
        split = section.get("tensor_split")
        if isinstance(split, list) and split and not any(
            isinstance(v, int | float) and v > 0 for v in split
        ):
            errors.append({
                "key": f"{_L}.tensor_split",
                "label": "Reparto entre GPUs",
                "message": "al menos una GPU tiene que llevar parte",
            })
    return errors


def unknown_keys(backend_type: str, extra: dict | None) -> list[str]:
    """Keys of the backend's section (and the top level) the schema does not know."""
    if not isinstance(extra, dict):
        return []
    known = {f.path for f in fields_for(backend_type)}
    section = section_for(backend_type)
    sections = {p[0] for p in known if len(p) > 1} | {section}
    out: list[str] = []
    for key, value in extra.items():
        if key in INTERNAL_KEYS or key.startswith("_"):
            continue
        if key in sections and isinstance(value, dict):
            for sub in value:
                if (key, sub) not in known and not any(
                    p[:2] == (key, sub) for p in known
                ):
                    out.append(f"{key}.{sub}")
            continue
        if (key,) not in known and not any(p[0] == key for p in known):
            out.append(key)
    return sorted(out)
