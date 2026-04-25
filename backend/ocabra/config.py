import secrets

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
    )

    # App
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    database_url: str = "postgresql+asyncpg://ocabra:ocabra@postgres:5432/ocabra"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Models storage
    models_dir: str = "/data/models"
    download_dir: str = ""  # default: {models_dir}/downloads
    # Bloque 15 — Modular backends install root (one subdir per backend).
    backends_dir: str = "/data/backends"
    # True on the fat image (all backends pre-installed): unknown backends
    # without metadata.json are reported as "built-in" so they stay usable.
    # Set to False on the slim image so the same backends show as
    # "not_installed" and the UI surfaces the install button.
    backends_fat_image: bool = True
    max_temperature_c: int = 88
    hf_cache_dir: str = "/data/hf_cache"
    ai_models_root: str = "/docker/ai-models"
    hf_token: str = ""
    ollama_base_url: str = "http://ollama:11434"
    ollama_keep_alive: str = "30m"
    ollama_inventory_sync_interval_seconds: int = 15

    # GPU scheduling
    default_gpu_index: int = 1
    worker_port_range_start: int = 18000
    worker_port_range_end: int = 19000
    vllm_gpu_memory_utilization: float = 0.85
    # Helps keep torch/vLLM GPU index mapping aligned with NVML/pynvml.
    cuda_device_order: str = "PCI_BUS_ID"
    # Runtime-stability knobs for vLLM on mixed consumer GPUs.
    vllm_enforce_eager: bool = True
    # Leave unset by default; vLLM will auto-pick a backend supported by runtime.
    vllm_attention_backend: str | None = None
    # Reuse KV cache for repeated prompt prefixes and shared system prompts.
    vllm_enable_prefix_caching: bool = True
    # Concurrency limits. Leave unset to use vLLM defaults.
    vllm_max_num_seqs: int | None = 16
    vllm_max_num_batched_tokens: int | None = 8192
    vllm_tensor_parallel_size: int | None = None
    vllm_max_model_len: int | None = None
    vllm_model_impl: str | None = None
    vllm_runner: str | None = None
    vllm_hf_overrides: str | None = None
    vllm_chat_template: str | None = None
    vllm_chat_template_content_format: str | None = None
    vllm_generation_config: str | None = None
    vllm_override_generation_config: str | None = None
    vllm_tool_call_parser: str | None = None
    vllm_tool_parser_plugin: str | None = None
    vllm_reasoning_parser: str | None = None
    vllm_language_model_only: bool | None = None
    vllm_enable_chunked_prefill: bool | None = None
    vllm_swap_space: float | None = None
    # Quantized KV cache can increase effective context capacity on Ampere/Ada/Hopper.
    # Leave unset to keep the runtime default.
    vllm_kv_cache_dtype: str | None = None
    # Needed for some custom-tokenizer/custom-model repos on Hugging Face.
    vllm_trust_remote_code: bool = False
    vllm_disable_log_requests: bool = True
    # llama.cpp / llama-server
    llama_cpp_server_bin: str = "/usr/local/bin/llama-server"
    llama_cpp_gpu_layers: int = 0
    llama_cpp_ctx_size: int = 4096
    llama_cpp_threads: int | None = None
    llama_cpp_batch_size: int = 512
    llama_cpp_ubatch_size: int = 128
    llama_cpp_flash_attn: bool = False
    llama_cpp_mlock: bool = True
    llama_cpp_embeddings: bool = False
    llama_cpp_startup_timeout_s: int = 30
    # SGLang
    sglang_python_bin: str = "/opt/sglang-venv/bin/python"
    sglang_server_module: str = "sglang.launch_server"
    sglang_tensor_parallel_size: int | None = None
    sglang_context_length: int | None = None
    sglang_mem_fraction_static: float = 0.9
    sglang_trust_remote_code: bool = False
    sglang_disable_radix_cache: bool = False
    sglang_startup_timeout_s: int = 120
    # Chatterbox TTS
    chatterbox_python_bin: str = "/opt/chatterbox-venv/bin/python"
    chatterbox_model_name: str = "ResembleAI/chatterbox-turbo"
    # Voxtral TTS (vllm-omni)
    voxtral_python_bin: str = "/opt/voxtral-venv/bin/python"
    voxtral_startup_timeout_s: int = 300
    # TensorRT-LLM
    tensorrt_llm_enabled: bool = False
    tensorrt_llm_launch_mode: str = "binary"
    tensorrt_llm_python_bin: str = "/usr/bin/python3"
    tensorrt_llm_serve_bin: str = "/usr/local/bin/trtllm-serve"
    tensorrt_llm_serve_module: str = "tensorrt_llm.commands.serve"
    tensorrt_llm_docker_bin: str = "/usr/bin/docker"
    tensorrt_llm_docker_image: str = "nvcr.io/nvidia/tensorrt-llm/release:latest"
    tensorrt_llm_host_helper_image: str = "ocabra-api"
    tensorrt_llm_docker_models_mount_host: str = "/docker/ai-models/ocabra/models"
    tensorrt_llm_docker_models_mount_container: str = "/data/models"
    tensorrt_llm_docker_hf_cache_mount_host: str | None = "/docker/ai-models/ocabra/hf_cache"
    tensorrt_llm_docker_hf_cache_mount_container: str = "/data/hf_cache"
    tensorrt_llm_engines_dir: str | None = None
    tensorrt_llm_backend: str = "trt"
    tensorrt_llm_tokenizer_path: str | None = None
    tensorrt_llm_max_batch_size: int | None = None
    tensorrt_llm_context_length: int | None = None
    tensorrt_llm_trust_remote_code: bool = False
    tensorrt_llm_startup_timeout_s: int = 120
    # BitNet / bitnet.cpp (llama-server compatible)
    bitnet_server_bin: str = "/usr/local/bin/bitnet-server"
    bitnet_gpu_layers: int = 0
    bitnet_ctx_size: int = 4096
    bitnet_threads: int | None = None
    bitnet_batch_size: int = 512
    bitnet_ubatch_size: int = 128
    bitnet_parallel: int = 1
    bitnet_flash_attn: bool = False
    bitnet_mlock: bool = True
    bitnet_startup_timeout_s: int = 30
    # Diffusers worker tuning.
    diffusers_torch_dtype: str = "auto"
    diffusers_enable_torch_compile: bool = False
    diffusers_enable_xformers: bool = False
    diffusers_offload_mode: str = "none"
    diffusers_allow_tf32: bool = True
    vram_buffer_mb: int = 512
    vram_pressure_threshold_pct: float = 90.0
    # 11.1: Proactive VRAM eviction threshold (fraction 0.0-1.0).
    # When used VRAM exceeds this fraction, the VRAM watchdog evicts LRU models.
    vram_eviction_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Fraction of total VRAM above which LRU models are proactively evicted.",
    )

    # OpenAI audio uploads
    # Maximum allowed size per multipart part (in MB) for /v1/audio/transcriptions.
    openai_audio_max_part_size_mb: int = 256
    # Whisper worker startup deadline (seconds). Diarization cold starts may need extra time.
    whisper_startup_timeout_s: int = 300

    # Realtime API defaults (STT/TTS models for /v1/realtime sessions)
    realtime_default_stt_model: str = ""
    realtime_default_tts_model: str = ""

    # 11.4: Busy timeout for individual requests
    busy_timeout_seconds: int = Field(
        default=300,
        ge=30,
        description="Max seconds for a single request before the model is marked ERROR.",
    )
    busy_timeout_action: str = Field(
        default="mark_error",
        description="Action on busy timeout: 'mark_error' or 'restart_worker'.",
    )

    # 11.2: Worker health monitoring and auto-restart
    worker_health_check_interval_seconds: int = Field(default=10, ge=1)
    auto_restart_workers: bool = Field(default=True)
    max_worker_restarts: int = Field(
        default=3, ge=0, description="Consecutive restarts before giving up."
    )
    worker_restart_backoff_seconds: float = Field(default=5.0, ge=1.0)

    # Model lifecycle
    idle_timeout_seconds: int = 300
    idle_eviction_check_interval_seconds: int = 15
    # How long (seconds) ensure_loaded waits for a model that is currently loading.
    # Covers: eviction of a busy model (drain) + actual model startup time.
    # Large models (vLLM 70B, Whisper large-v3 + diarization) can take 10+ minutes.
    model_load_wait_timeout_s: int = 720
    # Max seconds to wait for in-flight requests to drain before pressure-evicting a model.
    pressure_eviction_drain_timeout_s: int = 60

    # Interactive generation services
    hunyuan_base_url: str = "http://hunyuan:8080"
    comfyui_base_url: str = "http://comfyui:8188"
    a1111_base_url: str = "http://a1111:7860"
    hunyuan_ui_url: str = ""
    comfyui_ui_url: str = ""
    a1111_ui_url: str = ""
    hunyuan_preferred_gpu: int = 1
    comfyui_preferred_gpu: int = 1
    a1111_preferred_gpu: int = 1
    hunyuan_idle_unload_seconds: int = 300
    comfyui_idle_unload_seconds: int = 600
    a1111_idle_unload_seconds: int = 600
    hunyuan_docker_container: str = "ocabra-hunyuan-1"
    comfyui_docker_container: str = "ocabra-comfyui-1"
    a1111_docker_container: str = "ocabra-a1111-1"
    # Seconds to wait for an active generation to finish before forcing eviction.
    # 0 = evict immediately, -1 = wait indefinitely (not recommended for pressure eviction).
    hunyuan_generation_grace_period_s: int = 120
    comfyui_generation_grace_period_s: int = 120
    a1111_generation_grace_period_s: int = 120

    # LiteLLM
    litellm_base_url: str = "http://litellm:4000"
    litellm_admin_key: str = ""
    litellm_auto_sync: bool = False

    # ACE-Step music generation — backend (subprocess) mode
    acestep_project_dir: str = "/docker/ACE-Step-1.5"
    acestep_startup_timeout_s: int = 300
    acestep_generation_timeout_s: int = 600
    acestep_poll_interval_s: float = 1.0
    # ACE-Step — service/dashboard mode (standalone Gradio or API instance)
    # In Gradio mode (default): base_url = http://acestep:7860, health_path = /
    # In API mode: base_url = http://acestep:8001, health_path = /health
    acestep_base_url: str = "http://acestep:7860"
    acestep_ui_url: str = ""
    acestep_preferred_gpu: int = 1
    acestep_idle_unload_seconds: int = 1800  # 0 = disabled; default 30 min
    acestep_docker_container: str = "ocabra-acestep-1"
    acestep_generation_grace_period_s: int = 180
    # GPU utilization threshold (%) used to infer active generation for services that
    # have no dedicated generation-status endpoint (Hunyuan, ACE-Step).
    generation_gpu_util_threshold_pct: int = 50
    # Optional: if set, AceStepBackend will proxy to this external ACE-Step REST API
    # instead of spawning a local subprocess (use when running ACE-Step as a Docker service
    # with ACESTEP_MODE=api). Example: http://acestep:8001
    acestep_external_api_url: str = ""

    # Unsloth Studio — training/fine-tuning UI + chat + GGUF export
    # No-code WebUI servida por la imagen oficial `unsloth/unsloth`. Se trata
    # como un servicio más (mismo patrón que ACE-Step / ComfyUI): gestiona su
    # propio ciclo de vida y expone su UI vía iframe a través del gateway.
    unsloth_base_url: str = "http://unsloth-studio:8000"
    unsloth_ui_url: str = ""
    unsloth_preferred_gpu: int = 1
    # Training puede durar horas; 0 = nunca evictar por inactividad.
    unsloth_idle_unload_seconds: int = 0
    unsloth_docker_container: str = "ocabra-unsloth-studio-1"
    # -1 = esperar indefinidamente a que el training/generación activo termine
    # (capado a 30 s sólo para evicciones por presión de VRAM).
    unsloth_generation_grace_period_s: int = -1

    # Docker Compose project directory (for `docker compose up -d <service>`)
    compose_project_dir: str = "/opt/ocabra"

    # Langfuse observability (desactivado por defecto)
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = ""
    langfuse_secret_key: str | None = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_capture_content: bool = False
    langfuse_sample_rate: float = 1.0
    langfuse_flush_interval_s: float = 2.0

    # Energy
    energy_cost_eur_kwh: float = 0.15

    # OpenAI-equivalent pricing (USD per 1M tokens) for cost estimation
    openai_ref_small_input: float = 0.15
    openai_ref_small_output: float = 0.60
    openai_ref_medium_input: float = 2.50
    openai_ref_medium_output: float = 10.00
    openai_ref_large_input: float = 10.00
    openai_ref_large_output: float = 30.00
    openai_ref_embedding_input: float = 0.10
    openai_ref_audio_stt_input: float = 6.00
    openai_ref_tts_input: float = 15.00
    openai_ref_image_input: float = 40.00

    # Auth
    jwt_secret: str = Field(default_factory=lambda: secrets.token_hex(32))
    jwt_ttl_hours: int = 24
    jwt_remember_days: int = 30
    ocabra_admin_user: str = "ocabra"
    ocabra_admin_pass: str = "ocabra"
    use_https: bool = False

    # API access control
    require_api_key_openai: bool = True
    require_api_key_ollama: bool = True

    # OpenAI Files + Batches API
    openai_files_dir: str = "/data/openai_files"
    batch_max_concurrency: int = Field(
        default=4, ge=1, description="Max concurrent requests dispatched per batch."
    )
    batch_request_timeout_seconds: int = Field(
        default=600, ge=10, description="Per-request timeout when dispatching batch lines."
    )
    batch_poll_interval_seconds: int = Field(
        default=5, ge=2, description="How often the batch processor polls for pending batches."
    )

    # Profile resolution
    # When True, canonical model_id (containing '/') is resolved to its default
    # profile with a deprecation warning.  Set to False to enforce profile_id-only
    # access (returns 404 for raw model IDs).
    legacy_model_id_fallback: bool = True

    # Gateway service-to-service token (empty = disabled)
    gateway_service_token: str = ""

    # Federation (multi-node peer-to-peer inference)
    federation_enabled: bool = Field(
        default=False, description="Enable federation mode for multi-node inference"
    )
    federation_node_id: str = Field(
        default="", description="Unique identifier for this node (auto-generated if empty)"
    )
    federation_node_name: str = Field(
        default="", description="Human-readable name for this node"
    )
    federation_heartbeat_interval: int = Field(
        default=30, ge=5, description="Seconds between heartbeat polls to peers"
    )
    federation_proxy_timeout_s: int = Field(
        default=300, ge=10, description="Timeout for proxied requests to peers"
    )
    federation_verify_ssl: bool = Field(
        default=True, description="Verify TLS certificates for peer connections"
    )

    # Agents + MCP (plan: docs/tasks/agents-mcp-plan.md)
    mcp_tools_cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="TTL for cached tools/list responses from MCP servers.",
    )
    mcp_default_tool_timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="Default per-tool-call timeout when the agent does not override it.",
    )
    mcp_max_concurrent_tool_calls: int = Field(
        default=8,
        ge=1,
        description="Max concurrent tool_calls dispatched by the AgentExecutor per turn.",
    )
    mcp_result_max_bytes: int = Field(
        default=262144,
        ge=1024,
        description="Maximum size (bytes) of a tool_call result payload before truncation.",
    )
    mcp_stdio_allowed: bool = Field(
        default=True,
        description=(
            "If false, MCP servers with transport=stdio cannot be created even by admins."
        ),
    )

    @field_validator(
        "langfuse_public_key",
        "langfuse_secret_key",
        "vllm_attention_backend",
        "vllm_tensor_parallel_size",
        "vllm_max_model_len",
        "vllm_model_impl",
        "vllm_runner",
        "vllm_hf_overrides",
        "vllm_chat_template",
        "vllm_chat_template_content_format",
        "vllm_generation_config",
        "vllm_override_generation_config",
        "vllm_tool_call_parser",
        "vllm_tool_parser_plugin",
        "vllm_reasoning_parser",
        "vllm_language_model_only",
        "vllm_enable_chunked_prefill",
        "vllm_swap_space",
        "vllm_kv_cache_dtype",
        "llama_cpp_threads",
        "sglang_python_bin",
        "sglang_tensor_parallel_size",
        "sglang_context_length",
        "bitnet_threads",
        "tensorrt_llm_python_bin",
        "tensorrt_llm_serve_module",
        "tensorrt_llm_docker_hf_cache_mount_host",
        "tensorrt_llm_engines_dir",
        "tensorrt_llm_tokenizer_path",
        "tensorrt_llm_max_batch_size",
        "tensorrt_llm_context_length",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("diffusers_offload_mode")
    @classmethod
    def _validate_diffusers_offload_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"none", "model", "sequential"}:
            raise ValueError("diffusers_offload_mode must be one of: none, model, sequential")
        return normalized

    @field_validator("tensorrt_llm_launch_mode")
    @classmethod
    def _validate_tensorrt_launch_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"binary", "module", "docker"}:
            raise ValueError("tensorrt_llm_launch_mode must be one of: binary, module, docker")
        return normalized


settings = Settings()
