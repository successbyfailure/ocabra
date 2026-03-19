from pydantic import field_validator
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
    vram_buffer_mb: int = 512
    vram_pressure_threshold_pct: float = 90.0

    # Model lifecycle
    idle_timeout_seconds: int = 300
    idle_eviction_check_interval_seconds: int = 15

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
    a1111_docker_container: str = "ocabra-a1111-1"

    # LiteLLM
    litellm_base_url: str = "http://litellm:4000"
    litellm_admin_key: str = ""
    litellm_auto_sync: bool = False

    # Energy
    energy_cost_eur_kwh: float = 0.15

    @field_validator(
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
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


settings = Settings()
