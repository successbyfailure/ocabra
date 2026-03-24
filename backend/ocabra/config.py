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
    # TensorRT-LLM
    tensorrt_llm_enabled: bool = False
    tensorrt_llm_launch_mode: str = "binary"
    tensorrt_llm_python_bin: str = "/usr/bin/python3"
    tensorrt_llm_serve_bin: str = "/usr/local/bin/trtllm-serve"
    tensorrt_llm_serve_module: str = "tensorrt_llm.commands.serve"
    tensorrt_llm_docker_bin: str = "/usr/bin/docker"
    tensorrt_llm_docker_image: str = "nvcr.io/nvidia/tensorrt-llm/release:latest"
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

    # OpenAI audio uploads
    # Maximum allowed size per multipart part (in MB) for /v1/audio/transcriptions.
    openai_audio_max_part_size_mb: int = 256
    # Whisper worker startup deadline (seconds). Diarization cold starts may need extra time.
    whisper_startup_timeout_s: int = 300

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
