from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HFVLLMRuntimeProbe(BaseModel):
    status: Literal[
        "supported_native",
        "supported_transformers_backend",
        "supported_pooling",
        "needs_remote_code",
        "missing_chat_template",
        "missing_tool_parser",
        "missing_reasoning_parser",
        "needs_hf_overrides",
        "unsupported_tokenizer",
        "unsupported_architecture",
        "unavailable",
        "unknown",
    ] = "unknown"
    reason: str | None = None
    recommended_model_impl: Literal["auto", "vllm", "transformers"] | None = None
    recommended_runner: Literal["generate", "pooling"] | None = None
    tokenizer_load: bool | None = None
    config_load: bool | None = None
    observed_at: datetime | None = None


class HFVLLMSupport(BaseModel):
    classification: Literal[
        "native_vllm",
        "transformers_backend",
        "pooling",
        "unsupported",
        "unknown",
    ] = "unknown"
    label: str = "unknown"
    model_impl: Literal["auto", "vllm", "transformers"] | None = None
    runner: Literal["generate", "pooling"] | None = None
    task_mode: (
        Literal[
            "generate",
            "multimodal_generate",
            "pooling",
            "multimodal_pooling",
        ]
        | None
    ) = None
    required_overrides: list[str] = Field(default_factory=list)
    recipe_id: str | None = None
    recipe_notes: list[str] = Field(default_factory=list)
    recipe_model_impl: Literal["auto", "vllm", "transformers"] | None = None
    recipe_runner: Literal["generate", "pooling"] | None = None
    suggested_config: dict[str, Any] = Field(default_factory=dict)
    suggested_tuning: dict[str, Any] = Field(default_factory=dict)
    runtime_probe: HFVLLMRuntimeProbe | None = None


class HFModelCard(BaseModel):
    repo_id: str
    model_name: str
    task: str | None
    downloads: int
    likes: int
    size_gb: float | None
    tags: list[str]
    gated: bool
    suggested_backend: str
    compatibility: str = "unknown"
    compatibility_reason: str | None = None
    vllm_support: HFVLLMSupport | None = None


class HFModelDetail(HFModelCard):
    siblings: list[dict]
    readme_excerpt: str | None
    suggested_backend: str
    estimated_vram_gb: float | None


class HFModelVariant(BaseModel):
    variant_id: str
    label: str
    artifact: str | None
    size_gb: float | None
    format: str
    quantization: str | None
    backend_type: Literal["vllm", "acestep", "llama_cpp", "sglang", "tensorrt_llm", "bitnet", "diffusers", "whisper", "tts", "ollama"]
    is_default: bool = False
    installable: bool = True
    compatibility: str = "unknown"
    compatibility_reason: str | None = None
    vllm_support: HFVLLMSupport | None = None


class OllamaModelCard(BaseModel):
    name: str
    description: str
    tags: list[str]
    size_gb: float | None
    pulls: int


class OllamaModelVariant(BaseModel):
    name: str
    tag: str
    size_gb: float | None
    parameter_size: str | None
    quantization: str | None
    context_window: str | None
    modality: str | None
    updated_hint: str | None


class LocalModel(BaseModel):
    model_ref: str
    path: str
    source: Literal["huggingface", "gguf", "ollama"]
    backend_type: Literal["vllm", "acestep", "llama_cpp", "sglang", "tensorrt_llm", "bitnet", "diffusers", "whisper", "tts", "ollama"]
    size_gb: float | None


class DownloadJob(BaseModel):
    job_id: str
    source: Literal["huggingface", "ollama", "bitnet"]
    model_ref: str
    artifact: str | None = None
    register_config: dict[str, Any] | None = None
    status: Literal["queued", "downloading", "completed", "failed", "cancelled"]
    progress_pct: float
    speed_mb_s: float | None
    eta_seconds: int | None
    error: str | None
    started_at: datetime
    completed_at: datetime | None
