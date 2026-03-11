from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class HFModelCard(BaseModel):
    repo_id: str
    model_name: str
    task: str | None
    downloads: int
    likes: int
    size_gb: float | None
    tags: list[str]
    gated: bool


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
    backend_type: Literal["vllm", "diffusers", "whisper", "tts", "ollama"]
    is_default: bool = False


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
    backend_type: Literal["vllm", "diffusers", "whisper", "tts", "ollama"]
    size_gb: float | None


class DownloadJob(BaseModel):
    job_id: str
    source: Literal["huggingface", "ollama"]
    model_ref: str
    artifact: str | None = None
    status: Literal["queued", "downloading", "completed", "failed", "cancelled"]
    progress_pct: float
    speed_mb_s: float | None
    eta_seconds: int | None
    error: str | None
    started_at: datetime
    completed_at: datetime | None
