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


class OllamaModelCard(BaseModel):
    name: str
    description: str
    tags: list[str]
    size_gb: float | None
    pulls: int


class LocalModel(BaseModel):
    model_ref: str
    path: str
    source: Literal["huggingface", "gguf", "ollama"]
    backend_type: Literal["vllm", "diffusers", "whisper", "tts"]
    size_gb: float | None


class DownloadJob(BaseModel):
    job_id: str
    source: Literal["huggingface", "ollama"]
    model_ref: str
    status: Literal["queued", "downloading", "completed", "failed", "cancelled"]
    progress_pct: float
    speed_mb_s: float | None
    eta_seconds: int | None
    error: str | None
    started_at: datetime
    completed_at: datetime | None
