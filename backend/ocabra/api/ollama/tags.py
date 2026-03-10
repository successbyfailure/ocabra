"""
GET /api/tags — list installed models in Ollama-compatible format.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Request

from ocabra.config import settings

from ._mapper import OllamaNameMapper

router = APIRouter()
_mapper = OllamaNameMapper()


@router.get("/tags", summary="List models")
async def list_tags(request: Request) -> dict:
    """
    List all configured models in Ollama /api/tags format.

    Returns:
      {"models": [{"name": ..., "model": ..., "size": ..., "details": {...}}, ...]}
    """
    model_manager = request.app.state.model_manager
    states = await model_manager.list_states()

    models: list[dict] = []
    for state in states:
        ollama_name = _mapper.to_ollama(state.model_id)
        family = ollama_name.split(":", 1)[0]
        models.append(
            {
                "name": ollama_name,
                "model": ollama_name,
                "modified_at": _to_iso_z(state.loaded_at),
                "size": _estimate_size_bytes(state.model_id, state.vram_used_mb),
                "digest": f"sha256:{hashlib.sha256(state.model_id.encode('utf-8')).hexdigest()}",
                "details": {
                    "parent_model": "",
                    "format": _infer_format(state.model_id),
                    "family": family,
                    "families": [family],
                    "parameter_size": _infer_parameter_size(ollama_name),
                    "quantization_level": "F16",
                },
            }
        )

    return {"models": models}


def _to_iso_z(value: datetime | None) -> str:
    dt = value or datetime.now(UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _estimate_size_bytes(model_id: str, vram_used_mb: int) -> int:
    model_dir = Path(settings.models_dir) / model_id
    if model_dir.exists():
        size = sum(
            path.stat().st_size
            for path in model_dir.rglob("*")
            if path.is_file() and path.suffix in {".safetensors", ".bin", ".gguf"}
        )
        if size > 0:
            return int(size)

    return int(max(0, vram_used_mb) * 1024 * 1024)


def _infer_format(model_id: str) -> str:
    lower = model_id.lower()
    if lower.endswith(".gguf"):
        return "gguf"
    if "diffusion" in lower:
        return "safetensors"
    return "safetensors"


def _infer_parameter_size(ollama_name: str) -> str:
    if ":" not in ollama_name:
        return "unknown"
    tag = ollama_name.split(":", 1)[1].strip()
    return tag.upper()
