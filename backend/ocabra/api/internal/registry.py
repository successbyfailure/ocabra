from pathlib import Path

from fastapi import APIRouter, Query

from ocabra.config import settings
from ocabra.registry.huggingface import HuggingFaceRegistry
from ocabra.registry.local_scanner import LocalScanner
from ocabra.registry.ollama_registry import OllamaRegistry
from ocabra.schemas.registry import HFModelCard, HFModelDetail, LocalModel, OllamaModelCard

router = APIRouter(tags=["registry"])

_hf_registry = HuggingFaceRegistry()
_ollama_registry = OllamaRegistry()
_local_scanner = LocalScanner()


@router.get("/registry/hf/search", response_model=list[HFModelCard])
@router.get("/registry/hf", response_model=list[HFModelCard])
async def search_hf_models(
    q: str = Query(default="", description="Text query"),
    task: str | None = Query(default=None, description="HF pipeline task"),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[HFModelCard]:
    return await _hf_registry.search(query=q, task=task, limit=limit)


@router.get("/registry/hf/{repo_id:path}", response_model=HFModelDetail)
async def get_hf_model_detail(repo_id: str) -> HFModelDetail:
    return await _hf_registry.get_model_detail(repo_id)


@router.get("/registry/ollama/search", response_model=list[OllamaModelCard])
@router.get("/registry/ollama", response_model=list[OllamaModelCard])
async def search_ollama_models(
    q: str = Query(default="", description="Text query"),
) -> list[OllamaModelCard]:
    return await _ollama_registry.search(query=q)


@router.get("/registry/local", response_model=list[LocalModel])
async def list_local_models() -> list[LocalModel]:
    return await _local_scanner.scan(Path(settings.models_dir))
