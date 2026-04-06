from pathlib import Path

from fastapi import APIRouter, Depends, Query

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.registry.bitnet_registry import BitnetRegistry
from ocabra.registry.huggingface import HuggingFaceRegistry
from ocabra.registry.local_scanner import LocalScanner
from ocabra.registry.ollama_registry import OllamaRegistry
from ocabra.schemas.registry import (
    HFModelCard,
    HFModelDetail,
    HFModelVariant,
    LocalModel,
    OllamaModelCard,
    OllamaModelVariant,
)

router = APIRouter(tags=["registry"])

_hf_registry = HuggingFaceRegistry()
_ollama_registry = OllamaRegistry()
_local_scanner = LocalScanner()
_bitnet_registry = BitnetRegistry()


@router.get("/registry/hf/search", response_model=list[HFModelCard], summary="Search HuggingFace models")
@router.get("/registry/hf", response_model=list[HFModelCard], summary="Search HuggingFace models")
async def search_hf_models(
    q: str = Query(default="", description="Text query"),
    task: str | None = Query(default=None, description="HF pipeline task"),
    limit: int = Query(default=20, ge=1, le=100),
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[HFModelCard]:
    return await _hf_registry.search(query=q, task=task, limit=limit)


@router.get(
    "/registry/hf/{repo_id:path}/variants",
    response_model=list[HFModelVariant],
    summary="List HuggingFace model variants",
    description="Return downloadable variants (GGUF files, safetensors shards, etc.) for a HuggingFace repo.",
)
async def get_hf_variants(
    repo_id: str,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[HFModelVariant]:
    return await _hf_registry.get_variants(repo_id)


@router.get(
    "/registry/hf/{repo_id:path}",
    response_model=HFModelDetail,
    summary="Get HuggingFace model detail",
    description="Return detailed metadata for a single HuggingFace repository.",
)
async def get_hf_model_detail(
    repo_id: str,
    _user: UserContext = Depends(require_role("model_manager")),
) -> HFModelDetail:
    return await _hf_registry.get_model_detail(repo_id)




@router.get("/registry/bitnet/search", response_model=list[HFModelCard], summary="Search BitNet models")
@router.get("/registry/bitnet", response_model=list[HFModelCard], summary="Search BitNet models")
async def search_bitnet_models(
    q: str = Query(default="", description="Text query"),
    limit: int = Query(default=20, ge=1, le=100),
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[HFModelCard]:
    return await _bitnet_registry.search(query=q, limit=limit)


@router.get(
    "/registry/bitnet/{repo_id:path}/variants",
    response_model=list[HFModelVariant],
    summary="List BitNet model variants",
    description="Return downloadable GGUF variants for a BitNet repository.",
)
async def get_bitnet_variants(
    repo_id: str,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[HFModelVariant]:
    return await _bitnet_registry.get_variants(repo_id=repo_id)

@router.get("/registry/ollama/search", response_model=list[OllamaModelCard], summary="Search Ollama models")
@router.get("/registry/ollama", response_model=list[OllamaModelCard], summary="Search Ollama models")
async def search_ollama_models(
    q: str = Query(default="", description="Text query"),
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[OllamaModelCard]:
    return await _ollama_registry.search(query=q)


@router.get(
    "/registry/ollama/{model_name}/variants",
    response_model=list[OllamaModelVariant],
    summary="List Ollama model variants",
    description="Return available tags/variants for an Ollama model.",
)
async def get_ollama_variants(
    model_name: str,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[OllamaModelVariant]:
    return await _ollama_registry.get_variants(model_name=model_name)


@router.get(
    "/registry/local",
    response_model=list[LocalModel],
    summary="List local models",
    description="Scan the models directory and return all locally available model files.",
)
async def list_local_models(
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[LocalModel]:
    return await _local_scanner.scan(Path(settings.models_dir))
