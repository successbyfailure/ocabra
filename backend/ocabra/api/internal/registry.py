import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.core.model_ref import (
    KNOWN_BACKEND_TYPES,
    LLAMA_CPP_CONFIG_SECTION,
    build_model_ref,
    is_llama_cpp_family,
)
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
    description=(
        "Scan the models directory and return all locally available model "
        "files. Results are cached for ~30 s — pass ``?refresh=true`` to "
        "force a re-scan after a manual file change."
    ),
)
async def list_local_models(
    refresh: bool = Query(default=False, description="Bypass the 30-s scan cache"),
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[LocalModel]:
    if refresh:
        _local_scanner.invalidate()
    ollama_shared = (
        Path(settings.ollama_shared_models_dir)
        if settings.ollama_shared_models_dir
        else None
    )
    return await _local_scanner.scan(Path(settings.models_dir), ollama_shared)


def invalidate_local_scan() -> None:
    """Drop the cached LocalScanner result so the next scan re-walks disk.

    Call this from download/delete endpoints that mutate the models tree.
    Exported as a module-level helper so other routers can import it without
    poking at the private ``_local_scanner`` singleton directly.
    """
    _local_scanner.invalidate()


class RegisterLocalModelRequest(BaseModel):
    """Register a model whose weights are already on disk.

    Covers the two cases the download flow cannot: a GGUF that was fetched
    outside oCabra (or by an earlier download whose backend choice no longer
    fits), and an Ollama blob — Ollama downloads a GGUF that our llama.cpp
    backends can serve directly, so pointing at it avoids re-downloading
    several GB that are already on the disk.

    ``backend_type`` is explicit on purpose. The scanner can only guess one
    backend per file, and the same GGUF is servable by upstream llama.cpp, the
    Makespace fork or bitnet.
    """

    model_config = ConfigDict(extra="forbid")

    model_ref: str = Field(..., min_length=1, description="Backend-native id, e.g. 'gemma4-26b'")
    path: str = Field(..., min_length=1, description="Absolute path to the weights on disk")
    backend_type: str = Field(..., min_length=1)
    display_name: str | None = None
    load_policy: str = "on_demand"
    preferred_gpu: int | None = None


@router.post(
    "/registry/local/register",
    summary="Register an already-downloaded model under a chosen backend",
    description=(
        "Create a model configuration for weights that are already on disk, "
        "pinning the exact file so the backend does not have to guess it. "
        "Use this to attach an Ollama-downloaded GGUF to a llama.cpp backend "
        "without re-downloading it, or to serve the same file with a different "
        "backend than the one the scanner inferred."
    ),
    responses={
        400: {"description": "Unknown backend, or path outside the allowed roots"},
        404: {"description": "Path does not exist"},
        409: {"description": "That model id is already registered"},
    },
)
async def register_local_model(
    body: RegisterLocalModelRequest,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    if body.backend_type not in KNOWN_BACKEND_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown backend '{body.backend_type}'")

    path = Path(body.path)
    # Only paths under a configured root may be registered: this endpoint takes
    # a filesystem path from an API caller, and without the check it would
    # happily point a worker at any readable file on the host.
    roots = [Path(settings.models_dir)]
    if settings.ollama_shared_models_dir:
        roots.append(Path(settings.ollama_shared_models_dir))
    try:
        # Off-thread: resolve()/is_file() hit the filesystem, and this is an
        # async handler on the event loop.
        resolved = await asyncio.to_thread(path.resolve)
        if not any(resolved.is_relative_to(root.resolve()) for root in roots):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Refusing to register '{path}': outside the configured roots "
                    f"({', '.join(str(r) for r in roots)})"
                ),
            )
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path '{path}'") from exc
    if not await asyncio.to_thread(resolved.is_file):
        raise HTTPException(status_code=404, detail=f"No such file: '{path}'")

    model_id = build_model_ref(body.backend_type, body.model_ref)
    mm = request.app.state.model_manager
    if await mm.get_state(model_id) is not None:
        raise HTTPException(status_code=409, detail=f"'{model_id}' is already registered")

    # Pin the exact file. Both llama.cpp backends and bitnet read ``model_path``
    # from their own extra_config section; without it they fall back to
    # directory/stem heuristics that cannot find an Ollama blob (its filename
    # is a sha256 digest, unrelated to the model name).
    extra_config: dict = {}
    if is_llama_cpp_family(body.backend_type):
        extra_config = {LLAMA_CPP_CONFIG_SECTION: {"model_path": str(resolved)}}
    elif body.backend_type == "bitnet":
        extra_config = {"bitnet": {"model_path": str(resolved)}}
    else:
        extra_config = {"model_path": str(resolved)}

    try:
        state = await mm.add_model(
            model_id=model_id,
            backend_type=body.backend_type,
            display_name=body.display_name or body.model_ref,
            load_policy=body.load_policy,
            preferred_gpu=body.preferred_gpu,
            extra_config=extra_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Same hook the download flow uses: without it the model stays invisible to
    # /v1/models (clients resolve profiles, not canonical ids) until a restart.
    profile_registry = getattr(request.app.state, "profile_registry", None)
    if profile_registry is not None:
        await profile_registry.on_model_added(state.model_id)
    invalidate_local_scan()
    return state.to_dict()
