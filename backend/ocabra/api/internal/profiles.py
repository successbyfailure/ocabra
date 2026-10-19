"""Internal REST endpoints for model profile CRUD and asset management."""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.core.profile_registry import ProfileRegistry
from ocabra.core.runtime_preset_capacity import runtime_capacity_cache
from ocabra.database import AsyncSessionLocal
from ocabra.schemas.profiles import (
    ProfileAliasCreate,
    ProfileAliasOut,
    ProfileClone,
    ProfileCreate,
    ProfileOut,
    ProfileRename,
    ProfileUpdate,
    RouterCreate,
    RuntimePresetBenchmarkOut,
    RuntimePresetBenchmarkRequest,
    RuntimePresetCreate,
    RuntimePresetOut,
    RuntimePresetResolutionOut,
    RuntimePresetUpdate,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["profiles"])

PROFILES_ASSET_BASE = Path("/data/profiles")
ALLOWED_ASSET_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".safetensors",
    ".bin",
    ".pt",
    ".gguf",
    ".json",
}
MAX_ASSET_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


def _get_registry(request: Request) -> ProfileRegistry:
    registry = getattr(request.app.state, "profile_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Profile registry not available")
    return registry


def _profile_to_dict(profile) -> dict[str, Any]:
    return ProfileOut.model_validate(profile).model_dump(mode="json")


def _preset_to_dict(preset) -> dict[str, Any]:
    return RuntimePresetOut.model_validate(preset).model_dump(mode="json")


def _alias_to_dict(alias) -> dict[str, Any]:
    return ProfileAliasOut.model_validate(alias).model_dump(mode="json")


def _is_path_within_base(path: Path, base: Path) -> bool:
    """Ensure resolved path is within the base directory (path traversal protection)."""
    try:
        return path.resolve(strict=False).is_relative_to(base.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False


# ── Per-model profile endpoints ──────────────────────────────


@router.get(
    "/models/{model_id:path}/profiles",
    summary="List profiles for a model",
    description="Return all profiles associated with a given base model.",
)
async def list_model_profiles(
    model_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> list[dict]:
    """List all profiles for a specific base model."""
    registry = _get_registry(request)
    profiles = await registry.list_by_model(model_id)
    return [_profile_to_dict(p) for p in profiles]


@router.get(
    "/profiles",
    summary="List all profiles",
    description=(
        "Return every profile in the registry. Useful for admin surfaces "
        "that need to list routers (profiles with ``routing_targets``) "
        "across all base models."
    ),
)
async def list_all_profiles(
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> list[dict]:
    """List every profile."""
    registry = _get_registry(request)
    profiles = await registry.list_all()
    return [_profile_to_dict(p) for p in profiles]


@router.post(
    "/models/{model_id:path}/profiles",
    summary="Create a profile for a model",
    description="Create a new profile linked to the given base model.",
    status_code=201,
    responses={
        400: {"description": "Invalid profile data or duplicate profile_id"},
        404: {"description": "Base model not found"},
    },
)
async def create_model_profile(
    model_id: str,
    body: ProfileCreate,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Create a new profile for a base model."""
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        try:
            profile = await registry.create(
                session,
                profile_id=body.profile_id,
                base_model_id=model_id,
                profile_kind="execution",
                display_name=body.display_name,
                description=body.description,
                category=body.category,
                load_overrides=body.load_overrides,
                request_defaults=body.request_defaults,
                enabled=body.enabled,
                is_default=body.is_default,
                routing_targets=body.routing_targets,
                runtime_policy=(
                    body.runtime_policy.model_dump() if body.runtime_policy else None
                ),
            )
        except ValueError as exc:
            msg = str(exc)
            if "not found" in msg.lower():
                raise HTTPException(status_code=404, detail=msg) from exc
            raise HTTPException(status_code=400, detail=msg) from exc
    return _profile_to_dict(profile)


# ── Runtime preset endpoints ─────────────────────────────────


@router.get(
    "/profiles/{profile_id}/runtime-presets",
    summary="List profile runtime presets",
    description="Return the context/runtime presets configured for a public profile.",
    responses={404: {"description": "Profile not found"}},
)
async def list_runtime_presets(
    profile_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> list[dict]:
    """List a profile's runtime presets in selection/UI order."""
    registry = _get_registry(request)
    try:
        presets = await registry.list_runtime_presets(profile_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [_preset_to_dict(preset) for preset in presets]


@router.get(
    "/profiles/{profile_id}/runtime-presets/resolved",
    summary="Resolve profile runtime preset capacities",
    description=(
        "Calculate each preset's effective context against current model metadata "
        "and total GPU capacity without loading or unloading a model."
    ),
    responses={404: {"description": "Profile or base model not found"}},
)
async def resolve_runtime_presets(
    profile_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> list[dict]:
    """Return the current administrative capacity snapshot for a profile."""
    registry = _get_registry(request)
    profile = await registry.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    if profile.profile_kind != "execution" or profile.base_model_id is None:
        raise HTTPException(status_code=400, detail="Routers do not have runtime presets")
    model_manager = request.app.state.model_manager
    state = await model_manager.get_state(profile.base_model_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Base model '{profile.base_model_id}' not found",
        )
    gpu_manager = model_manager.gpu_manager
    gpu_states = await gpu_manager.get_all_states() if gpu_manager is not None else []
    rows = await runtime_capacity_cache.resolve(
        profile,
        state,
        gpu_states,
        force=True,
    )
    return [
        RuntimePresetResolutionOut.model_validate(row).model_dump(mode="json")
        for row in rows
    ]


@router.post(
    "/profiles/{profile_id}/runtime-presets/{preset_key}/benchmark",
    summary="Measure one runtime preset",
    response_model=RuntimePresetBenchmarkOut,
)
async def benchmark_runtime_preset(
    profile_id: str,
    preset_key: str,
    body: RuntimePresetBenchmarkRequest,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Measure load latency and generation throughput for a selected preset."""
    from ocabra.api.openai._deps import ensure_worker_loaded, to_backend_body
    from ocabra.core.runtime_coordinator import plan_profile_runtime
    from ocabra.core.runtime_presets import ContextCapacityError

    registry = _get_registry(request)
    profile = await registry.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    if profile.profile_kind != "execution" or profile.base_model_id is None:
        raise HTTPException(status_code=400, detail="Routers cannot be benchmarked")
    manager = request.app.state.model_manager
    try:
        plan = await plan_profile_runtime(
            profile,
            manager,
            {
                "messages": [{"role": "user", "content": body.prompt}],
                "max_tokens": body.max_tokens,
                "ocabra": {"context_preset": preset_key},
            },
        )
    except ContextCapacityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    before = await manager.get_state(plan.worker_key)
    was_cold = before is None or before.status.value != "loaded"
    if body.cold_start and before is not None and before.status.value == "loaded":
        if manager.is_busy(plan.worker_key):
            raise HTTPException(status_code=409, detail="Runtime is serving a request")
        await manager.unload(plan.worker_key, reason="runtime_benchmark")
        was_cold = True

    load_started = time.monotonic()
    state = await ensure_worker_loaded(
        manager,
        profile.base_model_id,
        plan.worker_key,
        plan.load_overrides,
        preferred_gpu=plan.preferred_gpu,
        load_policy=plan.load_policy,
        required_tokens=plan.required_tokens,
    )
    load_ms = int((time.monotonic() - load_started) * 1000)
    generation_body = {
        "model": state.backend_model_id,
        "messages": [{"role": "user", "content": body.prompt}],
        "max_tokens": body.max_tokens,
        "stream": False,
    }
    generation_started = time.monotonic()
    result = await request.app.state.worker_pool.forward_request(
        plan.worker_key,
        "/v1/chat/completions",
        to_backend_body(state, generation_body),
    )
    generation_ms = max(1, int((time.monotonic() - generation_started) * 1000))
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    output_tokens = usage.get("completion_tokens")
    tokens_per_second = (
        round(float(output_tokens) * 1000 / generation_ms, 2)
        if isinstance(output_tokens, int)
        else None
    )
    return RuntimePresetBenchmarkOut(
        preset_key=preset_key,
        effective_context=int(plan.effective_context or 0),
        worker_key=plan.worker_key,
        was_cold=was_cold,
        load_ms=load_ms,
        generation_ms=generation_ms,
        output_tokens=output_tokens,
        tokens_per_second=tokens_per_second,
    ).model_dump(mode="json")


@router.post(
    "/profiles/{profile_id}/runtime-presets",
    summary="Create profile runtime preset",
    description=(
        "Create a context/runtime preset. The first preset becomes the enabled "
        "default automatically. Configuration is applied on a later runtime load."
    ),
    status_code=201,
    responses={
        400: {"description": "Invalid or duplicate preset"},
        404: {"description": "Profile not found"},
    },
)
async def create_runtime_preset(
    profile_id: str,
    body: RuntimePresetCreate,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Create one runtime preset under ``profile_id``."""
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        try:
            preset = await registry.create_runtime_preset(
                session,
                profile_id,
                body.model_dump(),
            )
        except ValueError as exc:
            status = 404 if "profile" in str(exc).lower() and "not found" in str(exc).lower() else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
    return _preset_to_dict(preset)


@router.patch(
    "/profiles/{profile_id}/runtime-presets/{preset_key}",
    summary="Update profile runtime preset",
    description="Patch a runtime preset after validating the complete merged configuration.",
    responses={
        400: {"description": "Invalid preset patch"},
        404: {"description": "Profile or preset not found"},
    },
)
async def update_runtime_preset(
    profile_id: str,
    preset_key: str,
    body: RuntimePresetUpdate,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Update a profile runtime preset."""
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(status_code=400, detail="No fields to update")
    async with AsyncSessionLocal() as session:
        try:
            preset = await _get_registry(request).update_runtime_preset(
                session,
                profile_id,
                preset_key,
                patch,
            )
        except ValueError as exc:
            status = 404 if "not found" in str(exc).lower() else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
    return _preset_to_dict(preset)


@router.delete(
    "/profiles/{profile_id}/runtime-presets/{preset_key}",
    summary="Delete profile runtime preset",
    description=(
        "Delete a preset. A default preset cannot be deleted while another "
        "preset exists; make its replacement default first."
    ),
    responses={
        400: {"description": "Current default cannot be deleted"},
        404: {"description": "Profile or preset not found"},
    },
)
async def delete_runtime_preset(
    profile_id: str,
    preset_key: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Delete one runtime preset."""
    async with AsyncSessionLocal() as session:
        try:
            await _get_registry(request).delete_runtime_preset(
                session,
                profile_id,
                preset_key,
            )
        except ValueError as exc:
            status = 404 if "not found" in str(exc).lower() else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
    return {"ok": True, "profile_id": profile_id, "preset_key": preset_key}


@router.post(
    "/profiles/{profile_id}/aliases",
    summary="Create hidden profile alias",
    response_model=ProfileAliasOut,
)
async def create_profile_alias(
    profile_id: str,
    body: ProfileAliasCreate,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Map a compatibility id (normally ``-ctx*``) to a profile preset."""
    async with AsyncSessionLocal() as session:
        try:
            alias = await _get_registry(request).create_alias(
                session,
                alias_id=body.alias_id,
                profile_id=profile_id,
                preset_key=body.preset_key,
                hidden=body.hidden,
            )
        except ValueError as exc:
            status = 404 if "not found" in str(exc).lower() else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
    return _alias_to_dict(alias)


@router.delete(
    "/profiles/{profile_id}/aliases/{alias_id:path}",
    summary="Delete profile alias",
)
async def delete_profile_alias(
    profile_id: str,
    alias_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    registry = _get_registry(request)
    resolved = await registry.resolve_alias(alias_id)
    if resolved is None or resolved[0].profile_id != profile_id:
        raise HTTPException(status_code=404, detail=f"Alias '{alias_id}' not found.")
    async with AsyncSessionLocal() as session:
        await registry.delete_alias(session, alias_id)
    return {"ok": True, "profile_id": profile_id, "alias_id": alias_id}


# ── Direct profile endpoints ─────────────────────────────────


@router.post(
    "/profiles",
    summary="Create a router profile",
    description="Create a model-independent router with ordered profile targets.",
    status_code=201,
    responses={400: {"description": "Invalid router or targets"}},
)
async def create_router_profile(
    body: RouterCreate,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Create a router that owns no physical base model."""

    async with AsyncSessionLocal() as session:
        try:
            profile = await _get_registry(request).create(
                session,
                profile_id=body.profile_id,
                profile_kind="router",
                base_model_id=None,
                display_name=body.display_name,
                description=body.description,
                category=body.category,
                request_defaults=body.request_defaults,
                enabled=body.enabled,
                routing_targets=body.routing_targets,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _profile_to_dict(profile)


@router.get(
    "/profiles/{profile_id}",
    summary="Get profile detail",
    description="Return the full detail of a single profile.",
    responses={404: {"description": "Profile not found"}},
)
async def get_profile(
    profile_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Get a single profile by its id."""
    registry = _get_registry(request)
    profile = await registry.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    return _profile_to_dict(profile)


@router.patch(
    "/profiles/{profile_id}",
    summary="Update a profile",
    description="Patch mutable fields of a profile.",
    responses={
        400: {"description": "Invalid patch data"},
        404: {"description": "Profile not found"},
    },
)
async def update_profile(
    profile_id: str,
    body: ProfileUpdate,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Update an existing profile."""
    registry = _get_registry(request)
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(status_code=400, detail="No fields to update")
    async with AsyncSessionLocal() as session:
        try:
            profile = await registry.update(session, profile_id, patch)
        except ValueError as exc:
            msg = str(exc)
            if "not found" in msg.lower():
                raise HTTPException(status_code=404, detail=msg) from exc
            raise HTTPException(status_code=400, detail=msg) from exc
    return _profile_to_dict(profile)


@router.post(
    "/profiles/{profile_id}/rename",
    summary="Rename a profile",
    description=(
        "Change a profile's primary id and cascade the new id into every "
        "router whose ``routing_targets`` references it. Historical rows in "
        "``request_stats.via_router_profile_id`` are left alone on purpose "
        "(they describe which router handled a given request at the time)."
    ),
    responses={
        400: {"description": "Invalid new_profile_id or duplicate"},
        404: {"description": "Profile not found"},
    },
)
async def rename_profile(
    profile_id: str,
    body: ProfileRename,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        try:
            profile = await registry.rename(session, profile_id, body.new_profile_id)
        except ValueError as exc:
            msg = str(exc)
            if "not found" in msg.lower():
                raise HTTPException(status_code=404, detail=msg) from exc
            raise HTTPException(status_code=400, detail=msg) from exc
    return _profile_to_dict(profile)


@router.post(
    "/profiles/{profile_id}/clone",
    summary="Clone a profile",
    description=(
        "Create a new profile with the same fields as the source, under a "
        "fresh profile_id. ``is_default`` is forced to False in the clone "
        "to avoid two defaults for the same base model."
    ),
    status_code=201,
    responses={
        400: {"description": "Invalid new_profile_id or duplicate"},
        404: {"description": "Profile not found"},
    },
)
async def clone_profile(
    profile_id: str,
    body: ProfileClone,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        try:
            profile = await registry.clone(
                session, profile_id, body.new_profile_id, body.display_name
            )
        except ValueError as exc:
            msg = str(exc)
            if "not found" in msg.lower():
                raise HTTPException(status_code=404, detail=msg) from exc
            raise HTTPException(status_code=400, detail=msg) from exc
    return _profile_to_dict(profile)


@router.delete(
    "/profiles/{profile_id}",
    summary="Delete a profile",
    description="Remove a profile and its assets from disk.",
    responses={404: {"description": "Profile not found"}},
)
async def delete_profile(
    profile_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Delete a profile and its associated assets."""
    registry = _get_registry(request)
    async with AsyncSessionLocal() as session:
        try:
            await registry.delete(session, profile_id)
        except ValueError as exc:
            status = 404 if "not found" in str(exc).lower() else 409
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    # Clean up asset directory
    asset_dir = PROFILES_ASSET_BASE / profile_id
    if asset_dir.exists() and _is_path_within_base(asset_dir, PROFILES_ASSET_BASE):
        await asyncio.to_thread(shutil.rmtree, asset_dir)

    return {"ok": True, "profile_id": profile_id}


# ── Asset management ─────────────────────────────────────────


@router.post(
    "/profiles/{profile_id}/assets",
    summary="Upload an asset to a profile",
    description=(
        "Upload a file (audio reference, LoRA weights, etc.) to the profile's asset directory. "
        "The file is saved under /data/profiles/{profile_id}/ and registered in the assets JSONB."
    ),
    responses={
        400: {"description": "Invalid file or path traversal"},
        404: {"description": "Profile not found"},
        413: {"description": "File too large"},
    },
)
async def upload_profile_asset(
    profile_id: str,
    file: UploadFile,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Upload an asset file to a profile's storage directory."""
    registry = _get_registry(request)
    profile = await registry.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Sanitize filename
    safe_name = Path(file.filename).name
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_ASSET_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{suffix}' not allowed. "
            f"Allowed: {sorted(ALLOWED_ASSET_EXTENSIONS)}",
        )

    asset_dir = PROFILES_ASSET_BASE / profile_id
    target_path = asset_dir / safe_name

    # Path traversal protection
    if not _is_path_within_base(target_path, PROFILES_ASSET_BASE):
        raise HTTPException(status_code=400, detail="Path traversal refused")

    # Create directory
    await asyncio.to_thread(asset_dir.mkdir, parents=True, exist_ok=True)

    # Read the full file content with size check, then write in a thread
    chunks: list[bytes] = []
    total_bytes = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_ASSET_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum size of {MAX_ASSET_SIZE_BYTES} bytes",
                )
            chunks.append(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read upload: {exc}") from exc

    def _write_file() -> None:
        with open(target_path, "wb") as f:  # noqa: ASYNC230
            for c in chunks:
                f.write(c)

    try:
        await asyncio.to_thread(_write_file)
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc

    # Derive asset key from filename stem
    asset_key = Path(safe_name).stem

    # Update assets JSONB
    current_assets = dict(profile.assets or {})
    current_assets[asset_key] = {
        "filename": safe_name,
        "path": str(target_path),
        "size_bytes": total_bytes,
    }

    async with AsyncSessionLocal() as session:
        updated = await registry.update_assets(session, profile_id, current_assets)

    logger.info(
        "profile_asset_uploaded",
        profile_id=profile_id,
        asset_key=asset_key,
        size_bytes=total_bytes,
    )
    return {
        "ok": True,
        "asset_key": asset_key,
        "filename": safe_name,
        "size_bytes": total_bytes,
        "profile": _profile_to_dict(updated),
    }


@router.delete(
    "/profiles/{profile_id}/assets/{asset_key}",
    summary="Delete a profile asset",
    description="Remove a specific asset file and its reference from the profile.",
    responses={
        400: {"description": "Path traversal refused"},
        404: {"description": "Profile or asset not found"},
    },
)
async def delete_profile_asset(
    profile_id: str,
    asset_key: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Delete an asset from a profile."""
    registry = _get_registry(request)
    profile = await registry.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")

    current_assets = dict(profile.assets or {})
    asset_info = current_assets.get(asset_key)
    if asset_info is None:
        raise HTTPException(
            status_code=404, detail=f"Asset '{asset_key}' not found in profile '{profile_id}'"
        )

    # Delete file from disk
    file_path = Path(str(asset_info.get("path", "")))
    file_exists = await asyncio.to_thread(file_path.exists)
    if file_exists:
        if not _is_path_within_base(file_path, PROFILES_ASSET_BASE):
            raise HTTPException(status_code=400, detail="Path traversal refused")
        await asyncio.to_thread(file_path.unlink)

    # Update JSONB
    del current_assets[asset_key]
    async with AsyncSessionLocal() as session:
        updated = await registry.update_assets(
            session, profile_id, current_assets if current_assets else None
        )

    logger.info("profile_asset_deleted", profile_id=profile_id, asset_key=asset_key)
    return {
        "ok": True,
        "asset_key": asset_key,
        "profile": _profile_to_dict(updated),
    }
