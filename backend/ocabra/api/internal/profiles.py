"""Internal REST endpoints for model profile CRUD and asset management."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.core.profile_registry import ProfileRegistry
from ocabra.database import AsyncSessionLocal
from ocabra.schemas.profiles import ProfileCreate, ProfileOut, ProfileUpdate

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
                display_name=body.display_name,
                description=body.description,
                category=body.category,
                load_overrides=body.load_overrides,
                request_defaults=body.request_defaults,
                enabled=body.enabled,
                is_default=body.is_default,
            )
        except ValueError as exc:
            msg = str(exc)
            if "not found" in msg.lower():
                raise HTTPException(status_code=404, detail=msg) from exc
            raise HTTPException(status_code=400, detail=msg) from exc
    return _profile_to_dict(profile)


# ── Direct profile endpoints ─────────────────────────────────


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
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
