"""OpenAI-compatible Files API.

Implements ``/v1/files`` for uploading, listing, retrieving, and deleting
files used as inputs/outputs of the Batches API.

Files are stored on disk under ``settings.openai_files_dir`` (one file per
upload, named by UUID). Metadata is persisted in the ``openai_files`` table.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import desc, select

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import get_openai_user
from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.openai_batches import OpenAIBatch, OpenAIFile

router = APIRouter(tags=["OpenAI Files"])
logger = structlog.get_logger(__name__)

VALID_PURPOSES = {"batch", "batch_output", "user_data"}
MAX_FILE_BYTES = 512 * 1024 * 1024  # 512MB cap


def _strip_file_prefix(file_id: str) -> uuid.UUID:
    """Accept OpenAI-style ``file-<uuid>`` or raw UUID and return the UUID."""
    raw = file_id[5:] if file_id.startswith("file-") else file_id
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail={"error": {
            "message": f"No such file: '{file_id}'",
            "type": "invalid_request_error",
            "code": "file_not_found",
        }}) from exc


def _ensure_storage_dir() -> Path:
    path = Path(settings.openai_files_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


@router.post("/files", summary="Upload a file")
async def upload_file(
    user: Annotated[UserContext, Depends(get_openai_user)],
    file: UploadFile = File(...),
    purpose: str = Form(...),
) -> dict:
    """Upload a file. Required ``purpose=batch`` for use as Batches input."""
    if purpose not in VALID_PURPOSES:
        raise HTTPException(status_code=400, detail={"error": {
            "message": f"Invalid purpose '{purpose}'. Must be one of: {sorted(VALID_PURPOSES)}",
            "type": "invalid_request_error",
            "param": "purpose",
        }})
    if user.is_anonymous or user.user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    file_uuid = uuid.uuid4()
    storage_dir = _ensure_storage_dir()
    storage_path = storage_dir / f"{file_uuid}"

    total = 0
    try:
        with storage_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    out.close()
                    storage_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail={"error": {
                        "message": f"File exceeds maximum size of {MAX_FILE_BYTES} bytes",
                        "type": "invalid_request_error",
                    }})
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        storage_path.unlink(missing_ok=True)
        logger.exception("openai_file_upload_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="File upload failed") from exc

    async with AsyncSessionLocal() as session:
        record = OpenAIFile(
            id=file_uuid,
            user_id=uuid.UUID(user.user_id),
            filename=file.filename or f"upload-{file_uuid}",
            bytes=total,
            purpose=purpose,
            storage_path=str(storage_path),
            status="uploaded",
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record.to_openai_dict()


@router.get("/files", summary="List uploaded files")
async def list_files(
    user: Annotated[UserContext, Depends(get_openai_user)],
    purpose: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=10000),
) -> dict:
    if user.user_id is None:
        return {"object": "list", "data": [], "has_more": False}
    async with AsyncSessionLocal() as session:
        stmt = select(OpenAIFile).where(OpenAIFile.user_id == uuid.UUID(user.user_id))
        if purpose:
            stmt = stmt.where(OpenAIFile.purpose == purpose)
        stmt = stmt.order_by(desc(OpenAIFile.created_at)).limit(limit)
        rows = (await session.execute(stmt)).scalars().all()
        return {
            "object": "list",
            "data": [r.to_openai_dict() for r in rows],
            "has_more": len(rows) == limit,
        }


async def _load_file(file_id: str, user: UserContext) -> OpenAIFile:
    file_uuid = _strip_file_prefix(file_id)
    async with AsyncSessionLocal() as session:
        record = await session.get(OpenAIFile, file_uuid)
        if record is None:
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"No such file: '{file_id}'",
                "type": "invalid_request_error",
                "code": "file_not_found",
            }})
        if not user.is_admin and (user.user_id is None or record.user_id != uuid.UUID(user.user_id)):
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"No such file: '{file_id}'",
                "type": "invalid_request_error",
                "code": "file_not_found",
            }})
        return record


@router.get("/files/{file_id}", summary="Retrieve file metadata")
async def retrieve_file(
    file_id: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> dict:
    record = await _load_file(file_id, user)
    return record.to_openai_dict()


@router.get("/files/{file_id}/content", summary="Download file content")
async def file_content(
    file_id: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
):
    record = await _load_file(file_id, user)
    path = Path(record.storage_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail={"error": {
            "message": f"File content missing on disk for '{file_id}'",
            "type": "invalid_request_error",
            "code": "file_not_found",
        }})
    return FileResponse(
        path=path,
        filename=record.filename,
        media_type="application/octet-stream",
    )


@router.delete("/files/{file_id}", summary="Delete a file")
async def delete_file(
    file_id: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> dict:
    record = await _load_file(file_id, user)
    file_uuid = record.id
    async with AsyncSessionLocal() as session:
        # Refuse if any non-terminal batch references this file
        in_use = await session.execute(
            select(OpenAIBatch.id).where(
                (OpenAIBatch.input_file_id == file_uuid)
                & OpenAIBatch.status.in_(["validating", "in_progress", "finalizing", "cancelling"])
            )
        )
        if in_use.first() is not None:
            raise HTTPException(status_code=409, detail={"error": {
                "message": "File is in use by an active batch",
                "type": "invalid_request_error",
                "code": "file_in_use",
            }})
        record_in_session = await session.get(OpenAIFile, file_uuid)
        if record_in_session is not None:
            await session.delete(record_in_session)
            await session.commit()

    try:
        Path(record.storage_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("openai_file_disk_delete_failed", file_id=str(file_uuid), error=str(exc))

    return JSONResponse({
        "id": f"file-{file_uuid}",
        "object": "file",
        "deleted": True,
    })
