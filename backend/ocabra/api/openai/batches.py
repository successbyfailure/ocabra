"""OpenAI-compatible Batches API.

Implements ``/v1/batches`` for asynchronous batch processing of requests.
The processor that consumes ``validating`` and ``in_progress`` batches lives
in :mod:`ocabra.core.batch_processor` and runs as a startup task.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import get_openai_user
from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.openai_batches import OpenAIBatch, OpenAIFile

router = APIRouter(tags=["OpenAI Batches"])
logger = structlog.get_logger(__name__)

ALLOWED_ENDPOINTS = {"/v1/chat/completions", "/v1/embeddings", "/v1/completions"}


class BatchCreateBody(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str = "24h"
    metadata: dict[str, Any] | None = None


def _strip_id(prefixed: str, prefix: str) -> uuid.UUID:
    raw = prefixed[len(prefix):] if prefixed.startswith(prefix) else prefixed
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail={"error": {
            "message": f"No such resource: '{prefixed}'",
            "type": "invalid_request_error",
        }}) from exc


@router.post("/batches", summary="Create a batch")
async def create_batch(
    body: BatchCreateBody,
    user: Annotated[UserContext, Depends(get_openai_user)],
    request: Request,
) -> dict:
    """Create a new batch job from an uploaded JSONL file."""
    if user.is_anonymous or user.user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    if body.endpoint not in ALLOWED_ENDPOINTS:
        raise HTTPException(status_code=400, detail={"error": {
            "message": f"Unsupported endpoint '{body.endpoint}'. Allowed: {sorted(ALLOWED_ENDPOINTS)}",
            "type": "invalid_request_error",
            "param": "endpoint",
        }})
    if body.completion_window != "24h":
        raise HTTPException(status_code=400, detail={"error": {
            "message": "Only completion_window='24h' is supported",
            "type": "invalid_request_error",
            "param": "completion_window",
        }})

    file_uuid = _strip_id(body.input_file_id, "file-")
    user_uuid = uuid.UUID(user.user_id)

    async with AsyncSessionLocal() as session:
        input_file = await session.get(OpenAIFile, file_uuid)
        if input_file is None or (input_file.user_id != user_uuid and not user.is_admin):
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"No such file: '{body.input_file_id}'",
                "type": "invalid_request_error",
                "code": "file_not_found",
            }})
        if input_file.purpose != "batch":
            raise HTTPException(status_code=400, detail={"error": {
                "message": "Input file must have purpose='batch'",
                "type": "invalid_request_error",
                "param": "input_file_id",
            }})

        path = Path(input_file.storage_path)
        if not path.exists():
            raise HTTPException(status_code=400, detail={"error": {
                "message": "Input file content missing on disk",
                "type": "invalid_request_error",
            }})

        # Quick line count and basic JSONL validation
        request_total = 0
        validation_errors: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                request_total += 1
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    validation_errors.append({
                        "line": line_no,
                        "code": "invalid_json",
                        "message": str(exc),
                    })
                    continue
                if not isinstance(obj, dict):
                    validation_errors.append({
                        "line": line_no,
                        "code": "invalid_request",
                        "message": "Each line must be a JSON object",
                    })
                    continue
                if obj.get("url") and obj.get("url") != body.endpoint:
                    validation_errors.append({
                        "line": line_no,
                        "code": "endpoint_mismatch",
                        "message": f"Line url '{obj.get('url')}' does not match batch endpoint",
                    })

        if request_total == 0:
            raise HTTPException(status_code=400, detail={"error": {
                "message": "Input file contains no requests",
                "type": "invalid_request_error",
            }})
        if validation_errors:
            raise HTTPException(status_code=400, detail={"error": {
                "message": "Input file failed validation",
                "type": "invalid_request_error",
                "errors": validation_errors[:20],
            }})

        api_key_id: uuid.UUID | None = None
        api_key_name = getattr(user, "api_key_name", None)
        if api_key_name:
            from ocabra.db.auth import ApiKey

            row = (await session.execute(
                select(ApiKey.id)
                .where(ApiKey.user_id == user_uuid, ApiKey.name == api_key_name)
                .limit(1)
            )).first()
            if row:
                api_key_id = row[0]

        now = datetime.now(UTC)
        batch = OpenAIBatch(
            id=uuid.uuid4(),
            user_id=user_uuid,
            api_key_id=api_key_id,
            endpoint=body.endpoint,
            input_file_id=file_uuid,
            completion_window=body.completion_window,
            status="validating",
            request_total=request_total,
            request_completed=0,
            request_failed=0,
            batch_metadata=body.metadata,
            created_at=now,
            expires_at=now + timedelta(hours=24),
        )
        session.add(batch)
        await session.commit()
        await session.refresh(batch)

        # Nudge processor to pick this up immediately
        processor = getattr(request.app.state, "batch_processor", None)
        if processor is not None:
            processor.notify()

        return batch.to_openai_dict()


@router.get("/batches/{batch_id}", summary="Retrieve a batch")
async def retrieve_batch(
    batch_id: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> dict:
    batch_uuid = _strip_id(batch_id, "batch_")
    async with AsyncSessionLocal() as session:
        batch = await session.get(OpenAIBatch, batch_uuid)
        if batch is None:
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"No such batch: '{batch_id}'",
                "type": "invalid_request_error",
            }})
        if not user.is_admin and (user.user_id is None or batch.user_id != uuid.UUID(user.user_id)):
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"No such batch: '{batch_id}'",
                "type": "invalid_request_error",
            }})
        return batch.to_openai_dict()


@router.get("/batches", summary="List batches")
async def list_batches(
    user: Annotated[UserContext, Depends(get_openai_user)],
    after: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    if user.user_id is None:
        return {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}
    async with AsyncSessionLocal() as session:
        stmt = select(OpenAIBatch).where(OpenAIBatch.user_id == uuid.UUID(user.user_id))
        if after:
            cursor = await session.get(OpenAIBatch, _strip_id(after, "batch_"))
            if cursor is not None:
                stmt = stmt.where(OpenAIBatch.created_at < cursor.created_at)
        stmt = stmt.order_by(desc(OpenAIBatch.created_at)).limit(limit)
        rows = (await session.execute(stmt)).scalars().all()
        data = [r.to_openai_dict() for r in rows]
        return {
            "object": "list",
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": len(rows) == limit,
        }


@router.post("/batches/{batch_id}/cancel", summary="Cancel a batch")
async def cancel_batch(
    batch_id: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> dict:
    batch_uuid = _strip_id(batch_id, "batch_")
    async with AsyncSessionLocal() as session:
        batch = await session.get(OpenAIBatch, batch_uuid)
        if batch is None or (
            not user.is_admin
            and (user.user_id is None or batch.user_id != uuid.UUID(user.user_id))
        ):
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"No such batch: '{batch_id}'",
                "type": "invalid_request_error",
            }})
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch.to_openai_dict()
        if batch.status == "cancelling":
            return batch.to_openai_dict()
        batch.status = "cancelling"
        batch.cancelling_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(batch)
        return batch.to_openai_dict()


_ = settings  # keep import for forward-compat config use
