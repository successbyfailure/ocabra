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
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import get_openai_user
from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.auth import User
from ocabra.db.openai_batches import OpenAIBatch, OpenAIFile

router = APIRouter(tags=["OpenAI Batches"])
logger = structlog.get_logger(__name__)

ALLOWED_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/completions",
    "/v1/images/generations",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    # 2026-11-02: pooling/score/rerank/classify/audio-generate all take a
    # plain JSON body and return either JSON (pooling/score/rerank/classify)
    # or binary (audio/generate, same as audio/speech) — batch_processor's
    # generic _dispatch() already handles both shapes, so no per-endpoint
    # dispatch code was needed to add these, unlike audio/transcriptions.
    "/v1/pooling",
    "/v1/score",
    "/v1/rerank",
    "/v1/classify",
    "/v1/audio/generate",
    # 2026-11-02: same file_id-reference trick as /v1/audio/transcriptions
    # below, extended to the other multipart-only endpoint.
    "/v1/images/edits",
}

# oCabra extension (2026-10-30), NOT part of the OpenAI Batches spec: the
# real /v1/audio/transcriptions endpoint takes multipart/form-data with a
# binary file, which doesn't fit a JSON line. Batch lines targeting it
# reference an already-uploaded file instead of inlining audio — upload it
# first via POST /v1/files with purpose="batch_input_audio", then put its
# file_id in the line's body. batch_processor._dispatch() builds the real
# multipart request from that reference at execution time.
_BATCH_INPUT_AUDIO_PURPOSE = "batch_input_audio"

# oCabra extension (2026-11-02), same shape as _BATCH_INPUT_AUDIO_PURPOSE
# above: /v1/images/edits takes multipart/form-data (a base image, plus
# optional mask / up to 3 extra reference images), none of which fit a JSON
# line. Batch lines reference pre-uploaded files instead: body.image_file_id
# (required) and optionally body.mask_file_id / body.image_ref_1_file_id /
# body.image_ref_2_file_id / body.image_ref_3_file_id, all uploaded via
# POST /v1/files with purpose="batch_input_image".
_BATCH_INPUT_IMAGE_PURPOSE = "batch_input_image"
_IMAGE_EDIT_FILE_FIELDS = (
    "image_file_id",
    "mask_file_id",
    "image_ref_1_file_id",
    "image_ref_2_file_id",
    "image_ref_3_file_id",
)


class BatchCreateBody(BaseModel):
    input_file_id: str
    endpoint: str = Field(..., json_schema_extra={"enum": sorted(ALLOWED_ENDPOINTS)})
    completion_window: str = Field("24h", json_schema_extra={"enum": ["24h"]})
    metadata: dict[str, Any] | None = None


class TranscriptionBatchLineBody(BaseModel):
    """Shape of ``body`` in a ``/v1/audio/transcriptions`` batch line.

    Documentation-only (used for OpenAPI + line-shape validation, not as a
    FastAPI request param): the real endpoint takes multipart/form-data, so
    this describes the oCabra file_id-reference extension instead — see
    ``_BATCH_INPUT_AUDIO_PURPOSE`` above.
    """

    file_id: str = Field(
        ..., description="file-<uuid> uploaded via POST /v1/files with purpose='batch_input_audio'"
    )
    model: str = Field(..., description="Whisper profile_id")
    language: str | None = None
    prompt: str | None = None
    response_format: str | None = Field(
        None, json_schema_extra={"enum": ["json", "text", "srt", "vtt", "verbose_json"]}
    )
    temperature: float | None = None
    diarize: bool | None = None
    timestamp_granularities: list[str] | None = None
    vad_filter: bool | None = None
    condition_on_previous_text: bool | None = None

    model_config = ConfigDict(extra="allow")


class ImageEditsBatchLineBody(BaseModel):
    """Shape of ``body`` in a ``/v1/images/edits`` batch line.

    Documentation-only, same rationale as ``TranscriptionBatchLineBody`` —
    describes the oCabra file_id-reference extension (see
    ``_BATCH_INPUT_IMAGE_PURPOSE`` / ``_IMAGE_EDIT_FILE_FIELDS`` above), not
    the real multipart endpoint.
    """

    image_file_id: str = Field(
        ..., description="file-<uuid> uploaded via POST /v1/files with purpose='batch_input_image'"
    )
    mask_file_id: str | None = Field(None, description="Same purpose as image_file_id")
    image_ref_1_file_id: str | None = None
    image_ref_2_file_id: str | None = None
    image_ref_3_file_id: str | None = None
    model: str = Field(..., description="Profile ID del modelo de edicion de imagen")
    prompt: str = Field(..., description="Descripcion de la edicion")
    response_format: str | None = Field(None, json_schema_extra={"enum": ["b64_json", "url"]})
    n: int | None = None
    size: str | None = None
    negative_prompt: str | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    strength: float | None = None
    seed: int | None = None

    model_config = ConfigDict(extra="allow")


def _is_manager_or_admin(user: UserContext) -> bool:
    """model_manager and system_admin both get cross-user visibility into
    the Batches UI (list/cancel other users' batches) — matches the
    frontend's /batches page, which is gated at minRole="model_manager",
    not just system_admin like ``user.is_admin`` alone would allow."""
    return user.role in ("model_manager", "system_admin")


async def _with_username(session: AsyncSession, batch: OpenAIBatch) -> dict[str, Any]:
    d = batch.to_openai_dict()
    username = (await session.execute(
        select(User.username).where(User.id == batch.user_id)
    )).scalar_one_or_none()
    d["user"] = {"id": str(batch.user_id), "username": username}
    return d


async def _resolve_owned_batch_file(
    session: AsyncSession,
    raw_file_id: Any,
    *,
    user_uuid: uuid.UUID,
    is_admin: bool,
    purpose: str,
) -> OpenAIFile | None:
    """Resolve a file_id reference used by a multipart-only batch extension
    (transcriptions, images/edits) to an owned file with the right purpose.
    Returns None for any kind of mismatch — caller turns that into a single
    ``invalid_file_id`` validation error, same message shape either way."""
    try:
        file_uuid = uuid.UUID(
            str(raw_file_id)[5:] if str(raw_file_id).startswith("file-") else str(raw_file_id)
        )
    except ValueError:
        return None
    file = await session.get(OpenAIFile, file_uuid)
    if file is None or (file.user_id != user_uuid and not is_admin) or file.purpose != purpose:
        return None
    return file


def _line_body_shape_errors(
    model_cls: type[BaseModel], line_body: dict[str, Any], *, skip_fields: set[str]
) -> list[dict[str, Any]]:
    """Validate a batch line's ``body`` against its documented Pydantic
    shape (``TranscriptionBatchLineBody``/``ImageEditsBatchLineBody``),
    skipping fields that already get a friendlier file_id-specific error
    from the caller (``missing_file_id``/``invalid_file_id``) so the same
    problem isn't reported twice under two different codes."""
    try:
        model_cls.model_validate(line_body)
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else "body"
            if field in skip_fields:
                continue
            errors.append({"code": "invalid_body", "message": f"body.{field}: {err['msg']}"})
        return errors
    return []


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
    """Create a new batch job from an uploaded JSONL file.

    Each line's ``body`` is forwarded verbatim to ``endpoint``, except for
    the two oCabra file_id-reference extensions documented as their own
    schemas (see the Schemas section of this page): ``endpoint=
    "/v1/audio/transcriptions"`` lines must match
    ``TranscriptionBatchLineBody`` and ``endpoint="/v1/images/edits"`` lines
    must match ``ImageEditsBatchLineBody``.
    """
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
                    continue
                if body.endpoint == "/v1/audio/transcriptions":
                    line_body = obj.get("body") if isinstance(obj.get("body"), dict) else {}
                    for shape_err in _line_body_shape_errors(
                        TranscriptionBatchLineBody, line_body, skip_fields={"file_id"}
                    ):
                        validation_errors.append({"line": line_no, **shape_err})
                    audio_file_id = line_body.get("file_id")
                    if not audio_file_id:
                        validation_errors.append({
                            "line": line_no,
                            "code": "missing_file_id",
                            "message": "body.file_id is required for /v1/audio/transcriptions "
                            "lines (oCabra extension — upload the audio first via POST "
                            "/v1/files with purpose='batch_input_audio')",
                        })
                        continue
                    audio_file = await _resolve_owned_batch_file(
                        session, audio_file_id,
                        user_uuid=user_uuid, is_admin=user.is_admin,
                        purpose=_BATCH_INPUT_AUDIO_PURPOSE,
                    )
                    if audio_file is None:
                        validation_errors.append({
                            "line": line_no,
                            "code": "invalid_file_id",
                            "message": f"body.file_id '{audio_file_id}' is not a file you own "
                            f"with purpose='{_BATCH_INPUT_AUDIO_PURPOSE}'",
                        })
                elif body.endpoint == "/v1/images/edits":
                    line_body = obj.get("body") if isinstance(obj.get("body"), dict) else {}
                    for shape_err in _line_body_shape_errors(
                        ImageEditsBatchLineBody, line_body, skip_fields={"image_file_id"}
                    ):
                        validation_errors.append({"line": line_no, **shape_err})
                    image_file_id = line_body.get("image_file_id")
                    if not image_file_id:
                        validation_errors.append({
                            "line": line_no,
                            "code": "missing_file_id",
                            "message": "body.image_file_id is required for /v1/images/edits "
                            "lines (oCabra extension — upload the base image first via POST "
                            "/v1/files with purpose='batch_input_image')",
                        })
                        continue
                    for field in _IMAGE_EDIT_FILE_FIELDS:
                        ref_id = line_body.get(field)
                        if ref_id is None:
                            continue
                        ref_file = await _resolve_owned_batch_file(
                            session, ref_id,
                            user_uuid=user_uuid, is_admin=user.is_admin,
                            purpose=_BATCH_INPUT_IMAGE_PURPOSE,
                        )
                        if ref_file is None:
                            validation_errors.append({
                                "line": line_no,
                                "code": "invalid_file_id",
                                "message": f"body.{field} '{ref_id}' is not a file you own "
                                f"with purpose='{_BATCH_INPUT_IMAGE_PURPOSE}'",
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

        return await _with_username(session, batch)


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
        if not _is_manager_or_admin(user) and (
            user.user_id is None or batch.user_id != uuid.UUID(user.user_id)
        ):
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"No such batch: '{batch_id}'",
                "type": "invalid_request_error",
            }})
        return await _with_username(session, batch)


@router.get("/batches", summary="List batches")
async def list_batches(
    user: Annotated[UserContext, Depends(get_openai_user)],
    after: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    if user.user_id is None:
        return {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}
    manager_or_admin = _is_manager_or_admin(user)
    async with AsyncSessionLocal() as session:
        stmt = select(OpenAIBatch)
        if not manager_or_admin:
            stmt = stmt.where(OpenAIBatch.user_id == uuid.UUID(user.user_id))
        if after:
            cursor = await session.get(OpenAIBatch, _strip_id(after, "batch_"))
            if cursor is not None:
                stmt = stmt.where(OpenAIBatch.created_at < cursor.created_at)
        stmt = stmt.order_by(desc(OpenAIBatch.created_at)).limit(limit)
        rows = (await session.execute(stmt)).scalars().all()
        data = [await _with_username(session, r) for r in rows]
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
            not _is_manager_or_admin(user)
            and (user.user_id is None or batch.user_id != uuid.UUID(user.user_id))
        ):
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"No such batch: '{batch_id}'",
                "type": "invalid_request_error",
            }})
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return await _with_username(session, batch)
        if batch.status == "cancelling":
            return await _with_username(session, batch)
        batch.status = "cancelling"
        batch.cancelling_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(batch)
        return await _with_username(session, batch)


_ = settings  # keep import for forward-compat config use
