"""OpenAI-compatible Batches API.

Implements ``/v1/batches`` for asynchronous batch processing of requests.
The processor that consumes ``validating`` and ``in_progress`` batches lives
in :mod:`ocabra.core.batch_processor` and runs as a startup task.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import get_openai_user
from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.auth import User
from ocabra.db.openai_batches import OpenAIBatch, OpenAIFile
from ocabra.schemas.batches import Batch, BatchList

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


async def _with_username(
    session: AsyncSession, batch: OpenAIBatch, request: Request | None = None
) -> dict[str, Any]:
    d = batch.to_openai_dict()
    username = (await session.execute(
        select(User.username).where(User.id == batch.user_id)
    )).scalar_one_or_none()
    d["user"] = {"id": str(batch.user_id), "username": username}
    d["ocabra"] = await _live_field(session, batch, request) if request is not None else None
    return d


_TERMINAL = ("completed", "failed", "expired", "cancelled")
_NOT_READY = ("loading", "unloaded", "configured", "discovered", "unloading")


async def _model_wait(mm: Any, model: str) -> dict[str, Any]:
    """State of ``model`` across its workers (``<model>`` and ``<model>::<hash>``)."""
    states = [
        st
        for key, st in (getattr(mm, "_states", {}) or {}).items()
        if key == model or key.startswith(model + "::")
    ]
    statuses = {str(getattr(st.status, "value", st.status)) for st in states}
    if "loaded" in statuses:
        status = "loaded"
    elif "loading" in statuses:
        status = "loading"
    elif statuses:
        status = sorted(statuses)[0]
    else:
        status = None
    expected: int | None = None
    if status in ("loading", "unloaded", "configured"):
        try:
            typical = await mm.get_expected_load_seconds(model)
        except Exception:  # noqa: BLE001 - a hint, never an error
            typical = None
        if typical is not None:
            started = [
                t for t in (mm.load_started_at(st.model_id) for st in states) if t
            ] if status == "loading" else []
            elapsed = (time.time() - min(started)) if started else 0.0
            expected = max(0, int(typical - elapsed))
    return {"model": model, "status": status, "expected_wait_seconds": expected}


async def _live_field(
    session: AsyncSession, batch: OpenAIBatch, request: Request
) -> dict[str, Any] | None:
    """``ocabra``: why an unfinished batch is where it is (``BatchLiveStatus``).

    A batch at 0/144 for a minute looked the same whether it was queued behind
    another batch, waiting for its model to load or yielding to interactive
    traffic; for single requests oCabra already says so (X-Ocabra-Model-Status,
    X-Ocabra-Expected-Wait-Seconds), for a batch nothing did.
    """
    if batch.status in _TERMINAL:
        return None
    base: dict[str, Any] = {
        "queue_position": None, "in_flight": 0, "waiting": 0, "preemptions": 0,
        "retry_in_seconds": None, "models": [],
    }  # fmt: skip
    if batch.status == "finalizing":
        return {**base, "phase": "finalizing", "detail": "Writing the output files."}
    if batch.status == "cancelling":
        return {**base, "phase": "cancelling", "detail": "Stopping: no new lines are dispatched."}
    processor = getattr(request.app.state, "batch_processor", None)
    snap = processor.live_status(str(batch.id)) if processor is not None else None
    if snap is None:
        ahead = (await session.execute(
            select(func.count()).select_from(OpenAIBatch).where(
                OpenAIBatch.status.in_(("validating", "in_progress", "finalizing", "cancelling")),
                OpenAIBatch.created_at < batch.created_at,
            )
        )).scalar_one()
        return {
            **base,
            "phase": "queued",
            "queue_position": int(ahead),
            "detail": (
                f"Queued: {ahead} batch(es) ahead; batches run one at a time."
                if ahead
                else "Queued: it starts on the processor's next pass."
            ),
        }
    mm = getattr(request.app.state, "model_manager", None)
    names = sorted(set(snap["dispatch_models"]) | set(snap["cooldown_models"]) | set(snap["vram_models"]))
    models = [await _model_wait(mm, m) for m in names] if mm is not None else []
    by_name = {m["model"]: m for m in models}
    out = {
        **base,
        "in_flight": snap["in_flight"],
        "waiting": snap["waiting"],
        "preemptions": snap["preemptions"],
        "retry_in_seconds": snap["retry_in_seconds"],
        "models": models,
    }
    loading = [
        by_name[m] for m in snap["dispatch_models"]
        if by_name.get(m, {}).get("status") in _NOT_READY
    ]  # fmt: skip
    if snap["in_flight"] and loading:
        m = loading[0]
        wait = m.get("expected_wait_seconds")
        return {
            **out,
            "phase": "loading_model",
            "detail": f"Waiting for {m['model']} to load"
            + (f" (about {wait} s left)." if wait is not None else "."),
        }
    if snap["in_flight"]:
        return {**out, "phase": "running", "detail": f"Serving {snap['in_flight']} line(s)."}
    if snap["cooldown_models"]:
        retry = snap["retry_in_seconds"]
        return {
            **out,
            "phase": "yielding_to_interactive",
            "detail": (
                f"{', '.join(snap['cooldown_models'])} was lent to interactive requests; "
                f"{snap['waiting']} line(s) retry"
                + (f" in {retry} s." if retry is not None else " shortly.")
            ),
        }
    if snap["vram_models"]:
        return {
            **out,
            "phase": "waiting_for_vram",
            "detail": (
                f"No GPU memory free for {', '.join(snap['vram_models'])} yet; "
                f"{snap['waiting']} line(s) retry shortly."
            ),
        }
    return {**out, "phase": "starting", "detail": "Starting: reading the input file."}


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


def _read_jsonl_lines(path: Path) -> list[str]:
    """Lee un JSONL completo a memoria. Sincrono a proposito: se invoca con
    ``asyncio.to_thread`` porque el fichero lo sube el usuario y puede tener
    miles de lineas, y parsearlo en el event loop bloquea todo el servidor."""
    with path.open("r", encoding="utf-8") as fh:
        return fh.read().splitlines()


def _strip_id(prefixed: str, prefix: str) -> uuid.UUID:
    raw = prefixed[len(prefix):] if prefixed.startswith(prefix) else prefixed
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail={"error": {
            "message": f"No such resource: '{prefixed}'",
            "type": "invalid_request_error",
        }}) from exc


@router.post("/batches", summary="Create a batch", response_model=Batch)
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
        if not path.exists():  # noqa: ASYNC240 — solo metadatos (microsegundos); un to_thread costaria mas que la syscall
            raise HTTPException(status_code=400, detail={"error": {
                "message": "Input file content missing on disk",
                "type": "invalid_request_error",
            }})

        # Quick line count and basic JSONL validation
        request_total = 0
        validation_errors: list[dict[str, Any]] = []
        raw_lines = await asyncio.to_thread(_read_jsonl_lines, path)
        # The limit before validating each line (that can mean a DB lookup
        # per line): a file far over it is refused at once.
        lines_in_file = sum(1 for line in raw_lines if line.strip())
        if lines_in_file > settings.batch_max_requests:
            raise HTTPException(status_code=400, detail={"error": {
                "message": (
                    f"The input file has {lines_in_file} requests; a batch may hold at most "
                    f"{settings.batch_max_requests}. Split it into several batches."
                ),
                "type": "invalid_request_error",
                "param": "input_file_id",
                "code": "batch_too_large",
            }})
        for line_no, line in enumerate(raw_lines, start=1):
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
                "param": "input_file_id",
                "code": "empty_batch",
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

        return await _with_username(session, batch, request)


@router.get("/batches/{batch_id}", summary="Retrieve a batch", response_model=Batch)
async def retrieve_batch(
    batch_id: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
    request: Request,
) -> dict:
    """Return one batch (OpenAI ``Batch``).

    While it is unfinished, the oCabra field ``ocabra`` says why it is where
    it is (queued, its model loading, yielding to interactive traffic…), with
    an estimate of the wait when there is one. Poll this until ``status`` is
    ``completed``, ``failed``, ``expired`` or ``cancelled``; the results are in
    ``output_file_id`` and ``error_file_id`` (``GET /v1/files/{id}/content``).
    """
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
        return await _with_username(session, batch, request)


@router.get("/batches", summary="List batches", response_model=BatchList)
async def list_batches(
    user: Annotated[UserContext, Depends(get_openai_user)],
    request: Request,
    after: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    """List the caller's batches, newest first (all of them for
    model_manager and system_admin). Each item is a ``Batch`` with its
    ``ocabra`` live status."""
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
        data = [await _with_username(session, r, request) for r in rows]
        return {
            "object": "list",
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": len(rows) == limit,
        }


@router.post("/batches/{batch_id}/cancel", summary="Cancel a batch", response_model=Batch)
async def cancel_batch(
    batch_id: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
    request: Request,
) -> dict:
    """Cancel a batch: it goes to ``cancelling`` (no new lines are
    dispatched) and then to ``cancelled``; already-finished batches are
    returned unchanged."""
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
            return await _with_username(session, batch, request)
        if batch.status == "cancelling":
            return await _with_username(session, batch, request)
        batch.status = "cancelling"
        batch.cancelling_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(batch)
        return await _with_username(session, batch, request)


