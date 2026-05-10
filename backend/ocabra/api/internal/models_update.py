"""Re-pull / update endpoints for already installed models.

Supports the same source families as the download manager: ``ollama``,
``huggingface`` (full repo) and ``bitnet`` / ``llama_cpp`` single-file
GGUFs (when ``extra_config.model_path`` records the original filename).
Updates re-use :class:`DownloadManager`, which only fetches deltas — Ollama
checks digests, HuggingFace checks ETags, so unchanged files are skipped.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.api.internal.downloads import DownloadCreateRequest, download_manager
from ocabra.core.model_ref import parse_model_ref
from ocabra.registry.metadata import fetch_registry_metadata
from ocabra.schemas.registry import DownloadJob

router = APIRouter(tags=["models-update"])


class UpdateAllResponse(BaseModel):
    enqueued: list[DownloadJob]
    skipped: list[dict]


class RefreshMetadataResponse(BaseModel):
    refreshed: list[str]
    skipped: list[dict]


def _build_update_request(state) -> tuple[DownloadCreateRequest | None, str | None]:
    """Return (request, skip_reason). One of the two will be None."""
    backend_type = str(state.backend_type or "").strip().lower()
    try:
        _backend, raw = parse_model_ref(state.model_id)
    except ValueError:
        return None, "invalid model_id"

    if backend_type == "ollama":
        return DownloadCreateRequest(source="ollama", model_ref=raw), None

    repo_id = raw
    if "::" in raw:
        repo_id = raw.split("::", 1)[0]

    extra = state.extra_config or {}
    model_path = extra.get("model_path") if isinstance(extra, dict) else None
    artifact = Path(model_path).name if model_path else None

    if backend_type == "bitnet":
        if not artifact:
            return None, "missing extra_config.model_path; cannot infer artifact"
        return (
            DownloadCreateRequest(source="bitnet", model_ref=repo_id, artifact=artifact),
            None,
        )

    if backend_type == "llama_cpp":
        if not artifact:
            return None, "local-only GGUF (no remote artifact recorded)"
        return (
            DownloadCreateRequest(source="huggingface", model_ref=repo_id, artifact=artifact),
            None,
        )

    if backend_type in {"vllm", "sglang", "diffusers", "whisper", "tts", "chatterbox", "voxtral", "transformers"}:
        return (
            DownloadCreateRequest(source="huggingface", model_ref=repo_id, artifact=artifact),
            None,
        )

    return None, f"backend '{backend_type}' is not updatable from a remote registry"


async def _enqueue_update(req: DownloadCreateRequest) -> DownloadJob:
    return await download_manager.enqueue(
        source=req.source,
        model_ref=req.model_ref,
        artifact=req.artifact,
        kind="update",
    )


@router.post(
    "/models/{model_id:path}/update",
    response_model=DownloadJob,
    summary="Re-pull a single installed model",
    description=(
        "Enqueue a download job that re-fetches the model from its original "
        "registry (HuggingFace / Ollama / BitNet). Only deltas are transferred "
        "if the upstream content is unchanged."
    ),
    responses={
        404: {"description": "Model not found"},
        409: {"description": "Model cannot be updated from a remote registry"},
    },
)
async def update_single_model(
    model_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> DownloadJob:
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    update_req, skip_reason = _build_update_request(state)
    if update_req is None:
        raise HTTPException(status_code=409, detail=skip_reason or "model is not updatable")

    return await _enqueue_update(update_req)


@router.post(
    "/models/refresh-metadata",
    response_model=RefreshMetadataResponse,
    summary="Refresh registry metadata for installed models",
    description=(
        "Fetch ``release_date`` and ``last_updated`` from each model's upstream "
        "registry (HuggingFace / Ollama) and persist them under "
        "``extra_config.registry_metadata``. Local-only files are listed in "
        "``skipped``."
    ),
)
async def refresh_models_metadata(
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> RefreshMetadataResponse:
    mm = request.app.state.model_manager
    states = await mm.list_states()

    refreshed: list[str] = []
    skipped: list[dict] = []
    for state in states:
        try:
            _backend, raw = parse_model_ref(state.model_id)
        except ValueError:
            skipped.append({"model_id": state.model_id, "reason": "invalid model_id"})
            continue
        try:
            metadata = await fetch_registry_metadata(state.backend_type, raw)
        except Exception as exc:
            skipped.append({"model_id": state.model_id, "reason": f"fetch failed: {exc}"})
            continue
        if metadata is None:
            skipped.append({"model_id": state.model_id, "reason": "no upstream registry"})
            continue
        ok = await mm.set_registry_metadata(state.model_id, metadata)
        if ok:
            refreshed.append(state.model_id)
        else:
            skipped.append({"model_id": state.model_id, "reason": "model disappeared"})

    return RefreshMetadataResponse(refreshed=refreshed, skipped=skipped)


@router.post(
    "/models/update-all",
    response_model=UpdateAllResponse,
    summary="Re-pull every updatable model",
    description=(
        "Enqueue an update job for every configured model that has a known "
        "remote source. Models without a re-pullable source are listed in "
        "``skipped`` with the reason."
    ),
)
async def update_all_models(
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> UpdateAllResponse:
    mm = request.app.state.model_manager
    states = await mm.list_states()

    enqueued: list[DownloadJob] = []
    skipped: list[dict] = []
    for state in states:
        req, reason = _build_update_request(state)
        if req is None:
            skipped.append({"model_id": state.model_id, "reason": reason or "unknown"})
            continue
        try:
            job = await _enqueue_update(req)
            enqueued.append(job)
        except Exception as exc:
            skipped.append({"model_id": state.model_id, "reason": f"enqueue failed: {exc}"})

    return UpdateAllResponse(enqueued=enqueued, skipped=skipped)
