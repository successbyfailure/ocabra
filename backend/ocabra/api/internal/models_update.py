"""Re-pull / update endpoints for already installed models.

Supports the same source families as the download manager: ``ollama``,
``huggingface`` (full repo) and ``bitnet`` / ``llama_cpp`` single-file
GGUFs.  The structured ``ModelState.artifact`` descriptor is the source of
truth; ``extra_config.model_path`` is kept only as a legacy fallback.
Updates re-use :class:`DownloadManager`, which only fetches deltas — Ollama
checks digests, HuggingFace checks ETags, so unchanged files are skipped.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.api.internal.downloads import DownloadCreateRequest, download_manager
from ocabra.core.model_ref import is_llama_cpp_family, parse_model_ref
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

    repo_id = raw
    if "::" in raw:
        repo_id = raw.split("::", 1)[0]

    extra = state.extra_config or {}
    descriptor = state.artifact if isinstance(getattr(state, "artifact", None), dict) else {}

    # An explicit source is useful for local aliases whose public name is not
    # pullable.  Prefer it over all legacy inference.
    update_source = extra.get("update_source") if isinstance(extra, dict) else None
    if isinstance(update_source, dict):
        source = str(update_source.get("source") or "").strip().lower()
        model_ref = str(update_source.get("model_ref") or update_source.get("ref") or "").strip()
        if source in {"huggingface", "ollama", "bitnet"} and model_ref:
            return DownloadCreateRequest(
                source=source,
                model_ref=model_ref,
                artifact=update_source.get("artifact"),
            ), None

    artifact_ref = str(descriptor.get("ref") or "").strip()
    artifact_repo = str(descriptor.get("repo") or "").strip()
    artifact_file = str(descriptor.get("file") or "").strip() or None
    artifact_path = str(descriptor.get("path") or "").strip()
    artifact_source = str(descriptor.get("source") or "").strip().lower()
    artifact_kind = str(descriptor.get("kind") or "").strip().lower()

    if artifact_source == "ollama" and artifact_ref:
        return DownloadCreateRequest(source="ollama", model_ref=artifact_ref), None

    if artifact_source == "hf":
        resolved_repo = artifact_repo or artifact_ref.split("::", 1)[0]
        resolved_file = artifact_file or (Path(artifact_path).name if artifact_path else None)
        if resolved_repo and "/" in resolved_repo:
            return DownloadCreateRequest(
                source="bitnet" if backend_type == "bitnet" else "huggingface",
                model_ref=resolved_repo,
                artifact=resolved_file,
            ), None

    # Older repo/tree descriptors were recorded as source=local even though
    # their ref is the original HuggingFace repo.  Preserve that provenance.
    if artifact_kind in {"hf_repo", "ctranslate2"} and "/" in artifact_ref:
        return DownloadCreateRequest(source="huggingface", model_ref=artifact_ref), None

    model_path = extra.get("model_path") if isinstance(extra, dict) else None
    legacy_artifact = Path(model_path).name if model_path else None
    artifact_name = (
        artifact_file or (Path(artifact_path).name if artifact_path else None) or legacy_artifact
    )

    if backend_type == "ollama":
        return DownloadCreateRequest(source="ollama", model_ref=raw), None

    if backend_type == "bitnet":
        if artifact_repo and artifact_name:
            return DownloadCreateRequest(
                source="bitnet", model_ref=artifact_repo, artifact=artifact_name
            ), None
        if not artifact_name:
            return None, "missing extra_config.model_path; cannot infer artifact"
        return (
            DownloadCreateRequest(source="bitnet", model_ref=repo_id, artifact=artifact_name),
            None,
        )

    if is_llama_cpp_family(backend_type):
        if artifact_repo and artifact_name:
            return DownloadCreateRequest(
                source="huggingface", model_ref=artifact_repo, artifact=artifact_name
            ), None
        if not artifact_name:
            return None, "local-only GGUF (no remote artifact recorded)"
        return (
            DownloadCreateRequest(source="huggingface", model_ref=repo_id, artifact=artifact_name),
            None,
        )

    if backend_type == "piper":
        # Piper voices never go through DownloadManager at all — the worker
        # fetches/caches them itself via hf_hub_download() on first load
        # (workers/piper_worker.py:_download_voice_file), which already does
        # its own "only re-fetch if changed" check. Treating the voice name
        # as an HF repo_id (the generic branch below) 404s every time:
        # "es_ES-davefx-medium" is a filename pattern inside the shared
        # rhasspy/piper-voices repo, not a repo of its own. Reproduced live
        # 2026-10-29 via /ocabra/models/update-all.
        piper_cfg = extra.get("piper") if isinstance(extra, dict) else None
        if isinstance(piper_cfg, dict) and piper_cfg.get("onnx_path"):
            return None, "local-only Piper voice (custom onnx_path, no remote source)"
        return (
            None,
            "Piper voices refresh automatically from cache when the worker "
            "next loads; not updatable via the download queue",
        )

    if backend_type == "vibeasr":
        # model_id stores the repo with "/" flattened to "--" for local
        # directory naming (vibeasr_backend.py:flat_id) — un-flatten before
        # using it as an actual HF repo_id, or hf_hub_download 404s on the
        # literal flattened string. Reproduced live 2026-10-29.
        # Newer unified deployments have a friendly model_key while the
        # original flattened repo remains in artifact.ref.
        repo_id = (artifact_ref or repo_id).replace("--", "/")
        return (
            DownloadCreateRequest(source="huggingface", model_ref=repo_id, artifact=artifact_name),
            None,
        )

    if backend_type == "diffusers":
        diffusers_cfg = extra.get("diffusers") if isinstance(extra, dict) else None
        gguf_path = (
            diffusers_cfg.get("gguf_transformer_path") if isinstance(diffusers_cfg, dict) else None
        )
        if gguf_path:
            # Custom-assembled pipeline (GGUF transformer from one repo +
            # a separate text encoder from another) — the model_id is a
            # local alias, not a real HF repo_id, so the generic branch
            # below 404s. No single coherent remote source to re-pull from.
            # Reproduced live 2026-10-29 (qwen-image-2.1, qwen-image-edit-plus).
            return None, "custom-assembled diffusers pipeline (no single remote source)"

    if backend_type in {
        "vllm",
        "sglang",
        "diffusers",
        "whisper",
        "tts",
        "chatterbox",
        "voxtral",
        "transformers",
    }:
        return (
            DownloadCreateRequest(source="huggingface", model_ref=repo_id, artifact=artifact_name),
            None,
        )

    return None, f"backend '{backend_type}' is not updatable from a remote registry"


async def _enqueue_update(req: DownloadCreateRequest, state) -> DownloadJob:
    _backend, model_key = parse_model_ref(state.model_id)
    return await download_manager.enqueue(
        source=req.source,
        model_ref=req.model_ref,
        artifact=req.artifact,
        register_config={
            "backend_type": state.backend_type,
            "model_key": model_key,
        },
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

    return await _enqueue_update(update_req, state)


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
        update_req, reason = _build_update_request(state)
        if update_req is None:
            skipped.append({"model_id": state.model_id, "reason": reason or "no upstream registry"})
            continue
        try:
            metadata = await fetch_registry_metadata(
                "ollama" if update_req.source == "ollama" else "vllm",
                update_req.model_ref,
            )
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
    # Prefer a configured Ollama deployment as the owner of a shared pull;
    # llama.cpp deployments and discovered inventory entries often point at
    # the exact same Ollama artifact.
    states = sorted(
        states,
        key=lambda state: (
            state.backend_type != "ollama",
            getattr(state, "model_key", None) is None,
            state.model_id,
        ),
    )
    owners: dict[tuple[str, str, str | None], str] = {}
    for state in states:
        req, reason = _build_update_request(state)
        if req is None:
            skipped.append({"model_id": state.model_id, "reason": reason or "unknown"})
            continue
        key = (req.source, req.model_ref, req.artifact)
        owner = owners.get(key)
        if owner is not None:
            skipped.append(
                {
                    "model_id": state.model_id,
                    "reason": f"shares update source with '{owner}'",
                }
            )
            continue
        try:
            job = await _enqueue_update(req, state)
            enqueued.append(job)
            owners[key] = state.model_id
        except Exception as exc:
            skipped.append({"model_id": state.model_id, "reason": f"enqueue failed: {exc}"})

    return UpdateAllResponse(enqueued=enqueued, skipped=skipped)
