"""On-disk model-file helpers for the internal models API (REF-07 split).

Disk-size resolution and file deletion used by ``GET /ocabra/models`` and
``DELETE /ocabra/models/{id}``. The path-resolution primitives they build on
live in ``core/memory_estimate.py`` (shared with the memory estimate), so this
module stays a thin API-layer orchestration over them. Extracted verbatim from
``api/internal/models.py``; behaviour is unchanged.
"""

import asyncio
import shutil
from pathlib import Path

from fastapi import HTTPException

from ocabra.config import settings
from ocabra.core.memory_estimate import (
    _compute_path_size_bytes,
    _is_path_within_base,
    _resolve_extra_config_path,
    _resolve_local_model_path,
    _resolve_tensorrt_engine_path,
)
from ocabra.core.model_ref import parse_model_ref


def _unshared_download_dir(raw_path: str, other_paths: list[str]) -> Path | None:
    """Blocking part of :func:`_artifact_download_dir` (run it in a thread)."""
    root = (Path(settings.models_dir or "/data/models") / "huggingface").resolve()
    path = Path(raw_path).resolve()
    if path.is_file():
        path = path.parent
    if path.parent != root or not path.is_dir():
        return None
    for other in other_paths:
        if Path(other).resolve().is_relative_to(path):
            return None
    return path


async def _artifact_download_dir(model_id: str, artifact: dict | None) -> Path | None:
    """Download folder of a local GGUF artifact, if nothing else uses it.

    Since model ids are the public name (``llamacpp_mks/ravenx-chaos-agent``),
    the folder can no longer be derived from the id: the download manager
    names it after the repo and file (``<org>--<repo>--<stem>``) and the
    artifact keeps that path. Deleting by id left 16 GB behind (2026-11-21).

    Only a direct child of ``<models_dir>/huggingface`` is returned, and only
    when no other registered model points into it.
    """
    import sqlalchemy as sa

    from ocabra.database import AsyncSessionLocal
    from ocabra.db.model_config import ModelConfig

    if not isinstance(artifact, dict) or not artifact.get("path"):
        return None
    if artifact.get("source") == "ollama":
        return None  # Ollama's store: deleted through Ollama, not here
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                sa.select(ModelConfig.artifact).where(ModelConfig.model_id != model_id)
            )
        ).all()
    other_paths = [
        str(row[0]["path"]) for row in rows if isinstance(row[0], dict) and row[0].get("path")
    ]
    return await asyncio.to_thread(_unshared_download_dir, str(artifact["path"]), other_paths)


async def _delete_model_files(
    model_id: str,
    backend_type: str,
    extra_config: dict | None = None,
    artifact: dict | None = None,
) -> str | None:
    """Delete model files from disk based on backend type."""
    artifact_dir = await _artifact_download_dir(model_id, artifact)
    if artifact_dir is not None:
        await asyncio.to_thread(shutil.rmtree, artifact_dir)
        return str(artifact_dir)

    if backend_type == "comfyui_video":
        # La carpeta de pesos de vídeo es compartida entre modelos de vídeo y
        # con el ComfyUI interactivo: borrar uno solo quita su configuración.
        return None

    if backend_type == "piper":
        # Piper voices are registered with explicit absolute file paths
        # (extra_config.piper.onnx_path / config_path) rather than a
        # predictable HF-repo-derived directory name, so the generic
        # HF-layout cleanup below never matches them and silently leaves
        # the .onnx/.onnx.json orphaned on disk. Delete the two files
        # directly instead of the whole parent directory, since a shared
        # directory could in principle hold sibling voices.
        piper_cfg = (extra_config or {}).get("piper") or {}
        models_dir = Path(settings.models_dir or "/data/models")
        deleted_paths: list[str] = []
        for key in ("onnx_path", "config_path"):
            raw_path = piper_cfg.get(key)
            if not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.exists():  # noqa: ASYNC240 — solo metadatos (microsegundos); un to_thread costaria mas que la syscall
                continue
            if not _is_path_within_base(candidate, models_dir):
                raise HTTPException(
                    status_code=400,
                    detail=f"Refusing to delete '{candidate}' outside its configured base",
                )
            await asyncio.to_thread(candidate.unlink)
            deleted_paths.append(str(candidate))
        return ",".join(deleted_paths) if deleted_paths else None

    if backend_type == "ollama":
        # Ollama manages its own storage in a separate service, so we must call
        # its HTTP API — the ``ollama`` CLI is NOT installed in the api
        # container (a previous ``ollama rm`` subprocess silently no-op'd and
        # the model reappeared on the next inventory sync).
        import httpx

        parts = model_id.split("/", 1)
        ollama_name = parts[1] if len(parts) == 2 else model_id
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.request(
                    "DELETE",
                    f"{settings.ollama_base_url}/api/delete",
                    json={"name": ollama_name},
                )
        except Exception as exc:  # noqa: BLE001 — surface connectivity failures
            raise HTTPException(
                status_code=502,
                detail=f"Could not reach Ollama to delete '{ollama_name}': {exc}",
            ) from exc
        # 404 = already gone; treat as success. Anything else is a real error.
        if resp.status_code not in (200, 404):
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Ollama refused to delete '{ollama_name}': "
                    f"{resp.status_code} {resp.text[:200]}"
                ),
            )
        return f"ollama:{ollama_name}"

    # HuggingFace-backed models (vllm, diffusers, whisper, transformers, tensorrt_llm…)
    #
    # Blobs live in up to three places on our layout:
    #   1) ``<models_dir>/huggingface/<repo>--<subdir>``: the download manager
    #      target for first-time pulls of pipeline artifacts.
    #   2) ``<hf_cache_dir>/hub/models--<repo>``: the standard HuggingFace
    #      Hub cache used by ``huggingface_hub`` / ``transformers`` when
    #      loading pretrained checkpoints on the fly (this is where the
    #      heavy safetensors typically end up).
    #   3) ``<hf_cache_dir>/models--<repo>``: a legacy layout that the
    #      snapshot_download path used to write to before v0.30-ish; still
    #      appears in older installations. Left in place for compatibility.
    #
    # Prior to 2026-09-17 only (1) was cleaned up, so deleting a vLLM
    # model from the UI left tens of GB of orphaned blobs in (2)/(3) — the
    # dashboard would report the model gone while ``df`` insisted the
    # disk was full. Now we clean up whichever locations exist.
    #
    # 2026-11-01: a single-GGUF-file model_id (llama.cpp registry entries
    # for a repo with multiple quants, e.g.
    # ``llama_cpp/Qwen/Qwen2.5-0.5B-Instruct-GGUF::qwen2.5-0.5b-instruct-
    # q4_k_m``) embeds a SECOND separator, ``::``, between the repo and the
    # specific artifact stem — distinct from the ``/`` the code above
    # already handled. The download manager
    # (api/internal/downloads.py — ``f"{model_ref.replace('/','--')}--
    # {stem}"``) writes the directory with BOTH separators flattened to
    # ``--``, but this function only flattened ``/``, so the computed
    # candidate path for ``models_dir/huggingface/`` kept a literal ``::``
    # that never matched the real directory on disk — ``candidate.exists()``
    # silently failed and the GGUF (and its hf_cache metadata) was never
    # deleted. Confirmed live: ~5.5 GB left behind across 2 deleted models
    # after the API returned 200 OK for both. The hf_cache entries for a
    # single-GGUF-file download are also keyed by repo ALONE (no artifact
    # stem — one hub cache dir is shared by every quant of that repo), so
    # that candidate needs the repo-only variant, not the full name.
    parts = model_id.split("/", 1)
    raw_model = parts[1] if len(parts) == 2 else model_id
    repo_part, _, artifact_part = raw_model.partition("::")
    hf_dir_name = raw_model.replace("::", "--").replace("/", "--")
    hf_repo_dir_name = repo_part.replace("/", "--")
    hub_dir_name = f"models--{hf_dir_name}"
    hub_repo_dir_name = f"models--{hf_repo_dir_name}"
    models_dir = Path(settings.models_dir or "/data/models")
    hf_cache_dir = Path(settings.hf_cache_dir or "/data/hf_cache")

    deleted_paths: list[str] = []
    candidates: list[tuple[Path, Path]] = [
        # (path, base_dir_for_traversal_check)
        (models_dir / "huggingface" / hf_dir_name, models_dir / "huggingface"),
        (hf_cache_dir / "hub" / hub_dir_name, hf_cache_dir / "hub"),
        (hf_cache_dir / hub_dir_name, hf_cache_dir),
    ]
    if artifact_part:
        candidates.extend(
            [
                (hf_cache_dir / "hub" / hub_repo_dir_name, hf_cache_dir / "hub"),
                (hf_cache_dir / hub_repo_dir_name, hf_cache_dir),
            ]
        )
    for candidate, base in candidates:
        if not candidate.exists():
            continue
        if not _is_path_within_base(candidate, base):
            raise HTTPException(
                status_code=400,
                detail=f"Refusing to delete '{candidate}' outside its configured base",
            )
        await asyncio.to_thread(shutil.rmtree, candidate)
        deleted_paths.append(str(candidate))
    return ",".join(deleted_paths) if deleted_paths else None


async def _resolve_disk_size_bytes(
    state,
    ollama_sizes: dict[str, int],
) -> int | None:
    backend_type, backend_model_id = parse_model_ref(state.model_id)
    if backend_type == "ollama":
        return ollama_sizes.get(backend_model_id.strip().lower())
    if backend_type == "tensorrt_llm":
        path = _resolve_extra_config_path(
            state.extra_config, "engine_dir"
        ) or _resolve_tensorrt_engine_path(backend_model_id)
        if path is None or not path.exists():
            return None
        return await asyncio.to_thread(_compute_path_size_bytes, path)

    return await asyncio.to_thread(_local_disk_size_bytes, state.extra_config, backend_model_id)


def _local_disk_size_bytes(extra_config: dict | None, backend_model_id: str) -> int | None:
    """Bytes on disk of a local model's weights. Blocking: run it in a thread."""
    path = _resolve_extra_config_path(
        extra_config, "model_path", "base_model_id"
    ) or _resolve_local_model_path(backend_model_id)
    if path is None or not path.exists():
        return None
    return _compute_path_size_bytes(path)
