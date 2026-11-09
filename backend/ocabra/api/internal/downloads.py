import asyncio
import contextlib
import re
import shutil
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.core.background_tasks import spawn
from ocabra.core.compatibility import backends_for_artifact
from ocabra.core.model_ref import (
    KNOWN_BACKEND_TYPES,
    LLAMA_CPP_CONFIG_SECTION,
    build_model_ref,
    is_llama_cpp_family,
)
from ocabra.core.registration_defaults import launch_defaults
from ocabra.redis_client import get_key, lpush, publish, rpop, set_key, subscribe
from ocabra.registry.huggingface import HuggingFaceRegistry
from ocabra.registry.ollama_registry import OllamaRegistry
from ocabra.schemas.registry import DownloadJob

logger = structlog.get_logger(__name__)

QUEUE_NAME = "queue:download"
DOWNLOAD_PROGRESS_CHANNEL = "download:progress"


class DownloadCreateRequest(BaseModel):
    source: Literal["huggingface", "ollama", "bitnet"]
    model_ref: str
    artifact: str | None = None
    register_config: dict | None = None


def _rc_get(register_config: dict, snake: str, default=None):
    """Read a ``register_config`` key accepting snake_case or camelCase.

    The UI builds this payload client-side and ``api/client.ts`` ships it with
    ``JSON.stringify`` — no case conversion anywhere — so it arrives camelCased
    while every reader here expects snake_case. The keys therefore all missed:
    the load policy and display name picked in the Explore install modal were
    silently discarded, and worse, ``extraConfig`` carrying the whole vLLM
    recipe preconfiguration was dropped, so the "con preconfiguracion" toast
    was never true (verified 2026-11-06: zero registered vLLM models carry a
    recipeId). Accepting both spellings fixes it regardless of client version.
    """
    camel = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), snake)
    value = register_config.get(snake)
    if value is None:
        value = register_config.get(camel)
    return default if value is None else value


class DownloadManager:
    def __init__(self) -> None:
        self._hf_registry = HuggingFaceRegistry()
        self._ollama_registry = OllamaRegistry()
        self._worker_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._active_cancel_flags: dict[str, asyncio.Event] = {}
        self._app = None

    async def start(self, app=None) -> None:
        self._app = app
        if not self._worker_task or self._worker_task.done():
            await self._cleanup_stale_jobs()
            self._worker_task = asyncio.create_task(
                self._worker_loop(), name="download-manager-worker"
            )
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(), name="download-manager-cleanup"
            )

    async def _cleanup_stale_jobs(self) -> None:
        """Mark jobs left in 'downloading' state as failed (server was killed mid-download)."""
        jobs = await self.list_jobs()
        for job in jobs:
            if job.status == "downloading":
                failed = job.model_copy(
                    update={
                        "status": "failed",
                        "error": "interrupted (server restart)",
                        "completed_at": datetime.now(UTC),
                    }
                )
                await self._save_and_publish(failed)

    async def _cleanup_loop(self) -> None:
        """Periodically sweep terminal jobs whose result has had time to be
        seen — the queue would otherwise accumulate every download/update job
        ever run (111 stale entries found on 2026-11-02, days old)."""
        while True:
            await asyncio.sleep(max(60, int(settings.download_job_cleanup_poll_interval_s)))
            try:
                await self._sweep_old_jobs()
            except Exception:
                logger.exception("download_cleanup_sweep_failed")

    async def _sweep_old_jobs(self) -> int:
        now = datetime.now(UTC)
        completed_cutoff = now.timestamp() - settings.download_job_retention_completed_s
        failed_cutoff = now.timestamp() - settings.download_job_retention_failed_s
        redis = get_redis_safe()
        deleted = 0
        for job in await self.list_jobs():
            if job.status not in {"completed", "failed", "cancelled"} or job.completed_at is None:
                continue
            cutoff = completed_cutoff if job.status == "completed" else failed_cutoff
            if job.completed_at.timestamp() < cutoff:
                await redis.delete(f"download:job:{job.job_id}")
                deleted += 1
        return deleted

    async def stop(self) -> None:
        for task in (self._worker_task, self._cleanup_task):
            if not task:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._worker_task = None
        self._cleanup_task = None

    async def enqueue(
        self,
        source: str,
        model_ref: str,
        artifact: str | None = None,
        register_config: dict | None = None,
        kind: str = "install",
    ) -> DownloadJob:
        if source not in {"huggingface", "ollama", "bitnet"}:
            raise ValueError("source must be 'huggingface', 'ollama' or 'bitnet'")
        if kind not in {"install", "update"}:
            raise ValueError("kind must be 'install' or 'update'")

        now = datetime.now(UTC)
        job = DownloadJob(
            job_id=uuid.uuid4().hex,
            source=source,
            model_ref=model_ref,
            artifact=artifact,
            register_config=register_config,
            status="queued",
            progress_pct=0.0,
            speed_mb_s=None,
            eta_seconds=None,
            error=None,
            started_at=now,
            completed_at=None,
            kind=kind,
        )

        await set_key(f"download:job:{job.job_id}", job.model_dump(mode="json"))
        await lpush(QUEUE_NAME, {"job_id": job.job_id})
        await self._publish_job_update(job)
        return job

    async def cancel(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        if not job:
            raise KeyError(job_id)

        if job.status in {"completed", "failed", "cancelled"}:
            return

        cancel_event = self._active_cancel_flags.get(job_id)
        if cancel_event:
            cancel_event.set()

        updated = job.model_copy(update={"status": "cancelled", "completed_at": datetime.now(UTC)})
        await self._save_and_publish(updated)

    async def get_job(self, job_id: str) -> DownloadJob | None:
        raw = await get_key(f"download:job:{job_id}")
        if not raw:
            return None
        return DownloadJob.model_validate(raw)

    async def list_jobs(self) -> list[DownloadJob]:
        redis = get_redis_safe()
        keys = await redis.keys("download:job:*")

        jobs: list[DownloadJob] = []
        for key in keys:
            payload = await get_key(key)
            if payload:
                jobs.append(DownloadJob.model_validate(payload))
        jobs.sort(key=lambda j: j.started_at, reverse=True)
        return jobs

    async def _worker_loop(self) -> None:
        while True:
            payload = await rpop(QUEUE_NAME)
            if not payload:
                await asyncio.sleep(0.2)
                continue

            job_id = str(payload.get("job_id", ""))
            job = await self.get_job(job_id)
            if not job:
                continue
            if job.status == "cancelled":
                continue

            await self._execute_job(job)

    async def _execute_job(self, job: DownloadJob) -> None:
        cancel_flag = asyncio.Event()
        self._active_cancel_flags[job.job_id] = cancel_flag
        loop = asyncio.get_event_loop()

        # Preflight free-space check. Bail out before allocating anything if
        # the target dir is already below threshold — prevents the cascade of
        # 80 enqueued re-pulls all failing mid-stream when the first one
        # fills the disk.
        try:
            free_bytes = await asyncio.to_thread(
                lambda: shutil.disk_usage(settings.models_dir).free
            )
        except Exception:
            free_bytes = None
        if free_bytes is not None:
            free_gb = free_bytes / (1024**3)
            min_gb = float(settings.download_min_free_gb or 0)
            if free_gb < min_gb:
                failed = job.model_copy(
                    update={
                        "status": "failed",
                        "error": (
                            f"insufficient disk space: {free_gb:.1f} GB free in "
                            f"{settings.models_dir} is below the configured "
                            f"download_min_free_gb threshold ({min_gb:.0f} GB). "
                            f"Free space and retry."
                        ),
                        "completed_at": datetime.now(UTC),
                    }
                )
                await self._save_and_publish(failed)
                self._active_cancel_flags.pop(job.job_id, None)
                return

        downloading = job.model_copy(update={"status": "downloading"})
        await self._save_and_publish(downloading)

        try:

            def _progress(pct: float, speed_mb_s: float | None) -> None:
                if cancel_flag.is_set():
                    raise asyncio.CancelledError
                # Skip 100% update — _execute_job handles completed status to avoid race
                if pct >= 100:
                    return

                eta = None
                if speed_mb_s and speed_mb_s > 0:
                    remaining_mb = max(0.0, 1.0 - pct / 100.0) * 1024
                    eta = int(remaining_mb / speed_mb_s)

                updated = downloading.model_copy(
                    update={
                        "progress_pct": round(float(pct), 2),
                        "speed_mb_s": speed_mb_s,
                        "eta_seconds": eta,
                    }
                )
                try:
                    asyncio.get_running_loop()
                    spawn(self._save_and_publish(updated))
                except RuntimeError:
                    loop.call_soon_threadsafe(loop.create_task, self._save_and_publish(updated))

            target_dir = Path(settings.models_dir)
            if downloading.source == "huggingface":
                await self._hf_registry.download(
                    repo_id=downloading.model_ref,
                    target_dir=self._hf_download_dir(downloading),
                    progress_callback=_progress,
                    artifact=downloading.artifact,
                )
            elif downloading.source == "bitnet":
                if not downloading.artifact:
                    raise ValueError("bitnet downloads require a .gguf artifact")
                stem = Path(downloading.artifact).stem.replace("/", "_")
                download_dir = (
                    target_dir
                    / "huggingface"
                    / f"{downloading.model_ref.replace('/', '--')}--{stem}"
                )
                await asyncio.to_thread(download_dir.mkdir, parents=True, exist_ok=True)

                await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=downloading.model_ref,
                    filename=downloading.artifact,
                    local_dir=str(download_dir),
                    token=settings.hf_token or None,
                    cache_dir=settings.hf_cache_dir or None,
                )
                _progress(100.0, None)
            else:
                await self._ollama_registry.pull(
                    model_ref=downloading.model_ref,
                    progress_callback=_progress,
                )

            if cancel_flag.is_set():
                cancelled = downloading.model_copy(
                    update={"status": "cancelled", "completed_at": datetime.now(UTC)}
                )
                await self._save_and_publish(cancelled)
                return

            completed = downloading.model_copy(
                update={
                    "status": "completed",
                    "progress_pct": 100.0,
                    "eta_seconds": 0,
                    "completed_at": datetime.now(UTC),
                }
            )
            await self._save_and_publish(completed)
            # Updates refresh bytes and metadata for existing deployments.
            # Auto-registration belongs only to first-time installs; running it
            # here created duplicate logical models/backends on update-all.
            if downloading.kind == "install":
                await self._auto_register_model(downloading)
            await self._refresh_registry_metadata(downloading)
        except asyncio.CancelledError:
            cancelled = downloading.model_copy(
                update={"status": "cancelled", "completed_at": datetime.now(UTC)}
            )
            await self._save_and_publish(cancelled)
        except Exception as exc:
            failed = downloading.model_copy(
                update={
                    "status": "failed",
                    "error": str(exc),
                    "completed_at": datetime.now(UTC),
                }
            )
            await self._save_and_publish(failed)
        finally:
            self._active_cancel_flags.pop(job.job_id, None)

    async def _auto_register_model(self, job: DownloadJob) -> None:
        """Register downloaded model in the model manager if not already configured."""
        if not self._app:
            return
        try:
            mm = self._app.state.model_manager
            backend_model_id = self._job_model_id(job)
            register_config = job.register_config or {}
            backend_type = await self._resolve_job_backend(job)
            model_key = self._job_model_key(job)
            model_id = build_model_ref(backend_type, model_key)
            existing = await mm.get_state(model_id)
            if existing:
                return
            artifact = None
            if job.source == "ollama" and backend_type != "ollama":
                # An Ollama pull served by llama.cpp: read the tag's blob through
                # its manifest on every load (follows future ``ollama pull``s).
                from ocabra.core.compatibility import derive_artifact

                artifact = derive_artifact(
                    {"kind": "ollama_tag", "source": "ollama", "ref": job.model_ref}, backend_type
                )
            elif job.source == "ollama":
                artifact = {"kind": "ollama_tag", "source": "ollama", "ref": job.model_ref}
            extra_config = _rc_get(register_config, "extra_config") or {}
            # Single-file weights (BitNet / llama.cpp GGUF) are stored under
            # ``huggingface/<repo-->--<artifact stem>/<artifact>``. Pin the exact
            # path: the backends can't derive it from the ``repo::artifact``
            # model_id (the '::' separator and dots in version names defeat the
            # directory/stem heuristics), so without this the model registers
            # fine but fails to load with "GGUF model not found".
            if job.artifact and (
                job.source == "bitnet"
                or (is_llama_cpp_family(backend_type) or backend_type == "bitnet")
            ):
                stem = Path(job.artifact).stem.replace("/", "_")
                model_path = (
                    Path(settings.models_dir)
                    / "huggingface"
                    / f"{job.model_ref.replace('/', '--')}--{stem}"
                    / job.artifact
                )
                # Into the backend's own section, same as ``/registry/local/register``
                # does: both llama.cpp backends share ``llama_cpp`` and bitnet reads
                # ``bitnet``. Writing it at the top level worked for llama.cpp only
                # because its option lookup falls back to the root.
                section = (
                    LLAMA_CPP_CONFIG_SECTION if is_llama_cpp_family(backend_type) else "bitnet"
                )
                section_cfg = extra_config.get(section)
                section_cfg = dict(section_cfg) if isinstance(section_cfg, dict) else {}
                section_cfg["model_path"] = str(model_path)
                extra_config = {**extra_config, section: section_cfg}

            if job.source == "huggingface" and backend_type == "whisper":
                nemo_path = self._find_downloaded_nemo_path(job)
                if nemo_path is not None:
                    extra_config = {**extra_config, "base_model_id": str(nemo_path)}

            # Nothing else fills these in: installing from Explore sends no
            # extra_config at all, and llama.cpp without one runs on the CPU
            # with a 4096-token window (see registration_defaults).
            extra_config = launch_defaults(backend_type, extra_config, artifact=artifact)

            await mm.add_model(
                model_id=model_id,
                backend_type=backend_type,
                display_name=_rc_get(register_config, "display_name") or model_key,
                load_policy=_rc_get(register_config, "load_policy", "on_demand"),
                pin_permanent=bool(_rc_get(register_config, "pin_permanent", False)),
                preferred_gpu=_rc_get(register_config, "preferred_gpu"),
                extra_config=extra_config or None,
                model_key=model_key,
                backend_model_id_override=backend_model_id if artifact is None else None,
                artifact=artifact,
            )
            # Playground / /v1/models look up by profile_id; without a default
            # profile the freshly downloaded model can't be selected until the
            # next restart (only startup and the Ollama loop trigger a sweep).
            profile_registry = getattr(self._app.state, "profile_registry", None)
            if profile_registry is not None:
                await profile_registry.on_model_added(model_id)
                # Extra deployments of the same file (e.g. the Ollama copy of a
                # pull served by llama.cpp, or upstream llama.cpp next to the fork).
                from ocabra.core import model_deployments as md

                for extra_backend in self._job_extra_backends(job, backend_type):
                    try:
                        await md.add_deployment(
                            model_key,
                            extra_backend,
                            model_manager=mm,
                            profile_registry=profile_registry,
                            source_backend=backend_type,
                        )
                    except Exception:  # noqa: BLE001 — the primary deployment is already usable
                        logger.warning(
                            "download_extra_deployment_failed",
                            job_id=job.job_id,
                            model_key=model_key,
                            backend=extra_backend,
                            exc_info=True,
                        )
        except Exception:
            # Non-fatal (the model can be registered by hand), but it must be
            # visible: a silent ``pass`` here meant a finished download that
            # never showed up in Models, with nothing in the logs explaining why.
            logger.warning(
                "download_auto_register_failed",
                job_id=job.job_id,
                model_ref=job.model_ref,
                exc_info=True,
            )

    async def _resolve_job_backend(self, job: DownloadJob) -> str:
        """Backend a finished download is registered under.

        Shared by auto-registration and the metadata refresh so both address
        the same ``model_id``. The metadata refresh used to re-infer the
        backend on its own and ignore the user's explicit choice, so picking a
        non-default backend in Explore wrote the metadata to a model id that
        did not exist.

        An explicit choice from the UI wins over inference. The same repo can
        legitimately be served by more than one backend (safetensors + GGUF in
        one upload, or upstream llama.cpp vs the Makespace fork), and
        ``infer_backend_for_repo`` can only ever return one of them. Validated
        against KNOWN_BACKEND_TYPES so a typo degrades to inference instead of
        registering an unservable model id.
        """
        register_config = job.register_config or {}
        default_backend = _rc_get(register_config, "default_backend")
        if isinstance(default_backend, str) and default_backend in KNOWN_BACKEND_TYPES:
            return default_backend
        requested_backend = _rc_get(register_config, "backend_type")
        if isinstance(requested_backend, str) and requested_backend in KNOWN_BACKEND_TYPES:
            return requested_backend
        if job.source == "ollama":
            # Ollama is the distribution channel; serving goes to the
            # production llama.cpp when it is installed (user, 2026-11-07) —
            # but only for tags it can actually serve. A ternary/1-bit GGUF
            # (bonsai, BitNet, i2_s, q1_0, ...) runs on the bitnet backend
            # and nowhere else, and ``add_model`` refuses the mismatch, so
            # choosing llamacpp_mks blindly ended with the weights downloaded
            # and no model registered: "Backend 'llamacpp_mks' cannot serve
            # this artifact (ollama_tag); compatible: bitnet". Reproduced
            # 2026-11-09 pulling ``bonsai-2:27b``.
            #
            # Asking the same compatibility table ``add_model`` validates
            # against is what keeps the two from disagreeing.
            installed = self._installed_backends()
            compatibles = backends_for_artifact(
                {"kind": "ollama_tag", "source": "ollama", "ref": job.model_ref}, installed
            )
            instalados = [c["backend"] for c in compatibles if c["installed"]]
            if "llamacpp_mks" in instalados:
                return "llamacpp_mks"
            if instalados:
                return instalados[0]
            if compatibles:
                # Nothing compatible is installed: register on the backend
                # that could serve it anyway. "Backend not installed" at load
                # time is a state an operator can act on; a download that
                # vanishes is not.
                return compatibles[0]["backend"]
            return "ollama"
        if job.source == "bitnet":
            return "bitnet"
        backend_type = await self._hf_registry.infer_backend_for_repo(
            job.model_ref, artifact=job.artifact
        )
        if isinstance(requested_backend, str) and requested_backend:
            logger.warning(
                "download_register_unknown_backend_requested",
                requested=requested_backend,
                fell_back_to=backend_type,
            )
        return backend_type

    async def _refresh_registry_metadata(self, job: DownloadJob) -> None:
        """Fetch upstream release / last-modified dates and persist them on the model.

        Runs after a successful download (initial install or update). Skips
        silently on any failure — metadata is best-effort.
        """
        if not self._app:
            return
        try:
            from ocabra.registry.metadata import fetch_registry_metadata

            mm = self._app.state.model_manager
            backend_model_id = self._job_model_id(job)
            backend_type = await self._resolve_job_backend(job)

            model_id = build_model_ref(backend_type, self._job_model_key(job))
            metadata = await fetch_registry_metadata(
                "ollama" if job.source == "ollama" else backend_type, backend_model_id
            )
            if metadata is None:
                return
            await mm.set_registry_metadata(model_id, metadata)
        except Exception:
            pass  # Non-fatal: metadata can be backfilled later.

    def _installed_backends(self) -> set[str] | None:
        if not self._app:
            return None
        from ocabra.core.model_deployments import installed_backends

        return installed_backends(self._app.state)

    def _job_model_key(self, job: DownloadJob) -> str:
        """Public name of the downloaded model: ``register_config.model_key``,
        else the Ollama tag / repo reference (the pre-unification id)."""
        raw = _rc_get(job.register_config or {}, "model_key")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return self._job_model_id(job)

    def _job_extra_backends(self, job: DownloadJob, primary: str) -> list[str]:
        """Backends to deploy besides the primary one.

        ``register_config.backends`` when given; otherwise an Ollama pull
        served by llama.cpp also keeps its Ollama deployment as fallback.
        """
        raw = _rc_get(job.register_config or {}, "backends")
        if isinstance(raw, list):
            wanted = [str(b) for b in raw if str(b) in KNOWN_BACKEND_TYPES]
        elif job.source == "ollama" and primary != "ollama":
            wanted = ["ollama"]
        else:
            wanted = []
        installed = self._installed_backends()
        out = []
        for backend in wanted:
            if backend == primary or backend in out:
                continue
            if installed is not None and backend not in installed:
                continue
            out.append(backend)
        return out

    def _job_model_id(self, job: DownloadJob) -> str:
        if job.source not in {"huggingface", "bitnet"} or not job.artifact:
            return job.model_ref
        stem = Path(job.artifact).stem
        return f"{job.model_ref}::{stem}"

    def _hf_download_dir(self, job: DownloadJob) -> Path:
        base = Path(settings.models_dir) / "huggingface"
        suffix = ""
        if job.artifact:
            suffix = f"--{Path(job.artifact).stem.replace('/', '_')}"
        return base / f"{job.model_ref.replace('/', '--')}{suffix}"

    def _find_downloaded_nemo_path(self, job: DownloadJob) -> Path | None:
        download_dir = self._hf_download_dir(job)
        if not download_dir.exists() or not download_dir.is_dir():
            return None

        if job.artifact and job.artifact.lower().endswith(".nemo"):
            candidate = download_dir / job.artifact
            if candidate.exists() and candidate.is_file():
                return candidate

        nemo_files = sorted(download_dir.rglob("*.nemo"))
        if not nemo_files:
            return None
        return nemo_files[0]

    async def clear_jobs(self, statuses: set[str] | None = None) -> int:
        """Delete jobs by status from Redis. Returns count deleted."""
        redis = get_redis_safe()
        keys = await redis.keys("download:job:*")
        deleted = 0
        for key in keys:
            payload = await get_key(key)
            if not payload:
                continue
            job_status = payload.get("status", "")
            if statuses is None or job_status in statuses:
                await redis.delete(key)
                deleted += 1
        return deleted

    async def _save_and_publish(self, job: DownloadJob) -> None:
        await set_key(f"download:job:{job.job_id}", job.model_dump(mode="json"))
        await self._publish_job_update(job)

    async def _publish_job_update(self, job: DownloadJob) -> None:
        payload = job.model_dump(mode="json")
        await publish(DOWNLOAD_PROGRESS_CHANNEL, payload)
        await publish(f"{DOWNLOAD_PROGRESS_CHANNEL}:{job.job_id}", payload)


def get_redis_safe():
    from ocabra.redis_client import get_redis

    return get_redis()


download_manager = DownloadManager()


async def _lifespan(_app) -> AsyncGenerator[None, None]:
    await download_manager.start(app=_app)
    yield
    await download_manager.stop()


router = APIRouter(tags=["downloads"], lifespan=_lifespan)


@router.get(
    "/downloads",
    response_model=list[DownloadJob],
    summary="List download jobs",
    description="Return all download jobs (active, queued, completed, failed) ordered by most recent first.",
)
async def list_downloads(
    # Lectura abierta para el indicador de descargas de la cabecera
    # (2026-11-09). Lanzar o cancelar sigue pidiendo model_manager.
    _user: UserContext = Depends(require_role("user")),
) -> list[DownloadJob]:
    return await download_manager.list_jobs()


@router.post(
    "/downloads",
    response_model=DownloadJob,
    summary="Enqueue a download",
    description=(
        "Queue a model download from HuggingFace, Ollama, or BitNet. "
        "The model is auto-registered on completion if register_config is provided."
    ),
    responses={400: {"description": "Invalid source or missing artifact for bitnet"}},
)
async def enqueue_download(
    request: DownloadCreateRequest,
    _user: UserContext = Depends(require_role("model_manager")),
) -> DownloadJob:
    try:
        return await download_manager.enqueue(
            source=request.source,
            model_ref=request.model_ref,
            artifact=request.artifact,
            register_config=request.register_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/downloads/{job_id}",
    response_model=DownloadJob,
    summary="Get download job",
    description="Return the current state of a single download job by its ID.",
    responses={404: {"description": "Download job not found"}},
)
async def get_download(
    job_id: str,
    _user: UserContext = Depends(require_role("model_manager")),
) -> DownloadJob:
    """Get a single download job by ID."""
    job = await download_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found")
    return job


@router.delete(
    "/downloads",
    summary="Clear download history",
    description=(
        "Delete historical download jobs by status (comma-separated). "
        "Active jobs (queued, downloading) are never deleted."
    ),
)
async def clear_downloads(
    status: str = "failed,cancelled,completed",
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, int]:
    """Delete historical download jobs by status (comma-separated). Active jobs are not affected."""
    requested = {s.strip() for s in status.split(",")}
    safe = requested - {"queued", "downloading"}  # never delete active jobs
    deleted = await download_manager.clear_jobs(statuses=safe)
    return {"deleted": deleted}


@router.delete(
    "/downloads/{job_id}",
    summary="Cancel a download",
    description="Cancel an active or queued download job. Already-completed jobs are left unchanged.",
    responses={404: {"description": "Download job not found"}},
)
async def cancel_download(
    job_id: str,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, bool]:
    try:
        await download_manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Download job not found") from exc
    return {"ok": True}


@router.get(
    "/downloads/{job_id}/stream",
    summary="Stream download progress",
    description="Subscribe to real-time Server-Sent Events for download progress updates.",
    responses={404: {"description": "Download job not found"}},
)
async def stream_download_progress(
    job_id: str,
    _user: UserContext = Depends(require_role("model_manager")),
) -> StreamingResponse:
    if await download_manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Download job not found")

    channel = f"download:progress:{job_id}"

    async def event_stream() -> AsyncGenerator[str, None]:
        async with subscribe(channel) as pubsub:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    continue
                data = message.get("data")
                if not isinstance(data, str):
                    continue
                yield f"data: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
