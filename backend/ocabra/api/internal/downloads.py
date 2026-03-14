import asyncio
import contextlib
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from ocabra.config import settings
from ocabra.redis_client import get_key, lpush, publish, rpop, set_key, subscribe
from ocabra.registry.huggingface import HuggingFaceRegistry
from ocabra.registry.ollama_registry import OllamaRegistry
from ocabra.schemas.registry import DownloadJob

QUEUE_NAME = "queue:download"


class DownloadCreateRequest(BaseModel):
    source: Literal["huggingface", "ollama"]
    model_ref: str
    artifact: str | None = None
    register_config: dict | None = None


class DownloadManager:
    def __init__(self) -> None:
        self._hf_registry = HuggingFaceRegistry()
        self._ollama_registry = OllamaRegistry()
        self._worker_task: asyncio.Task | None = None
        self._active_cancel_flags: dict[str, asyncio.Event] = {}
        self._app = None

    async def start(self, app=None) -> None:
        self._app = app
        if self._worker_task and not self._worker_task.done():
            return
        await self._cleanup_stale_jobs()

        self._worker_task = asyncio.create_task(self._worker_loop(), name="download-manager-worker")

    async def _cleanup_stale_jobs(self) -> None:
        """Mark jobs left in 'downloading' state as failed (server was killed mid-download)."""
        jobs = await self.list_jobs()
        for job in jobs:
            if job.status == "downloading":
                failed = job.model_copy(
                    update={"status": "failed", "error": "interrupted (server restart)", "completed_at": datetime.now(UTC)}
                )
                await self._save_and_publish(failed)

    async def stop(self) -> None:
        if not self._worker_task:
            return
        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task
        self._worker_task = None

    async def enqueue(
        self,
        source: str,
        model_ref: str,
        artifact: str | None = None,
        register_config: dict | None = None,
    ) -> DownloadJob:
        if source not in {"huggingface", "ollama"}:
            raise ValueError("source must be 'huggingface' or 'ollama'")

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
        )

        await set_key(f"download:job:{job.job_id}", job.model_dump(mode="json"))
        await lpush(QUEUE_NAME, {"job_id": job.job_id})
        await publish(f"download:progress:{job.job_id}", job.model_dump(mode="json"))
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
                    asyncio.create_task(self._save_and_publish(updated))
                except RuntimeError:
                    loop.call_soon_threadsafe(loop.create_task, self._save_and_publish(updated))

            target_dir = Path(settings.models_dir)
            if downloading.source == "huggingface":
                folder_suffix = ""
                if downloading.artifact:
                    stem = Path(downloading.artifact).stem.replace("/", "_")
                    folder_suffix = f"--{stem}"
                await self._hf_registry.download(
                    repo_id=downloading.model_ref,
                    target_dir=target_dir / "huggingface" / (downloading.model_ref.replace("/", "--") + folder_suffix),
                    progress_callback=_progress,
                    artifact=downloading.artifact,
                )
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
            await self._auto_register_model(downloading)
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
            model_id = self._job_model_id(job)
            existing = await mm.get_state(model_id)
            if existing:
                return
            if job.source == "ollama":
                backend_type = "ollama"
            else:
                backend_type = await self._hf_registry.infer_backend_for_repo(job.model_ref, artifact=job.artifact)
            await mm.add_model(
                model_id=model_id,
                backend_type=backend_type,
                display_name=(job.register_config or {}).get("display_name")
                or (model_id.split("/")[-1] if "/" in model_id else model_id),
                load_policy=(job.register_config or {}).get("load_policy", "on_demand"),
                auto_reload=bool((job.register_config or {}).get("auto_reload", False)),
                preferred_gpu=(job.register_config or {}).get("preferred_gpu"),
                extra_config=(job.register_config or {}).get("extra_config"),
            )
        except Exception:
            pass  # Non-fatal: model can be registered manually

    def _job_model_id(self, job: DownloadJob) -> str:
        if job.source != "huggingface" or not job.artifact:
            return job.model_ref
        stem = Path(job.artifact).stem
        return f"{job.model_ref}::{stem}"

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
        await publish(f"download:progress:{job.job_id}", job.model_dump(mode="json"))


def get_redis_safe():
    from ocabra.redis_client import get_redis

    return get_redis()


download_manager = DownloadManager()


async def _lifespan(_app) -> AsyncGenerator[None, None]:
    await download_manager.start(app=_app)
    yield
    await download_manager.stop()


router = APIRouter(tags=["downloads"], lifespan=_lifespan)


@router.get("/downloads", response_model=list[DownloadJob])
async def list_downloads() -> list[DownloadJob]:
    return await download_manager.list_jobs()


@router.post("/downloads", response_model=DownloadJob)
async def enqueue_download(request: DownloadCreateRequest) -> DownloadJob:
    try:
        return await download_manager.enqueue(
            source=request.source,
            model_ref=request.model_ref,
            artifact=request.artifact,
            register_config=request.register_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/downloads/{job_id}", response_model=DownloadJob)
async def get_download(job_id: str) -> DownloadJob:
    """Get a single download job by ID."""
    job = await download_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found")
    return job


@router.delete("/downloads")
async def clear_downloads(status: str = "failed,cancelled,completed") -> dict[str, int]:
    """Delete historical download jobs by status (comma-separated). Active jobs are not affected."""
    requested = {s.strip() for s in status.split(",")}
    safe = requested - {"queued", "downloading"}  # never delete active jobs
    deleted = await download_manager.clear_jobs(statuses=safe)
    return {"deleted": deleted}


@router.delete("/downloads/{job_id}")
async def cancel_download(job_id: str) -> dict[str, bool]:
    try:
        await download_manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Download job not found") from exc
    return {"ok": True}


@router.get("/downloads/{job_id}/stream")
async def stream_download_progress(job_id: str) -> StreamingResponse:
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
