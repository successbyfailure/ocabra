"""Cola de trabajos de ``/v1/videos``.

La generación de vídeo dura de 45 s a varios minutos y ocupa una GPU de 24 GB
entera, así que la API es asíncrona como la de OpenAI: ``POST`` encola y
contesta al momento, el cliente consulta el estado y descarga el MP4.

* Un trabajo a la vez en todo el nodo. Dos generaciones a la vez no caben en
  una tarjeta y alternarlas solo añade cargas.
* El prompt no se guarda en la base de datos: vive en memoria mientras el
  trabajo espera o se genera. Si el api se reinicia, los trabajos pendientes
  se marcan como fallidos (``server_restarted``): no hay con qué rehacerlos.
* El MP4 se guarda en ``<openai_files_dir>/videos`` y se borra al caducar
  (``video_retention_hours``, por defecto 24 h, como los vídeos de OpenAI).
* Mientras genera, el trabajo cuenta como petición en curso del worker, de
  modo que el árbitro de VRAM no lo expulsa a mitad.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import select, update

from ocabra.config import settings
from ocabra.db.videos import VideoJob

logger = structlog.get_logger(__name__)

_PROGRESS_POLL_S = 2.0
_CLEANUP_INTERVAL_S = 600.0
# Pasos del ancho y el alto: el workflow genera a media resolución y escala
# x2, y LTX trabaja en múltiplos de 32 → la salida va en múltiplos de 64.
SIZE_STEP = 64


def videos_dir() -> Path:
    return Path(settings.openai_files_dir) / "videos"


def normalize_size(width: int, height: int) -> tuple[int, int]:
    """Redondea a múltiplos de 64 (mínimo 256), conservando la proporción."""

    def snap(value: int) -> int:
        return max(256, int(round(value / SIZE_STEP)) * SIZE_STEP)

    return snap(width), snap(height)


@dataclass
class PendingVideo:
    job_id: uuid.UUID
    model: str
    params: dict[str, Any]
    user: Any
    request: Any  # Request original: solo para la fila de estadísticas.
    submitted_at: float = field(default_factory=time.monotonic)
    submitted_wall: datetime = field(default_factory=lambda: datetime.now(UTC))


class VideoJobManager:
    def __init__(self, app: Any, session_factory: Any) -> None:
        self._app = app
        self._session_factory = session_factory
        self._queue: asyncio.Queue[PendingVideo] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._current: uuid.UUID | None = None

    # ── Ciclo de vida ───────────────────────────────────────────────────
    async def start(self) -> None:
        # Ni el disco ni la BD pueden impedir que arranque el api: sin ellos
        # /v1/videos fallará al encolar, pero el resto sigue funcionando.
        try:
            videos_dir().mkdir(parents=True, exist_ok=True)
            await self._fail_orphans()
        except Exception:  # noqa: BLE001
            logger.warning("video_jobs_start_degraded", exc_info=True)
        self._tasks = [
            asyncio.create_task(self._runner(), name="video_jobs_runner"),
            asyncio.create_task(self._cleanup_loop(), name="video_jobs_cleanup"),
        ]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks = []

    async def _fail_orphans(self) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(VideoJob)
                .where(VideoJob.status.in_(("queued", "in_progress")))
                .values(
                    status="failed",
                    completed_at=datetime.now(UTC),
                    error={
                        "code": "server_restarted",
                        "message": "The server restarted before the video was generated; submit it again.",
                    },
                )
            )
            await session.commit()

    # ── API ─────────────────────────────────────────────────────────────
    def queue_position(self, job_id: uuid.UUID) -> int | None:
        if self._current == job_id:
            return 0
        for index, pending in enumerate(list(self._queue._queue)):  # noqa: SLF001
            if pending.job_id == job_id:
                return index + 1
        return None

    async def submit(self, pending: PendingVideo) -> None:
        await self._queue.put(pending)

    # ── Ejecución ───────────────────────────────────────────────────────
    async def _set(self, job_id: uuid.UUID, **values: Any) -> None:
        async with self._session_factory() as session:
            await session.execute(update(VideoJob).where(VideoJob.id == job_id).values(**values))
            await session.commit()

    async def _runner(self) -> None:
        while True:
            pending = await self._queue.get()
            self._current = pending.job_id
            try:
                await self._run(pending)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — un trabajo roto no para la cola
                logger.exception("video_job_crashed", job_id=str(pending.job_id))
            finally:
                self._current = None
                self._queue.task_done()

    async def _run(self, pending: PendingVideo) -> None:
        from fastapi import HTTPException

        from ocabra.api.openai._deps import resolve_profile
        from ocabra.stats.collector import record_direct_request

        app = self._app
        model_manager = app.state.model_manager
        worker_pool = app.state.worker_pool
        job_id = pending.job_id
        started_at = datetime.now(UTC)
        started = time.monotonic()
        # Tiempos como los de cualquier petición: la cola es lo que esperó
        # detrás de otros vídeos, la carga lo que tardó en tener el modelo
        # listo (incluida la expulsión de lo que ocupaba la GPU) y la
        # generación lo que tardó el worker. La duración total cuenta desde el
        # POST, como la del resto de peticiones.
        queue_ms = int((started - pending.submitted_at) * 1000)
        load_ms: int | None = None
        generation_ms: int | None = None
        serving_gpus: list[int] = []
        await self._set(job_id, status="in_progress", started_at=started_at, progress=1)

        status_code, error_message = 200, None
        worker_key: str | None = None
        work: dict[str, Any] = {
            "output_w": pending.params["width"],
            "output_h": pending.params["height"],
            "seconds": pending.params["seconds"],
            "mode": pending.params.get("mode", "t2v"),
            "input_frames": pending.params["seconds"] * pending.params.get("fps", 24) + 1,
        }
        inflight_id = None
        try:
            # Carga (y expulsa lo necesario) igual que cualquier petición /v1.
            load_started = time.monotonic()
            profile, state = await resolve_profile(
                pending.model,
                model_manager,
                app.state.profile_registry,
                user=pending.user,
                router_resolver=getattr(app.state, "router_resolver", None),
                required_capability="video_generation",
            )
            if not getattr(state.capabilities, "video_generation", False):
                raise _JobError(
                    "model_not_capable", f"'{pending.model}' does not generate video", 400
                )
            load_ms = int((time.monotonic() - load_started) * 1000)
            worker_key = state.model_id
            serving_gpus = list(getattr(state, "current_gpu", None) or [])
            backend = await worker_pool.get_backend(state.backend_type)
            inflight_id = model_manager.begin_request(worker_key, user=pending.user)
            progress_task = asyncio.create_task(
                self._follow_progress(job_id, backend, state.backend_model_id)
            )
            generation_started = time.monotonic()
            try:
                body, stats = await backend.generate_video(
                    state.backend_model_id, {**pending.params, "job_id": job_id.hex}
                )
                generation_ms = int((time.monotonic() - generation_started) * 1000)
            finally:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            path = videos_dir() / f"{job_id.hex}.mp4"
            await asyncio.to_thread(path.write_bytes, body)
            clean = {k.removeprefix("x-ocabra-"): v for k, v in stats.items()}
            seed = _int_or_none(clean.get("seed"))
            now = datetime.now(UTC)
            await self._set(
                job_id,
                status="completed",
                progress=100,
                completed_at=now,
                expires_at=now + timedelta(hours=max(1, int(settings.video_retention_hours))),
                storage_path=str(path),
                bytes=len(body),
                seed=seed,
                stats=clean,
            )
            work.update(
                {
                    "output_bytes": len(body),
                    "generation_s": clean.get("generation-s"),
                    # Velocidad: fotogramas generados por segundo de generación.
                    "generation_fps": round(work["input_frames"] / (generation_ms / 1000), 3)
                    if generation_ms
                    else None,
                }
            )
            logger.info("video_job_completed", job_id=str(job_id), model=pending.model, **clean)
        except _JobError as exc:
            status_code, error_message = exc.status, exc.message
            await self._fail(job_id, exc.code, exc.message)
        except HTTPException as exc:
            detail = exc.detail
            message = detail if isinstance(detail, str) else str(detail)
            if isinstance(detail, dict):
                err = detail.get("error") if isinstance(detail.get("error"), dict) else {}
                message = str(err.get("message") or detail)
            status_code, error_message = exc.status_code, message
            await self._fail(job_id, _code_for(exc.status_code), message)
        except Exception as exc:  # noqa: BLE001
            import httpx

            status = 500
            message = str(exc) or type(exc).__name__
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                message = _worker_detail(exc.response) or message
            status_code, error_message = status, message
            await self._fail(job_id, _code_for(status), message)
            logger.warning("video_job_failed", job_id=str(job_id), error=message)
        finally:
            if inflight_id is not None and worker_key is not None:
                model_manager.end_request(worker_key, inflight_id)
            if pending.request is not None:
                with contextlib.suppress(Exception):
                    from ocabra.core.request_telemetry import RequestRuntimeTelemetry

                    pending.request.state.runtime_telemetry = RequestRuntimeTelemetry(
                        started_monotonic=pending.submitted_at,
                        queue_ms=queue_ms,
                        load_ms=load_ms,
                        gpu_indices=serving_gpus or None,
                    )
                    await record_direct_request(
                        pending.request,
                        model_id=pending.model,
                        started_at=pending.submitted_wall,
                        duration_ms=(time.monotonic() - pending.submitted_at) * 1000,
                        generation_ms=generation_ms,
                        endpoint_path="/v1/videos",
                        request_kind="video_generation",
                        status_code=status_code,
                        error_message=error_message,
                        work_size_meta=work,
                    )

    async def _fail(self, job_id: uuid.UUID, code: str, message: str) -> None:
        await self._set(
            job_id,
            status="failed",
            completed_at=datetime.now(UTC),
            error={"code": code, "message": message[:2000]},
        )

    async def _follow_progress(
        self, job_id: uuid.UUID, backend: Any, backend_model_id: str
    ) -> None:
        last = -1
        while True:
            await asyncio.sleep(_PROGRESS_POLL_S)
            info = await backend.get_progress(backend_model_id)
            if not info or info.get("job_id") != job_id.hex:
                continue
            value = int(min(99, max(1, float(info.get("percent") or 0))))
            if value != last:
                last = value
                await self._set(job_id, progress=value)

    # ── Caducidad ───────────────────────────────────────────────────────
    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await self.purge_expired()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("video_jobs_cleanup_failed", exc_info=True)
            await asyncio.sleep(_CLEANUP_INTERVAL_S)

    async def purge_expired(self) -> int:
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(VideoJob).where(
                            VideoJob.expires_at.is_not(None), VideoJob.expires_at < now
                        )
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                if row.storage_path:
                    await asyncio.to_thread(Path(row.storage_path).unlink, missing_ok=True)
                await session.delete(row)
            await session.commit()
        return len(rows)


class _JobError(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _code_for(status: int) -> str:
    return {
        400: "invalid_request",
        404: "model_not_found",
        503: "model_unavailable",
        507: "insufficient_vram",
    }.get(status, "generation_failed")


def _worker_detail(response: Any) -> str | None:
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        return response.text[:500] if getattr(response, "text", None) else None
    detail = data.get("detail") if isinstance(data, dict) else None
    return str(detail) if detail else None
