"""TensorRT-LLM engine compilation manager.

Manages the lifecycle of compile jobs: queuing, Docker execution (convert + build
phases), progress publishing via Redis, and DB persistence.

One job runs at a time — compilation locks the target GPU(s).
"""
from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.trtllm import TrtllmCompileJob
from ocabra.redis_client import publish

logger = structlog.get_logger(__name__)

# Redis channel prefix for compile progress events
_CHANNEL_PREFIX = "trtllm:compile:"


@dataclass
class CompileRequest:
    """Input parameters for a compile job."""

    source_model: str
    engine_name: str
    gpu_indices: list[int]
    dtype: str
    max_batch_size: int
    max_input_len: int
    max_seq_len: int


@dataclass
class CompileJobState:
    """In-memory state mirroring the DB row while the job is active."""

    job_id: str
    source_model: str
    engine_name: str
    gpu_indices: list[int]
    dtype: str
    config: dict[str, Any]
    status: str = "pending"
    phase: str | None = None
    progress_pct: int = 0
    error_detail: str | None = None
    engine_dir: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_model": self.source_model,
            "engine_name": self.engine_name,
            "gpu_indices": self.gpu_indices,
            "dtype": self.dtype,
            "config": self.config,
            "status": self.status,
            "phase": self.phase,
            "progress_pct": self.progress_pct,
            "error_detail": self.error_detail,
            "engine_dir": self.engine_dir,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class TrtllmCompileManager:
    """Manages TensorRT-LLM engine compilation jobs.

    Usage::

        manager = TrtllmCompileManager()
        await manager.start()
        job = await manager.enqueue(request)
        # subscribe to redis channel trtllm:compile:{job_id} for progress
        await manager.stop()
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[CompileJobState] = asyncio.Queue()
        self._active: CompileJobState | None = None
        self._history: dict[str, CompileJobState] = {}
        self._lock = asyncio.Lock()
        self._worker_task: asyncio.Task | None = None

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background worker loop."""
        await self._recover_stale_jobs()
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="trtllm-compile-worker"
        )

    async def stop(self) -> None:
        """Cancel the background worker loop gracefully."""
        if not self._worker_task:
            return
        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task
        self._worker_task = None

    # ── Public API ───────────────────────────────────────────────

    async def enqueue(self, req: CompileRequest) -> CompileJobState:
        """Create a new compile job and add it to the queue.

        Returns:
            CompileJobState with status='pending' and the assigned job_id.
        """
        job_id = uuid.uuid4().hex
        config = {
            "max_batch_size": req.max_batch_size,
            "max_input_len": req.max_input_len,
            "max_seq_len": req.max_seq_len,
        }
        state = CompileJobState(
            job_id=job_id,
            source_model=req.source_model,
            engine_name=req.engine_name,
            gpu_indices=req.gpu_indices,
            dtype=req.dtype,
            config=config,
        )

        await self._save_to_db(state)
        self._history[job_id] = state
        await self._queue.put(state)

        logger.info(
            "trtllm_compile_enqueued",
            job_id=job_id,
            source_model=req.source_model,
            engine_name=req.engine_name,
            gpu_indices=req.gpu_indices,
        )
        return state

    async def cancel(self, job_id: str) -> CompileJobState:
        """Cancel a pending or running job.

        Returns:
            Updated CompileJobState.
        Raises:
            KeyError: if job_id not found.
            ValueError: if job is already finished.
        """
        state = self._history.get(job_id)
        if state is None:
            state = await self._load_from_db(job_id)
            if state is None:
                raise KeyError(f"Compile job '{job_id}' not found")
            self._history[job_id] = state

        if state.status in {"done", "failed", "cancelled"}:
            raise ValueError(f"Job '{job_id}' is already in terminal state '{state.status}'")

        state._cancel_event.set()
        if state.status == "pending":
            state.status = "cancelled"
            state.finished_at = datetime.now(UTC)
            await self._save_to_db(state)
            await self._publish_progress(state)
        # If running, the worker loop will detect the cancel event

        return state

    def get_job(self, job_id: str) -> CompileJobState | None:
        """Return in-memory job state, or None if not cached."""
        return self._history.get(job_id)

    async def list_jobs(self) -> list[CompileJobState]:
        """Return all jobs from the DB, most recent first."""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select, desc
            result = await session.execute(
                select(TrtllmCompileJob).order_by(desc(TrtllmCompileJob.started_at))
            )
            rows = result.scalars().all()
        return [self._row_to_state(row) for row in rows]

    # ── Worker loop ──────────────────────────────────────────────

    async def _worker_loop(self) -> None:
        logger.info("trtllm_compile_worker_started")
        while True:
            try:
                state = await self._queue.get()
            except asyncio.CancelledError:
                break

            # Skip jobs that were cancelled while queued
            if state.status == "cancelled" or state._cancel_event.is_set():
                self._queue.task_done()
                continue

            async with self._lock:
                self._active = state
                try:
                    await self._run_job(state)
                except asyncio.CancelledError:
                    state.status = "cancelled"
                    state.finished_at = datetime.now(UTC)
                    await self._save_to_db(state)
                    await self._publish_progress(state)
                    raise
                except Exception as exc:
                    logger.exception("trtllm_compile_unexpected_error", job_id=state.job_id, error=str(exc))
                    state.status = "failed"
                    state.error_detail = f"Unexpected error: {exc}"
                    state.finished_at = datetime.now(UTC)
                    await self._save_to_db(state)
                    await self._publish_progress(state)
                finally:
                    self._active = None
                    self._queue.task_done()

        logger.info("trtllm_compile_worker_stopped")

    async def _run_job(self, state: CompileJobState) -> None:
        """Execute the full convert → build pipeline for a job."""
        state.status = "running"
        state.started_at = datetime.now(UTC)
        await self._save_to_db(state)
        await self._publish_progress(state)

        try:
            # ── Phase 1: convert ────────────────────────────────
            await self._run_phase(state, "convert")
            if state._cancel_event.is_set():
                state.status = "cancelled"
                state.finished_at = datetime.now(UTC)
                await self._save_to_db(state)
                await self._publish_progress(state)
                return

            # ── Phase 2: build ──────────────────────────────────
            await self._run_phase(state, "build")
            if state._cancel_event.is_set():
                state.status = "cancelled"
                state.finished_at = datetime.now(UTC)
                await self._save_to_db(state)
                await self._publish_progress(state)
                return

            # ── Success ─────────────────────────────────────────
            engine_dir = self._engine_dir(state.engine_name)
            state.status = "done"
            state.phase = None
            state.progress_pct = 100
            state.engine_dir = engine_dir
            state.finished_at = datetime.now(UTC)
            await self._save_to_db(state)
            await self._publish_progress(state)
            logger.info(
                "trtllm_compile_done",
                job_id=state.job_id,
                engine_dir=engine_dir,
            )

        except Exception as exc:
            state.status = "failed"
            state.error_detail = str(exc)
            state.finished_at = datetime.now(UTC)
            await self._save_to_db(state)
            await self._publish_progress(state)
            logger.error(
                "trtllm_compile_failed",
                job_id=state.job_id,
                error=str(exc),
            )

    async def _run_phase(self, state: CompileJobState, phase: str) -> None:
        """Run a single Docker phase (convert or build) and stream its output."""
        state.phase = phase
        state.progress_pct = 5 if phase == "convert" else 50
        await self._save_to_db(state)
        await self._publish_progress(state)

        cmd = self._build_docker_cmd(state, phase)
        logger.info(
            "trtllm_compile_phase_start",
            job_id=state.job_id,
            phase=phase,
            cmd=" ".join(cmd),
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        # Stream output lines, publish to Redis, watch for cancel
        lines_buf: list[str] = []
        assert proc.stdout is not None
        try:
            async for raw_line in proc.stdout:
                if state._cancel_event.is_set():
                    proc.kill()
                    await proc.wait()
                    return

                line = raw_line.decode("utf-8", errors="replace").rstrip()
                lines_buf.append(line)
                if len(lines_buf) > 200:
                    lines_buf = lines_buf[-200:]

                await self._publish_log_line(state, phase, line)

        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise

        rc = await proc.wait()
        if rc != 0:
            tail = "\n".join(lines_buf[-40:])
            raise RuntimeError(
                f"Docker {phase} exited with code {rc}.\nOutput tail:\n{tail}"
            )

        state.progress_pct = 45 if phase == "convert" else 95
        await self._save_to_db(state)
        await self._publish_progress(state)

    # ── Docker command builders ──────────────────────────────────

    def _build_docker_cmd(self, state: CompileJobState, phase: str) -> list[str]:
        docker_bin = settings.tensorrt_llm_docker_bin or "docker"
        image = settings.tensorrt_llm_docker_image or "nvcr.io/nvidia/tensorrt-llm/release:latest"
        models_host = settings.tensorrt_llm_docker_models_mount_host or "/docker/ai-models/ocabra/models"
        models_container = settings.tensorrt_llm_docker_models_mount_container or "/data/models"

        gpu_spec = ",".join(str(i) for i in state.gpu_indices)
        tp_size = len(state.gpu_indices)

        # Source model path inside container
        # model_id format: "vllm/Org/Model" or "Org/Model"
        raw_model = state.source_model
        # Strip backend prefix like "vllm/" or "tensorrt_llm/"
        if "/" in raw_model:
            parts = raw_model.split("/", 1)
            if parts[0] in {"vllm", "tensorrt_llm", "llama_cpp", "sglang"}:
                raw_model = parts[1]
        hf_dir_name = raw_model.replace("/", "--")
        model_dir = f"{models_container}/huggingface/{hf_dir_name}"

        ckpt_dir = f"{models_container}/tensorrt_llm/{state.engine_name}/tllm_ckpt"
        engine_dir = f"{models_container}/tensorrt_llm/{state.engine_name}/engine"

        base_cmd = [
            docker_bin, "run", "--rm",
            "--gpus", f"device={gpu_spec}",
            "-v", f"{models_host}:{models_container}",
        ]

        if phase == "convert":
            return base_cmd + [
                image,
                "trtllm-convert",
                "--model_dir", model_dir,
                "--output_dir", ckpt_dir,
                "--dtype", state.dtype,
                "--tp_size", str(tp_size),
            ]
        else:  # build
            cfg = state.config
            return base_cmd + [
                image,
                "trtllm-build",
                "--checkpoint_dir", ckpt_dir,
                "--output_dir", engine_dir,
                "--max_batch_size", str(cfg.get("max_batch_size", 1)),
                "--max_input_len", str(cfg.get("max_input_len", 2048)),
                "--max_seq_len", str(cfg.get("max_seq_len", 4096)),
                "--tp_size", str(tp_size),
            ]

    def _engine_dir(self, engine_name: str) -> str:
        models_host = settings.tensorrt_llm_docker_models_mount_host or "/docker/ai-models/ocabra/models"
        return f"{models_host}/tensorrt_llm/{engine_name}/engine"

    # ── Redis helpers ────────────────────────────────────────────

    async def _publish_progress(self, state: CompileJobState) -> None:
        channel = f"{_CHANNEL_PREFIX}{state.job_id}"
        payload = {
            "type": "progress",
            **state.to_dict(),
        }
        with contextlib.suppress(Exception):
            await publish(channel, payload)

    async def _publish_log_line(self, state: CompileJobState, phase: str, line: str) -> None:
        channel = f"{_CHANNEL_PREFIX}{state.job_id}"
        payload = {"type": "log", "phase": phase, "line": line}
        with contextlib.suppress(Exception):
            await publish(channel, payload)

    # ── DB helpers ───────────────────────────────────────────────

    async def _save_to_db(self, state: CompileJobState) -> None:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(TrtllmCompileJob).where(TrtllmCompileJob.id == uuid.UUID(state.job_id))
            )
            row = result.scalar_one_or_none()
            if row is None:
                row = TrtllmCompileJob(id=uuid.UUID(state.job_id))
                session.add(row)
            row.source_model = state.source_model
            row.engine_name = state.engine_name
            row.gpu_indices = state.gpu_indices
            row.dtype = state.dtype
            row.config = state.config
            row.status = state.status
            row.phase = state.phase
            row.progress_pct = state.progress_pct
            row.error_detail = state.error_detail
            row.engine_dir = state.engine_dir
            row.started_at = state.started_at
            row.finished_at = state.finished_at
            await session.commit()

    async def _load_from_db(self, job_id: str) -> CompileJobState | None:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(TrtllmCompileJob).where(TrtllmCompileJob.id == uuid.UUID(job_id))
            )
            row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._row_to_state(row)

    def _row_to_state(self, row: TrtllmCompileJob) -> CompileJobState:
        return CompileJobState(
            job_id=str(row.id).replace("-", ""),
            source_model=row.source_model,
            engine_name=row.engine_name,
            gpu_indices=list(row.gpu_indices),
            dtype=row.dtype,
            config=dict(row.config),
            status=row.status,
            phase=row.phase,
            progress_pct=row.progress_pct or 0,
            error_detail=row.error_detail,
            engine_dir=row.engine_dir,
            started_at=row.started_at,
            finished_at=row.finished_at,
        )

    async def _recover_stale_jobs(self) -> None:
        """Mark jobs left in 'running' state as failed on startup."""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(TrtllmCompileJob).where(TrtllmCompileJob.status == "running")
            )
            rows = result.scalars().all()
            for row in rows:
                row.status = "failed"
                row.error_detail = "interrupted (server restart)"
                row.finished_at = datetime.now(UTC)
            if rows:
                await session.commit()
                logger.warning(
                    "trtllm_compile_stale_jobs_recovered", count=len(rows)
                )
