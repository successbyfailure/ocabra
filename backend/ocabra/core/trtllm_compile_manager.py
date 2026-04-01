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

    def __init__(self, model_manager: Any = None) -> None:
        self._queue: asyncio.Queue[CompileJobState] = asyncio.Queue()
        self._active: CompileJobState | None = None
        self._history: dict[str, CompileJobState] = {}
        self._lock = asyncio.Lock()
        self._worker_task: asyncio.Task | None = None
        self._model_manager = model_manager

    def set_model_manager(self, model_manager: Any) -> None:
        """Inject model manager for auto-registration after compile."""
        self._model_manager = model_manager

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
            # AWQ models use same pipeline but convert script passes use_autoawq=True
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
            await self._register_engine(state)
            # Remove temporary checkpoint (not needed after engine is built)
            await self._cleanup_checkpoint(state.engine_name)

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

    # TRT-LLM 1.0 uses python3 -c for the convert step (no trtllm-convert binary).
    # The script detects architecture from config.json and uses the right model class.
    # Passed as a -c argument so no shell escaping is needed (subprocess list form).
    _CONVERT_SCRIPT = (
        "import argparse,json,sys;"
        "p=argparse.ArgumentParser();"
        "p.add_argument('--model_dir');"
        "p.add_argument('--output_dir');"
        "p.add_argument('--dtype',default='float16');"
        "p.add_argument('--tp_size',type=int,default=1);"
        "a=p.parse_args(sys.argv[1:]);"
        "cfg=json.load(open(a.model_dir+'/config.json'));"
        "arch=cfg.get('architectures',['?'])[0];"
        "print('[convert] arch='+arch,flush=True);"
        "from tensorrt_llm.models import MODEL_MAP;"
        "cls=MODEL_MAP.get(arch);"
        "cls or (print('[convert] ERROR: arch '+arch+' not supported by TRT-LLM 1.0',file=sys.stderr) or sys.exit(1));"
        "print('[convert] class='+cls.__name__,flush=True);"
        "from tensorrt_llm.mapping import Mapping;"
        "qcfg=cfg.get('quantization_config',{});"
        "qm=qcfg.get('quant_method','').lower();"
        "from tensorrt_llm.models.modeling_utils import QuantConfig;"
        "from tensorrt_llm.quantization import QuantAlgo;"
        "is_awq='awq' in qm\n"
        "is_gptq='gptq' in qm or is_awq\n"
        "quant=None\nextra_kwargs={}\n"
        "if is_awq:\n"
        "    print('[convert] AWQ detected — W4A16_GPTQ+use_autoawq',flush=True)\n"
        "    quant=QuantConfig(quant_algo=QuantAlgo.W4A16_GPTQ)\n"
        "    quant.group_size=qcfg.get('group_size',128)\n"
        "    quant.has_zero_point=bool(qcfg.get('zero_point',False))\n"
        "    extra_kwargs={'use_autoawq':True}\n"
        "elif is_gptq:\n"
        "    print('[convert] GPTQ detected',flush=True)\n"
        "    quant=QuantConfig(quant_algo=QuantAlgo.W4A16_GPTQ)\n"
        "    quant.group_size=qcfg.get('group_size',128)\n"
        "import os\nos.makedirs(a.output_dir,exist_ok=True)\n"
        "for rank in range(a.tp_size):\n"
        "    print(f'[convert] rank {rank}/{a.tp_size} loading model on cpu ...',flush=True)\n"
        "    m=cls.from_hugging_face(a.model_dir,dtype=a.dtype,"
        "mapping=Mapping(world_size=a.tp_size,tp_size=a.tp_size,rank=rank),"
        "quant_config=quant,load_model_on_cpu=True,**extra_kwargs)\n"
        "    print(f'[convert] rank {rank}/{a.tp_size} saving checkpoint ...',flush=True)\n"
        "    m.save_checkpoint(a.output_dir)\n"
        "    del m\n"
        "print('[convert] done all ranks',flush=True)"
    )

    # dtype names differ: TRT-LLM uses 'float16', 'bfloat16'; user picks 'fp16', 'bf16'
    _DTYPE_MAP = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "int8": "int8",
        "fp8": "fp8",
    }

    def _build_docker_cmd(self, state: CompileJobState, phase: str) -> list[str]:
        docker_bin = settings.tensorrt_llm_docker_bin or "docker"
        image = settings.tensorrt_llm_docker_image or "nvcr.io/nvidia/tensorrt-llm/release:latest"
        models_host = settings.tensorrt_llm_docker_models_mount_host or "/docker/ai-models/ocabra/models"
        models_container = settings.tensorrt_llm_docker_models_mount_container or "/data/models"

        tp_size = len(state.gpu_indices)
        dtype_trtllm = self._DTYPE_MAP.get(state.dtype, state.dtype)
        # Docker requires quoted value "device=0,1" for multiple GPU IDs;
        # single GPU can use bare device=N format.
        gpu_ids = ",".join(str(i) for i in state.gpu_indices)
        gpu_spec = f'"device={gpu_ids}"' if tp_size > 1 else f"device={gpu_ids}"

        # Source model path inside container
        # model_id format: "vllm/Org/Model" or "Org/Model"
        raw_model = state.source_model
        if "/" in raw_model:
            parts = raw_model.split("/", 1)
            if parts[0] in {"vllm", "tensorrt_llm", "llama_cpp", "sglang"}:
                raw_model = parts[1]
        hf_dir_name = raw_model.replace("/", "--")
        model_dir = f"{models_container}/huggingface/{hf_dir_name}"

        ckpt_dir = f"{models_container}/tensorrt_llm/{state.engine_name}/tllm_ckpt"
        engine_dir = f"{models_container}/tensorrt_llm/{state.engine_name}/engine"

        # Both phases need GPU access (TRT-LLM python libs require CUDA even for convert)
        base_cmd = [
            docker_bin, "run", "--rm",
            "--gpus", gpu_spec,
            "-v", f"{models_host}:{models_container}",
            "--ipc", "host",
        ]

        cfg = state.config
        build_common = [
            "--max_batch_size", str(cfg.get("max_batch_size", 1)),
            "--max_input_len", str(cfg.get("max_input_len", 2048)),
            "--max_seq_len", str(cfg.get("max_seq_len", 4096)),
        ]

        if phase == "convert":
            # TRT-LLM 1.0: no trtllm-convert binary — use Python API via python3 -c
            return base_cmd + [
                image,
                "python3", "-c", self._CONVERT_SCRIPT,
                "--model_dir", model_dir,
                "--output_dir", ckpt_dir,
                "--dtype", dtype_trtllm,
                "--tp_size", str(tp_size),
            ]
        elif phase == "build":
            # For TP>1, use --workers N so trtllm-build launches one worker per rank
            workers_args = ["--workers", str(tp_size)] if tp_size > 1 else []
            return base_cmd + [
                image,
                "trtllm-build",
                "--checkpoint_dir", ckpt_dir,
                "--output_dir", engine_dir,
                *workers_args,
                *build_common,
            ]
        else:
            raise ValueError(f"Unknown build phase: {phase}")

    def _is_awq_model(self, state: CompileJobState) -> bool:
        """Check if the source model is AWQ-quantized by reading its config.json."""
        import json as _json
        from pathlib import Path as _Path
        # Use container-side path (where the API runs)
        models_dir = settings.models_dir or (
            settings.tensorrt_llm_engines_dir.rsplit("/tensorrt_llm", 1)[0]
            if settings.tensorrt_llm_engines_dir and "/tensorrt_llm" in settings.tensorrt_llm_engines_dir
            else "/data/models"
        )
        raw = state.source_model
        parts = raw.split("/", 1)
        if parts[0] in {"vllm", "tensorrt_llm", "llama_cpp", "sglang"}:
            raw = parts[1]
        hf_dir_name = raw.replace("/", "--")
        cfg_path = _Path(models_dir) / "huggingface" / hf_dir_name / "config.json"
        try:
            cfg = _json.loads(cfg_path.read_text())
            qcfg = cfg.get("quantization_config") or {}
            qm = str(qcfg.get("quant_type") or qcfg.get("quant_method") or "").lower()
            return "awq" in qm
        except Exception:
            return False

    def _resolve_hf_tokenizer_path(self, source_model: str) -> str | None:
        """Derive the HuggingFace model directory from source_model (used as tokenizer)."""
        raw = source_model
        parts = raw.split("/", 1)
        if parts[0] in {"vllm", "tensorrt_llm", "llama_cpp", "sglang"}:
            raw = parts[1]
        hf_dir_name = raw.replace("/", "--")
        models_container = settings.tensorrt_llm_engines_dir.rsplit("/tensorrt_llm", 1)[0] \
            if settings.tensorrt_llm_engines_dir and "/tensorrt_llm" in settings.tensorrt_llm_engines_dir \
            else (settings.models_dir or "/data/models")
        candidate = f"{models_container}/huggingface/{hf_dir_name}"
        from pathlib import Path
        if Path(candidate).exists():
            return candidate
        return None

    def _engine_dir(self, engine_name: str) -> str:
        """Return the host-side engine path (for Docker mounts and DB storage)."""
        models_host = settings.tensorrt_llm_docker_models_mount_host or "/docker/ai-models/ocabra/models"
        return f"{models_host}/tensorrt_llm/{engine_name}/engine"

    def _engine_dir_container(self, engine_name: str) -> str:
        """Return the container-side engine path (used by the backend at runtime)."""
        if settings.tensorrt_llm_engines_dir:
            return f"{settings.tensorrt_llm_engines_dir}/{engine_name}/engine"
        models_container = settings.tensorrt_llm_docker_models_mount_container or "/data/models"
        return f"{models_container}/tensorrt_llm/{engine_name}/engine"

    async def _cleanup_checkpoint(self, engine_name: str) -> None:
        """Delete the TRT-LLM checkpoint dir after a successful build to free disk space."""
        import shutil
        from pathlib import Path
        models_host = settings.tensorrt_llm_docker_models_mount_host or "/docker/ai-models/ocabra/models"
        ckpt_path = Path(models_host) / "tensorrt_llm" / engine_name / "tllm_ckpt"
        if ckpt_path.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, ckpt_path)
                logger.info("trtllm_checkpoint_cleaned", path=str(ckpt_path))
            except Exception as exc:
                logger.warning("trtllm_checkpoint_cleanup_failed", path=str(ckpt_path), error=str(exc))

    async def _register_engine(self, state: CompileJobState) -> None:
        """Register a successfully compiled engine as a tensorrt_llm model."""
        if self._model_manager is None:
            logger.warning("trtllm_no_model_manager_skipping_registration", engine=state.engine_name)
            return
        model_id = f"tensorrt_llm/{state.engine_name}"
        extra_config = {
            "engine_dir": self._engine_dir_container(state.engine_name),
            "launch_mode": "docker",
            "max_batch_size": state.config.get("max_batch_size", 1),
            "context_length": state.config.get("max_seq_len", 4096),
            "tokenizer_path": self._resolve_hf_tokenizer_path(state.source_model),
        }
        try:
            await self._model_manager.add_model(
                model_id=model_id,
                backend_type="tensorrt_llm",
                display_name=state.engine_name,
                preferred_gpu=state.gpu_indices[0] if state.gpu_indices else None,
                extra_config=extra_config,
            )
            logger.info("trtllm_engine_registered", model_id=model_id)
        except Exception as exc:
            logger.warning("trtllm_engine_registration_failed", model_id=model_id, error=str(exc))

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
