"""Monitors active workers and restarts those that die.

Does NOT replace launch logic in each backend.
Limited to:
- Periodic health polling of each registered worker
- Dead process detection (pid check + HTTP health)
- Transition model to ERROR if worker dies
- Optional auto-restart with exponential backoff
"""

from __future__ import annotations

import asyncio
import os

import httpx
import structlog

logger = structlog.get_logger(__name__)


class BackendProcessManager:
    """Monitors active workers and restarts those that die."""

    def __init__(self, model_manager, worker_pool, settings) -> None:
        self._model_manager = model_manager
        self._worker_pool = worker_pool
        self._settings = settings
        self._restart_counts: dict[str, int] = {}  # model_id -> consecutive restarts
        self._health_fail_counts: dict[str, int] = {}  # model_id -> consecutive failures
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Launch the health check loop as an asyncio.Task."""
        self._task = asyncio.create_task(self._health_loop(), name="backend-process-health")

    async def stop(self) -> None:
        """Cancel the loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _health_loop(self) -> None:
        """Periodically check every registered worker.

        For each (model_id, worker_info) in worker_pool._workers:
          1. Check if PID is alive via os.kill(pid, 0)
          2. If dead PID -> _handle_worker_death(model_id)
          3. If alive -> HTTP GET http://127.0.0.1:{port}/health with timeout 5s
          4. If health fails -> increment consecutive failure counter
          5. If 3 consecutive failures -> _handle_worker_death(model_id)
          6. If health OK -> reset failure counter and restart counter
        """
        interval = max(
            1,
            int(getattr(self._settings, "worker_health_check_interval_seconds", 10)),
        )
        while True:
            try:
                await asyncio.sleep(interval)
                # Snapshot workers to avoid dict-changed-during-iteration
                workers = dict(self._worker_pool._workers)
                for model_id, worker_info in workers.items():
                    # Skip ollama workers (external process, not managed)
                    if worker_info.backend_type == "ollama":
                        continue
                    # Skip pid=0 (placeholder workers)
                    if worker_info.pid <= 0:
                        continue
                    try:
                        await self._check_worker(model_id, worker_info)
                    except Exception as exc:
                        logger.warning(
                            "worker_health_check_error",
                            model_id=model_id,
                            error=str(exc),
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("health_loop_error", error=str(exc))

    async def _check_worker(self, model_id: str, worker_info) -> None:
        """Check a single worker: PID alive then HTTP health."""
        pid = worker_info.pid
        port = worker_info.port

        # Step 1: Check if PID is alive
        if not self._is_pid_alive(pid):
            logger.error(
                "worker_pid_dead",
                model_id=model_id,
                pid=pid,
            )
            await self._handle_worker_death(model_id, reason="pid_dead")
            return

        # Step 2: HTTP health check
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/health")
                resp.raise_for_status()
            # Health OK -- reset counters
            self._health_fail_counts.pop(model_id, None)
            self._restart_counts.pop(model_id, None)
        except Exception:
            count = self._health_fail_counts.get(model_id, 0) + 1
            self._health_fail_counts[model_id] = count
            if count >= 3:
                logger.error(
                    "worker_health_failed",
                    model_id=model_id,
                    pid=pid,
                    consecutive_failures=count,
                )
                await self._handle_worker_death(model_id, reason="health_check_failed")

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """Return True if a process with the given PID exists."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    async def _handle_worker_death(self, model_id: str, reason: str = "unknown") -> None:
        """Handle a dead or unresponsive worker.

        Steps:
        1. Log the death
        2. Mark model as ERROR
        3. Clean up worker from worker_pool
        4. Release VRAM lock in gpu_manager
        5. Reset GPU/VRAM state on model
        6. Emit system alert via Redis
        7. Publish model event
        8. Auto-restart with exponential backoff if enabled
        """
        from ocabra.core.model_manager import ModelStatus
        from ocabra.redis_client import publish_system_alert

        self._health_fail_counts.pop(model_id, None)

        # 1. Log
        logger.error("worker_death_detected", model_id=model_id, reason=reason)

        # 2. Mark model as ERROR
        state = await self._model_manager.get_state(model_id)
        if state:
            state.status = ModelStatus.ERROR
            state.error_message = f"Worker died: {reason}"

        # 3. Clean up worker from worker_pool
        self._worker_pool.remove_worker(model_id)

        # 4. Release VRAM lock in gpu_manager
        if state and self._model_manager._gpu_manager:
            for gpu_idx in state.current_gpu or []:
                await self._model_manager._gpu_manager.unlock_vram(gpu_idx, model_id)

        # 5. Reset GPU/VRAM state on model
        if state:
            state.current_gpu = []
            state.vram_used_mb = 0
            state.worker_info = None

        # 6. Emit system alert via Redis
        try:
            await publish_system_alert(
                "error",
                f"Worker for '{model_id}' died ({reason}). Model marked as ERROR.",
            )
        except Exception:
            pass

        # 7. Publish model event
        try:
            await self._model_manager._publish_event(model_id, "worker_death")
        except Exception:
            pass

        # 8. Auto-restart if enabled
        max_restarts = max(0, int(getattr(self._settings, "max_worker_restarts", 3)))
        auto_restart = bool(getattr(self._settings, "auto_restart_workers", True))
        backoff_base = float(getattr(self._settings, "worker_restart_backoff_seconds", 5.0))

        restart_count = self._restart_counts.get(model_id, 0)
        if auto_restart and restart_count < max_restarts:
            backoff = backoff_base * (2**restart_count)
            logger.info(
                "worker_auto_restart_attempt",
                model_id=model_id,
                attempt=restart_count + 1,
                max_restarts=max_restarts,
                backoff_seconds=backoff,
            )
            await asyncio.sleep(backoff)
            self._restart_counts[model_id] = restart_count + 1
            try:
                await self._model_manager.load(model_id)
                logger.info("worker_auto_restart_success", model_id=model_id)
            except Exception as exc:
                logger.error(
                    "worker_auto_restart_failed",
                    model_id=model_id,
                    error=str(exc),
                )
        elif auto_restart:
            logger.error(
                "worker_max_restarts_exceeded",
                model_id=model_id,
                max_restarts=max_restarts,
            )
