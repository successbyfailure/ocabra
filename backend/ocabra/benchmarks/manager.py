"""Persistent, traffic-aware benchmark worker.

One process evaluates one ``application + suite + model`` unit.  A process is
never frozen in the middle of inference: pause means terminate at the client
boundary, retain its checkout/results, then invoke the runner again with its
resume flag.  Completed units that overlap non-benchmark traffic are rejected
and repeated from a fresh checkout.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import platform
import re
import secrets
import signal
import sys
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

import sqlalchemy as sa
import structlog
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ocabra.benchmarks.catalog import APPS, AppSpec, SuiteSpec
from ocabra.config import APP_VERSION, settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.auth import ApiKey, User, UserRole
from ocabra.db.benchmarks import BenchmarkAttempt, BenchmarkRun
from ocabra.db.stats import RequestStat

logger = structlog.get_logger(__name__)

_SERVICE_USERNAME = "__ocabra_benchmark__"
_SERVICE_KEY_NAME = "ocabra-benchmark"
_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def git_authorization_header(token: str) -> str:
    """Build GitHub's HTTPS Git authentication header without changing the URL."""
    credentials = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return f"Authorization: Basic {credentials}"


@dataclass(frozen=True)
class BenchmarkIdentity:
    user_id: uuid.UUID
    api_key_name: str
    raw_api_key: str


@dataclass(frozen=True)
class ProcessOutcome:
    return_code: int | None
    interruption: str | None
    external_requests: list[dict[str, object]]


def external_active_requests(model_manager, identity: BenchmarkIdentity) -> list[dict[str, object]]:
    """Return inference requests not owned by the internal benchmark user."""
    if model_manager is None:
        return []
    external: list[dict[str, object]] = []
    for request in model_manager.active_requests_snapshot():
        if str(request.user_id or "") == str(identity.user_id):
            continue
        external.append(
            {
                "requestId": request.request_id,
                "model": request.profile_id or request.model_id,
                "apiKeyName": request.api_key_name,
                "username": request.username,
                "startedAt": datetime.fromtimestamp(request.started_at, UTC).isoformat(),
                "source": "active",
            }
        )
    return external


def execution_observations(rows, gpu_states: dict[int, object]) -> dict[str, object]:
    """Summarize persisted benchmark requests into reproducible runtime facts."""
    gpu_indices: set[int] = set()
    for row in rows:
        telemetry = row.runtime_telemetry if isinstance(row.runtime_telemetry, dict) else {}
        observed = telemetry.get("gpu_indices")
        if isinstance(observed, list):
            gpu_indices.update(int(index) for index in observed if isinstance(index, int))
        elif row.gpu_index is not None:
            gpu_indices.add(int(row.gpu_index))
    gpus = []
    for index in sorted(gpu_indices):
        state = gpu_states.get(index)
        gpus.append(
            {
                "index": index,
                "name": getattr(state, "name", None),
                "totalVramMb": getattr(state, "total_vram_mb", None),
            }
        )
    return {
        "model": {
            "backends": sorted({row.backend_type for row in rows if row.backend_type}),
            "runtimePresets": sorted(
                {row.runtime_preset for row in rows if row.runtime_preset}
            ),
            "effectiveContexts": sorted(
                {row.effective_context for row in rows if row.effective_context}
            ),
        },
        "hardware": {
            "gpus": gpus,
            "nodes": sorted({row.remote_node_name or "local" for row in rows}),
        },
        "requests": {
            "count": len(rows),
            "errors": sum(1 for row in rows if row.error or (row.status_code or 0) >= 400),
            "inputTokens": sum(row.input_tokens or 0 for row in rows),
            "outputTokens": sum(row.output_tokens or 0 for row in rows),
        },
    }


def backend_version_observations(
    backend_types: list[str],
    nodes: list[str],
    installer,
    source_commits: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Describe installed backend artifacts without failing on remote/unknown modules."""
    modules: list[dict[str, object]] = []
    for backend_type in backend_types:
        state = None
        if installer is not None and nodes == ["local"]:
            with suppress(KeyError):
                state = installer.get_state(backend_type)
        modules.append(
            {
                "type": backend_type,
                "installedVersion": getattr(state, "installed_version", None),
                "packages": dict(getattr(state, "packages", {}) or {}),
                "ociDigest": getattr(state, "oci_digest", None),
                "gitRef": getattr(state, "git_ref", None),
                "sourceCommit": (
                    (source_commits or {}).get(backend_type) if nodes == ["local"] else None
                ),
                "node": "local" if nodes == ["local"] else None,
            }
        )
    return modules


class BenchmarkManager:
    """Single-worker queue for reproducible application benchmarks."""

    def __init__(self, app) -> None:
        self._app = app
        self._root = Path(settings.benchmark_dir).resolve()
        self._identity: BenchmarkIdentity | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._process: asyncio.subprocess.Process | None = None
        self._process_lock = asyncio.Lock()

    async def start(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._identity = await self._create_service_identity()
        await self._recover_interrupted_runs()
        self._stop.clear()
        self._task = asyncio.create_task(self._worker_loop(), name="benchmark-manager")
        logger.info("benchmark_manager_started", root=str(self._root))

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        await self._terminate_process()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("benchmark_manager_stopped")

    def notify(self) -> None:
        self._wake.set()

    async def request_pause(self, run_id: uuid.UUID) -> BenchmarkRun:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.status in _TERMINAL:
                raise ValueError("La ejecución ya terminó")
            if run.status in {"queued", "paused"}:
                run.status = "paused"
            else:
                run.status = "pausing"
            run.pause_reason = "manual"
            run.phase = "pausing" if run.status == "pausing" else "paused"
            await session.commit()
            await session.refresh(run)
        self.notify()
        return run

    async def request_resume(self, run_id: uuid.UUID) -> BenchmarkRun:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.status != "paused":
                raise ValueError("Solo se puede reanudar una ejecución pausada")
            run.status = "queued"
            run.pause_reason = None
            run.error = None
            run.phase = "queued"
            await session.commit()
            await session.refresh(run)
        self.notify()
        return run

    async def request_cancel(self, run_id: uuid.UUID) -> BenchmarkRun:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.status in _TERMINAL:
                raise ValueError("La ejecución ya terminó")
            if run.status in {"queued", "paused"}:
                run.status = "cancelled"
                run.finished_at = datetime.now(UTC)
                run.phase = "cancelled"
            else:
                run.status = "cancelling"
                run.phase = "cancelling"
            run.pause_reason = None
            await session.commit()
            await session.refresh(run)
        self.notify()
        return run

    async def request_retry(self, run_id: uuid.UUID) -> BenchmarkRun:
        """Retry only the failed/current model and keep its runner checkpoint."""
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.status != "failed":
                raise ValueError("Solo se puede reintentar una ejecución fallida")
            run.status = "queued"
            run.phase = "queued"
            run.pause_reason = None
            run.current_model = None
            run.error = None
            run.finished_at = None
            await session.commit()
            await session.refresh(run)
        self.notify()
        return run

    async def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._resume_after_quiet_period()
                run_id = await self._next_queued_run()
                if run_id is not None:
                    await self._execute_run(run_id)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — one broken run must not kill the queue
                logger.exception("benchmark_worker_error", error=str(exc))
            try:
                await asyncio.wait_for(
                    self._wake.wait(), timeout=max(0.5, settings.benchmark_poll_interval_s)
                )
            except TimeoutError:
                pass
            self._wake.clear()

    async def _next_queued_run(self) -> uuid.UUID | None:
        async with AsyncSessionLocal() as session:
            run_id = await session.scalar(
                select(BenchmarkRun.id)
                .where(BenchmarkRun.status == "queued")
                .order_by(BenchmarkRun.created_at)
                .limit(1)
            )
            return run_id

    async def _resume_after_quiet_period(self) -> None:
        if self._identity is None:
            return
        if external_active_requests(self._model_manager, self._identity):
            async with AsyncSessionLocal() as session:
                await session.execute(
                    sa.update(BenchmarkRun)
                    .where(
                        BenchmarkRun.status == "paused",
                        BenchmarkRun.pause_reason == "external_traffic",
                    )
                    .values(external_activity_at=datetime.now(UTC))
                )
                await session.commit()
            return
        cutoff = datetime.now(UTC) - timedelta(seconds=settings.benchmark_quiet_period_s)
        async with AsyncSessionLocal() as session:
            runs = (
                await session.execute(
                    select(BenchmarkRun).where(
                        BenchmarkRun.status == "paused",
                        BenchmarkRun.pause_reason == "external_traffic",
                        BenchmarkRun.auto_pause.is_(True),
                        BenchmarkRun.external_activity_at <= cutoff,
                    )
                )
            ).scalars()
            changed = False
            for run in runs:
                run.status = "queued"
                run.pause_reason = None
                run.phase = "queued"
                changed = True
            if changed:
                await session.commit()

    async def _execute_run(self, run_id: uuid.UUID) -> None:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.status != "queued":
                return
            if run.current_model_index >= len(run.models):
                run.status = "completed"
                run.phase = "completed"
                run.current_model = None
                run.finished_at = datetime.now(UTC)
                await session.commit()
                return
            app = APPS[run.app]
            suite = app.suites[run.suite]
            model = run.models[run.current_model_index]
            if run.auto_pause and self._identity is not None:
                active = external_active_requests(self._model_manager, self._identity)
                if active:
                    run.status = "paused"
                    run.phase = "paused_external_traffic"
                    run.pause_reason = "external_traffic"
                    run.external_activity_at = datetime.now(UTC)
                    await session.commit()
                    return
            run.status = "running"
            run.phase = "checkout"
            run.current_model = model
            run.started_at = run.started_at or datetime.now(UTC)
            run.error = None
            await session.commit()

        try:
            await self._execute_model(run_id, app, suite, model)
        except Exception as exc:  # noqa: BLE001 — persist the failure for the UI
            logger.exception("benchmark_model_failed", run_id=str(run_id), model=model)
            await self._fail_run(run_id, str(exc))

    async def _execute_model(
        self, run_id: uuid.UUID, app: AppSpec, suite: SuiteSpec, model: str
    ) -> None:
        if self._identity is None:
            raise RuntimeError("La identidad interna de benchmark no está inicializada")
        attempt_number, workspace, resume = await self._attempt_workspace(run_id, model)
        source = workspace / "source"
        log_path = workspace / "runner.log"
        workspace.mkdir(parents=True, exist_ok=True)
        await self._set_phase(run_id, "checkout")
        await self._ensure_checkout(source, app, await self._run_revision(run_id), log_path)
        dataset_root = source / app.dataset_root
        script = (
            Path(__file__).with_name("runners") / suite.local_runner
            if suite.local_runner
            else dataset_root / str(suite.script)
        )
        if not script.is_file():
            raise RuntimeError(f"La revisión no contiene el runner esperado: {script}")

        if suite.prepare_script and not (workspace / ".prepared").exists():
            await self._set_phase(run_id, "preparing")
            prepare = dataset_root / suite.prepare_script
            outcome = await self._run_process(
                run_id,
                [sys.executable, "-I", str(prepare)],
                source,
                log_path,
                detect_external=False,
            )
            if outcome.interruption:
                await self._persist_interrupted_without_attempt(run_id, outcome.interruption)
                return
            if outcome.return_code != 0:
                raise RuntimeError(f"La preparación terminó con código {outcome.return_code}")
            (workspace / ".prepared").touch()

        environment = await self._base_environment(run_id, source, app, suite, model)
        attempt = await self._create_attempt(
            run_id, model, attempt_number, workspace, log_path, environment
        )
        before = self._result_snapshot(dataset_root, suite.result_glob)
        command = [sys.executable, "-I", str(script)]
        if suite.local_runner:
            command.extend(("--source-root", str(source)))
        command.extend((suite.models_flag, model, *suite.default_args))
        command.extend(await self._suite_args(run_id))
        if resume:
            command.extend(suite.resume_args)
        await self._set_phase(run_id, "running")
        started_at = datetime.now(UTC)
        outcome = await self._run_process(run_id, command, source, log_path, detect_external=True)
        finished_at = datetime.now(UTC)
        await self._set_phase(run_id, "finalizing")
        # StatsMiddleware records at response completion. Give its short DB
        # write a scheduling turn before checking for fast external requests
        # that may have started and ended between active-request polls.
        await asyncio.sleep(0.1)
        environment = await self._complete_environment(
            environment, started_at, finished_at, source
        )
        historical = await self._historical_external_requests(started_at, finished_at)
        external = self._deduplicate_external([*outcome.external_requests, *historical])

        if outcome.interruption:
            await self._finish_interrupted_attempt(attempt.id, outcome, external, environment)
            return
        result_path = self._changed_result(dataset_root, suite.result_glob, before)
        result = await asyncio.to_thread(self._read_result, result_path) if result_path else None
        if outcome.return_code != 0 or result is None:
            message = (
                f"El runner terminó con código {outcome.return_code}"
                if outcome.return_code != 0
                else f"No generó un resultado que coincida con {suite.result_glob}"
            )
            await self._finish_failed_attempt(attempt.id, outcome.return_code, message, environment)
            await self._fail_run(run_id, message)
            return
        if external:
            await self._finish_contaminated_attempt(attempt.id, result, external, environment)
            if await self._run_auto_pause(run_id):
                await self._pause_for_external_traffic(run_id, external)
            else:
                await self._retry_contaminated(run_id)
            return
        await self._finish_clean_attempt(attempt.id, result, outcome.return_code, environment)
        await self._advance_run(run_id, model, result)

    async def _run_process(
        self,
        run_id: uuid.UUID,
        command: list[str],
        cwd: Path,
        log_path: Path,
        *,
        detect_external: bool,
    ) -> ProcessOutcome:
        env = dict(os.environ)
        assert self._identity is not None
        env.update(
            {
                "OPENAI_COMPATIBLE_URL": settings.benchmark_base_url,
                "OPENAI_COMPATIBLE_KEY": self._identity.raw_api_key,
                "UV_CACHE_DIR": str(self._root / "uv" / "cache"),
                "UV_PYTHON_INSTALL_DIR": str(self._root / "uv" / "python"),
            }
        )
        if await self._run_app(run_id) == "corrala":
            env["CORRALA_BENCH_DATABASE_URL"] = await self._corrala_database_url()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        external: list[dict[str, object]] = []
        with log_path.open("a", encoding="utf-8") as log:
            self._write_command(log, command)
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            async with self._process_lock:
                self._process = process
            interruption: str | None = None
            try:
                while process.returncode is None:
                    status, auto_pause = await self._control_state(run_id)
                    if status == "cancelling":
                        interruption = "cancelled"
                        break
                    if status == "pausing":
                        interruption = "manual"
                        break
                    if detect_external and auto_pause:
                        active = external_active_requests(self._model_manager, self._identity)
                        if active:
                            external.extend(active)
                            interruption = "external_traffic"
                            break
                    try:
                        await asyncio.wait_for(
                            process.wait(), timeout=max(0.25, settings.benchmark_poll_interval_s)
                        )
                    except TimeoutError:
                        pass
                if interruption:
                    await self._terminate_process(process)
                else:
                    await process.wait()
                    # A control request can race with a runner's last output.
                    # Honour it even when the child exited between polls: the
                    # result is retried/resumed instead of silently advancing.
                    status, _ = await self._control_state(run_id)
                    if status == "cancelling":
                        interruption = "cancelled"
                    elif status == "pausing":
                        interruption = "manual"
            finally:
                async with self._process_lock:
                    if self._process is process:
                        self._process = None
        return ProcessOutcome(process.returncode, interruption, external)

    async def _control_state(self, run_id: uuid.UUID) -> tuple[str, bool]:
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(BenchmarkRun.status, BenchmarkRun.auto_pause).where(
                        BenchmarkRun.id == run_id
                    )
                )
            ).one()
            return row[0], row[1]

    async def _attempt_workspace(self, run_id: uuid.UUID, model: str) -> tuple[int, Path, bool]:
        async with AsyncSessionLocal() as session:
            attempts = (
                (
                    await session.execute(
                        select(BenchmarkAttempt)
                        .where(BenchmarkAttempt.run_id == run_id, BenchmarkAttempt.model == model)
                        .order_by(BenchmarkAttempt.attempt_number.desc())
                    )
                )
                .scalars()
                .all()
            )
        number = max((attempt.attempt_number for attempt in attempts), default=0) + 1
        latest = attempts[0] if attempts else None
        if (
            latest
            and latest.status in {"interrupted", "failed"}
            and Path(latest.workspace_path).is_dir()  # noqa: ASYNC240 — stat local mínimo
        ):
            return number, Path(latest.workspace_path), True
        model_part = _SAFE_COMPONENT.sub("_", model).strip("._")[:100] or "model"
        return number, self._root / "runs" / str(run_id) / model_part / f"attempt-{number}", False

    async def _ensure_checkout(
        self, source: Path, app: AppSpec, revision: str, log_path: Path
    ) -> None:
        if (source / ".git").is_dir():
            return
        source.parent.mkdir(parents=True, exist_ok=True)
        clone_env = dict(os.environ)
        if settings.benchmark_git_token:
            clone_env.update(
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.extraHeader",
                    "GIT_CONFIG_VALUE_0": git_authorization_header(settings.benchmark_git_token),
                }
            )
        clone = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "--filter=blob:none",
            *(("--no-checkout",) if app.checkout_paths else ()),
            app.repository,
            str(source),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=clone_env,
        )
        output = await clone.communicate()
        await asyncio.to_thread(log_path.write_bytes, output[0] or b"")
        if clone.returncode != 0:
            raise RuntimeError(f"No se pudo clonar {app.repository}; consulta runner.log")
        if app.checkout_paths:
            sparse = await asyncio.create_subprocess_exec(
                "git", "-C", str(source), "sparse-checkout", "set", *app.checkout_paths,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            output = await sparse.communicate()
            with log_path.open("ab") as log:
                log.write(output[0] or b"")
            if sparse.returncode != 0:
                raise RuntimeError("No se pudo preparar el checkout parcial del benchmark")
        checkout = await asyncio.create_subprocess_exec(
            "git", "-C", str(source), "checkout", "--detach", revision,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )  # fmt: skip
        output = await checkout.communicate()
        with log_path.open("ab") as log:
            log.write(output[0] or b"")
        if checkout.returncode != 0:
            raise RuntimeError(f"No se pudo fijar la revisión {revision!r}")

    async def _create_service_identity(self) -> BenchmarkIdentity:
        from ocabra.core.auth_manager import generate_api_key, hash_password_async

        raw, key_hash, prefix = generate_api_key()
        async with AsyncSessionLocal() as session:
            user = await session.scalar(select(User).where(User.username == _SERVICE_USERNAME))
            if user is None:
                user = User(
                    username=_SERVICE_USERNAME,
                    email=None,
                    hashed_password=await hash_password_async(secrets.token_urlsafe(48)),
                    role=UserRole.system_admin.value,
                    is_active=True,
                )
                session.add(user)
                await session.flush()
            else:
                user.is_active = True
                user.role = UserRole.system_admin.value
            await session.execute(
                sa.delete(ApiKey).where(
                    ApiKey.user_id == user.id,
                    ApiKey.name == _SERVICE_KEY_NAME,
                )
            )
            session.add(
                ApiKey(
                    user_id=user.id,
                    name=_SERVICE_KEY_NAME,
                    key_hash=key_hash,
                    key_prefix=prefix,
                )
            )
            await session.commit()
            return BenchmarkIdentity(user.id, _SERVICE_KEY_NAME, raw)

    async def _recover_interrupted_runs(self) -> None:
        now = datetime.now(UTC)
        async with AsyncSessionLocal() as session:
            attempts = (
                await session.execute(
                    select(BenchmarkAttempt).where(BenchmarkAttempt.status == "running")
                )
            ).scalars()
            for attempt in attempts:
                attempt.status = "interrupted"
                attempt.error = "API reiniciada durante el intento; se reanudará"
                attempt.finished_at = now
            runs = (
                await session.execute(
                    select(BenchmarkRun).where(
                        BenchmarkRun.status.in_(["running", "pausing", "cancelling"])
                    )
                )
            ).scalars()
            for run in runs:
                if run.status == "cancelling":
                    run.status = "cancelled"
                    run.phase = "cancelled"
                    run.finished_at = now
                elif run.pause_reason == "manual":
                    run.status = "paused"
                    run.phase = "paused"
                else:
                    run.status = "queued"
                    run.phase = "queued"
            await session.commit()

    @property
    def _model_manager(self):
        return getattr(self._app.state, "model_manager", None)

    async def _terminate_process(self, process: asyncio.subprocess.Process | None = None) -> None:
        if process is None:
            async with self._process_lock:
                process = self._process
        if process is None or process.returncode is not None:
            return
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    @staticmethod
    def _write_command(log: TextIO, command: list[str]) -> None:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()

    @staticmethod
    def _result_snapshot(root: Path, pattern: str) -> dict[Path, tuple[int, int]]:
        return {
            path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
            for path in root.glob(pattern)
            if path.is_file()
        }

    @staticmethod
    def _changed_result(
        root: Path, pattern: str, before: dict[Path, tuple[int, int]]
    ) -> Path | None:
        changed = [
            path
            for path in root.glob(pattern)
            if path.is_file()
            and before.get(path.resolve()) != (path.stat().st_mtime_ns, path.stat().st_size)
        ]
        return max(changed, key=lambda path: path.stat().st_mtime_ns) if changed else None

    @staticmethod
    def _read_result(path: Path) -> dict:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("El resultado del benchmark no es un objeto JSON")
        return value

    async def _historical_external_requests(
        self, started_at: datetime, finished_at: datetime
    ) -> list[dict[str, object]]:
        assert self._identity is not None
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(RequestStat).where(
                        RequestStat.started_at >= started_at - timedelta(hours=1),
                        RequestStat.started_at <= finished_at,
                        sa.or_(
                            RequestStat.user_id.is_(None),
                            RequestStat.user_id != self._identity.user_id,
                        ),
                    )
                )
            ).scalars()
            result = []
            for row in rows:
                row_finished = row.started_at + timedelta(milliseconds=row.duration_ms or 0)
                if row_finished < started_at:
                    continue
                result.append(
                    {
                        "requestId": str(row.id),
                        "model": row.requested_model or row.model_id,
                        "apiKeyName": row.api_key_name,
                        "username": None,
                        "startedAt": row.started_at.isoformat(),
                        "source": "request_stats",
                    }
                )
            return result

    async def _base_environment(
        self,
        run_id: uuid.UUID,
        source: Path,
        app: AppSpec,
        suite: SuiteSpec,
        model: str,
    ) -> dict[str, object]:
        """Describe immutable software and dataset versions before inference starts."""
        commit = await self._git_value(source, "rev-parse", "HEAD")
        repository = await self._git_value(source, "remote", "get-url", "origin")
        runner_version: str | None = None
        if app.name == "bfcl":
            pyproject = source / app.dataset_root / "pyproject.toml"
            if pyproject.is_file():
                try:
                    import tomllib

                    raw = await asyncio.to_thread(pyproject.read_bytes)
                    runner_version = str(tomllib.loads(raw.decode())["project"]["version"])
                except (KeyError, OSError, TypeError, ValueError):
                    runner_version = None
        return {
            "software": {
                "ocabra": APP_VERSION,
                "python": platform.python_version(),
                **({"scorer": runner_version} if runner_version else {}),
            },
            "benchmark": {
                "app": app.name,
                "suite": suite.name,
                "requestedRevision": await self._run_revision(run_id),
                "commit": commit,
                "repository": repository or app.repository,
                "runner": "ocabra" if suite.local_runner else "application",
            },
            "model": {"profile": model},
            "hardware": {"gpus": [], "nodes": []},
            "requests": {"count": 0},
        }

    @staticmethod
    async def _git_value(source: Path, *arguments: str) -> str | None:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(source),
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await process.communicate()
        return output.decode(errors="replace").strip() if process.returncode == 0 else None

    async def _complete_environment(
        self,
        base: dict[str, object],
        started_at: datetime,
        finished_at: datetime,
        source: Path,
    ) -> dict[str, object]:
        """Attach the backends, nodes and physical GPUs observed in benchmark requests."""
        assert self._identity is not None
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(RequestStat).where(
                        RequestStat.user_id == self._identity.user_id,
                        RequestStat.started_at >= started_at - timedelta(hours=1),
                        RequestStat.started_at <= finished_at,
                    )
                )
            ).scalars().all()
        rows = [
            row
            for row in rows
            if row.started_at + timedelta(milliseconds=row.duration_ms or 0) >= started_at
        ]
        gpu_manager = getattr(self._app.state, "gpu_manager", None)
        gpu_states = {}
        if gpu_manager is not None:
            states = await gpu_manager.get_all_states()
            gpu_states = {state.index: state for state in states}
        observed = execution_observations(rows, gpu_states)
        environment = dict(base)
        software = dict(environment.get("software") or {})
        installer = getattr(self._app.state, "backend_installer", None)
        model_observations = observed["model"]
        hardware_observations = observed["hardware"]
        assert isinstance(model_observations, dict)
        assert isinstance(hardware_observations, dict)
        source_commits: dict[str, str] = {}
        if hardware_observations["nodes"] == ["local"]:
            for backend_type in model_observations["backends"]:
                source = Path(settings.backends_dir) / backend_type / "src"
                commit = await self._git_value(source, "rev-parse", "HEAD")
                if commit:
                    source_commits[backend_type] = commit
        software["backends"] = backend_version_observations(
            model_observations["backends"],
            hardware_observations["nodes"],
            installer,
            source_commits,
        )
        benchmark = environment.get("benchmark")
        if isinstance(benchmark, dict) and benchmark.get("app") == "bfcl":
            metadata_files = list(
                (source / "berkeley-function-call-leaderboard" / ".venv").glob(
                    "lib/python*/site-packages/bfcl_eval-*.dist-info/METADATA"
                )
            )
            if metadata_files:
                try:
                    text = await asyncio.to_thread(metadata_files[0].read_text, encoding="utf-8")
                    version = next(
                        line.removeprefix("Version: ").strip()
                        for line in text.splitlines()
                        if line.startswith("Version: ")
                    )
                    software["scorer"] = version
                except (OSError, StopIteration):
                    pass
        environment["software"] = software
        model = dict(environment.get("model") or {})
        model.update(observed["model"])
        environment["model"] = model
        environment["hardware"] = observed["hardware"]
        environment["requests"] = observed["requests"]
        return environment

    @staticmethod
    def _deduplicate_external(items: list[dict[str, object]]) -> list[dict[str, object]]:
        return list({str(item.get("requestId")): item for item in items}.values())

    async def _corrala_database_url(self) -> str:
        url = settings.benchmark_corrala_database_url.strip()
        if not url:
            from sqlalchemy.engine import make_url

            main = make_url(settings.database_url)
            url = str(main.set(database=f"{main.database or 'ocabra'}_bench"))
        await self._ensure_postgres_database(url)
        return url

    @staticmethod
    async def _ensure_postgres_database(url: str) -> None:
        import asyncpg
        from sqlalchemy.engine import make_url

        parsed = make_url(url)
        database = parsed.database
        if not database or not any(
            token in database.lower() for token in ("bench", "test", "scratch", "tmp")
        ):
            raise RuntimeError(
                "La base de Corrala debe incluir bench/test/scratch/tmp en el nombre"
            )
        connection = await asyncpg.connect(
            user=parsed.username,
            password=parsed.password,
            host=parsed.host,
            port=parsed.port or 5432,
            database="postgres",
        )
        try:
            exists = await connection.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", database
            )
            if not exists:
                quoted = database.replace('"', '""')
                await connection.execute(f'CREATE DATABASE "{quoted}"')
        finally:
            await connection.close()

    async def _get_run(self, session, run_id: uuid.UUID) -> BenchmarkRun:
        run = await session.get(BenchmarkRun, run_id)
        if run is None:
            raise LookupError("Ejecución no encontrada")
        return run

    async def _run_revision(self, run_id: uuid.UUID) -> str:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            return run.revision or APPS[run.app].default_revision

    async def _run_app(self, run_id: uuid.UUID) -> str:
        async with AsyncSessionLocal() as session:
            return str(
                await session.scalar(select(BenchmarkRun.app).where(BenchmarkRun.id == run_id))
            )

    async def _suite_args(self, run_id: uuid.UUID) -> list[str]:
        async with AsyncSessionLocal() as session:
            value = await session.scalar(
                select(BenchmarkRun.suite_args).where(BenchmarkRun.id == run_id)
            )
            return list(value or [])

    async def _run_auto_pause(self, run_id: uuid.UUID) -> bool:
        async with AsyncSessionLocal() as session:
            return bool(
                await session.scalar(
                    select(BenchmarkRun.auto_pause).where(BenchmarkRun.id == run_id)
                )
            )

    async def _set_phase(self, run_id: uuid.UUID, phase: str) -> None:
        async with AsyncSessionLocal() as session:
            await session.execute(
                sa.update(BenchmarkRun).where(BenchmarkRun.id == run_id).values(phase=phase)
            )
            await session.commit()

    async def _create_attempt(
        self,
        run_id: uuid.UUID,
        model: str,
        number: int,
        workspace: Path,
        log_path: Path,
        environment: dict[str, object],
    ) -> BenchmarkAttempt:
        attempt = BenchmarkAttempt(
            run_id=run_id,
            model=model,
            attempt_number=number,
            status="running",
            workspace_path=str(workspace),
            log_path=str(log_path),
            environment=environment,
            started_at=datetime.now(UTC),
        )
        async with AsyncSessionLocal() as session:
            session.add(attempt)
            await session.commit()
            await session.refresh(attempt)
        return attempt

    async def _persist_interrupted_without_attempt(self, run_id: uuid.UUID, reason: str) -> None:
        if reason == "cancelled":
            await self._mark_cancelled(run_id)
        elif reason == "manual":
            await self._mark_paused(run_id, "manual")
        else:
            await self._pause_for_external_traffic(run_id, [])

    async def _finish_interrupted_attempt(
        self,
        attempt_id: uuid.UUID,
        outcome: ProcessOutcome,
        external: list[dict[str, object]],
        environment: dict[str, object],
    ) -> None:
        async with AsyncSessionLocal() as session:
            attempt = await session.get(BenchmarkAttempt, attempt_id)
            if attempt is not None:
                attempt.status = "interrupted"
                attempt.return_code = outcome.return_code
                attempt.external_requests = external
                attempt.environment = environment
                attempt.error = f"Interrumpido: {outcome.interruption}"
                attempt.finished_at = datetime.now(UTC)
                await session.commit()
        run_id = await self._attempt_run_id(attempt_id)
        if outcome.interruption == "cancelled":
            await self._mark_cancelled(run_id)
        elif outcome.interruption == "manual":
            await self._mark_paused(run_id, "manual")
        else:
            await self._pause_for_external_traffic(run_id, external)

    async def _attempt_run_id(self, attempt_id: uuid.UUID) -> uuid.UUID:
        async with AsyncSessionLocal() as session:
            run_id = await session.scalar(
                select(BenchmarkAttempt.run_id).where(BenchmarkAttempt.id == attempt_id)
            )
            assert run_id is not None
            return run_id

    async def _finish_failed_attempt(
        self,
        attempt_id: uuid.UUID,
        return_code: int | None,
        error: str,
        environment: dict[str, object],
    ) -> None:
        async with AsyncSessionLocal() as session:
            attempt = await session.get(BenchmarkAttempt, attempt_id)
            if attempt:
                attempt.status = "failed"
                attempt.return_code = return_code
                attempt.error = error
                attempt.environment = environment
                attempt.finished_at = datetime.now(UTC)
                await session.commit()

    async def _finish_contaminated_attempt(
        self,
        attempt_id: uuid.UUID,
        result: dict,
        external: list[dict[str, object]],
        environment: dict[str, object],
    ) -> None:
        async with AsyncSessionLocal() as session:
            attempt = await session.get(BenchmarkAttempt, attempt_id)
            if attempt:
                attempt.status = "contaminated"
                attempt.return_code = 0
                attempt.result = result
                attempt.external_requests = external
                attempt.environment = environment
                attempt.finished_at = datetime.now(UTC)
                await session.commit()

    async def _finish_clean_attempt(
        self,
        attempt_id: uuid.UUID,
        result: dict,
        return_code: int | None,
        environment: dict[str, object],
    ) -> None:
        async with AsyncSessionLocal() as session:
            attempt = await session.get(BenchmarkAttempt, attempt_id)
            if attempt:
                attempt.status = "completed"
                attempt.return_code = return_code
                attempt.result = result
                attempt.environment = environment
                attempt.finished_at = datetime.now(UTC)
                await session.commit()

    async def _pause_for_external_traffic(
        self, run_id: uuid.UUID, external: list[dict[str, object]]
    ) -> None:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            run.status = "paused"
            run.phase = "paused_external_traffic"
            run.pause_reason = "external_traffic"
            run.external_activity_at = datetime.now(UTC)
            if external:
                run.contaminated_attempts += 1
            await session.commit()

    async def _mark_paused(self, run_id: uuid.UUID, reason: str) -> None:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            run.status = "paused"
            run.phase = "paused"
            run.pause_reason = reason
            await session.commit()

    async def _mark_cancelled(self, run_id: uuid.UUID) -> None:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            run.status = "cancelled"
            run.phase = "cancelled"
            run.finished_at = datetime.now(UTC)
            await session.commit()

    async def _advance_run(
        self,
        run_id: uuid.UUID,
        model: str,
        result: dict,
    ) -> None:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            values = dict(run.result or {})
            values[model] = result
            run.result = values
            run.current_model_index += 1
            run.current_model = None
            if run.current_model_index >= len(run.models):
                run.status = "completed"
                run.phase = "completed"
                run.finished_at = datetime.now(UTC)
            else:
                run.status = "queued"
                run.phase = "queued"
            await session.commit()
        self.notify()

    async def _retry_contaminated(self, run_id: uuid.UUID) -> None:
        """Repeat the same model immediately when automatic pausing is disabled."""
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            run.status = "queued"
            run.phase = "queued"
            run.current_model = None
            run.contaminated_attempts += 1
            await session.commit()
        self.notify()

    async def _fail_run(self, run_id: uuid.UUID, error: str) -> None:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.status not in _TERMINAL:
                now = datetime.now(UTC)
                await session.execute(
                    sa.update(BenchmarkAttempt)
                    .where(
                        BenchmarkAttempt.run_id == run_id,
                        BenchmarkAttempt.status == "running",
                    )
                    .values(status="failed", error=error[:4000], finished_at=now)
                )
                run.status = "failed"
                run.phase = "failed"
                run.error = error[:4000]
                run.finished_at = now
                await session.commit()


async def load_run(run_id: uuid.UUID) -> BenchmarkRun | None:
    """Load a run with attempts for API serialization."""
    async with AsyncSessionLocal() as session:
        return await session.scalar(
            select(BenchmarkRun)
            .options(selectinload(BenchmarkRun.attempts))
            .where(BenchmarkRun.id == run_id)
        )


async def list_runs(limit: int = 100) -> list[BenchmarkRun]:
    """Load newest runs with attempts for the dashboard."""
    async with AsyncSessionLocal() as session:
        return list(
            (
                await session.execute(
                    select(BenchmarkRun)
                    .options(selectinload(BenchmarkRun.attempts))
                    .order_by(BenchmarkRun.created_at.desc())
                    .limit(limit)
                )
            ).scalars()
        )
