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
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO
from zoneinfo import ZoneInfo

import sqlalchemy as sa
import structlog
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ocabra.benchmarks.catalog import APPS, AppSpec, SuiteSpec
from ocabra.benchmarks.diagnosis.jobs import DiagnosisRunner, DiagnosisStore
from ocabra.benchmarks.models import bench_profile, ensure_deployment, pending_deployment
from ocabra.config import APP_VERSION, settings
from ocabra.core.cloud import timeline as cloud_timeline
from ocabra.database import AsyncSessionLocal
from ocabra.db.auth import ApiKey, User, UserRole
from ocabra.db.benchmarks import BenchmarkAttempt, BenchmarkRun, BenchmarkSchedule
from ocabra.db.stats import RequestStat

logger = structlog.get_logger(__name__)

_SERVICE_USERNAME = "__ocabra_benchmark__"
_SERVICE_KEY_NAME = "ocabra-benchmark"
_TERMINAL = frozenset({"completed", "failed", "cancelled"})
# Lo que deja un pod que se apaga sin llegar a servir (core/cloud/manager.py):
# GPU averiada, o la máquina no llegó a arrancar el nodo (imagen, red).
POD_RETRY_MARKS = ("no funciona para cómputo", "No llegó a unirse a la federación")
# GPU averiada en el pod: se vuelve a pedir pasado este tiempo, y un tipo de
# GPU que falla tantas veces se deja de pedir para esa batería.
BROKEN_POD_RETRY_S = 300
BROKEN_GPU_STRIKES = 3
# Respuestas pasajeras que los runners reintentan: no cuentan como fallo del modelo.
_TRANSIENT_STATUS = frozenset({429, 502, 503, 504})
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def node_base_url(base_url: str, node: str | None) -> str:
    """URL base del runner: ``<base>/_node/<nodo>`` fuerza el nodo de cada petición."""
    if not node:
        return base_url
    from urllib.parse import quote

    return f"{base_url.rstrip('/')}/_node/{quote(node, safe='')}"


def next_schedule_occurrence(
    weekdays: list[int], time_of_day: str, timezone: str, after: datetime
) -> datetime:
    """Return the first configured weekly occurrence strictly after ``after``."""
    zone = ZoneInfo(timezone)
    hour, minute = (int(part) for part in time_of_day.split(":"))
    local_after = after.astimezone(zone)
    for offset in range(8):
        candidate_date: date = local_after.date() + timedelta(days=offset)
        if candidate_date.weekday() not in weekdays:
            continue
        candidate = datetime.combine(candidate_date, time(hour, minute), tzinfo=zone)
        if candidate > local_after:
            return candidate.astimezone(UTC)
    raise ValueError("No se pudo calcular la próxima ejecución")


def run_from_schedule(schedule: BenchmarkSchedule, *, now: datetime | None = None) -> BenchmarkRun:
    """Create a queued run snapshot; later edits to the plan cannot mutate it."""
    run_id = uuid.uuid4()
    return BenchmarkRun(
        id=run_id,
        app=schedule.app,
        suite=schedule.suite,
        revision=schedule.revision,
        models=list(schedule.models),
        suite_args=list(schedule.suite_args),
        auto_pause=schedule.auto_pause,
        node=schedule.node,
        sampling=getattr(schedule, "sampling", None),
        cloud=getattr(schedule, "cloud", None),
        phase="queued",
        schedule_id=schedule.id,
        workspace_path=str(Path(settings.benchmark_dir).resolve() / "runs" / str(run_id)),
        created_by=schedule.created_by,
        created_at=now or datetime.now(UTC),
    )


def runs_from_schedule(
    schedule: BenchmarkSchedule, *, now: datetime | None = None
) -> list[BenchmarkRun]:
    """La ejecución principal del plan y una más por cada prueba de su batería."""
    first = run_from_schedule(schedule, now=now)
    extras = [
        item
        for item in (getattr(schedule, "battery", None) or [])
        if isinstance(item, dict) and item.get("app") and item.get("suite")
    ]
    if not extras:
        return [first]
    first.batch_id = uuid.uuid4()
    runs = [first]
    for item in extras:
        run = run_from_schedule(schedule, now=now)
        run.app = str(item["app"])
        run.suite = str(item["suite"])
        run.suite_args = list(item.get("suite_args") or [])
        run.revision = None
        run.batch_id = first.batch_id
        runs.append(run)
    return runs


def git_authorization_header(token: str) -> str:
    """Build GitHub's HTTPS Git authentication header without changing the URL."""
    credentials = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return f"Authorization: Basic {credentials}"


def corrala_benchmark_database_url(database_url: str) -> str:
    """Derive Corrala's scratch database URL without redacting credentials."""
    from sqlalchemy.engine import make_url

    main = make_url(database_url)
    return main.set(database=f"{main.database or 'ocabra'}_bench").render_as_string(
        hide_password=False
    )


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


def belongs_to_run(row, node: str | None) -> bool:
    """Si una petición del benchmark es de la ejecución que mide en *node*.

    Varias ejecuciones miden a la vez en máquinas distintas con la misma
    identidad de servicio (y a menudo el mismo modelo). Las del runner van
    forzadas a su nodo (``/_node/<nodo>/`` → ruta ``forced``): una forzada a
    otro nodo es de otra ejecución. En una medida local cuenta todo lo que no
    va forzado a un nodo.
    """
    route = getattr(row, "federation_route", None)
    target = (getattr(row, "remote_node_name", None) or "").lower()
    if node:
        # El runner va siempre forzado a su nodo, y una petición forzada ya no
        # se sirve en otro sitio (503 si el nodo no está): lo que no va forzado
        # aquí es de otra ejecución o de otra suite (un juez sin /_node/).
        return route == "forced" and target == node.lower()
    return route != "forced"


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
            "runtimePresets": sorted({row.runtime_preset for row in rows if row.runtime_preset}),
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
        # Procesos de runner vivos (uno por ejecución en curso).
        self._processes: set[asyncio.subprocess.Process] = set()
        self._process_lock = asyncio.Lock()
        # Una ejecución a la vez por máquina («» = este servidor): las de
        # nodos distintos no comparten hardware y avanzan en paralelo.
        self._running: dict[str, asyncio.Task[None]] = {}
        self._diagnoses = DiagnosisStore(self._root)
        self._diagnosis_runner: DiagnosisRunner | None = None
        self._provision_task: asyncio.Task[None] | None = None
        # Una sola pasada de aprovisionamiento a la vez: dos a la vez pedirían
        # dos pods para la misma ejecución.
        self._provision_lock = asyncio.Lock()
        # Overrides de muestreo activos (perfil, parámetros y clave del
        # runner), uno por ejecución en curso; «diagnosis» es el del diagnóstico.
        self._sampling_overrides: dict[str, dict[str, object]] = {}
        self._last_diagnosis_purge = 0.0
        # (nodo de la nube, modelo) ya borrados del pod: no se vuelve a pedir.
        self._freed_pod_models: set[tuple[str, str]] = set()
        # Base de pruebas de Corrala de cada máquina que mide a la vez (hueco).
        self._corrala_slots: dict[str, int] = {}
        # Muestreo que manda el runner, por (máquina, modelo): combinación → peticiones.
        self._observed_sampling: dict[tuple[str, str], dict[str, int]] = {}

    @property
    def service_api_key(self) -> str | None:
        """La clave del runner (el middleware reconoce así sus peticiones)."""
        return self._identity.raw_api_key if self._identity is not None else None

    def observe_sampling(self, node: str, model: str, params: dict[str, object]) -> None:
        """Anota el muestreo de una petición del runner (lo llama el middleware)."""
        counts = self._observed_sampling.setdefault((node.lower(), model), {})
        key = json.dumps(params, sort_keys=True)
        if key in counts or len(counts) < 50:
            counts[key] = counts.get(key, 0) + 1

    def _sampling_summary(self, node: str | None, model: str) -> list[dict[str, object]]:
        counts = self._observed_sampling.pop(((node or "").lower(), model), {})
        ranked = sorted(counts.items(), key=lambda item: -item[1])[:10]
        return [{"params": json.loads(key), "requests": n} for key, n in ranked]

    @property
    def sampling_overrides(self) -> list[dict[str, object]]:
        """Overrides de muestreo de todo lo que está midiendo ahora."""
        return list(self._sampling_overrides.values())

    @property
    def sampling_override(self) -> dict[str, object] | None:
        """El override del diagnóstico en curso (lo pone ``DiagnosisRunner``)."""
        return self._sampling_overrides.get("diagnosis")

    @sampling_override.setter
    def sampling_override(self, value: dict[str, object] | None) -> None:
        if value:
            self._sampling_overrides["diagnosis"] = value
        else:
            self._sampling_overrides.pop("diagnosis", None)

    @property
    def diagnoses(self) -> DiagnosisStore:
        return self._diagnoses

    async def start(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._identity = await self._create_service_identity()
        await self._recover_interrupted_runs()
        await asyncio.to_thread(self._diagnoses.recover_interrupted)
        self._stop.clear()
        self._task = asyncio.create_task(self._worker_loop(), name="benchmark-manager")
        # Aparte del worker: mientras una ejecución mide (puede durar horas),
        # los pods de las siguientes se tienen que ir pidiendo y preparando.
        self._provision_task = asyncio.create_task(
            self._provision_loop(), name="benchmark-provisioning"
        )
        logger.info("benchmark_manager_started", root=str(self._root))

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        await self._terminate_process()
        for task in (self._task, self._provision_task, *self._running.values()):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        self._task = None
        self._provision_task = None
        self._running.clear()
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
        if run.status == "cancelled":
            await self._release_cloud(run_id)
        self.notify()
        return run

    async def request_retry(self, run_id: uuid.UUID) -> BenchmarkRun:
        """Retry only the failed/current model and keep its runner checkpoint."""
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.status != "failed":
                raise ValueError("Solo se puede reintentar una ejecución fallida")
            if run.cloud and run.cloud_node_id is not None:
                # Si su pod ya no está (se apagó al fallar), se pide otro.
                cloud = self._cloud()
                nodes = await cloud.list_nodes() if cloud is not None else []
                live = any(
                    item["id"] == str(run.cloud_node_id) and item["status"] != "terminated"
                    for item in nodes
                )
                if not live:
                    run.node = None
                    run.cloud_node_id = None
                    run.cloud_request_id = None
            elif run.cloud and run.node is None:
                run.cloud_request_id = None
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

    async def request_schedule_run(self, schedule_id: uuid.UUID) -> BenchmarkRun:
        """Queue a plan now unless its previous run is still active."""
        async with AsyncSessionLocal() as session:
            schedule = await session.scalar(
                select(BenchmarkSchedule)
                .where(BenchmarkSchedule.id == schedule_id)
                .with_for_update()
            )
            if schedule is None:
                raise LookupError("Plan de benchmark no encontrado")
            active = await session.scalar(
                select(BenchmarkRun.id)
                .where(
                    BenchmarkRun.schedule_id == schedule_id,
                    BenchmarkRun.status.not_in(_TERMINAL),
                )
                .limit(1)
            )
            if active is not None:
                raise ValueError("Este plan ya tiene una ejecución activa")
            now = datetime.now(UTC)
            runs = runs_from_schedule(schedule, now=now)
            run = runs[0]
            session.add_all(runs)
            schedule.last_run_at = now
            schedule.last_run_id = run.id
            await session.commit()
            await session.refresh(run)
        self.notify()
        return run

    async def request_diagnosis(
        self, attempt_id: uuid.UUID, options: dict[str, object]
    ) -> dict[str, object]:
        """Encola el diagnóstico de los fallos de un intento completado."""
        from ocabra.benchmarks.diagnosis.cases import EXTRACTORS

        async with AsyncSessionLocal() as session:
            attempt = await session.get(BenchmarkAttempt, attempt_id)
            if attempt is None:
                raise LookupError("Intento de benchmark no encontrado")
            run = await session.get(BenchmarkRun, attempt.run_id)
            if attempt.status != "completed" or run is None:
                raise ValueError("Solo se diagnostican intentos completados")
            if run.app not in EXTRACTORS:
                raise ValueError(f"El diagnóstico aún no sabe leer los casos de '{run.app}'")
            request = {
                **options,
                "attempt_id": str(attempt.id),
                "run_id": str(run.id),
                "app": run.app,
                "suite": run.suite,
                "model": attempt.model,
            }
        job = await asyncio.to_thread(self._diagnoses.create, request)
        self.notify()
        return job

    async def cancel_diagnosis(self, job_id: str) -> dict[str, object]:
        """Cancela un diagnóstico en cola o en curso."""
        job = await asyncio.to_thread(self._diagnoses.load, job_id)
        if job is None:
            raise LookupError("Diagnóstico no encontrado")
        if job["status"] == "queued":
            return await asyncio.to_thread(self._diagnoses.cancel_queued, job_id) or job
        runner = self._diagnosis_runner
        if job["status"] == "running" and runner is not None and runner.job_id == job_id:
            runner.cancel()
            return job
        raise ValueError("El diagnóstico ya terminó")

    async def delete_diagnosis(self, job_id: str) -> None:
        if not await asyncio.to_thread(self._diagnoses.delete, job_id):
            raise LookupError("Diagnóstico no encontrado")

    async def _purge_diagnoses(self) -> None:
        """Retención de los diagnósticos terminados, como mucho una vez por hora."""
        now = asyncio.get_running_loop().time()
        if self._last_diagnosis_purge and now - self._last_diagnosis_purge < 3600:
            return
        self._last_diagnosis_purge = now
        removed = await asyncio.to_thread(
            self._diagnoses.purge, settings.benchmark_diagnosis_retention_days
        )
        if removed:
            logger.info("benchmark_diagnoses_purged", removed=removed)

    async def request_baseline(self, run_id: uuid.UUID) -> BenchmarkRun:
        """Make one completed run the unique reference for its application suite."""
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.status != "completed":
                raise ValueError("Solo una ejecución completada puede ser referencia")
            await session.execute(
                sa.update(BenchmarkRun)
                .where(BenchmarkRun.app == run.app, BenchmarkRun.suite == run.suite)
                .values(is_baseline=False)
            )
            run.is_baseline = True
            await session.commit()
            await session.refresh(run)
            return run

    async def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._enqueue_due_schedules()
                await self._resume_after_quiet_period()
                started = await self._start_queued_runs()
                if started:
                    continue
                # Los diagnósticos repiten casos en este servidor: solo sin
                # nada local en cola ni midiendo.
                await self._purge_diagnoses()
                job = None
                if "" not in self._running and not await self._local_run_queued():
                    job = await asyncio.to_thread(self._diagnoses.next_queued)
                if job is not None and self._identity is not None:
                    runner = DiagnosisRunner(self._app, self._diagnoses, self._identity.raw_api_key)
                    self._diagnosis_runner = runner
                    try:
                        await runner.run(job)
                    finally:
                        self._diagnosis_runner = None
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

    async def _enqueue_due_schedules(self) -> None:
        """Materialize due plans once and advance their clock atomically."""
        now = datetime.now(UTC)
        queued = False
        async with AsyncSessionLocal() as session:
            schedules = (
                (
                    await session.execute(
                        select(BenchmarkSchedule)
                        .where(
                            BenchmarkSchedule.enabled.is_(True),
                            BenchmarkSchedule.next_run_at.is_not(None),
                            BenchmarkSchedule.next_run_at <= now,
                        )
                        .with_for_update(skip_locked=True)
                    )
                )
                .scalars()
                .all()
            )
            for schedule in schedules:
                schedule.next_run_at = next_schedule_occurrence(
                    list(schedule.weekdays), schedule.time_of_day, schedule.timezone, now
                )
                active = await session.scalar(
                    select(BenchmarkRun.id)
                    .where(
                        BenchmarkRun.schedule_id == schedule.id,
                        BenchmarkRun.status.not_in(_TERMINAL),
                    )
                    .limit(1)
                )
                if active is not None:
                    continue
                runs = runs_from_schedule(schedule, now=now)
                session.add_all(runs)
                schedule.last_run_at = now
                schedule.last_run_id = runs[0].id
                queued = True
            await session.commit()
        if queued:
            self.notify()

    async def _start_queued_runs(self) -> int:
        """Arranca la siguiente ejecución de cada máquina libre. Devuelve cuántas."""
        started = 0
        while True:
            picked = await self._next_queued(set(self._running))
            if picked is None:
                return started
            run_id, target = picked
            task = asyncio.create_task(
                self._execute_run_guarded(run_id), name=f"benchmark-run-{target or 'local'}"
            )
            self._running[target] = task
            task.add_done_callback(lambda _t, target=target: self._run_finished(target, _t))
            started += 1

    def _run_finished(self, target: str, task: asyncio.Task[None]) -> None:
        if self._running.get(target) is task:
            self._running.pop(target, None)
        self._wake.set()

    async def _execute_run_guarded(self, run_id: uuid.UUID) -> None:
        try:
            await self._execute_run(run_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — one broken run must not kill the queue
            logger.exception("benchmark_worker_error", run_id=str(run_id), error=str(exc))

    async def _next_queued(self, busy: set[str]) -> tuple[uuid.UUID, str] | None:
        """La ejecución en cola más antigua de una máquina que no esté midiendo ya."""
        async with AsyncSessionLocal() as session:
            query = (
                select(BenchmarkRun.id, BenchmarkRun.node)
                .where(
                    BenchmarkRun.status == "queued",
                    # Las que esperan su pod no ocupan la cola: las mueve
                    # _advance_provisioning hasta que el nodo está listo.
                    sa.or_(BenchmarkRun.cloud.is_(None), BenchmarkRun.node.is_not(None)),
                )
                .order_by(BenchmarkRun.created_at)
            )
            if "" in busy:
                query = query.where(BenchmarkRun.node.is_not(None))
            remote = [name for name in busy if name]
            if remote:
                query = query.where(
                    sa.or_(BenchmarkRun.node.is_(None), BenchmarkRun.node.not_in(remote))
                )
            row = (await session.execute(query.limit(1))).first()
            return (row[0], row[1] or "") if row else None

    async def _next_queued_run(self) -> uuid.UUID | None:
        picked = await self._next_queued(set())
        return picked[0] if picked else None

    async def _local_run_queued(self) -> bool:
        async with AsyncSessionLocal() as session:
            found = await session.scalar(
                select(BenchmarkRun.id)
                .where(
                    BenchmarkRun.status == "queued",
                    BenchmarkRun.node.is_(None),
                    BenchmarkRun.cloud.is_(None),
                )
                .limit(1)
            )
            return found is not None

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
            skipped = (run.cloud or {}).get("skipped") or {}
            if model in skipped:
                reason = f"no se pudo copiar al pod: {skipped[model]}"
                await session.commit()
                await self._model_failed(run_id, model, reason)
                return
            # En un nodo remoto el tráfico del hub no comparte hardware con la
            # medida: solo cuenta el que llega a ese nodo (se mira al terminar).
            if run.auto_pause and self._identity is not None and not run.node:
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

        node = await self._run_node(run_id)
        if not node and await ensure_deployment(self._app.state, model):
            # Un backend compatible que aquí no estaba desplegado.
            logger.info("benchmark_deployment_created", run_id=str(run_id), model=model)
        environment = await self._base_environment(run_id, source, app, suite, model)
        if node:
            environment["node"] = await self._node_description(node)
        sampling = await self._recommended_sampling(run_id, model)
        if sampling is not None:
            environment["sampling"] = {"mode": "recommended", **sampling}
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
        self._observed_sampling.pop(((node or "").lower(), model), None)
        if sampling is not None and self._identity is not None:
            self._sampling_overrides[str(run_id)] = {
                "profile": model,
                "params": sampling["params"],
                "api_key": self._identity.raw_api_key,
            }
        try:
            outcome = await self._run_process(
                run_id,
                command,
                source,
                log_path,
                detect_external=not node,
                progress_path=workspace / "progress.json",
                node=node,
            )
        finally:
            self._sampling_overrides.pop(str(run_id), None)
        finished_at = datetime.now(UTC)
        await self._set_phase(run_id, "finalizing")
        # StatsMiddleware records at response completion. Give its short DB
        # write a scheduling turn before checking for fast external requests
        # that may have started and ended between active-request polls.
        await asyncio.sleep(0.1)
        environment = await self._complete_environment(
            environment, started_at, finished_at, source, node=node
        )
        observed = self._sampling_summary(node, model)
        if observed:
            environment.setdefault("model", {})["requestSampling"] = observed
        historical = await self._historical_external_requests(started_at, finished_at, node=node)
        if node:
            peer_external, verified = await self._node_local_requests(node, started_at, finished_at)
            historical.extend(peer_external)
            hardware = environment.get("hardware")
            if isinstance(hardware, dict):
                hardware["nodeTrafficVerified"] = verified
        external = self._deduplicate_external([*outcome.external_requests, *historical])

        if outcome.interruption:
            await self._finish_interrupted_attempt(attempt.id, outcome, external, environment)
            return
        served_elsewhere = self._served_elsewhere(environment, node)
        if served_elsewhere and outcome.return_code == 0:
            message = (
                f"Se pidió medir en '{node or 'local'}' pero las peticiones se sirvieron en "
                f"{', '.join(served_elsewhere)}: el resultado no corresponde a ese nodo"
            )
            await self._finish_failed_attempt(attempt.id, outcome.return_code, message, environment)
            await self._model_failed(run_id, model, message)
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
            await self._model_failed(run_id, model, message)
            return
        failed_requests = await self._measured_failures(started_at, finished_at, model, node=node)
        if failed_requests is not None:
            await self._finish_failed_attempt(
                attempt.id, outcome.return_code, failed_requests, environment
            )
            await self._model_failed(run_id, model, failed_requests)
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
        progress_path: Path | None = None,
        node: str | None = None,
    ) -> ProcessOutcome:
        env = dict(os.environ)
        assert self._identity is not None
        env.update(
            {
                "OPENAI_COMPATIBLE_URL": node_base_url(settings.benchmark_base_url, node),
                "OPENAI_COMPATIBLE_KEY": self._identity.raw_api_key,
                "UV_CACHE_DIR": str(self._root / "uv" / "cache"),
                "UV_PYTHON_INSTALL_DIR": str(self._root / "uv" / "python"),
            }
        )
        if await self._run_app(run_id) == "corrala":
            env["CORRALA_BENCH_DATABASE_URL"] = await self._corrala_database_url(node or "")
        if progress_path is not None:
            env["OCABRA_BENCHMARK_PROGRESS_PATH"] = str(progress_path)
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
                self._processes.add(process)
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
                    self._processes.discard(process)
        if interruption is None and self._stop.is_set():
            # El api se está apagando y ha terminado el runner (código -15):
            # no es un fallo del modelo; al arrancar se retoma desde su punto.
            interruption = "shutdown"
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
                "git",
                "-C",
                str(source),
                "sparse-checkout",
                "set",
                *app.checkout_paths,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
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
        """Termina *process*, o todos los runners vivos si no se indica."""
        if process is None:
            async with self._process_lock:
                alive = list(self._processes)
            for item in alive:
                await self._terminate_process(item)
            return
        if process.returncode is not None:
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

    async def _measured_failures(
        self, started_at: datetime, finished_at: datetime, model: str, *, node: str | None = None
    ) -> str | None:
        """Por qué el resultado no vale si el modelo medido no llegó a responder.

        Algunos runners (Escriba) anotan cada petición fallida como un caso
        más y terminan bien: un modelo que no cabía en la GPU (409
        ``insufficient_vram`` en cada petición) salía como «completado» con
        una nota que no mide nada. Si al menos la mitad de las peticiones del
        benchmark a ese modelo fallaron, el intento falla con el error más
        repetido. Los pasajeros que el runner reintenta (429, 502–504) no
        cuentan.
        """
        if self._identity is None:
            return None
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(
                        RequestStat.status_code,
                        RequestStat.error,
                        RequestStat.federation_route,
                        RequestStat.remote_node_name,
                    ).where(
                        RequestStat.user_id == self._identity.user_id,
                        RequestStat.started_at >= started_at,
                        RequestStat.started_at <= finished_at,
                        sa.or_(
                            RequestStat.requested_model == model,
                            RequestStat.model_id == model,
                        ),
                    )
                )
            ).all()
        counted = [
            (row.status_code, row.error)
            for row in rows
            if row.status_code not in _TRANSIENT_STATUS and belongs_to_run(row, node)
        ]
        failed = [(code, error) for code, error in counted if code is not None and code >= 400]
        if not failed or len(failed) * 2 < len(counted):
            return None
        reasons: dict[str, int] = {}
        for code, error in failed:
            key = f"{code} {(error or '').strip()[:200]}".strip()
            reasons[key] = reasons.get(key, 0) + 1
        top = max(reasons, key=reasons.get)
        return (
            f"El modelo no respondió en {len(failed)} de {len(counted)} peticiones del "
            f"benchmark; el resultado no lo mide. Error más repetido: {top}"
        )

    async def _historical_external_requests(
        self, started_at: datetime, finished_at: datetime, *, node: str | None = None
    ) -> list[dict[str, object]]:
        """Peticiones ajenas que coincidieron con la medida en el mismo hardware.

        En local, cualquier petición de otro usuario en este nodo. En un nodo
        remoto, solo las que este hub le mandó a ese nodo (las de la GPU local
        no comparten hardware con la medida).
        """
        assert self._identity is not None
        conditions = [
            RequestStat.started_at >= started_at - timedelta(hours=1),
            RequestStat.started_at <= finished_at,
            sa.or_(
                RequestStat.user_id.is_(None),
                RequestStat.user_id != self._identity.user_id,
            ),
        ]
        if node:
            conditions.append(sa.func.lower(RequestStat.remote_node_name) == node.lower())
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(select(RequestStat).where(*conditions))).scalars()
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
        *,
        node: str | None = None,
    ) -> dict[str, object]:
        """Attach the backends, nodes and physical GPUs observed in benchmark requests."""
        assert self._identity is not None
        async with AsyncSessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(RequestStat).where(
                            RequestStat.user_id == self._identity.user_id,
                            RequestStat.started_at >= started_at - timedelta(hours=1),
                            RequestStat.started_at <= finished_at,
                        )
                    )
                )
                .scalars()
                .all()
            )
        rows = [
            row
            for row in rows
            if row.started_at + timedelta(milliseconds=row.duration_ms or 0) >= started_at
        ]
        if not node:
            rows = [row for row in rows if belongs_to_run(row, None)]
        auxiliary_nodes: list[str] = []
        if node:
            # En un nodo remoto se describe el modelo medido. Los auxiliares
            # que fija la suite (el reranker de Escriba, jueces…) se sirven
            # donde estén y solo se anotan: no son lo que se mide.
            profile = str((base.get("model") or {}).get("profile") or "")
            measured = [
                row
                for row in rows
                if (row.requested_model or row.model_id) == profile and belongs_to_run(row, node)
            ]
            auxiliary_nodes = sorted(
                {
                    row.remote_node_name or "local"
                    for row in rows
                    if (row.requested_model or row.model_id) != profile
                }
            )
            rows = measured
        gpu_manager = getattr(self._app.state, "gpu_manager", None)
        gpu_states = {}
        if node:
            # Los índices de GPU de las filas son los del nodo, no los nuestros.
            peer = self._peer(node)
            for gpu in getattr(peer, "gpus", None) or []:
                if isinstance(gpu, dict) and isinstance(gpu.get("index"), int):
                    gpu_states[gpu["index"]] = SimpleNamespace(
                        name=gpu.get("name"), total_vram_mb=gpu.get("total_vram_mb")
                    )
        elif gpu_manager is not None:
            states = await gpu_manager.get_all_states()
            gpu_states = {state.index: state for state in states}
        observed = execution_observations(rows, gpu_states)
        node_side: dict[str, object] | None = None
        if node:
            profile = str((base.get("model") or {}).get("profile") or "")
            node_side = await self._node_served(node, profile, started_at, finished_at, gpu_states)
            if node_side.get("rows"):
                # Lo que dice el propio nodo manda: las filas del hub toman el
                # backend y las GPU del estado local del hub, no del nodo.
                observed["model"]["backends"] = node_side["backends"]
                observed["hardware"]["gpus"] = node_side["gpus"]
                observed["hardware"]["source"] = "node"
            else:
                observed["hardware"]["source"] = "hub"
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
        if node:
            software["backends"] = await self._node_backend_versions(
                node, list(model_observations["backends"])
            )
            details = await self._node_details(node)
            node_env = (
                dict(environment.get("node") or {})
                if isinstance(environment.get("node"), dict)
                else {}
            )
            node_env.update({key: value for key, value in details.items() if value is not None})
            environment["node"] = node_env
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
        if auxiliary_nodes:
            environment["hardware"]["auxiliaryNodes"] = auxiliary_nodes
        environment["requests"] = observed["requests"]
        profile_name = str((environment.get("model") or {}).get("profile") or "")
        snapshot = await self._model_snapshot(profile_name, node) if profile_name else None
        if snapshot is not None:
            environment["model"]["snapshot"] = snapshot
        return environment

    async def _model_snapshot(self, name: str, node: str | None) -> dict | None:
        """Configuración, presets, muestreo y fichero del modelo donde se midió."""
        from ocabra.benchmarks import model_snapshot

        try:
            if node:
                peer = self._peer(node)
                if peer is None:
                    return None
                return await model_snapshot.capture(lambda path: self._peer_json(peer, path), name)
            if self._identity is None:
                return None
            import httpx

            base = settings.benchmark_base_url.rstrip("/")
            headers = {"Authorization": f"Bearer {self._identity.raw_api_key}"}
            async with httpx.AsyncClient(timeout=30.0) as client:

                async def get_json(path: str):
                    response = await client.get(f"{base}{path}", headers=headers)
                    return response.json() if response.status_code == 200 else None

                return await model_snapshot.capture(get_json, name)
        except Exception as exc:  # noqa: BLE001 — la foto es un extra, no tumba el intento
            logger.warning("benchmark_model_snapshot_failed", model=name, error=str(exc))
            return None

    @staticmethod
    def _deduplicate_external(items: list[dict[str, object]]) -> list[dict[str, object]]:
        return list({str(item.get("requestId")): item for item in items}.values())

    # ── Nodos de la federación ──────────────────────────────────────────
    def _peer(self, node: str):
        manager = getattr(self._app.state, "federation_manager", None)
        if manager is None:
            return None
        wanted = node.lower()
        return next(
            (p for p in manager.get_all_peers() if wanted in {p.name.lower(), str(p.peer_id)}),
            None,
        )

    async def _recommended_sampling(self, run_id: uuid.UUID, model: str) -> dict | None:
        """Muestreo del fabricante para ``model`` si la ejecución lo pide.

        Sin ficha oficial en Hugging Face no hay con qué comparar: falla en vez
        de medir con el muestreo de siempre y etiquetarlo como recomendado.
        """
        async with AsyncSessionLocal() as session:
            mode = getattr(await self._get_run(session, run_id), "sampling", None)
        if mode != "recommended":
            return None
        from ocabra.benchmarks.diagnosis.reference import load_reference, resolve_reference_repo

        state_obj = self._app.state
        profile = await bench_profile(state_obj.profile_registry, model)
        state = await state_obj.model_manager.get_state(profile.base_model_id) if profile else None
        repo = resolve_reference_repo(
            str(getattr(state, "model_key", None) or model),
            getattr(state, "artifact", None),
            getattr(state, "extra_config", None),
        )
        reference = await load_reference(repo)
        if reference.error or not reference.sampling:
            raise RuntimeError(
                f"Sin muestreo del fabricante para {model}: {reference.error or 'ficha vacía'}. "
                "Fija extra_config.diagnosis.reference_repo o mide con el del benchmark."
            )
        return {"reference": reference.repo, "params": reference.sampling}

    async def _run_node(self, run_id: uuid.UUID) -> str | None:
        async with AsyncSessionLocal() as session:
            return (await self._get_run(session, run_id)).node

    async def _node_description(self, node: str) -> dict[str, object]:
        """Nodo, versión de oCabra y GPU del nodo remoto donde se mide."""
        peer = self._peer(node)
        description: dict[str, object] = {
            "name": node,
            "online": bool(getattr(peer, "online", False)),
            "gpus": [
                {
                    "index": g.get("index"),
                    "name": g.get("name"),
                    "totalVramMb": g.get("total_vram_mb"),
                }
                for g in (getattr(peer, "gpus", None) or [])
                if isinstance(g, dict)
            ],
        }
        payload = await self._peer_json(peer, "/health")
        if isinstance(payload, dict) and payload.get("version"):
            description["ocabra"] = payload["version"]
        return description

    async def _node_served(
        self,
        node: str,
        profile: str,
        started_at: datetime,
        finished_at: datetime,
        gpu_states: dict[int, object],
    ) -> dict[str, object]:
        """Backend y GPU con los que el nodo atendió las peticiones del modelo medido.

        Se leen de las estadísticas del propio nodo (las que le llegaron por la
        federación desde aquí), que son las únicas que conocen su worker.
        """
        payload = await self._peer_json(self._peer(node), "/ocabra/stats/recent?limit=200")
        rows = payload.get("requests") if isinstance(payload, dict) else None
        backends: set[str] = set()
        indices: set[int] = set()
        count = 0
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not row.get("inboundFrom"):
                continue
            if profile and (row.get("requestedModel") or row.get("modelId")) != profile:
                continue
            try:
                started = datetime.fromisoformat(str(row.get("startedAt")))
            except ValueError:
                continue
            ended = started + timedelta(milliseconds=int(row.get("durationMs") or 0))
            if ended < started_at or started > finished_at:
                continue
            count += 1
            if row.get("backendType"):
                backends.add(str(row["backendType"]))
            telemetry = (
                row.get("runtimeTelemetry") if isinstance(row.get("runtimeTelemetry"), dict) else {}
            )
            for index in telemetry.get("gpu_indices") or []:
                if isinstance(index, int):
                    indices.add(index)
        inferred = False
        if count and not indices and len(gpu_states) == 1:
            # Nodos antiguos no publican los índices; con una sola GPU no hay duda.
            indices = set(gpu_states)
            inferred = True
        gpus = [
            {
                "index": index,
                "name": getattr(gpu_states.get(index), "name", None),
                "totalVramMb": getattr(gpu_states.get(index), "total_vram_mb", None),
                **({"inferred": True} if inferred else {}),
            }
            for index in sorted(indices)
        ]
        return {"rows": count, "backends": sorted(backends), "gpus": gpus}

    async def _node_backend_versions(
        self, node: str, backend_types: list[str]
    ) -> list[dict[str, object]]:
        """Versiones de los backends instalados en el nodo (``/ocabra/backends``)."""
        payload = await self._peer_json(self._peer(node), "/ocabra/backends")
        states = {
            str(item.get("backend_type")): item
            for item in (payload if isinstance(payload, list) else [])
            if isinstance(item, dict)
        }
        modules = []
        for backend_type in backend_types:
            state = states.get(backend_type) or {}
            modules.append(
                {
                    "type": backend_type,
                    "installedVersion": state.get("installed_version"),
                    "packages": dict(state.get("packages") or {}),
                    "ociDigest": state.get("oci_digest"),
                    "gitRef": state.get("git_ref"),
                    "sourceCommit": None,
                    "installSource": state.get("install_source"),
                    "node": node,
                    "verified": bool(state),
                }
            )
        return modules

    async def _node_details(self, node: str) -> dict[str, object]:
        """Driver/CUDA, versión y, si es un pod, sus datos de nube."""
        peer = self._peer(node)
        details: dict[str, object] = {}
        cuda = getattr(peer, "cuda", None)
        if isinstance(cuda, dict):
            details["cuda"] = {
                key: cuda.get(key)
                for key in ("ok", "driver_cuda", "devices", "error")
                if key in cuda
            }
        payload = await self._peer_json(peer, "/health")
        if isinstance(payload, dict) and payload.get("version"):
            details["ocabraAtEnd"] = payload["version"]
        peer_id = getattr(peer, "peer_id", None)
        if peer_id:
            details["peerId"] = str(peer_id)
            details["cloud"] = await self._cloud_metadata(str(peer_id))
        return details

    async def _cloud_metadata(self, peer_id: str) -> dict[str, object] | None:
        from ocabra.db.cloud import CloudNode

        try:
            async with AsyncSessionLocal() as session:
                row = await session.scalar(
                    select(CloudNode).where(CloudNode.peer_id == uuid.UUID(peer_id)).limit(1)
                )
        except Exception:  # noqa: BLE001 — un peer que no es pod
            return None
        if row is None:
            return None
        return {
            "provider": row.provider,
            "podId": row.provider_pod_id,
            "gpuTypeId": row.gpu_type_id,
            "gpuCount": row.gpu_count,
            "cloudType": row.cloud_type,
            "image": row.image,
            "costPerHr": row.cost_per_hr,
            "dataCenterId": row.data_center_id,
            "cloudNodeId": str(row.id),
            # Cuánto tardó en estar listo (espera, arranque, conexión, preparación).
            "readiness": cloud_timeline.summary(row.timeline) if row.timeline else None,
        }

    async def _peer_json(self, peer, path: str) -> object | None:
        manager = getattr(self._app.state, "federation_manager", None)
        if manager is None or peer is None:
            return None
        try:
            response = await manager.proxy_raw(
                peer, "GET", path, b"", {"accept": "application/json"}
            )
            try:
                if response.status_code != 200:
                    return None
                return json.loads(await response.aread())
            finally:
                await response.aclose()
        except Exception:  # noqa: BLE001 — informativo
            return None

    async def _node_local_requests(
        self, node: str, started_at: datetime, finished_at: datetime
    ) -> tuple[list[dict[str, object]], bool]:
        """Tráfico que el nodo atendió por su cuenta durante la medida.

        Lo que el hub le mandó ya está en nuestras estadísticas; aquí se leen
        las peticiones recientes del propio nodo que no llegaron por la
        federación. ``verified`` es False si no se pudieron leer (nodo con
        acceso solo de inferencia o caído): la medida no queda limpia ni sucia,
        queda sin comprobar.
        """
        peer = self._peer(node)
        payload = await self._peer_json(peer, "/ocabra/stats/recent?limit=200")
        rows = payload.get("requests") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return [], False
        external: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, dict) or row.get("inboundFrom"):
                continue
            try:
                started = datetime.fromisoformat(str(row.get("startedAt")))
            except ValueError:
                continue
            ended = started + timedelta(milliseconds=int(row.get("durationMs") or 0))
            if ended < started_at or started > finished_at:
                continue
            external.append(
                {
                    "requestId": f"{node}:{row.get('id')}",
                    "model": row.get("requestedModel") or row.get("modelId"),
                    "apiKeyName": row.get("apiKeyName"),
                    "username": row.get("username"),
                    "startedAt": started.isoformat(),
                    "source": f"node:{node}",
                }
            )
        return external, True

    @staticmethod
    def _served_elsewhere(environment: dict[str, object], node: str | None) -> list[str]:
        """Nodos donde se sirvió la medida que no son el pedido.

        Solo con un nodo pedido: un benchmark local sigue el reparto normal de
        la federación (un modelo que solo tiene un peer se mide allí), como
        hasta ahora, y su entorno ya dice dónde se sirvió.
        """
        if not node:
            return []
        hardware = environment.get("hardware")
        nodes = hardware.get("nodes") if isinstance(hardware, dict) else None
        if not isinstance(nodes, list):
            return []
        if not nodes:
            # Ninguna petición del modelo medido: no hay nada que lo respalde.
            return ["ninguna parte (no hay peticiones del modelo medido)"]
        wanted = node.lower()
        return [str(n) for n in nodes if str(n).lower() != wanted]

    async def _corrala_database_url(self, target: str = "") -> str:
        """La base de pruebas de Corrala para la máquina *target* («» = aquí).

        Corrala vacía y rellena su base en cada medida: dos ejecuciones a la vez
        (en máquinas distintas) sobre la misma se borraban los asistentes la
        una a la otra (``conversations_assistant_id_fkey``). Cada máquina que
        mide a la vez recibe un hueco propio (``…_bench``, ``…_bench_2``…), que
        se reutiliza en cuanto queda libre.
        """
        url = settings.benchmark_corrala_database_url.strip()
        if not url:
            url = corrala_benchmark_database_url(settings.database_url)
        busy = {
            slot
            for other, slot in self._corrala_slots.items()
            if other != target and other in self._running
        }
        slot = self._corrala_slots.get(target)
        if slot is None or slot in busy:
            slot = next(i for i in range(len(busy) + 1) if i not in busy)
        self._corrala_slots[target] = slot
        if slot:
            from sqlalchemy.engine import make_url

            parsed = make_url(url)
            url = parsed.set(database=f"{parsed.database}_{slot + 1}").render_as_string(
                hide_password=False
            )
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
                sa.update(BenchmarkRun)
                .where(BenchmarkRun.id == run_id, BenchmarkRun.phase != phase)
                .values(phase=phase)
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
        if reason == "shutdown":
            return  # Sigue «running»: el arranque siguiente la vuelve a la cola.
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
                attempt.error = (
                    "API reiniciada durante el intento; se reanudará"
                    if outcome.interruption == "shutdown"
                    else f"Interrumpido: {outcome.interruption}"
                )
                attempt.finished_at = datetime.now(UTC)
                await session.commit()
        run_id = await self._attempt_run_id(attempt_id)
        if outcome.interruption == "shutdown":
            return  # Sigue «running»: el arranque siguiente la vuelve a la cola.
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
        await self._release_cloud(run_id)

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
                if run.cloud:
                    run.node = None  # El siguiente modelo puede seguir descargándose.
            await session.commit()
            finished = run.status == "completed"
        await self._free_pod_models(run_id)
        if finished:
            await self._release_cloud(run_id)
        self.notify()

    async def _model_failed(self, run_id: uuid.UUID, model: str, error: str) -> None:
        """Un modelo que falla no tumba a los que le siguen en la misma ejecución.

        Se anota en ``run.error`` y se pasa al siguiente. Solo falla la ejecución
        entera si era el último y ninguno ha dado resultado: así «Reintentar»
        sigue retomando ese modelo con su checkpoint.
        """
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            last = run.current_model_index + 1 >= len(run.models)
            if last and not run.result:
                failed_before = run.error
            else:
                note = f"{model}: {error}"
                run.error = f"{run.error}\n{note}"[:4000] if run.error else note[:4000]
                run.current_model_index += 1
                run.current_model = None
                if last:
                    run.status = "completed"
                    run.phase = "completed"
                    run.finished_at = datetime.now(UTC)
                else:
                    run.status = "queued"
                    run.phase = "queued"
                    if run.cloud:
                        run.node = None
                await session.commit()
                await self._free_pod_models(run_id)
                if last:
                    await self._release_cloud(run_id)
                self.notify()
                return
        message = f"{failed_before}\n{model}: {error}" if failed_before else error
        await self._fail_run(run_id, message)

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
        await self._release_cloud(run_id)

    # ── Pod propio ──────────────────────────────────────────────────────

    def _cloud(self):
        federation = getattr(self._app.state, "federation_manager", None)
        return getattr(federation, "cloud", None)

    async def _provision_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._sync_reserved_pods()
                await self._advance_provisioning()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — el bucle no se cae por una ejecución
                logger.exception("benchmark_provision_loop_error", error=str(exc))
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=15)

    async def _sync_reserved_pods(self) -> None:
        """Aparta del reparto automático los pods de benchmarks que no han terminado.

        Un pod con modelos cargados era un destino más para la federación:
        tráfico de usuarios acababa en la máquina alquilada, contaminando la
        medida (y mandando sus datos a ese pod). Reservado, solo le llega lo
        que se le manda por ``/_node/<pod>/``, que es como mide el runner.
        """
        fm = getattr(self._app.state, "federation_manager", None)
        cloud = getattr(fm, "cloud", None)
        if fm is None or cloud is None or not hasattr(fm, "reserved_peers"):
            return
        async with AsyncSessionLocal() as session:
            node_ids = {
                str(node_id)
                for node_id in (
                    await session.execute(
                        select(BenchmarkRun.cloud_node_id).where(
                            BenchmarkRun.cloud_node_id.is_not(None),
                            BenchmarkRun.status.not_in(_TERMINAL),
                        )
                    )
                ).scalars()
            }
        names: set[str] = set()
        if node_ids:
            for node in await cloud.list_nodes():
                if str(node.get("id")) in node_ids and node.get("peer_name"):
                    names.add(str(node["peer_name"]).lower())
        if names != fm.reserved_peers:
            logger.info("benchmark_pods_reserved", pods=sorted(names))
        fm.reserved_peers = names

    async def _advance_provisioning(self) -> None:
        """Mueve las ejecuciones que esperan su pod: pedirlo, esperarlo, darlo por listo."""
        async with self._provision_lock:
            await self._advance_provisioning_locked()

    async def _advance_provisioning_locked(self) -> None:
        async with AsyncSessionLocal() as session:
            run_ids = (
                (
                    await session.execute(
                        select(BenchmarkRun.id).where(
                            BenchmarkRun.status == "queued",
                            BenchmarkRun.cloud.is_not(None),
                            BenchmarkRun.node.is_(None),
                        )
                    )
                )
                .scalars()
                .all()
            )
        for run_id in run_ids:
            try:
                await self._provision(run_id)
            except Exception as exc:  # noqa: BLE001 — se informa en la ejecución
                logger.exception("benchmark_provisioning_failed", run_id=str(run_id))
                await self._fail_run(run_id, f"No se pudo preparar el pod: {exc}")

    async def _provision(self, run_id: uuid.UUID) -> None:
        cloud = self._cloud()
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            spec = dict(run.cloud or {})
            request_id = run.cloud_request_id
            models = list(run.models)
            cursor = run.current_model_index
            waiting_since = run.updated_at if cursor > 0 else None
        if cloud is None:
            await self._fail_run(run_id, "La nube no está disponible (federación apagada)")
            return
        if request_id is None:
            retry_after = spec.get("retry_after")
            if retry_after and datetime.fromisoformat(str(retry_after)) > datetime.now(UTC):
                # A la anterior le tocó una GPU averiada: se espera un poco.
                await self._set_phase(run_id, "waiting_capacity")
                return
            sibling = await self._batch_request(run_id)
            if sibling is None:
                await self._request_pod(run_id, cloud, spec, models)
                return
            # Otra prueba de la misma batería ya pidió la máquina: se comparte.
            async with AsyncSessionLocal() as session:
                run = await self._get_run(session, run_id)
                run.cloud_request_id = sibling
                await session.commit()
            request_id = sibling
        request = await cloud.requests.get(str(request_id))
        if request["status"] == "waiting":
            await self._set_phase(run_id, "waiting_capacity")
            return
        if request["status"] != "launched":
            reason = request.get("lastError") or request["status"]
            await self._fail_run(run_id, f"No hubo máquina para el benchmark: {reason}")
            return
        node_id = request["cloudNodeId"]
        node = next((item for item in await cloud.list_nodes() if item["id"] == node_id), None)
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            run.cloud_node_id = uuid.UUID(node_id)
            await session.commit()
        if node is None or node["status"] in ("terminated", "error"):
            if node is not None and await self._retry_broken_pod(run_id, spec, node):
                return
            await self._fail_run(
                run_id, f"El pod se cayó antes de medir: {(node or {}).get('error') or 'terminado'}"
            )
            return
        items = node.get("preload_models") or []
        selected = cursor
        if any(item.get("benchmark_prefetch") for item in items):
            # Cada perfil puede ser un alias del mismo modelo base. Solo la
            # disponibilidad de un modelo pendiente de ESTA prueba la desbloquea.
            by_id = {item["model_id"]: item for item in items}
            prerequisites = [item for item in items if item.get("kind") == "backend"]
            relevant = []
            relevant_index: list[int] = []
            skipped = spec.get("skipped") or {}
            for index in range(cursor, len(models)):
                if models[index] in skipped:
                    continue  # No va al pod: se da por fallido al llegarle el turno.
                item = by_id.get(await self._pod_model_id(models[index]) or "")
                if item is None:
                    await self._fail_run(run_id, f"Falta la precarga de {models[index]}")
                    return
                if item["status"] == "removed":
                    # Ya se borró al acabar sus pruebas (un «Reintentar» tardío).
                    item = {**item, "status": "error", "error": "ya se borró del pod"}
                relevant.append(item)
                relevant_index.append(index)
                if item["status"] in ("ready", "loaded"):
                    selected = index
                    relevant = [item]
                    break
            # Un fallo ajeno no impide medir los modelos aún en preparación.
            waiting = [i for i in relevant if i["status"] != "error"]
            if relevant and not waiting:
                # Lo que queda no se pudo preparar: el primero se da por
                # fallido con su motivo y se sigue con el siguiente, en vez de
                # tumbar la ejecución entera (y la medida de los demás).
                await self._fail_unprepared_model(
                    run_id, models, cursor, relevant_index[0], relevant[0]
                )
                return
            items = prerequisites + (waiting or relevant)
        broken = [item for item in items if item.get("status") == "error"]
        if broken:
            detail = "; ".join(f"{i.get('model_id')}: {i.get('error')}" for i in broken)[:1000]
            await self._fail_run(run_id, f"No se pudo preparar el pod: {detail}")
            return
        pending = [item for item in items if item.get("status") not in ("loaded", "ready")]
        if node.get("peer_online") and node.get("peer_name") and not pending:
            async with AsyncSessionLocal() as session:
                run = await self._get_run(session, run_id)
                if run.status != "queued" or run.node is not None:
                    return
                if selected != cursor:
                    # El cursor y los checkpoints siguen apuntando al modelo medido.
                    models.insert(cursor, models.pop(selected))
                    run.models = models
                run.node = node["peer_name"]
                run.phase = "queued"
                await session.commit()
            logger.info("benchmark_pod_ready", run_id=str(run_id), node=node["peer_name"])
            self.notify()
            return
        launched_at = request.get("finishedAt")
        timeout = int(spec.get("ready_timeout_minutes") or 90)
        since = waiting_since or (datetime.fromisoformat(launched_at) if launched_at else None)
        if since and datetime.now(UTC) - since > timedelta(minutes=timeout):
            if waiting_since is not None and cursor < len(models):
                # Ya se midió algo en este pod: el que no llega falla solo y se
                # sigue con los demás (antes se perdía el resto de la batería).
                await self._fail_unprepared_model(
                    run_id,
                    models,
                    cursor,
                    cursor,
                    {"error": f"no estuvo listo en {timeout} min"},
                )
                return
            await self._fail_run(run_id, f"El pod no estuvo listo en {timeout} min")
            return
        await self._set_phase(run_id, "preparing_pod")

    async def _batch_request(self, run_id: uuid.UUID) -> uuid.UUID | None:
        """La petición de máquina que ya hizo otra prueba de la misma batería."""
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.batch_id is None:
                return None
            return await session.scalar(
                select(BenchmarkRun.cloud_request_id)
                .where(
                    BenchmarkRun.batch_id == run.batch_id,
                    BenchmarkRun.id != run.id,
                    BenchmarkRun.cloud_request_id.is_not(None),
                )
                .limit(1)
            )

    async def _batch_models(self, run_id: uuid.UUID, models: list[str]) -> list[str]:
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.batch_id is None:
                return models
            rows = (
                await session.execute(
                    select(BenchmarkRun.models).where(BenchmarkRun.batch_id == run.batch_id)
                )
            ).scalars()
            merged = list(models)
            for listed in rows:
                for model in listed or []:
                    if model not in merged:
                        merged.append(model)
            return merged

    async def _request_pod(self, run_id: uuid.UUID, cloud, spec: dict, models: list[str]) -> None:
        """Pide la máquina: los modelos de toda la batería se replican en el pod."""
        registry = self._app.state.profile_registry
        base_ids: list[str] = []
        bench_of: dict[str, list[str]] = {}
        for model in await self._batch_models(run_id, models):
            profile = await bench_profile(registry, model)
            base_id = profile.base_model_id if profile is not None else None
            if base_id is None:
                # Un backend aún sin desplegar aquí: el pod lo monta con él.
                try:
                    pending = await pending_deployment(model)
                except ValueError as exc:
                    await self._fail_run(run_id, str(exc))
                    return
                base_id = pending.model_id if pending is not None else None
            if base_id is None:
                await self._fail_run(run_id, f"Perfil desconocido: {model}")
                return
            if base_id not in base_ids:
                base_ids.append(base_id)
            bench_of.setdefault(base_id, []).append(model)
        # Lo que no se puede copiar al pod (ficheros solo de aquí, un backend
        # que el pod no sabe montar) no tumba la batería: se mide lo demás y
        # ese modelo queda fallido con su motivo.
        check = getattr(cloud, "replication_problems", None)
        problems = await check(base_ids) if check is not None else {}
        if problems and len(problems) == len(base_ids):
            await self._fail_run(
                run_id, "No hubo máquina para el benchmark: " + "; ".join(problems.values())
            )
            return
        skipped = {
            model: reason for base, reason in problems.items() for model in bench_of.get(base, [])
        }
        base_ids = [base for base in base_ids if base not in problems]
        if skipped:
            await self._mark_skipped(run_id, skipped)
        async with AsyncSessionLocal() as session:
            app = (await self._get_run(session, run_id)).app
        launch = {
            "gpu_count": int(spec.get("gpu_count") or 1),
            "cloud_type": spec.get("cloud_type"),
            "name": f"bench-{app}"[:40],
            "models": base_ids,
            "benchmark_prefetch": len(base_ids) > 1,
            "recipes": list(spec.get("recipes") or []),
            "backends": list(spec.get("backends") or []),
            "volume_id": spec.get("volume_id"),
            "privacy_mode": spec.get("privacy_mode") or "functional",
            "max_runtime_minutes": int(spec.get("max_runtime_minutes") or 480),
            "container_disk_gb": spec.get("container_disk_gb"),
            # Entre que queda listo y la primera petición no hay tráfico: margen.
            "idle_shutdown_minutes": 60,
        }
        request = await cloud.requests.create(
            gpu_type_ids=list(spec.get("gpu_type_ids") or []),
            launch=launch,
            wait_minutes=int(spec.get("wait_minutes") or 360),
            retry_interval_s=120,
            purpose="benchmark",
            benchmark_run_id=run_id,
        )
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            run.cloud_request_id = uuid.UUID(request["id"])
            if request.get("cloudNodeId"):
                run.cloud_node_id = uuid.UUID(request["cloudNodeId"])
            run.phase = "preparing_pod" if request["status"] == "launched" else "waiting_capacity"
            await session.commit()
        if request["status"] in ("failed", "expired", "cancelled"):
            await self._fail_run(
                run_id, f"No hubo máquina para el benchmark: {request.get('lastError')}"
            )
        self.notify()

    async def _pod_model_id(self, model: str) -> str | None:
        """El modelo nuestro (o el despliegue aún por crear) que el pod monta para *model*."""
        profile = await bench_profile(self._app.state.profile_registry, model)
        if profile is not None and profile.base_model_id:
            return profile.base_model_id
        try:
            pending = await pending_deployment(model)
        except ValueError:
            return None
        return pending.model_id if pending is not None else None

    async def _free_pod_models(self, run_id: uuid.UUID) -> None:
        """Borra del pod los modelos que ya no necesita ninguna prueba de la batería.

        El disco del pod no da para todos los modelos de una batería grande:
        en cuanto un modelo ha pasado por todas sus pruebas se borra (con sus
        ficheros) y deja sitio a las descargas de los siguientes. Se protege
        por modelo, no por despliegue: ``llamacpp_mks/x`` y ``ollama/x`` leen
        el mismo fichero, así que no se borra ninguno mientras quede el otro.
        """
        try:
            async with AsyncSessionLocal() as session:
                run = await self._get_run(session, run_id)
                if not run.cloud or run.cloud_node_id is None:
                    return
                node_id = run.cloud_node_id
                mates = select(BenchmarkRun).where(BenchmarkRun.cloud_node_id == node_id)
                if run.batch_id is not None:
                    mates = select(BenchmarkRun).where(
                        sa.or_(
                            BenchmarkRun.cloud_node_id == node_id,
                            BenchmarkRun.batch_id == run.batch_id,
                        )
                    )
                rows = list((await session.execute(mates)).scalars())
            pending: set[str] = set()
            done: set[str] = set()
            for row in rows:
                models = list(row.models or [])
                cut = len(models) if row.status in _TERMINAL else row.current_model_index
                done.update(models[:cut])
                pending.update(models[cut:])
            pending_ids = {mid for mid in [await self._pod_model_id(m) for m in pending] if mid}
            busy_keys = {mid.split("/", 1)[-1] for mid in pending_ids}
            to_free: list[str] = []
            for model in sorted(done - pending):
                mid = await self._pod_model_id(model)
                if (
                    mid
                    and mid not in pending_ids
                    and mid.split("/", 1)[-1] not in busy_keys
                    and (str(node_id), mid) not in self._freed_pod_models
                    and mid not in to_free
                ):
                    to_free.append(mid)
            if not to_free:
                return
            peer = await self._cloud_peer(str(node_id))
            if peer is None:
                return
            from urllib.parse import quote

            fm = self._app.state.federation_manager
            for mid in to_free:
                response = await fm.proxy_raw(
                    peer, "DELETE", f"/ocabra/models/{quote(mid, safe='/:')}?delete_files=true", b""
                )
                try:
                    await response.aread()
                finally:
                    await response.aclose()
                if response.status_code < 400 or response.status_code == 404:
                    self._freed_pod_models.add((str(node_id), mid))
                    mark = getattr(getattr(fm, "cloud", None), "mark_preload_removed", None)
                    if callable(mark):
                        mark(str(node_id), [mid])
                    logger.info("benchmark_pod_model_freed", node=peer.name, model=mid)
                else:
                    logger.warning(
                        "benchmark_pod_model_free_failed",
                        node=peer.name,
                        model=mid,
                        status=response.status_code,
                    )
        except Exception as exc:  # noqa: BLE001 — liberar disco no debe parar la medida
            logger.warning("benchmark_pod_free_error", run_id=str(run_id), error=str(exc))

    async def _cloud_peer(self, node_id: str):
        """El peer de federación de un nodo de la nube, si está conectado."""
        fm = getattr(self._app.state, "federation_manager", None)
        cloud = getattr(fm, "cloud", None)
        if fm is None or cloud is None:
            return None
        name = next(
            (n.get("peer_name") for n in await cloud.list_nodes() if str(n.get("id")) == node_id),
            None,
        )
        if not name:
            return None
        return next((p for p in fm.get_online_peers() if p.name.lower() == str(name).lower()), None)

    async def _retry_broken_pod(self, run_id: uuid.UUID, spec: dict, node: dict) -> bool:
        """Pide otra máquina si el pod no sirvió (GPU averiada, no arrancó). ``True`` si lo hizo.

        RunPod da a veces máquinas cuya GPU no inicializa (``cuInit`` 999): el
        pod lo detecta y se apaga (core/cloud/manager.py). Antes eso tumbaba
        la batería entera. Ahora se vuelve a pedir pasados
        ``BROKEN_POD_RETRY_S`` (la máquina siguiente suele ser otra) con las
        mismas GPUs; un tipo de GPU que falla ``BROKEN_GPU_STRIKES`` veces se
        deja de pedir. Toda la batería pasa a la máquina nueva.
        """
        error = str(node.get("error") or "")
        if not any(mark in error for mark in POD_RETRY_MARKS):
            return False
        gpu = str(node.get("gpu_type_id") or "")
        strikes = (
            dict(spec.get("broken_gpus") or {}) if isinstance(spec.get("broken_gpus"), dict) else {}
        )
        strikes[gpu] = int(strikes.get(gpu) or 0) + 1
        excluded = {g for g, n in strikes.items() if n >= BROKEN_GPU_STRIKES}
        remaining = [g for g in spec.get("gpu_type_ids") or [] if g not in excluded]
        if not remaining:
            return False
        retry_at = (datetime.now(UTC) + timedelta(seconds=BROKEN_POD_RETRY_S)).isoformat()
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            query = select(BenchmarkRun).where(BenchmarkRun.id == run_id)
            if run.batch_id is not None:
                query = select(BenchmarkRun).where(
                    BenchmarkRun.batch_id == run.batch_id,
                    BenchmarkRun.status.not_in(_TERMINAL),
                )
            for row in (await session.execute(query)).scalars():
                cloud = dict(row.cloud or {})
                cloud["gpu_type_ids"] = [
                    g for g in cloud.get("gpu_type_ids") or [] if g not in excluded
                ]
                cloud["broken_gpus"] = strikes
                cloud["retry_after"] = retry_at
                row.cloud = cloud
                row.cloud_request_id = None
                row.cloud_node_id = None
                row.node = None
                row.status = "queued"
                row.phase = "waiting_capacity"
            await session.commit()
        logger.warning(
            "benchmark_pod_gpu_broken_retry",
            run_id=str(run_id),
            gpu=gpu,
            strikes=strikes[gpu],
            excluded=sorted(excluded),
            retry_at=retry_at,
        )
        self.notify()
        return True

    async def _fail_unprepared_model(
        self, run_id: uuid.UUID, models: list[str], cursor: int, index: int, item: dict
    ) -> None:
        """Da por fallido un modelo que el pod no pudo preparar y sigue con el resto."""
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            if run.status != "queued" or run.node is not None:
                return
            if index != cursor:
                # El cursor apunta siempre al modelo del que se habla.
                models.insert(cursor, models.pop(index))
                run.models = models
                await session.commit()
        model = models[cursor]
        reason = f"no se pudo preparar en el pod: {item.get('error') or 'error'}"
        await self._model_failed(run_id, model, reason[:1000])

    async def _mark_skipped(self, run_id: uuid.UUID, skipped: dict[str, str]) -> None:
        """Anota en la ejecución (y su batería) los modelos que no irán al pod."""
        async with AsyncSessionLocal() as session:
            run = await self._get_run(session, run_id)
            query = select(BenchmarkRun).where(BenchmarkRun.id == run_id)
            if run.batch_id is not None:
                query = select(BenchmarkRun).where(BenchmarkRun.batch_id == run.batch_id)
            for row in (await session.execute(query)).scalars():
                if row.cloud:
                    row.cloud = {**row.cloud, "skipped": skipped}
            await session.commit()

    async def _release_cloud(self, run_id: uuid.UUID) -> None:
        """Al terminar (bien, mal o cancelada): suelta la petición y, si se pidió, el pod."""
        try:
            async with AsyncSessionLocal() as session:
                run = await self._get_run(session, run_id)
                spec = dict(run.cloud or {}) if run.cloud else None
                request_id = run.cloud_request_id
                node_id = run.cloud_node_id
                if run.batch_id is not None:
                    # Con batería, la máquina es de todas: se suelta con la última.
                    pending = await session.scalar(
                        select(BenchmarkRun.id)
                        .where(
                            BenchmarkRun.batch_id == run.batch_id,
                            BenchmarkRun.status.not_in(_TERMINAL),
                        )
                        .limit(1)
                    )
                    if pending is not None:
                        return
            cloud = self._cloud()
            if not spec or cloud is None:
                return
            if request_id is not None:
                request = await cloud.requests.get(str(request_id))
                if request["status"] == "waiting":
                    await cloud.requests.cancel(str(request_id))
                if node_id is None and request.get("cloudNodeId"):
                    node_id = uuid.UUID(request["cloudNodeId"])
            if node_id is not None and spec.get("terminate_after", True):
                node = next(
                    (item for item in await cloud.list_nodes() if item["id"] == str(node_id)), None
                )
                if node is not None and node["status"] not in ("terminated",):
                    await cloud.terminate(str(node_id), reason="benchmark_done")
                    logger.info(
                        "benchmark_pod_terminated", run_id=str(run_id), node_id=str(node_id)
                    )
        except Exception:  # noqa: BLE001 — no dejar el error del pod tapar el de la medida
            logger.exception("benchmark_pod_release_failed", run_id=str(run_id))


async def load_run(run_id: uuid.UUID) -> BenchmarkRun | None:
    """Load a run with attempts for API serialization."""
    async with AsyncSessionLocal() as session:
        return await session.scalar(
            select(BenchmarkRun)
            .options(selectinload(BenchmarkRun.attempts))
            .where(BenchmarkRun.id == run_id)
        )


async def list_runs(limit: int = 100) -> list[BenchmarkRun]:
    """Load newest runs plus older baselines, all with attempts."""
    async with AsyncSessionLocal() as session:
        recent = list(
            (
                await session.execute(
                    select(BenchmarkRun)
                    .options(selectinload(BenchmarkRun.attempts))
                    .order_by(BenchmarkRun.created_at.desc())
                    .limit(limit)
                )
            ).scalars()
        )
        known = {run.id for run in recent}
        baselines = list(
            (
                await session.execute(
                    select(BenchmarkRun)
                    .options(selectinload(BenchmarkRun.attempts))
                    .where(BenchmarkRun.is_baseline.is_(True), BenchmarkRun.id.not_in(known))
                    .order_by(BenchmarkRun.created_at.desc())
                )
            ).scalars()
        )
        return [*recent, *baselines]
