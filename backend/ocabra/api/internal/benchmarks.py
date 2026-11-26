"""Administration API for application-owned benchmark runs."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.benchmarks.catalog import APPS, catalog_payload, scoring_issue
from ocabra.benchmarks.manager import list_runs, load_run, next_schedule_occurrence
from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.benchmarks import BenchmarkRun, BenchmarkSchedule

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])

_RESERVED_SUITE_ARGS = {
    "--api-key-env",
    "--base-url",
    "--database-url",
    "--model",
    "--models",
    "--output",
    "--report",
    "--results-dir",
    "--summary-json",
}


class BenchmarkCloudSpec(BaseModel):
    """Pod propio para medir: se pide, se prepara, se mide y (si se quiere) se apaga."""

    gpu_type_ids: list[str] = Field(
        ..., min_length=1, max_length=8, description="GPUs aceptables, por orden de preferencia"
    )
    gpu_count: int = Field(default=1, ge=1, le=8)
    cloud_type: Literal["COMMUNITY", "SECURE", "BOTH"] | None = None
    backends: list[str] = Field(default_factory=list, max_length=20)
    recipes: list[str] = Field(default_factory=list, max_length=10)
    volume_id: str | None = Field(default=None, max_length=64)
    privacy_mode: Literal["functional", "zero_data"] = "functional"
    wait_minutes: int = Field(default=360, ge=1, le=7 * 24 * 60, description="Espera de capacidad")
    ready_timeout_minutes: int = Field(
        default=90, ge=5, le=24 * 60, description="Tope desde que sale el pod hasta que está listo"
    )
    max_runtime_minutes: int = Field(default=480, ge=30, le=7 * 24 * 60)
    terminate_after: bool = True


def _check_suite_args(value: list[str]) -> list[str]:
    for item in value:
        if not item or "\x00" in item or len(item) > 500:
            raise ValueError("Argumento de suite no válido")
        if item.split("=", 1)[0] in _RESERVED_SUITE_ARGS:
            raise ValueError(f"oCabra controla el argumento {item.split('=', 1)[0]}")
    return value


class BenchmarkBatteryItem(BaseModel):
    """Otra prueba de la misma batería (mismos modelos, nodo o pod)."""

    app: str
    suite: str
    suite_args: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("suite_args")
    @classmethod
    def validate_suite_args(cls, value: list[str]) -> list[str]:
        return _check_suite_args(value)


class BenchmarkRunCreate(BaseModel):
    app: str
    suite: str
    models: Annotated[list[str], Field(min_length=1, max_length=64)]
    revision: str | None = None
    suite_args: list[str] = Field(default_factory=list, max_length=32)
    auto_pause: bool = True
    # Nodo de la federación donde medir (nombre del peer); None = este nodo.
    node: str | None = Field(default=None, max_length=64)
    # "benchmark": el muestreo que mande cada runner; "recommended": el del
    # fabricante (ficha de HF), aplicado en el servidor a las peticiones del runner.
    sampling: Literal["benchmark", "recommended"] = "benchmark"
    # Lanzar un pod propio para medir (en vez de un nodo que ya existe).
    cloud: BenchmarkCloudSpec | None = None
    # Más pruebas en el mismo pedido: se crea una ejecución por prueba y, con
    # pod propio, comparten la máquina.
    battery: list[BenchmarkBatteryItem] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_sampling_node(self):
        if self.sampling == "recommended" and (self.node or self.cloud):
            raise ValueError("El muestreo del fabricante solo se aplica en este nodo")
        if self.cloud and self.node:
            raise ValueError("Elige un nodo existente o un pod nuevo, no los dos")
        return self

    @field_validator("node")
    @classmethod
    def validate_node(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or value.lower() in ("local", "hub"):
            return None
        return value

    @field_validator("models")
    @classmethod
    def validate_models(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("Selecciona al menos un modelo")
        if any(
            len(item) > 200 or item.startswith("-") or any(ord(char) < 32 for char in item)
            for item in cleaned
        ):
            raise ValueError("Identificador de modelo no válido")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("No repitas modelos")
        return cleaned

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or len(value) > 200 or value.startswith("-"):
            raise ValueError("Revisión Git no válida")
        return value

    @field_validator("suite_args")
    @classmethod
    def validate_suite_args(cls, value: list[str]) -> list[str]:
        return _check_suite_args(value)

    def battery_items(self) -> list[tuple[str, str, list[str]]]:
        """La prueba principal y las de la batería, sin repetir."""
        items = [(self.app, self.suite, list(self.suite_args))]
        for extra in self.battery:
            entry = (extra.app, extra.suite, list(extra.suite_args))
            if entry not in items:
                items.append(entry)
        return items


class BenchmarkScheduleCreate(BenchmarkRunCreate):
    name: str = Field(min_length=1, max_length=120)
    enabled: bool = False
    weekdays: Annotated[list[int], Field(min_length=1, max_length=7)] = Field(
        default_factory=lambda: [0]
    )
    time_of_day: str = "02:00"
    timezone: str = "Europe/Madrid"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El plan necesita un nombre")
        return value

    @field_validator("weekdays")
    @classmethod
    def validate_weekdays(cls, value: list[int]) -> list[int]:
        if any(day < 0 or day > 6 for day in value):
            raise ValueError("Los días deben estar entre 0 y 6")
        return sorted(set(value))

    @field_validator("time_of_day")
    @classmethod
    def validate_time(cls, value: str) -> str:
        try:
            hour, minute = (int(part) for part in value.split(":"))
        except (ValueError, TypeError) as exc:
            raise ValueError("La hora debe tener formato HH:MM") from exc
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError("La hora debe tener formato HH:MM")
        return f"{hour:02d}:{minute:02d}"

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("Zona horaria desconocida") from exc
        return value


def _attempt_payload(attempt) -> dict[str, object]:
    return {
        "id": str(attempt.id),
        "model": attempt.model,
        "attemptNumber": attempt.attempt_number,
        "status": attempt.status,
        "environment": attempt.environment or {},
        "returnCode": attempt.return_code,
        "externalRequests": attempt.external_requests,
        "error": attempt.error,
        "startedAt": attempt.started_at.isoformat(),
        "finishedAt": attempt.finished_at.isoformat() if attempt.finished_at else None,
        "result": attempt.result,
    }


def _test_progress_payload(run: BenchmarkRun) -> dict[str, object] | None:
    """Read the bounded runner progress contract from the current attempt workspace."""
    if not run.current_model:
        return None
    attempt = next(
        (item for item in reversed(run.attempts) if item.model == run.current_model),
        None,
    )
    if attempt is None:
        return None
    try:
        root = Path(settings.benchmark_dir).resolve()
        workspace = Path(attempt.workspace_path).resolve()
        if not workspace.is_relative_to(root):
            return None
        path = workspace / "progress.json"
        if not path.is_file() or path.stat().st_size > 4096:
            return None
        progress = json.loads(path.read_text(encoding="utf-8"))
        completed = progress.get("completed")
        total = progress.get("total")
        unit = progress.get("unit")
        stage = progress.get("stage")
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or total <= 0
            or completed < 0
            or completed > total
            or not isinstance(unit, str)
            or not 0 < len(unit) <= 30
            or not isinstance(stage, str)
            or not 0 < len(stage) <= 40
        ):
            return None
        return {
            "completed": completed,
            "total": total,
            "unit": unit,
            "stage": stage,
            "updatedAt": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _run_payload(run: BenchmarkRun, *, include_attempts: bool = True) -> dict[str, object]:
    return {
        "id": str(run.id),
        "app": run.app,
        "suite": run.suite,
        "revision": run.revision or APPS[run.app].default_revision,
        "models": run.models,
        "suiteArgs": run.suite_args,
        "status": run.status,
        "phase": run.phase,
        "scheduleId": str(run.schedule_id) if run.schedule_id else None,
        "isBaseline": run.is_baseline,
        "autoPause": run.auto_pause,
        "node": getattr(run, "node", None),
        "sampling": getattr(run, "sampling", None) or "benchmark",
        "cloud": getattr(run, "cloud", None),
        "batchId": str(run.batch_id) if getattr(run, "batch_id", None) else None,
        "cloudRequestId": str(run.cloud_request_id)
        if getattr(run, "cloud_request_id", None)
        else None,
        "cloudNodeId": str(run.cloud_node_id) if getattr(run, "cloud_node_id", None) else None,
        "pauseReason": run.pause_reason,
        "currentModelIndex": run.current_model_index,
        "currentModel": run.current_model,
        "testProgress": _test_progress_payload(run) if include_attempts else None,
        "result": run.result,
        "contaminatedAttempts": run.contaminated_attempts,
        "error": run.error,
        "externalActivityAt": (
            run.external_activity_at.isoformat() if run.external_activity_at else None
        ),
        "createdAt": run.created_at.isoformat(),
        "startedAt": run.started_at.isoformat() if run.started_at else None,
        "finishedAt": run.finished_at.isoformat() if run.finished_at else None,
        "attempts": [_attempt_payload(item) for item in run.attempts] if include_attempts else [],
        "scoringIssue": _run_scoring_issue(run, include_attempts),
    }


def _run_scoring_issue(run: BenchmarkRun, include_attempts: bool) -> dict[str, str] | None:
    versions: list[str | None] = []
    if include_attempts:
        for attempt in run.attempts:
            if attempt.status == "completed":
                software = (attempt.environment or {}).get("software") or {}
                versions.append(software.get("ocabra") if isinstance(software, dict) else None)
    return scoring_issue(run.app, run.suite, versions, run.created_at)


def _schedule_payload(schedule: BenchmarkSchedule) -> dict[str, object]:
    return {
        "id": str(schedule.id),
        "name": schedule.name,
        "app": schedule.app,
        "suite": schedule.suite,
        "revision": schedule.revision or APPS[schedule.app].default_revision,
        "models": schedule.models,
        "suiteArgs": schedule.suite_args,
        "autoPause": schedule.auto_pause,
        "node": getattr(schedule, "node", None),
        "sampling": getattr(schedule, "sampling", None) or "benchmark",
        "cloud": getattr(schedule, "cloud", None),
        "battery": getattr(schedule, "battery", None) or [],
        "enabled": schedule.enabled,
        "weekdays": schedule.weekdays,
        "timeOfDay": schedule.time_of_day,
        "timezone": schedule.timezone,
        "nextRunAt": schedule.next_run_at.isoformat() if schedule.next_run_at else None,
        "lastRunAt": schedule.last_run_at.isoformat() if schedule.last_run_at else None,
        "lastRunId": str(schedule.last_run_id) if schedule.last_run_id else None,
        "createdAt": schedule.created_at.isoformat(),
        "updatedAt": schedule.updated_at.isoformat(),
    }


def _validate_catalog(app_name: str, suite_name: str) -> None:
    app = APPS.get(app_name)
    if app is None:
        raise HTTPException(status_code=400, detail="Aplicación de benchmark desconocida")
    if suite_name not in app.suites:
        raise HTTPException(status_code=400, detail="Suite de benchmark desconocida")


def _canonical_node(request: Request, node: str | None) -> str | None:
    """Nombre del peer tal como lo conoce la federación, o 400 si no existe.

    El benchmark fuerza las peticiones a ese nodo; un nombre que no casa con
    ningún peer haría que se sirvieran aquí y el resultado mediría otro
    hardware.
    """
    if node is None:
        return None
    manager = getattr(request.app.state, "federation_manager", None)
    peers = manager.get_all_peers() if manager is not None else []
    wanted = node.lower()
    for peer in peers:
        if wanted in {peer.name.lower(), str(peer.peer_id)}:
            if not peer.enabled:
                raise HTTPException(
                    status_code=400, detail=f"El nodo '{peer.name}' está desactivado"
                )
            return peer.name
    raise HTTPException(status_code=400, detail=f"No hay ningún nodo federado llamado '{node}'")


def _user_uuid(user: UserContext) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(user.user_id))
    except (TypeError, ValueError, AttributeError):
        return None


def _manager(request: Request):
    manager = getattr(request.app.state, "benchmark_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="El ejecutor de benchmarks no está disponible")
    return manager


@router.get("/catalog", summary="List benchmark applications and suites")
async def get_benchmark_catalog(
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict[str, object]]:
    """Return the trusted application suites that can be queued from the UI.

    Returns:
        Applications, their suites and resume/preparation capabilities.
    """
    return catalog_payload()


@router.post("/runs", status_code=202, summary="Queue a benchmark run")
async def create_benchmark_run(
    body: BenchmarkRunCreate,
    request: Request,
    user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Queue models for one application suite.

    Parameters:
        body: Application, suite, models, optional revision and runner arguments.

    Returns:
        Persisted run state. Models execute sequentially and can be resumed.
    """
    items = body.battery_items()
    for app_name, suite_name, _args in items:
        _validate_catalog(app_name, suite_name)
    node = _canonical_node(request, body.node)
    batch_id = uuid.uuid4() if len(items) > 1 else None
    root = Path(settings.benchmark_dir).resolve()  # noqa: ASYNC240 — cálculo local de ruta
    runs = []
    for app_name, suite_name, suite_args in items:
        run_id = uuid.uuid4()
        runs.append(
            BenchmarkRun(
                id=run_id,
                app=app_name,
                suite=suite_name,
                # La revisión fijada solo vale para la app principal.
                revision=body.revision if app_name == body.app else None,
                models=body.models,
                suite_args=suite_args,
                auto_pause=body.auto_pause,
                node=node,
                sampling=None if body.sampling == "benchmark" else body.sampling,
                cloud=body.cloud.model_dump() if body.cloud else None,
                batch_id=batch_id,
                phase="queued",
                workspace_path=str(root / "runs" / str(run_id)),
                created_by=_user_uuid(user),
            )
        )
    async with AsyncSessionLocal() as session:
        session.add_all(runs)
        await session.commit()
        for run in runs:
            await session.refresh(run)
    _manager(request).notify()
    payload = _run_payload(runs[0], include_attempts=False)
    payload["batchRuns"] = [str(run.id) for run in runs]
    return payload


@router.get("/schedules", summary="List reusable benchmark plans")
async def get_benchmark_schedules(
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict[str, object]]:
    """Return templates and their next/last scheduled executions."""
    async with AsyncSessionLocal() as session:
        rows = (
            (
                await session.execute(
                    select(BenchmarkSchedule).order_by(BenchmarkSchedule.created_at.desc())
                )
            )
            .scalars()
            .all()
        )
    return [_schedule_payload(row) for row in rows]


@router.post("/schedules", status_code=201, summary="Create a benchmark plan")
async def create_benchmark_schedule(
    body: BenchmarkScheduleCreate,
    request: Request,
    user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Create a reusable template and optionally enable its weekly schedule."""
    for app_name, suite_name, _args in body.battery_items():
        _validate_catalog(app_name, suite_name)
    now = datetime.now(UTC)
    schedule = BenchmarkSchedule(
        name=body.name,
        app=body.app,
        suite=body.suite,
        revision=body.revision,
        models=body.models,
        suite_args=body.suite_args,
        auto_pause=body.auto_pause,
        node=_canonical_node(request, body.node),
        sampling=None if body.sampling == "benchmark" else body.sampling,
        cloud=body.cloud.model_dump() if body.cloud else None,
        battery=[item.model_dump() for item in body.battery] or None,
        enabled=body.enabled,
        weekdays=body.weekdays,
        time_of_day=body.time_of_day,
        timezone=body.timezone,
        next_run_at=(
            next_schedule_occurrence(body.weekdays, body.time_of_day, body.timezone, now)
            if body.enabled
            else None
        ),
        created_by=_user_uuid(user),
    )
    async with AsyncSessionLocal() as session:
        session.add(schedule)
        await session.commit()
        await session.refresh(schedule)
    _manager(request).notify()
    return _schedule_payload(schedule)


@router.put("/schedules/{schedule_id}", summary="Replace a benchmark plan")
async def update_benchmark_schedule(
    schedule_id: uuid.UUID,
    body: BenchmarkScheduleCreate,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Replace a plan definition and recalculate its next occurrence."""
    for app_name, suite_name, _args in body.battery_items():
        _validate_catalog(app_name, suite_name)
    node = _canonical_node(request, body.node)
    async with AsyncSessionLocal() as session:
        schedule = await session.get(BenchmarkSchedule, schedule_id)
        if schedule is None:
            raise HTTPException(status_code=404, detail="Plan de benchmark no encontrado")
        for field in (
            "name",
            "app",
            "suite",
            "revision",
            "models",
            "suite_args",
            "auto_pause",
            "enabled",
            "weekdays",
            "time_of_day",
            "timezone",
        ):
            setattr(schedule, field, getattr(body, field))
        schedule.node = node
        schedule.sampling = None if body.sampling == "benchmark" else body.sampling
        schedule.cloud = body.cloud.model_dump() if body.cloud else None
        schedule.battery = [item.model_dump() for item in body.battery] or None
        schedule.next_run_at = (
            next_schedule_occurrence(
                body.weekdays, body.time_of_day, body.timezone, datetime.now(UTC)
            )
            if body.enabled
            else None
        )
        await session.commit()
        await session.refresh(schedule)
    _manager(request).notify()
    return _schedule_payload(schedule)


@router.delete("/schedules/{schedule_id}", status_code=204, summary="Delete a benchmark plan")
async def delete_benchmark_schedule(
    schedule_id: uuid.UUID,
    _user: UserContext = Depends(require_role("model_manager")),
) -> None:
    """Delete a plan without deleting its historical runs."""
    async with AsyncSessionLocal() as session:
        schedule = await session.get(BenchmarkSchedule, schedule_id)
        if schedule is None:
            raise HTTPException(status_code=404, detail="Plan de benchmark no encontrado")
        await session.delete(schedule)
        await session.commit()


@router.post("/schedules/{schedule_id}/run", status_code=202, summary="Run a benchmark plan now")
async def run_benchmark_schedule(
    schedule_id: uuid.UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Queue a template immediately, rejecting overlap with its active run."""
    manager = _manager(request)
    try:
        run = await manager.request_schedule_run(schedule_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _run_payload(run, include_attempts=False)


@router.get("/runs", summary="List benchmark runs")
async def get_benchmark_runs(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict[str, object]]:
    """Return newest benchmark runs with attempts and summarized results."""
    return [_run_payload(run) for run in await list_runs(limit)]


@router.get("/runs/{run_id}", summary="Get one benchmark run")
async def get_benchmark_run(
    run_id: uuid.UUID,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Return one benchmark run, every attempt and its raw result."""
    run = await load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    return _run_payload(run)


@router.get("/runs/{run_id}/log", summary="Read the current benchmark log")
async def get_benchmark_log(
    run_id: uuid.UUID,
    lines: Annotated[int, Query(ge=10, le=5000)] = 500,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Return the tail of the newest attempt log without exposing its path."""
    run = await load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    if not run.attempts:
        return {"lines": [], "attemptId": None}
    attempt = run.attempts[-1]
    path = Path(attempt.log_path)
    if not path.is_file():  # noqa: ASYNC240 — stat local mínimo antes de to_thread
        return {"lines": [], "attemptId": str(attempt.id)}
    content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
    return {"lines": content.splitlines()[-lines:], "attemptId": str(attempt.id)}


async def _control_run(run_id: uuid.UUID, request: Request, action: str) -> dict[str, object]:
    manager = _manager(request)
    try:
        run = await getattr(manager, f"request_{action}")(run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _run_payload(run, include_attempts=False)


@router.post("/runs/{run_id}/retry", summary="Retry a failed benchmark from its current model")
async def retry_benchmark_run(
    run_id: uuid.UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Requeue a failed run without repeating models already accepted.

    Parameters:
        run_id: Failed benchmark execution identifier.

    Returns:
        Requeued run, positioned at the failed model and retaining its checkpoint.
    """
    return await _control_run(run_id, request, "retry")


@router.post("/runs/{run_id}/baseline", summary="Set a completed run as suite baseline")
async def baseline_benchmark_run(
    run_id: uuid.UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Make this run the unique historical reference for its application suite."""
    return await _control_run(run_id, request, "baseline")


@router.post("/runs/{run_id}/pause", summary="Pause a benchmark run")
async def pause_benchmark_run(
    run_id: uuid.UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Stop the current runner safely and keep its workspace for resume."""
    return await _control_run(run_id, request, "pause")


@router.post("/runs/{run_id}/resume", summary="Resume a benchmark run")
async def resume_benchmark_run(
    run_id: uuid.UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Put a manually or automatically paused run back in the queue."""
    return await _control_run(run_id, request, "resume")


@router.post("/runs/{run_id}/cancel", summary="Cancel a benchmark run")
async def cancel_benchmark_run(
    run_id: uuid.UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Cancel queued work or terminate the current benchmark subprocess."""
    return await _control_run(run_id, request, "cancel")


# ── Diagnóstico de fallos ────────────────────────────────────────────────

_DIAGNOSIS_FILES = {
    "report.md": "text/markdown; charset=utf-8",
    "candidates.train.jsonl": "application/x-ndjson",
    "candidates.eval.jsonl": "application/x-ndjson",
    "replays.jsonl": "application/x-ndjson",
}


class BenchmarkDiagnosisCreate(BaseModel):
    """Opciones de un diagnóstico «¿es el modelo o es cómo lo servimos?»."""

    repeats: int = Field(default=3, ge=1, le=10)
    max_cases: int = Field(default=30, ge=1, le=500)
    compare_model: str | None = Field(default=None, max_length=200)
    reference_repo: str | None = Field(default=None, max_length=200)
    allow_load: bool = False

    @field_validator("compare_model", "reference_repo")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        value = (value or "").strip()
        return value or None


def _diagnosis_payload(job: dict[str, object], report: dict | None = None) -> dict[str, object]:
    request = job.get("request") or {}
    payload: dict[str, object] = {
        "id": job["id"],
        "status": job["status"],
        "phase": job.get("phase"),
        "progress": job.get("progress"),
        "error": job.get("error"),
        "createdAt": job.get("created_at"),
        "startedAt": job.get("started_at"),
        "finishedAt": job.get("finished_at"),
        "attemptId": request.get("attempt_id"),
        "runId": request.get("run_id"),
        "app": request.get("app"),
        "suite": request.get("suite"),
        "model": request.get("model"),
        "options": {
            key: request.get(key)
            for key in ("repeats", "max_cases", "compare_model", "reference_repo", "allow_load")
        },
    }
    if report is not None:
        payload["report"] = report
    return payload


@router.post(
    "/attempts/{attempt_id}/diagnoses",
    status_code=202,
    summary="Diagnose why a benchmark attempt's cases failed",
)
async def create_benchmark_diagnosis(
    attempt_id: uuid.UUID,
    request: Request,
    body: BenchmarkDiagnosisCreate | None = None,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Encola el diagnóstico de los casos fallidos de un intento completado.

    Clasifica cada fallo en plantilla, sampling, límites, cuantización/servido o
    modelo. Lo ejecuta la cola de benchmarks cuando no hay ninguno pendiente, y
    sus repeticiones entran por la cola normal de oCabra.

    Args:
        attempt_id: intento completado (``attempts[].id`` de una ejecución).
        body: ``repeats`` (repeticiones con el muestreo recomendado, 3),
            ``max_cases`` (casos repetidos como máximo, muestra estratificada,
            30), ``compare_model`` (perfil con otra cuantización para comparar),
            ``reference_repo`` (repo de HF de la ficha oficial si no se deduce)
            y ``allow_load`` (repetir aunque el modelo no esté cargado; puede
            expulsar otros modelos).

    Returns:
        El diagnóstico encolado (``status: queued``).
    """
    options = (body or BenchmarkDiagnosisCreate()).model_dump()
    try:
        job = await _manager(request).request_diagnosis(attempt_id, options)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _diagnosis_payload(job)


@router.get("/diagnoses", summary="List benchmark failure diagnoses")
async def list_benchmark_diagnoses(
    request: Request,
    attempt_id: uuid.UUID | None = None,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict[str, object]]:
    """Lista los diagnósticos, los más recientes primero.

    Args:
        attempt_id: solo los de ese intento.

    Returns:
        Diagnósticos sin el informe (pídelo en ``GET /diagnoses/{id}``).
    """
    jobs = await asyncio.to_thread(_manager(request).diagnoses.all)
    if attempt_id is not None:
        jobs = [
            job for job in jobs if (job.get("request") or {}).get("attempt_id") == str(attempt_id)
        ]
    return [_diagnosis_payload(job) for job in jobs]


@router.get("/diagnoses/{diagnosis_id}", summary="Read a benchmark failure diagnosis")
async def get_benchmark_diagnosis(
    diagnosis_id: uuid.UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Devuelve el estado de un diagnóstico y, si terminó, su informe completo.

    Args:
        diagnosis_id: id del diagnóstico.

    Returns:
        Estado, progreso y ``report`` (resumen por categoría, parámetros
        efectivos frente a recomendados, sondas de plantilla y la tabla de
        casos con su evidencia).
    """
    store = _manager(request).diagnoses
    job = await asyncio.to_thread(store.load, str(diagnosis_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Diagnóstico no encontrado")
    report = await asyncio.to_thread(store.report, str(diagnosis_id))
    return _diagnosis_payload(job, report)


@router.get(
    "/diagnoses/{diagnosis_id}/files/{name}",
    summary="Download a diagnosis report or training candidates",
)
async def get_benchmark_diagnosis_file(
    diagnosis_id: uuid.UUID,
    name: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
):
    """Descarga un fichero del diagnóstico.

    Args:
        diagnosis_id: id del diagnóstico.
        name: ``report.md``, ``candidates.train.jsonl``,
            ``candidates.eval.jsonl`` (reservados para medir, no para entrenar)
            o ``replays.jsonl``.

    Returns:
        El fichero tal cual.
    """
    from fastapi.responses import Response

    media_type = _DIAGNOSIS_FILES.get(name)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Fichero no disponible")
    path = _manager(request).diagnoses.path(str(diagnosis_id)) / name
    try:
        content = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Fichero no disponible") from exc
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{diagnosis_id}-{name}"'},
    )


@router.post("/diagnoses/{diagnosis_id}/cancel", summary="Cancel a benchmark failure diagnosis")
async def cancel_benchmark_diagnosis(
    diagnosis_id: uuid.UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, object]:
    """Cancela un diagnóstico en cola o en curso.

    Si está repitiendo casos, se corta el script de repetición; lo hecho hasta
    ese momento queda en disco pero no se genera informe.

    Args:
        diagnosis_id: id del diagnóstico.

    Returns:
        El diagnóstico (``cancelled`` si estaba en cola; en curso pasa a
        ``cancelled`` en cuanto se detiene).
    """
    try:
        job = await _manager(request).cancel_diagnosis(str(diagnosis_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _diagnosis_payload(job)


@router.delete(
    "/diagnoses/{diagnosis_id}", status_code=204, summary="Delete a benchmark failure diagnosis"
)
async def delete_benchmark_diagnosis(
    diagnosis_id: uuid.UUID,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> None:
    """Borra un diagnóstico terminado con su informe y candidatos.

    Los terminados se borran también solos pasados
    ``benchmark_diagnosis_retention_days`` días (30 por defecto).

    Args:
        diagnosis_id: id del diagnóstico.

    Returns:
        Nada (204). 409 si sigue en cola o en curso: cancélalo antes.
    """
    try:
        await _manager(request).delete_diagnosis(str(diagnosis_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
