"""Administration API for application-owned benchmark runs."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.benchmarks.catalog import APPS, catalog_payload
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


class BenchmarkRunCreate(BaseModel):
    app: str
    suite: str
    models: Annotated[list[str], Field(min_length=1, max_length=64)]
    revision: str | None = None
    suite_args: list[str] = Field(default_factory=list, max_length=32)
    auto_pause: bool = True

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
        for item in value:
            if not item or "\x00" in item or len(item) > 500:
                raise ValueError("Argumento de suite no válido")
            if item.split("=", 1)[0] in _RESERVED_SUITE_ARGS:
                raise ValueError(f"oCabra controla el argumento {item.split('=', 1)[0]}")
        return value


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
    }


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
    _validate_catalog(body.app, body.suite)
    run_id = uuid.uuid4()
    root = Path(settings.benchmark_dir).resolve()  # noqa: ASYNC240 — cálculo local de ruta
    workspace = str(root / "runs" / str(run_id))
    run = BenchmarkRun(
        id=run_id,
        app=body.app,
        suite=body.suite,
        revision=body.revision,
        models=body.models,
        suite_args=body.suite_args,
        auto_pause=body.auto_pause,
        phase="queued",
        workspace_path=workspace,
        created_by=_user_uuid(user),
    )
    async with AsyncSessionLocal() as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)
    _manager(request).notify()
    return _run_payload(run, include_attempts=False)


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
    _validate_catalog(body.app, body.suite)
    now = datetime.now(UTC)
    schedule = BenchmarkSchedule(
        name=body.name,
        app=body.app,
        suite=body.suite,
        revision=body.revision,
        models=body.models,
        suite_args=body.suite_args,
        auto_pause=body.auto_pause,
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
    _validate_catalog(body.app, body.suite)
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
