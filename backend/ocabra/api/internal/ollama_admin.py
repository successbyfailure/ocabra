"""Ollama daemon admin endpoints: version probe + server update."""
from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.registry.ollama_registry import OllamaRegistry

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["ollama-admin"])

_registry = OllamaRegistry()

UpdateStatus = Literal["idle", "pulling", "restarting", "done", "error"]


class _UpdateState:
    def __init__(self) -> None:
        self.status: UpdateStatus = "idle"
        self.detail: str | None = None
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.from_version: str | None = None
        self.to_version: str | None = None
        self._lock = asyncio.Lock()

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "detail": self.detail,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "from_version": self.from_version,
            "to_version": self.to_version,
        }


_update_state = _UpdateState()


_VERSION_RE = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def _parse_version(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = _VERSION_RE.match(value.strip())
    if not match:
        return None
    return (
        int(match.group(1) or 0),
        int(match.group(2) or 0),
        int(match.group(3) or 0),
    )


def _is_update_available(current: str | None, latest: str | None) -> bool:
    if not current or not latest:
        return False
    cur = _parse_version(current)
    lat = _parse_version(latest)
    if cur is None or lat is None:
        return current.strip() != latest.strip()
    return lat > cur


class VersionInfo(BaseModel):
    current: str | None
    latest: str | None
    update_available: bool


class UpdateStartResponse(BaseModel):
    status: UpdateStatus
    detail: str | None = None


@router.get(
    "/ollama/version",
    response_model=VersionInfo,
    summary="Get Ollama daemon version",
    description="Return the local Ollama daemon version and the latest released upstream version.",
)
async def get_ollama_version(
    _user: UserContext = Depends(require_role("model_manager")),
) -> VersionInfo:
    current, latest = await asyncio.gather(
        _registry.get_version(),
        _registry.get_latest_version(),
    )
    return VersionInfo(
        current=current,
        latest=latest,
        update_available=_is_update_available(current, latest),
    )


@router.get(
    "/ollama/server/update",
    summary="Get Ollama server update status",
    description="Return the status of the most recent server update operation.",
)
async def get_server_update_status(
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    return _update_state.snapshot()


@router.post(
    "/ollama/server/update",
    response_model=UpdateStartResponse,
    summary="Update the Ollama daemon container",
    description=(
        "Pull the latest ``ollama/ollama`` image and recreate the service. "
        "Runs in the background; poll ``GET /ollama/server/update`` for status."
    ),
)
async def start_server_update(
    _user: UserContext = Depends(require_role("system_admin")),
) -> UpdateStartResponse:
    if _update_state.status in {"pulling", "restarting"}:
        raise HTTPException(status_code=409, detail="An Ollama update is already running")

    asyncio.create_task(_run_server_update(), name="ollama-server-update")
    return UpdateStartResponse(status="pulling", detail="update started")


async def _run_server_update() -> None:
    async with _update_state._lock:
        _update_state.status = "pulling"
        _update_state.detail = "pulling latest image"
        _update_state.started_at = datetime.now(UTC)
        _update_state.finished_at = None
        _update_state.from_version = await _registry.get_version()
        _update_state.to_version = None

        try:
            compose_dir = settings.compose_project_dir
            compose_args_base = (
                "-f", f"{compose_dir}/docker-compose.yml",
                "--env-file", f"{compose_dir}/.env",
            )

            code, stdout, stderr = await _run_compose(*compose_args_base, "pull", "ollama")
            if code != 0:
                _update_state.status = "error"
                _update_state.detail = (stderr or stdout or f"docker compose pull exit_code={code}")[:1000]
                _update_state.finished_at = datetime.now(UTC)
                logger.warning("ollama_update_pull_failed", error=_update_state.detail)
                return

            _update_state.status = "restarting"
            _update_state.detail = "recreating container"
            code, stdout, stderr = await _run_compose(
                *compose_args_base,
                "up", "-d", "--no-deps", "--force-recreate", "ollama",
            )
            if code != 0:
                _update_state.status = "error"
                _update_state.detail = (stderr or stdout or f"docker compose up exit_code={code}")[:1000]
                _update_state.finished_at = datetime.now(UTC)
                logger.warning("ollama_update_up_failed", error=_update_state.detail)
                return

            new_version = None
            for _ in range(30):
                await asyncio.sleep(2.0)
                new_version = await _registry.get_version()
                if new_version:
                    break

            _update_state.to_version = new_version
            _update_state.status = "done"
            _update_state.detail = "update completed"
            _update_state.finished_at = datetime.now(UTC)
            logger.info(
                "ollama_update_done",
                from_version=_update_state.from_version,
                to_version=new_version,
            )
        except Exception as exc:
            _update_state.status = "error"
            _update_state.detail = str(exc)[:1000]
            _update_state.finished_at = datetime.now(UTC)
            logger.exception("ollama_update_unexpected_error")


async def _run_compose(*args: str) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        "docker", "compose",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return (
        process.returncode or 0,
        stdout.decode("utf-8", errors="ignore").strip(),
        stderr.decode("utf-8", errors="ignore").strip(),
    )
