"""Ollama daemon admin endpoints: version probe + server update."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.core.background_tasks import spawn
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

    spawn(_run_server_update(), name="ollama-server-update")
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

            # Services that layer a local Dockerfile over ``ollama/ollama``
            # (e.g. adding ffmpeg for the 0.32+ multimodal path) can't be
            # updated with ``compose pull`` — the target ``ocabra-ollama``
            # image isn't in any registry, so the pull fails with
            # "pull access denied" but with exit code 0 (compose swallows
            # the individual-service failure), and force-recreate then just
            # rebrings the stale local image. Detect ``build:`` and switch
            # to ``compose build --pull`` which refreshes the base image
            # and rebuilds the layer.
            uses_build = await _service_uses_build_key(compose_args_base, "ollama")
            if uses_build:
                code, stdout, stderr = await _run_compose(
                    *compose_args_base, "build", "--pull", "ollama"
                )
                fail_verb = "build"
            else:
                code, stdout, stderr = await _run_compose(
                    *compose_args_base, "pull", "ollama"
                )
                fail_verb = "pull"
            if code != 0:
                _update_state.status = "error"
                _update_state.detail = (
                    stderr or stdout or f"docker compose {fail_verb} exit_code={code}"
                )[:1000]
                _update_state.finished_at = datetime.now(UTC)
                logger.warning(
                    "ollama_update_pull_failed",
                    step=fail_verb,
                    error=_update_state.detail,
                )
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


async def _service_uses_build_key(compose_args: tuple[str, ...], service: str) -> bool:
    """Return whether ``service`` in the resolved compose config has a
    ``build:`` key (as opposed to only ``image:``).

    Used by the Ollama updater to decide between ``compose pull`` (upstream
    image) and ``compose build --pull`` (local Dockerfile that layers on top
    of an upstream base). Returns ``False`` on any parse/exec error — the
    caller falls back to plain pull, which matches the previous behaviour.
    """
    code, stdout, _ = await _run_compose(*compose_args, "config", "--format=json")
    if code != 0 or not stdout:
        return False
    try:
        cfg = json.loads(stdout)
    except (ValueError, json.JSONDecodeError):
        return False
    svc = (cfg.get("services") or {}).get(service) or {}
    return "build" in svc


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


# ── Global server settings (one shared daemon) ──────────────────────────────
#
# Ollama runs as a single daemon for every Ollama model (the per-GPU pools
# were retired on 2026-11-12). Its parallelism and default context are
# process-wide environment variables, so they are global settings of oCabra:
# ``settings.ollama_num_parallel`` / ``settings.ollama_max_context``, persisted
# like any other config override. Applying them writes ``ollama_env_file``
# (read by the ``ollama`` compose service through ``env_file``) and recreates
# the container, which drops every model Ollama has loaded.


class _ApplyState:
    def __init__(self) -> None:
        self.status: Literal["idle", "applying", "done", "error"] = "idle"
        self.detail: str | None = None
        self.finished_at: datetime | None = None
        self._lock = asyncio.Lock()


_apply_state = _ApplyState()


def render_ollama_env(num_parallel: int, max_context: int) -> str:
    return (
        "# Escrito por oCabra (Ajustes > Ollama). Lo lee el servicio ollama de\n"
        "# docker-compose.yml; cambiarlo a mano requiere recrear el contenedor.\n"
        f"OLLAMA_NUM_PARALLEL={int(num_parallel)}\n"
        f"OLLAMA_CONTEXT_LENGTH={int(max_context)}\n"
    )


def write_ollama_env_file(num_parallel: int, max_context: int, path: str | None = None) -> None:
    from pathlib import Path

    target = Path(path or settings.ollama_env_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_ollama_env(num_parallel, max_context), encoding="utf-8")


def parse_ollama_env(env: list[str]) -> dict[str, int | None]:
    """``{"numParallel", "maxContext"}`` from a container's ``KEY=value`` list."""
    values = dict(item.split("=", 1) for item in env if "=" in item)

    def as_int(key: str) -> int | None:
        try:
            return int(values[key])
        except (KeyError, ValueError):
            return None

    return {
        "numParallel": as_int("OLLAMA_NUM_PARALLEL"),
        "maxContext": as_int("OLLAMA_CONTEXT_LENGTH"),
    }


async def _running_ollama_env() -> dict[str, int | None] | None:
    """What the live ollama container was started with (docker inspect)."""
    compose_dir = settings.compose_project_dir
    code, out, _err = await _run_compose(
        "-f", f"{compose_dir}/docker-compose.yml", "--env-file", f"{compose_dir}/.env",
        "ps", "-q", "ollama",
    )
    container = out.splitlines()[0].strip() if code == 0 and out else ""
    if not container:
        return None
    process = await asyncio.create_subprocess_exec(
        "docker", "inspect", "-f", "{{json .Config.Env}}", container,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    if process.returncode:
        return None
    try:
        return parse_ollama_env(json.loads(stdout.decode() or "[]"))
    except ValueError:
        return None


class OllamaServerSettings(BaseModel):
    numParallel: int
    maxContext: int
    running: dict[str, int | None] | None = None
    pending: bool = False
    applyStatus: str = "idle"
    applyDetail: str | None = None


@router.get(
    "/ollama/server/settings",
    response_model=OllamaServerSettings,
    summary="Global settings of the shared Ollama server",
)
async def get_server_settings(
    _user: UserContext = Depends(require_role("user")),
) -> OllamaServerSettings:
    """Configured parallelism and maximum context of the Ollama daemon, what
    the running container actually has (``running``; null when it cannot be
    inspected) and whether they differ (``pending``: apply to take effect)."""
    running = await _running_ollama_env()
    want = {"numParallel": settings.ollama_num_parallel, "maxContext": settings.ollama_max_context}
    return OllamaServerSettings(
        **want,
        running=running,
        pending=bool(running) and any(running.get(k) != v for k, v in want.items()),
        applyStatus=_apply_state.status,
        applyDetail=_apply_state.detail,
    )


@router.post(
    "/ollama/server/apply-settings",
    response_model=UpdateStartResponse,
    summary="Apply the global Ollama settings (recreates the daemon)",
)
async def apply_server_settings(
    _user: UserContext = Depends(require_role("system_admin")),
) -> UpdateStartResponse:
    """Write the env file with the configured values and recreate the ollama
    container. Every model Ollama has loaded is dropped. Runs in the
    background; poll ``GET /ollama/server/settings``."""
    if _apply_state.status == "applying" or _update_state.status in {"pulling", "restarting"}:
        raise HTTPException(status_code=409, detail="The Ollama container is already being recreated")
    spawn(_run_apply_settings(), name="ollama-apply-settings")
    return UpdateStartResponse(status="restarting", detail="recreating the Ollama container")


async def _run_apply_settings() -> None:
    async with _apply_state._lock:
        _apply_state.status = "applying"
        _apply_state.detail = "writing settings and recreating the container"
        try:
            write_ollama_env_file(settings.ollama_num_parallel, settings.ollama_max_context)
            compose_dir = settings.compose_project_dir
            code, stdout, stderr = await _run_compose(
                "-f", f"{compose_dir}/docker-compose.yml", "--env-file", f"{compose_dir}/.env",
                "up", "-d", "--no-deps", "--force-recreate", "ollama",
            )
            if code != 0:
                _apply_state.status = "error"
                _apply_state.detail = (stderr or stdout or f"exit_code={code}")[:1000]
                return
            for _ in range(30):
                await asyncio.sleep(2.0)
                if await _registry.get_version():
                    break
            _apply_state.status = "done"
            _apply_state.detail = "applied"
            logger.info(
                "ollama_settings_applied",
                num_parallel=settings.ollama_num_parallel,
                max_context=settings.ollama_max_context,
            )
        except Exception as exc:  # noqa: BLE001
            _apply_state.status = "error"
            _apply_state.detail = str(exc)[:1000]
        finally:
            _apply_state.finished_at = datetime.now(UTC)
