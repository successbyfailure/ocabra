"""The Docker operations oCabra is allowed to perform, and nothing else.

Every Docker call of the api goes through :func:`run` or :func:`stream` by
operation name with typed arguments; the argv is built here, from validated
values. The docker agent (``python -m ocabra.docker_agent``, its own container,
the only one meant to hold ``/var/run/docker.sock``) exposes exactly these
operations over HTTP; the api uses this module directly only when no agent is
configured (see :mod:`ocabra.core.docker_ops`).

Why: with the socket, whoever controls the api could ``docker run -v /:/host``
and own the machine (SEG-03 in docs/REVISION-2026-11-16.md). Through the agent
the most they get is starting, stopping or updating oCabra's own services. Same
idea as hw-monitor, which alone may change GPU power limits and only accepts
``{gpu_index, limit_w}``.

Anything that would let a caller choose an image, a mount, a container outside
the catalogue or a shell command is rejected with :class:`OpError`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from ocabra.config import settings

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$")
_MODEL_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*){0,3}$")
_INSPECT_FIELDS = {"running": "{{.State.Running}}", "status": "{{.State.Status}}"}
_CONTAINER_ACTIONS = ("start", "stop", "restart")
_TRTLLM_PHASES = ("convert", "build")
_TRTLLM_BUILD_INTS = ("max_batch_size", "max_input_len", "max_seq_len")

#: Set by the agent's entry point. The api, running the same module locally,
#: is the api container itself; the agent has to find it.
AGENT_ENV = "OCABRA_DOCKER_AGENT"


class OpError(ValueError):
    """A request outside what an operation allows (answered as HTTP 400)."""


# ── helpers ──────────────────────────────────────────────────────────────────


def _result(code: int, stdout: str = "", stderr: str = "", **extra: Any) -> dict[str, Any]:
    return {"code": code, "stdout": stdout, "stderr": stderr, **extra}


async def _exec(argv: list[str], timeout: float | None = 120.0) -> dict[str, Any]:  # noqa: ASYNC109
    """Run ``argv``; exit 127 (never raises) without the docker CLI, 124 on timeout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError:
        return _result(127, "", "docker CLI not available")
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return _result(124, "", "timeout")
    return _result(
        proc.returncode or 0,
        out.decode("utf-8", errors="replace").strip(),
        err.decode("utf-8", errors="replace").strip(),
    )


def _str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise OpError(f"'{key}' must be a non-empty string")
    return value


def _compose(*, services_profile: bool = False) -> list[str]:
    """``docker compose`` against oCabra's own project files, never caller-given ones."""
    project_dir = settings.compose_project_dir
    argv = ["docker", "compose"]
    if services_profile:
        argv += ["--profile", "services"]
    return argv + [
        "-f",
        f"{project_dir}/docker-compose.yml",
        "--env-file",
        f"{project_dir}/.env",
    ]


def _catalogue() -> tuple[set[str], set[str]]:
    """Container names and compose services of the optional services catalogue."""
    from ocabra.core.service_manager import ServiceManager

    states = ServiceManager()._states.values()
    containers = {s.docker_container_name for s in states if s.docker_container_name}
    services = {s.compose_service_name for s in states if s.compose_service_name}
    return containers, services


def _catalogue_container(args: dict[str, Any]) -> str:
    name = _str(args, "container")
    if not _NAME.match(name) or name not in _catalogue()[0]:
        raise OpError(f"container '{name}' is not one of oCabra's services")
    return name


# ── optional services (ComfyUI, A1111, TRELLIS, ACE-Step, …) ─────────────────


async def _container_inspect(args: dict[str, Any]) -> dict[str, Any]:
    container = _catalogue_container(args)
    field = args.get("field")
    if field not in _INSPECT_FIELDS:
        raise OpError(f"'field' must be one of {sorted(_INSPECT_FIELDS)}")
    return await _exec(["docker", "inspect", "-f", _INSPECT_FIELDS[field], container])


async def _container_stats(args: dict[str, Any]) -> dict[str, Any]:
    container = _catalogue_container(args)
    return await _exec(["docker", "stats", "--no-stream", "--format", "{{json .}}", container])


async def _container_action(args: dict[str, Any]) -> dict[str, Any]:
    container = _catalogue_container(args)
    action = args.get("action")
    if action not in _CONTAINER_ACTIONS:
        raise OpError(f"'action' must be one of {list(_CONTAINER_ACTIONS)}")
    return await _exec(["docker", action, container])


async def _compose_up_service(args: dict[str, Any]) -> dict[str, Any]:
    service = _str(args, "service")
    if service not in _catalogue()[1]:
        raise OpError(f"service '{service}' is not one of oCabra's services")
    argv = _compose(services_profile=True) + [
        "up", "-d", "--build", "--pull", "missing", service,
    ]  # fmt: skip
    return await _exec(argv, timeout=None)  # first build of a service can take long


# ── the shared Ollama daemon ─────────────────────────────────────────────────


async def _ollama_ps(_args: dict[str, Any]) -> dict[str, Any]:
    return await _exec(_compose() + ["ps", "-q", "ollama"])


async def _ollama_env(_args: dict[str, Any]) -> dict[str, Any]:
    """``.Config.Env`` of the running ollama container (what it was started with)."""
    ps = await _ollama_ps({})
    container = ps["stdout"].splitlines()[0].strip() if ps["code"] == 0 and ps["stdout"] else ""
    if not _CONTAINER_ID.match(container):
        return _result(1, "", "ollama container not running")
    return await _exec(["docker", "inspect", "-f", "{{json .Config.Env}}", container])


async def _ollama_uses_build(_args: dict[str, Any]) -> dict[str, Any]:
    """Whether the ollama service is built (``build:``) rather than pulled.

    Answers only the boolean: the api used to read ``compose config`` itself,
    which is the whole resolved configuration with every service's secrets.
    """
    config = await _exec(_compose() + ["config", "--format=json"])
    uses_build = False
    if config["code"] == 0 and config["stdout"]:
        try:
            svc = (json.loads(config["stdout"]).get("services") or {}).get("ollama") or {}
            uses_build = "build" in svc
        except ValueError:
            uses_build = False
    return _result(config["code"], "", config["stderr"], uses_build=uses_build)


async def _ollama_pull(_args: dict[str, Any]) -> dict[str, Any]:
    return await _exec(_compose() + ["pull", "ollama"], timeout=None)


async def _ollama_build_pull(_args: dict[str, Any]) -> dict[str, Any]:
    return await _exec(_compose() + ["build", "--pull", "ollama"], timeout=None)


async def _ollama_recreate(_args: dict[str, Any]) -> dict[str, Any]:
    argv = _compose() + ["up", "-d", "--no-deps", "--force-recreate", "ollama"]
    return await _exec(argv, timeout=600)


# ── self-update ──────────────────────────────────────────────────────────────


async def _api_container() -> str | None:
    """The api container: this one, or (from the agent) its compose sibling."""
    if os.environ.get(AGENT_ENV) != "1":
        return socket.gethostname()
    own = await _exec(["docker", "inspect", socket.gethostname()])
    try:
        labels = (json.loads(own["stdout"])[0].get("Config") or {}).get("Labels") or {}
    except (ValueError, IndexError):
        return None
    project = labels.get("com.docker.compose.project")
    if not project:
        return None
    found = await _exec(
        [
            "docker", "ps", "-q",
            "--filter", f"label=com.docker.compose.project={project}",
            "--filter", "label=com.docker.compose.service=api",
        ]
    )  # fmt: skip
    first = found["stdout"].splitlines()[0].strip() if found["stdout"] else ""
    return first if _CONTAINER_ID.match(first) else None


async def _deployment():
    from ocabra.core.self_update import Deployment, deployment_from_inspect

    container = await _api_container()
    if not container:
        return Deployment("manual", "No se encontró el contenedor de la API.")
    inspected = await _exec(["docker", "inspect", container])
    if inspected["code"] != 0:
        return Deployment(
            "manual", f"No se pudo inspeccionar el contenedor: {inspected['stderr'][:200]}"
        )
    try:
        info = json.loads(inspected["stdout"])[0]
    except (ValueError, IndexError):
        return Deployment("manual", "Respuesta inesperada de docker inspect.")
    return deployment_from_inspect(info, container)


async def _self_deployment(_args: dict[str, Any]) -> dict[str, Any]:
    from dataclasses import asdict

    return _result(0, deployment=asdict(await _deployment()))


async def _updater_status(_args: dict[str, Any]) -> dict[str, Any]:
    from ocabra.core.self_update import UPDATER_NAME

    inspected = await _exec(
        ["docker", "inspect", "--format", "{{json .State}}", UPDATER_NAME], timeout=10
    )
    if inspected["code"] != 0:
        return _result(0, status=None)
    try:
        state = json.loads(inspected["stdout"])
    except ValueError:
        return _result(0, status=None)
    logs = await _exec(["docker", "logs", "--tail", "40", UPDATER_NAME], timeout=10)
    text = logs["stdout"] + (("\n" + logs["stderr"]) if logs["stderr"].strip() else "")
    return _result(
        0,
        status={
            "running": state.get("Running"),
            "exit_code": state.get("ExitCode"),
            "finished_at": state.get("FinishedAt"),
            "log": text.splitlines()[-40:],
        },
    )


async def _updater_remove(_args: dict[str, Any]) -> dict[str, Any]:
    from ocabra.core.self_update import UPDATER_NAME

    return await _exec(["docker", "rm", "-f", UPDATER_NAME], timeout=20)


async def _updater_launch(args: dict[str, Any]) -> dict[str, Any]:
    """Start the updater for a published ``X.Y.Z`` release of this same deployment.

    The caller picks the version only; the image, mounts and script are built
    here from the api container's own compose labels.
    """
    from ocabra.core.self_update import parse_version, updater_launch_argv

    target, current = _str(args, "version"), _str(args, "current")
    if not parse_version(target) or not parse_version(current) or target.startswith("v"):
        raise OpError("'version' and 'current' must be X.Y.Z")
    dep = await _deployment()
    if not dep.can_apply:
        raise OpError(dep.reason or "this deployment does not update itself")
    return await _exec(updater_launch_argv(dep, target, current), timeout=60)


# ── TensorRT-LLM engine compilation ──────────────────────────────────────────


def _trtllm_job(args: dict[str, Any]) -> SimpleNamespace:
    from ocabra.core.trtllm_compile_manager import TrtllmCompileManager

    engine = _str(args, "engine_name")
    if not _NAME.match(engine):
        raise OpError("'engine_name' may only contain letters, digits, '.', '_' and '-'")
    model = _str(args, "source_model")
    if not _MODEL_REF.match(model) or ".." in model:
        raise OpError("'source_model' is not a model reference")
    dtype = args.get("dtype")
    if dtype not in TrtllmCompileManager._DTYPE_MAP:
        raise OpError(f"'dtype' must be one of {sorted(TrtllmCompileManager._DTYPE_MAP)}")
    gpus = args.get("gpu_indices")
    if (
        not isinstance(gpus, list)
        or not gpus
        or not all(isinstance(g, int) and not isinstance(g, bool) and 0 <= g < 64 for g in gpus)
    ):
        raise OpError("'gpu_indices' must be a non-empty list of GPU indices")
    raw_config = args.get("config") or {}
    if not isinstance(raw_config, dict):
        raise OpError("'config' must be an object")
    config: dict[str, int] = {}
    for key in _TRTLLM_BUILD_INTS:
        if key in raw_config:
            value = raw_config[key]
            if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= 1 << 20:
                raise OpError(f"config.{key} must be a positive integer")
            config[key] = value
    return SimpleNamespace(
        engine_name=engine, source_model=model, dtype=dtype, gpu_indices=gpus, config=config
    )


async def _trtllm_phase(args: dict[str, Any]) -> AsyncIterator[str]:
    """Run one compile phase, yielding its output lines; the last is ``\\x00exit:<code>``."""
    from ocabra.core.trtllm_compile_manager import build_trtllm_docker_cmd

    phase = args.get("phase")
    if phase not in _TRTLLM_PHASES:
        raise OpError(f"'phase' must be one of {list(_TRTLLM_PHASES)}")
    argv = build_trtllm_docker_cmd(_trtllm_job(args), phase)
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    assert proc.stdout is not None
    try:
        async for raw in proc.stdout:
            yield raw.decode("utf-8", errors="replace").rstrip()
        yield f"{EXIT_MARKER}{await proc.wait()}"
    finally:
        if proc.returncode is None:  # cancelled or the client went away
            proc.kill()
            await proc.wait()


async def _host_helper(entrypoint: str, *args: str) -> dict[str, Any]:
    """A throwaway container in the host's PID namespace, with a fixed command."""
    image = str(settings.tensorrt_llm_host_helper_image).strip() or "ocabra-api:slim"
    return await _exec(
        ["docker", "run", "--rm", "--pid=host", "--entrypoint", entrypoint, image, *args],
        timeout=60,
    )


async def _trtllm_orphan_groups() -> tuple[dict[str, Any], set[int]]:
    from ocabra.backends.tensorrt_llm_backend import TensorRTLLMBackend

    ps = await _host_helper("ps", "-eo", "pid=,pgid=,args=")
    groups: set[int] = set()
    for line in ps["stdout"].splitlines() if ps["code"] == 0 else ():
        parsed = TensorRTLLMBackend._parse_host_ps_line(line)
        if parsed and parsed[1] > 1 and TensorRTLLMBackend._is_orphaned_trt_command(parsed[2]):
            groups.add(parsed[1])
    return ps, groups


async def _trtllm_host_orphans(_args: dict[str, Any]) -> dict[str, Any]:
    ps, groups = await _trtllm_orphan_groups()
    return _result(ps["code"], "", ps["stderr"], pgids=sorted(groups))


async def _trtllm_kill_orphans(args: dict[str, Any]) -> dict[str, Any]:
    """Kill orphaned TRT-LLM process groups on the host.

    Only groups that are orphaned ``trtllm-serve`` runs right now (same rule as
    the backend) are killed, whatever ``pgids`` asks for: a caller cannot use
    this to kill an arbitrary host process.
    """
    raw = args.get("pgids")
    if not isinstance(raw, list) or not all(isinstance(p, int) and p > 1 for p in raw):
        raise OpError("'pgids' must be a list of process group ids")
    _ps, orphans = await _trtllm_orphan_groups()
    targets = sorted(set(raw) & orphans)
    if not targets:
        return _result(0, killed=[])
    groups = " ".join(f"-{pgid}" for pgid in targets)
    script = f"kill -TERM -- {groups} 2>/dev/null || true; sleep 2; kill -KILL -- {groups} 2>/dev/null || true"
    killed = await _host_helper("sh", "-lc", script)
    return _result(killed["code"], "", killed["stderr"], killed=targets)


EXIT_MARKER = "\x00exit:"

OPERATIONS: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = {
    "container.inspect": _container_inspect,
    "container.stats": _container_stats,
    "container.action": _container_action,
    "compose.up_service": _compose_up_service,
    "ollama.ps": _ollama_ps,
    "ollama.env": _ollama_env,
    "ollama.uses_build": _ollama_uses_build,
    "ollama.pull": _ollama_pull,
    "ollama.build_pull": _ollama_build_pull,
    "ollama.recreate": _ollama_recreate,
    "self.deployment": _self_deployment,
    "updater.status": _updater_status,
    "updater.remove": _updater_remove,
    "updater.launch": _updater_launch,
    "trtllm.host_orphans": _trtllm_host_orphans,
    "trtllm.kill_orphans": _trtllm_kill_orphans,
}

STREAM_OPERATIONS: dict[str, Callable[[dict[str, Any]], AsyncIterator[str]]] = {
    "trtllm.phase": _trtllm_phase,
}


async def run(op: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run operation ``op``; raises :class:`OpError` for anything not allowed."""
    handler = OPERATIONS.get(op)
    if handler is None:
        raise OpError(f"unknown operation '{op}'")
    return await handler(dict(args or {}))


def stream(op: str, args: dict[str, Any] | None = None) -> AsyncIterator[str]:
    """Start streaming operation ``op`` (validation errors raise on first iteration)."""
    handler = STREAM_OPERATIONS.get(op)
    if handler is None:
        raise OpError(f"unknown operation '{op}'")
    return handler(dict(args or {}))
