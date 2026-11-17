"""oCabra updating itself.

Releases are git tags ``vX.Y.Z``; CI (``.github/workflows/release.yml``)
publishes ``ocabra-api`` and ``ocabra-frontend`` with that version as the image
tag. A node knows the latest release by listing the tags of
``<self_update_image_repo>/ocabra-api`` and comparing them with ``APP_VERSION``.

How a node applies one depends on how it runs (``detect_deployment``):

* ``compose`` — the docker-compose stack (hub, Mapache) whose api/frontend
  images come from the release registry. The API cannot restart itself and keep
  going, so it starts a one-shot *updater* container (its own image, which has
  the docker CLI and git) with the docker socket and the project folder. The
  updater fast-forwards the git checkout to the tag (``docker-compose.yml`` and
  the rest move with the release), points ``OCABRA_API_IMAGE`` and
  ``OCABRA_FRONTEND_IMAGE`` in ``.env`` at the new tag, pulls, recreates api and
  frontend and waits for ``/health`` to report the new version; if it does
  not, it restores ``.env`` and the previous commit and recreates them again.
  Migrations run in the new API's entrypoint, as on every start.
* ``local_build`` — the same stack building its images from source (the hub
  today). It is told about new releases but updated by hand.
* ``node`` — the ``ocabra-node`` image on a rented GPU. It is replaced from the
  hub (a new pod image), not from inside.
* ``manual`` — anything else (no docker socket, not a compose container).

The policy (``self_update_policy``) is per node: ``off``, ``notify`` (default:
show it, apply on request) or ``auto`` (apply once the node has been idle —
no requests in flight, no active batches, no downloads — for
``self_update_idle_minutes``).
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from ocabra.config import APP_VERSION, settings

logger = structlog.get_logger(__name__)

POLICIES = ("off", "notify", "auto")
UPDATER_NAME = "ocabra-updater"
_SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
_LATEST_TTL_S = 600.0


def parse_version(text: str | None) -> tuple[int, int, int] | None:
    match = _SEMVER.match((text or "").strip())
    return tuple(int(g) for g in match.groups()) if match else None  # type: ignore[return-value]


def newest(tags: list[str]) -> str | None:
    """Highest ``X.Y.Z`` among *tags* (``v`` prefix allowed), as ``X.Y.Z``."""
    versions = [v for v in (parse_version(t) for t in tags) if v]
    if not versions:
        return None
    return ".".join(str(n) for n in max(versions))


def is_newer(candidate: str | None, current: str = APP_VERSION) -> bool:
    a, b = parse_version(candidate), parse_version(current)
    return bool(a and b and a > b)


def image_for(component: str, version: str) -> str:
    repo = settings.self_update_image_repo.rstrip("/")
    return f"{repo}/ocabra-{component}:{version}"


# ── deployment ───────────────────────────────────────────────────


@dataclass
class Deployment:
    kind: str  # compose | local_build | node | manual
    reason: str = ""
    project: str | None = None
    working_dir: str | None = None
    config_files: list[str] = field(default_factory=list)
    container: str | None = None
    api_image: str | None = None
    services: list[str] = field(default_factory=lambda: ["api", "frontend"])

    @property
    def can_apply(self) -> bool:
        return self.kind == "compose"


async def detect_deployment() -> Deployment:
    """How this API runs, read from its own container's compose labels."""
    if os.environ.get("OCABRA_NODE_IMAGE") == "1":
        return Deployment("node", "Imagen ocabra-node: se reemplaza desde el hub.")
    from ocabra.core import docker_ops

    if not docker_ops.available():
        return Deployment(
            "manual", "Sin acceso a Docker: no hay agente de Docker ni /var/run/docker.sock."
        )
    try:
        result = await docker_ops.call("self.deployment")
    except docker_ops.AgentUnavailable as exc:
        return Deployment("manual", f"El agente de Docker no responde ({exc}).")
    return Deployment(**result["deployment"])


def deployment_from_inspect(info: dict, fallback_name: str) -> Deployment:
    """A :class:`Deployment` from the ``docker inspect`` of the api container."""
    labels = (info.get("Config") or {}).get("Labels") or {}
    project = labels.get("com.docker.compose.project")
    working_dir = labels.get("com.docker.compose.project.working_dir")
    if not project or not working_dir:
        return Deployment("manual", "El contenedor no lo gestiona docker compose.")
    image = (info.get("Config") or {}).get("Image") or ""
    files = [
        f for f in (labels.get("com.docker.compose.project.config_files") or "").split(",") if f
    ]
    dep = Deployment(
        "compose",
        project=project,
        working_dir=working_dir,
        config_files=files,
        container=info.get("Name", "").lstrip("/") or fallback_name,
        api_image=image,
    )
    repo = settings.self_update_image_repo.rstrip("/") + "/ocabra-api"
    if not image.startswith(repo + ":"):
        dep.kind = "local_build"
        dep.reason = (
            f"Las imágenes se construyen en local ({image or '?'}): para actualizarse sola, "
            f"OCABRA_API_IMAGE y OCABRA_FRONTEND_IMAGE tienen que apuntar a {repo}."
        )
    return dep


def updater_launch_argv(dep: Deployment, target: str, current: str) -> list[str]:
    """``docker run`` argv that starts the two-stage updater for ``target``.

    Two stages. This image only pulls the target release and hands over to
    it: the update itself runs with the *new* image's docker compose. With the
    running one, a node whose compose was too old for the new
    docker-compose.yml (v2.31 vs `gpus: all`) could never update itself.
    """
    script = updater_script(dep, target, current)
    new_image = image_for("api", target)
    mounts = (
        f"-v /var/run/docker.sock:/var/run/docker.sock "
        f"-v {shlex.quote(str(dep.working_dir))}:{shlex.quote(str(dep.working_dir))} "
        f"-w {shlex.quote(str(dep.working_dir))}"
    )
    stage1 = (
        f'echo "[ocabra-updater] descargando {new_image}"; '
        f"docker pull -q {shlex.quote(new_image)} 2>&1 || "
        f'{{ echo "[ocabra-updater] ERROR: no se pudo descargar {new_image}"; exit 5; }}; '
        f"exec docker run --rm --label org.ocabra.updater=2 {mounts} "
        f"--entrypoint sh {shlex.quote(new_image)} -c {shlex.quote(script)}"
    )
    return [
        "docker", "run", "-d", "--name", UPDATER_NAME,
        "--label", "org.ocabra.updater=1",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "--entrypoint", "sh",
        str(dep.api_image),
        "-c", stage1,
    ]  # fmt: skip


# ── idle check ───────────────────────────────────────────────────


async def busy_reasons(app: Any) -> list[str]:
    """What would be cut by a restart now (empty list = idle)."""
    reasons: list[str] = []
    mm = getattr(app.state, "model_manager", None)
    if mm is not None:
        inflight = sum(mm.inflight_snapshot().values())
        if inflight:
            reasons.append(f"{inflight} peticiones en curso")
    try:
        import sqlalchemy as sa

        from ocabra.database import AsyncSessionLocal
        from ocabra.db.openai_batches import OpenAIBatch

        async with AsyncSessionLocal() as session:
            active = (
                await session.execute(
                    sa.select(sa.func.count())
                    .select_from(OpenAIBatch)
                    .where(OpenAIBatch.status.in_(("validating", "in_progress", "finalizing")))
                )
            ).scalar_one()
        if active:
            reasons.append(f"{active} batches activos")
    except Exception as exc:  # noqa: BLE001 - unknown is not idle
        reasons.append(f"no se pudo comprobar los batches ({exc})")
    try:
        from ocabra.api.internal.downloads import download_manager

        downloads = len(getattr(download_manager, "_active_cancel_flags", {}) or {})
        if downloads:
            reasons.append(f"{downloads} descargas en curso")
    except Exception:  # noqa: BLE001
        pass
    return reasons


# ── updater container ────────────────────────────────────────────


def updater_script(dep: Deployment, version: str, current: str, health_tries: int = 90) -> str:
    """Shell run by the updater container (in the project folder)."""
    api_new = image_for("api", version)
    front_new = image_for("frontend", version)
    files = " ".join(f"-f {shlex.quote(f)}" for f in dep.config_files)
    compose = f"docker compose -p {shlex.quote(dep.project or 'ocabra')} {files}".strip()
    services = shlex.quote(" ".join(dep.services))
    tag = f"v{version}"
    return f"""
set -u
# docker compose reports on stderr; keep it in the same log the panel shows.
exec 2>&1
log() {{ echo "[ocabra-updater] $*"; }}
owner=$(stat -c %u:%g .)
as_owner() {{ HOME=/tmp setpriv --reuid "${{owner%%:*}}" --regid "${{owner##*:}}" --clear-groups "$@"; }}
set_env() {{
  key=$1; value=$2
  if grep -q "^$key=" .env 2>/dev/null; then
    sed -i "s|^$key=.*|$key=$value|" .env
  else
    printf '%s=%s\\n' "$key" "$value" >> .env
  fi
}}
prev_commit=""
if [ -d .git ]; then
  prev_commit=$(as_owner git rev-parse HEAD)
  log "git: {tag} (desde $prev_commit)"
  if ! as_owner git fetch --quiet origin tag {shlex.quote(tag)}; then
    log "ERROR: no se pudo descargar la etiqueta {tag}"; exit 2
  fi
  if ! as_owner git merge --ff-only --quiet {shlex.quote(tag)}; then
    log "ERROR: el repositorio tiene cambios locales o no avanza en línea recta hasta {tag}"; exit 3
  fi
fi
cp -p .env .env.before-update 2>/dev/null || true
set_env OCABRA_API_IMAGE {shlex.quote(api_new)}
set_env OCABRA_FRONTEND_IMAGE {shlex.quote(front_new)}
# docker-agent (the container that holds the Docker socket and launched this
# updater) uses the api image too, so it is pulled and recreated with it when
# the checked-out docker-compose.yml defines it. Asked of the compose file in
# use at each step: the release may add it, and a rollback to a version
# without it must not ask for it.
with_agent() {{
  svc={services}
  if {compose} config --services 2>/dev/null | grep -qx docker-agent; then
    case " $svc " in *" docker-agent "*) ;; *) svc="$svc docker-agent" ;; esac
  fi
  echo "$svc"
}}
rollback() {{
  log "ROLLBACK a {current}: $1"
  [ -f .env.before-update ] && mv -f .env.before-update .env
  [ -n "$prev_commit" ] && as_owner git reset --quiet --hard "$prev_commit"
  {compose} up -d --no-build $(with_agent)
  exit 4
}}
services=$(with_agent)
log "pull {api_new} {front_new}"
{compose} pull $services || rollback "pull falló"
log "recreando $services"
{compose} up -d --no-build $services || rollback "up falló"
for i in $(seq 1 {int(health_tries)}); do
  sleep 4
  got=$({compose} exec -T api curl -sf localhost:8000/health 2>/dev/null || true)
  case "$got" in *'"{version}"'*) log "OK: {version} en marcha"; rm -f .env.before-update; exit 0;; esac
done
rollback "la API nueva no respondió con {version} en {int(health_tries) * 4} s"
"""


@dataclass
class UpdateState:
    phase: str = "idle"  # idle | checking | waiting_idle | updating | failed | done
    current: str = APP_VERSION
    latest: str | None = None
    checked_at: float | None = None
    target: str | None = None
    started_at: float | None = None
    message: str | None = None
    busy: list[str] = field(default_factory=list)
    idle_since: float | None = None
    # Release whose update failed: "auto" does not retry it on its own (a
    # newer one, or the button, does). Without this a failing pull repeated
    # every hour.
    failed_target: str | None = None


class SelfUpdater:
    def __init__(self, app: Any) -> None:
        self._app = app
        self.state = UpdateState()
        self._deployment: Deployment | None = None
        self._lock = asyncio.Lock()

    async def deployment(self, refresh: bool = False) -> Deployment:
        if self._deployment is None or refresh:
            self._deployment = await detect_deployment()
        return self._deployment

    async def check(self, force: bool = False) -> str | None:
        if (
            not force
            and self.state.checked_at
            and time.time() - self.state.checked_at < _LATEST_TTL_S
        ):
            return self.state.latest
        from ocabra.core.oci_registry import OciRef, RegistryClient

        ref = OciRef.parse(image_for("api", "latest"))
        try:
            async with RegistryClient(ref, pull_only=True) as client:
                self.state.latest = newest(await client.tags())
        except Exception as exc:  # noqa: BLE001
            # GHCR answers "denied" for a package that does not exist yet.
            if "DENIED" in str(exc) or "denied" in str(exc):
                self.state.latest = None
                self.state.checked_at = time.time()
                self.state.message = (
                    f"Todavía no hay versiones publicadas en {ref.registry}/{ref.repository}."
                )
                return None
            logger.info("self_update_check_failed", error=str(exc))
            self.state.message = f"No se pudo consultar {ref.registry}: {exc}"
            return self.state.latest
        self.state.checked_at = time.time()
        if self.state.message and self.state.message.startswith(
            ("No se pudo consultar", "Todavía no hay")
        ):
            self.state.message = None
        return self.state.latest

    async def snapshot(self) -> dict[str, Any]:
        dep = await self.deployment()
        updater = await self._updater_status()
        if self.state.phase == "updating" and updater and not updater.get("running"):
            # The updater finished and this API is still the old one: it
            # rolled back (or failed before touching anything). A successful
            # update replaces this process, so it never gets here.
            code = updater.get("exit_code")
            self.state.phase = "failed"
            self.state.failed_target = self.state.target
            self.state.message = (
                f"La actualización a {self.state.target} falló y se volvió a {APP_VERSION}"
                if code == 4
                else f"La actualización a {self.state.target} falló (código {code}); no se cambió nada"
            )
        out = asdict(self.state)
        out.update(
            policy=settings.self_update_policy,
            idle_minutes=settings.self_update_idle_minutes,
            update_available=is_newer(self.state.latest),
            deployment={
                "kind": dep.kind,
                "reason": dep.reason,
                "can_apply": dep.can_apply,
                "api_image": dep.api_image,
                "project": dep.project,
            },
            updater=updater,
        )
        return out

    async def _updater_status(self) -> dict[str, Any] | None:
        from ocabra.core import docker_ops

        try:
            return (await docker_ops.call("updater.status")).get("status")
        except docker_ops.AgentUnavailable:
            return None  # the panel still shows the rest; apply() reports the agent

    async def apply(self, version: str | None = None, *, force: bool = False) -> dict[str, Any]:
        """Start the updater container. Raises ``RuntimeError`` with the reason
        when it cannot (not a managed deployment, busy, nothing newer)."""
        async with self._lock:
            dep = await self.deployment(refresh=True)
            if not dep.can_apply:
                raise RuntimeError(dep.reason or "Este despliegue no se actualiza solo.")
            target = version or await self.check(force=True)
            if not target or not parse_version(target):
                raise RuntimeError("No hay ninguna versión publicada.")
            if not force and not is_newer(target):
                raise RuntimeError(f"Ya está en {APP_VERSION} (publicada: {target}).")
            busy = await busy_reasons(self._app)
            if busy and not force:
                raise RuntimeError("Ocupado: " + ", ".join(busy))
            status = await self._updater_status()
            if status and status.get("running"):
                raise RuntimeError("Ya hay una actualización en marcha.")
            from ocabra.core import docker_ops

            await docker_ops.call("updater.remove")
            result = await docker_ops.call(
                "updater.launch", version=target, current=APP_VERSION
            )
            code, out, err = result["code"], result.get("stdout", ""), result.get("stderr", "")
            if code != 0:
                self.state.phase = "failed"
                self.state.message = f"No se pudo lanzar el actualizador: {err.strip()[:300]}"
                raise RuntimeError(self.state.message)
            self.state.phase = "updating"
            self.state.idle_since = None
            self.state.target = target
            self.state.started_at = time.time()
            self.state.message = f"Actualizando a {target}: la API se reiniciará."
            logger.info("self_update_started", target=target, container=out.strip()[:12])
            return await self.snapshot()

    async def tick(self) -> None:
        """One iteration of the background loop."""
        policy = settings.self_update_policy
        if policy == "off":
            return
        latest = await self.check()
        if policy != "auto" or not is_newer(latest):
            self.state.idle_since = None
            return
        await self.snapshot()  # closes a finished update attempt
        if latest == self.state.failed_target or self.state.phase == "updating":
            return
        dep = await self.deployment()
        if not dep.can_apply:
            return
        busy = await busy_reasons(self._app)
        self.state.busy = busy
        now = time.time()
        if busy:
            self.state.idle_since = None
            self.state.phase = "waiting_idle"
            return
        self.state.idle_since = self.state.idle_since or now
        if now - self.state.idle_since < settings.self_update_idle_minutes * 60:
            self.state.phase = "waiting_idle"
            return
        try:
            await self.apply(latest)
        except RuntimeError as exc:
            self.state.message = str(exc)

    async def run(self, stop: asyncio.Event) -> None:
        # First look a minute after start: a node fresh from an update does not
        # chain into another one before it has settled.
        interval = max(60, int(settings.self_update_check_interval_s or 3600))
        delay = 60.0
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
                return
            except TimeoutError:
                pass
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001
                logger.warning("self_update_tick_failed", error=str(exc))
            # Waiting for idle polls faster than looking for releases.
            delay = 60.0 if self.state.phase == "waiting_idle" else float(interval)
