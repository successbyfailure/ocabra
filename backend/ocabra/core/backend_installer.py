"""Backend installer — Bloque 15 Fase 1.

Runtime install/uninstall of inference backends as modular units under
``settings.backends_dir``.  Each backend lives in ``<backends_dir>/<backend_type>/``
with an isolated Python ``venv`` (for source installs) and a ``metadata.json``
describing the installed version.

This module only covers Fase 1:

* discovery of already-installed modules (scan of metadata.json),
* "built-in" detection for backends that are pre-installed in the image,
* installation from source (``venv + pip``),
* uninstallation (refuses if any model using the backend is loaded),
* concurrent-install safety (per-backend :class:`asyncio.Lock`).

Prebuilt installs (``method="oci"``) download the backend's folder as
published in ``settings.backend_oci_repository`` (see ``core/oci_registry.py``)
— the exact versions that run on the server that published it, without pip or
compilers. ``method="auto"`` uses it when the backend is published and falls
back to the source install otherwise (or if the download fails). Installing
over an installed backend (an *update*) unpacks next to it and swaps the
folders, so a failure leaves the old one in place.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ocabra.backends.base import BackendInstallSpec, BackendInterface

if TYPE_CHECKING:
    from ocabra.core.worker_pool import WorkerPool

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants and data types
# ---------------------------------------------------------------------------

METADATA_FILENAME = "metadata.json"
METADATA_SCHEMA_VERSION = 1
INSTALL_LOG_MAX_LINES = 500


class BackendInstallStatus(StrEnum):
    """Lifecycle status of a modular backend install."""

    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    UNINSTALLING = "uninstalling"
    ERROR = "error"


@dataclass
class BackendModuleState:
    """Public state exposed through the API for a single backend module.

    The dataclass mirrors the JSON shape consumed by the frontend.  Any new
    field added here must be JSON-serialisable via :func:`dataclasses.asdict`
    combined with ISO-8601 conversion for datetime values (see
    :meth:`BackendInstaller._state_to_dict`).
    """

    backend_type: str
    display_name: str
    description: str
    tags: list[str]
    install_status: BackendInstallStatus
    installed_version: str | None = None
    installed_at: datetime | None = None
    install_source: str | None = None  # "oci" | "source" | "built-in"
    estimated_size_mb: int = 0
    actual_size_mb: int | None = None
    error: str | None = None
    install_progress: float | None = None
    install_detail: str | None = None
    # Prebuilt installs: the image it came from and its digest.
    oci_ref: str | None = None
    oci_digest: str | None = None
    # Installed versions of the packages the spec names (venv backends) and
    # what the spec asks for: "declared vs installed" at a glance.
    packages: dict[str, str] = field(default_factory=dict)
    declared_packages: list[str] = field(default_factory=list)
    git_ref: str | None = None


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


class BackendAlreadyInstallingError(RuntimeError):
    """Raised when a second concurrent install is attempted for the same backend."""


class BackendInstaller:
    """Manage runtime install/uninstall of inference backends.

    The installer holds its own view of each known backend's state and acts as
    the authoritative source for the ``/ocabra/backends`` API.  The worker pool
    is consulted (and mutated) only through the two well-known entry points:

    * :meth:`WorkerPool.register_backend` on a successful install / start,
    * :meth:`WorkerPool.register_disabled_backend` when a backend is not (yet)
      installed — the model manager then surfaces a readable error when the
      user tries to use a model assigned to that backend.
    """

    def __init__(
        self,
        backends_dir: Path | str,
        worker_pool: WorkerPool,
        *,
        backend_registry: dict[str, BackendInterface] | None = None,
        assume_fat_image: bool = True,
    ) -> None:
        self._backends_dir = Path(backends_dir)
        self._worker_pool = worker_pool
        # On the fat image the oCabra container has every backend's pip deps
        # pre-installed, so a backend with an install_spec but no metadata is
        # treated as "built-in" to stay backwards compatible.  On the slim
        # image the same situation means the backend is genuinely not
        # installed and the UI should offer an install button.
        self._assume_fat_image = assume_fat_image
        # backend_type → instance (lazy-instantiated source of truth for specs
        # + a way to re-register after an install).  Fase 1 expects the caller
        # to pre-populate this from ``main.py`` using the instances that are
        # already wired into the worker pool, so we avoid re-importing every
        # backend class here (those imports are heavy).
        self._backends: dict[str, BackendInterface] = dict(backend_registry or {})
        self._states: dict[str, BackendModuleState] = {}
        self._install_locks: dict[str, asyncio.Lock] = {}
        # Detached install tasks per backend.  Decouples the actual install
        # from the SSE consumer so that a client disconnection does not orphan
        # the state machine half-way through pip (deuda 9f).
        self._install_tasks: dict[str, asyncio.Task[None]] = {}
        # Last install log per backend (in-memory ring buffer) — exposed via
        # ``GET /ocabra/backends/{type}/logs``.
        self._install_logs: dict[str, list[str]] = {}
        # Set to False to make ``_run_pip_install`` delegate to
        # ``_run_subprocess`` (i.e. no per-line parsing).  Tests that
        # monkeypatch ``_run_subprocess`` set this to False so they don't have
        # to mock the pip-progress path separately.
        self._pip_progress_enabled = True
        # backend_type -> (monotonic time, published summary | None)
        self._published_cache: dict[str, tuple[float, dict | None]] = {}

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def register_backend(self, backend_type: str, backend: BackendInterface) -> None:
        """Record a backend instance without yet touching the worker pool.

        Call this before :meth:`start` for every backend that should appear in
        the modular UI.  The installer still works if some backends are missing
        from the registry — they simply will not show up in
        :meth:`list_states`.
        """
        self._backends[backend_type] = backend

    @property
    def backends_dir(self) -> Path:
        return self._backends_dir

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Scan the backends directory and reconcile with the in-memory registry.

        Side effects:
            * Creates ``backends_dir`` if it does not exist.
            * Populates ``self._states`` with a :class:`BackendModuleState`
              entry for every known backend.
            * Does **not** register backends with the worker pool — the caller
              (``main.py``) continues to own that registration during Fase 1
              while backends remain baked into the fat image.  We only keep
              track of who is installed vs. not.
        """
        await asyncio.to_thread(self._backends_dir.mkdir, parents=True, exist_ok=True)

        for backend_type, backend in self._backends.items():
            spec = backend.install_spec
            meta = await self._load_metadata(backend_type)

            if spec is None:
                # Backend is always available (e.g. ollama) — mark as built-in.
                builtin = _BUILTIN_INFO.get(backend_type, {})
                self._states[backend_type] = BackendModuleState(
                    backend_type=backend_type,
                    display_name=builtin.get("display_name", backend_type),
                    description=builtin.get("description", "External or always-available backend."),
                    tags=list(builtin.get("tags", [])),
                    install_status=BackendInstallStatus.INSTALLED,
                    install_source="built-in",
                    installed_at=None,
                )
                continue

            if meta is not None:
                self._states[backend_type] = await self._installed_state(
                    backend_type, spec, meta
                )
            elif self._assume_fat_image:
                # No metadata on disk but the fat image has every backend
                # pre-installed — keep it usable.  Fase 2 will migrate each
                # backend to write its own metadata on first boot.
                self._states[backend_type] = BackendModuleState(
                    backend_type=backend_type,
                    display_name=spec.display_name or backend_type,
                    description=spec.description,
                    tags=list(spec.tags),
                    install_status=BackendInstallStatus.INSTALLED,
                    install_source="built-in",
                    estimated_size_mb=spec.estimated_size_mb,
                )
            else:
                # Slim image and no metadata → genuinely not installed.
                self._states[backend_type] = BackendModuleState(
                    backend_type=backend_type,
                    display_name=spec.display_name or backend_type,
                    description=spec.description,
                    tags=list(spec.tags),
                    install_status=BackendInstallStatus.NOT_INSTALLED,
                    estimated_size_mb=spec.estimated_size_mb,
                )
                self._worker_pool.register_disabled_backend(
                    backend_type, "not installed"
                )

        logger.info(
            "backend_installer_ready",
            backends=sorted(self._states.keys()),
            backends_dir=str(self._backends_dir),
        )

    async def _installed_state(
        self, backend_type: str, spec: BackendInstallSpec, meta: dict[str, Any]
    ) -> BackendModuleState:
        from ocabra.core.oci_registry import installed_packages, package_names

        packages = meta.get("packages")
        if not isinstance(packages, dict):
            packages = await asyncio.to_thread(
                installed_packages, meta.get("python_bin"), package_names(spec.pip_packages)
            )
        return BackendModuleState(
            backend_type=backend_type,
            display_name=spec.display_name or backend_type,
            description=spec.description,
            tags=list(spec.tags),
            install_status=BackendInstallStatus.INSTALLED,
            installed_version=display_version(
                backend_type, list(spec.pip_packages), dict(packages), meta.get("version")
            ),
            installed_at=_parse_iso(meta.get("installed_at")),
            install_source=meta.get("install_source"),
            estimated_size_mb=spec.estimated_size_mb,
            actual_size_mb=meta.get("size_mb"),
            oci_ref=meta.get("oci_ref"),
            oci_digest=meta.get("oci_digest"),
            packages=dict(packages),
            declared_packages=list(spec.pip_packages),
            git_ref=meta.get("git_ref"),
        )

    async def published(self, backend_type: str, *, refresh: bool = False) -> dict | None:
        """What the OCI repository offers for *backend_type* (cached 10 min).

        ``None`` when nothing is published or OCI is off; raises
        :class:`~ocabra.core.oci_registry.OciError` when the registry fails.
        """
        import time

        from ocabra.config import settings
        from ocabra.core.oci_registry import inspect_backend

        if not settings.backend_oci_repository:
            return None
        cached = self._published_cache.get(backend_type)
        if cached and not refresh and time.monotonic() - cached[0] < 600:
            return cached[1]
        found = await inspect_backend(settings.backend_oci_repository, backend_type)
        summary = found.summary() if found else None
        self._published_cache[backend_type] = (time.monotonic(), summary)
        return summary

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_state(self, backend_type: str) -> BackendModuleState:
        """Return the current :class:`BackendModuleState` for *backend_type*."""
        if backend_type not in self._states:
            raise KeyError(f"Backend '{backend_type}' is not registered")
        return self._states[backend_type]

    def list_states(self) -> list[BackendModuleState]:
        """Return all known backend states sorted by ``backend_type``."""
        return [self._states[bt] for bt in sorted(self._states.keys())]

    def get_logs(self, backend_type: str) -> list[str]:
        """Return the captured install logs for *backend_type* (may be empty)."""
        if backend_type not in self._states:
            raise KeyError(f"Backend '{backend_type}' is not registered")
        return list(self._install_logs.get(backend_type, []))

    # ------------------------------------------------------------------
    # Install / Uninstall
    # ------------------------------------------------------------------

    async def install(
        self, backend_type: str, method: str = "auto"
    ) -> AsyncIterator[BackendModuleState]:
        """Install *backend_type* and yield :class:`BackendModuleState` updates.

        The generator is suitable for driving an SSE stream: each yielded
        state can be serialised to JSON and forwarded to the client.

        Args:
            backend_type: Registered backend identifier.
            method: ``"source"`` (venv + pip / build), ``"oci"`` (the
                prebuilt image) or ``"auto"`` (prebuilt when published,
                source otherwise or if the download fails).

        Yields:
            Successive :class:`BackendModuleState` snapshots with
            ``install_progress`` and ``install_detail`` set while the
            installation is running, followed by a final ``INSTALLED`` or
            ``ERROR`` state.

        Raises:
            KeyError: If *backend_type* is not registered.
            BackendAlreadyInstallingError: If another install is already in
                progress for the same backend.
            RuntimeError: Replacing an installed backend while models use it.
        """
        if backend_type not in self._backends:
            raise KeyError(f"Backend '{backend_type}' is not registered")

        existing_task = self._install_tasks.get(backend_type)
        if existing_task is not None and not existing_task.done():
            raise BackendAlreadyInstallingError(
                f"Backend '{backend_type}' is already being installed"
            )

        if method not in {"source", "oci", "auto"}:
            raise ValueError(f"Unknown install method '{method}'")

        backend = self._backends[backend_type]
        spec = backend.install_spec
        if spec is None:
            raise ValueError(
                f"Backend '{backend_type}' is always-available and cannot be installed"
            )
        # Swapping a venv under a running worker breaks its lazy imports.
        blocker = self.reinstall_blocker(backend_type)
        if blocker:
            raise RuntimeError(blocker)

        # Run the actual install as a detached task: a client disconnect must
        # not orphan the state machine (deuda 9f).  We bridge the task and
        # the SSE consumer through an unbounded asyncio.Queue.  The task
        # always emits a terminal sentinel (``None``) so consumers exit.
        queue: asyncio.Queue[BackendModuleState | None] = asyncio.Queue()
        lock = self._install_locks.setdefault(backend_type, asyncio.Lock())

        async def _runner() -> None:
            async with lock:
                self._install_logs[backend_type] = []
                try:
                    use_oci = method == "oci"
                    if method == "auto":
                        use_oci = await self._oci_available(backend_type)
                    if use_oci:
                        failed = False
                        async for state in self._install_from_oci(backend_type, backend, spec):
                            failed = state.install_status == BackendInstallStatus.ERROR
                            if failed and method == "auto":
                                break  # fall back to source below
                            queue.put_nowait(state)
                        if not (failed and method == "auto"):
                            return
                        self._log(backend_type, "Prebuilt install failed: building from source")
                    async for state in self._install_from_source(
                        backend_type, backend, spec
                    ):
                        # _install_from_source already mutates self._states.
                        queue.put_nowait(state)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "backend_install_runner_failed", backend_type=backend_type
                    )
                    err_state = self._states.get(backend_type)
                    if err_state is None:
                        err_state = BackendModuleState(
                            backend_type=backend_type,
                            display_name=spec.display_name or backend_type,
                            description=spec.description,
                            tags=list(spec.tags),
                            install_status=BackendInstallStatus.ERROR,
                            estimated_size_mb=spec.estimated_size_mb,
                        )
                    err_state.install_status = BackendInstallStatus.ERROR
                    err_state.error = str(exc)
                    err_state.install_progress = None
                    err_state.install_detail = None
                    self._states[backend_type] = err_state
                    queue.put_nowait(err_state)
                finally:
                    queue.put_nowait(None)

        task = asyncio.create_task(
            _runner(), name=f"install-{backend_type}"
        )
        self._install_tasks[backend_type] = task

        try:
            while True:
                state = await queue.get()
                if state is None:
                    return
                yield state
        except asyncio.CancelledError:
            # Client disconnected; the install task keeps running on its
            # own and will land the final state in self._states.
            raise

    def reinstall_blocker(self, backend_type: str) -> str | None:
        """Why *backend_type* can't be (re)installed right now, if it can't."""
        current = self._states.get(backend_type)
        if (
            current is not None
            and current.install_status == BackendInstallStatus.INSTALLED
            and _count_models_for_backend(self._worker_pool, backend_type) > 0
        ):
            return f"Cannot reinstall '{backend_type}' while models use it: unload them first"
        return None

    async def _oci_available(self, backend_type: str) -> bool:
        try:
            return await self.published(backend_type, refresh=True) is not None
        except Exception as exc:  # noqa: BLE001 — registry down: source still works
            self._log(backend_type, f"Prebuilt image unavailable ({exc}): building from source")
            return False

    async def _install_from_oci(
        self,
        backend_type: str,
        backend: BackendInterface,
        spec: BackendInstallSpec,
    ) -> AsyncIterator[BackendModuleState]:
        """Download the published folder, check it runs here, swap it in."""
        import uuid

        from ocabra.config import settings
        from ocabra.core.oci_registry import inspect_backend, pull_backend

        target_dir = self._backends_dir / backend_type
        work = self._backends_dir / f".oci-{backend_type}-{uuid.uuid4().hex[:8]}"
        state = self._states[backend_type] = BackendModuleState(
            backend_type=backend_type,
            display_name=spec.display_name or backend_type,
            description=spec.description,
            tags=list(spec.tags),
            install_status=BackendInstallStatus.INSTALLING,
            estimated_size_mb=spec.estimated_size_mb,
            install_progress=0.0,
            install_detail="Looking up the prebuilt image...",
        )
        yield self._snapshot(state)
        try:
            published = await inspect_backend(settings.backend_oci_repository, backend_type)
            if published is None:
                raise RuntimeError(f"No prebuilt image for {backend_type}")
            from ocabra.core.cuda_probe import probe_cuda
            from ocabra.core.oci_registry import cuda_tuple

            need_cuda = published.labels.get("org.ocabra.backend.cuda")
            have_cuda = (await asyncio.to_thread(probe_cuda)).get("driver_cuda")
            if need_cuda and have_cuda and cuda_tuple(have_cuda) < cuda_tuple(need_cuda):
                raise RuntimeError(
                    f"The prebuilt image needs CUDA {need_cuda}; this machine's driver "
                    f"supports up to {have_cuda}"
                )
            want_py = published.labels.get("org.ocabra.backend.python")
            have_py = f"{sys.version_info.major}.{sys.version_info.minor}"
            if want_py and want_py != have_py:
                raise RuntimeError(
                    f"The prebuilt image needs Python {want_py}; this server runs {have_py}"
                )
            self._log(
                backend_type,
                f"Pulling {published.ref} ({published.digest}, "
                f"{published.size / 1e9:.2f} GB in {len(published.layers)} layers)",
            )
            # Progress is pushed from the download callbacks: snapshot it
            # every couple of seconds instead of on every chunk.
            loop_state = {"done": 0, "total": published.size or 1}

            def _progress(done: int, total: int) -> None:
                loop_state.update(done=done, total=total)

            pull = asyncio.create_task(pull_backend(published, work, _progress))
            while not pull.done():
                await asyncio.wait({pull}, timeout=2.0)
                frac = loop_state["done"] / loop_state["total"]
                state.install_progress = round(0.02 + 0.8 * frac, 3)
                state.install_detail = (
                    f"Downloading {loop_state['done'] / 1e9:.1f} / "
                    f"{loop_state['total'] / 1e9:.1f} GB"
                    if frac < 1
                    else "Unpacking..."
                )
                yield self._snapshot(state)
            tree = pull.result()

            state.install_progress = 0.86
            state.install_detail = "Checking it runs here..."
            yield self._snapshot(state)
            meta = json.loads(await asyncio.to_thread((tree / METADATA_FILENAME).read_text))
            await self._check_prebuilt(backend_type, tree, target_dir, meta)

            # Swap: the old folder stays until the new one is in place.
            old = None
            if target_dir.exists():
                old = target_dir.with_name(f".old-{backend_type}-{uuid.uuid4().hex[:8]}")
                await asyncio.to_thread(target_dir.rename, old)
            await asyncio.to_thread(tree.rename, target_dir)
            if old is not None:
                await asyncio.to_thread(shutil.rmtree, old, True)

            from ocabra.core.oci_registry import installed_packages, package_names

            python_bin = meta.get("python_bin")
            meta.update(
                installed_at=datetime.now(UTC).isoformat(),
                install_source="oci",
                oci_ref=str(published.ref),
                oci_digest=published.digest,
                published_from=published.labels.get("org.opencontainers.image.created"),
                packages=await asyncio.to_thread(
                    installed_packages, python_bin, package_names(spec.pip_packages)
                ),
                size_mb=await asyncio.to_thread(_compute_dir_size_mb, target_dir),
            )
            await self._write_metadata(backend_type, meta)
            self._register(backend_type, backend)
            final = await self._installed_state(backend_type, spec, meta)
            final.install_progress = 1.0
            final.install_detail = "Installed (prebuilt)"
            self._states[backend_type] = final
            self._log(backend_type, f"Installed {published.ref} ({published.digest})")
            yield self._snapshot(final)
        except Exception as exc:  # noqa: BLE001
            logger.exception("backend_oci_install_failed", backend_type=backend_type)
            self._log(backend_type, f"ERROR: {exc}")
            state.install_status = BackendInstallStatus.ERROR
            state.error = str(exc)
            state.install_detail = f"Install failed: {exc}"
            yield self._snapshot(state)
        finally:
            await asyncio.to_thread(shutil.rmtree, work, True)

    async def _check_prebuilt(
        self, backend_type: str, tree: Path, target_dir: Path, meta: dict[str, Any]
    ) -> None:
        """Refuse an image that wouldn't run from *target_dir* on this machine.

        Paths in a venv and in ``metadata.json`` are absolute: the image must
        have been built for this same location. Its interpreter must exist
        here, and every library a native binary needs must resolve except
        ``libcuda`` (the driver, provided by the host at run time).
        """
        if meta.get("backend_type") not in (None, backend_type):
            raise RuntimeError(f"The image holds {meta.get('backend_type')}, not {backend_type}")
        staged_bins = await asyncio.to_thread(_check_prebuilt_files, tree, target_dir, meta)
        if shutil.which("ldd") is None:
            return
        for name, staged in staged_bins.items():
            proc = await asyncio.create_subprocess_exec(
                "ldd", str(staged), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            out, _ = await proc.communicate()
            missing = [
                line.split("=>", 1)[0].strip()
                for line in out.decode(errors="replace").splitlines()
                if "not found" in line and "libcuda.so" not in line
            ]
            if missing:
                raise RuntimeError(f"{name} needs libraries this machine lacks: {', '.join(missing)}")

    async def _install_from_source(
        self,
        backend_type: str,
        backend: BackendInterface,
        spec: BackendInstallSpec,
    ) -> AsyncIterator[BackendModuleState]:
        target_dir = self._backends_dir / backend_type
        venv_dir = target_dir / "venv"
        bin_dir = target_dir / "bin"
        src_dir = target_dir / "src"

        # Only create a venv when there are Python deps to install (or the
        # core runtime opt-in). Native backends (llama_cpp, bitnet) declare
        # neither and skip the venv entirely; their post-install script just
        # builds a binary into BIN_DIR. Backends whose post-install needs the
        # venv (acestep / uv sync) declare pip_packages so the venv exists.
        needs_venv = bool(spec.pip_packages or spec.include_core_runtime)

        state = self._states[backend_type] = BackendModuleState(
            backend_type=backend_type,
            display_name=spec.display_name or backend_type,
            description=spec.description,
            tags=list(spec.tags),
            install_status=BackendInstallStatus.INSTALLING,
            estimated_size_mb=spec.estimated_size_mb,
            install_progress=0.0,
            install_detail="Preparing install directory...",
        )
        yield self._snapshot(state)

        try:
            await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(bin_dir.mkdir, parents=True, exist_ok=True)

            # Step 1 — apt packages (best-effort: only if apt is on PATH).
            if spec.apt_packages:
                state.install_progress = 0.05
                state.install_detail = "Installing system packages..."
                yield self._snapshot(state)
                await self._install_apt_packages(backend_type, spec.apt_packages)

            # Step 2 — git clone for backends that build from upstream sources.
            if spec.git_repo:
                state.install_progress = 0.12
                state.install_detail = f"Cloning {spec.git_repo}..."
                yield self._snapshot(state)
                await self._git_clone(
                    backend_type,
                    repo=spec.git_repo,
                    ref=spec.git_ref,
                    recursive=spec.git_recursive,
                    dest=src_dir,
                )

            # Step 3 — venv (skipped for native backends with no Python deps).
            python_bin: Path | None = None
            if needs_venv:
                state.install_progress = 0.18
                state.install_detail = "Creating venv..."
                self._log(backend_type, f"Creating venv at {venv_dir}")
                yield self._snapshot(state)
                await self._run_subprocess(
                    backend_type,
                    [sys.executable, "-m", "venv", str(venv_dir)],
                )
                python_bin = venv_dir / "bin" / "python"

                # Step 4 — pip install
                state.install_progress = 0.3
                state.install_detail = "Installing pip packages..."
                yield self._snapshot(state)

                pip_bin = venv_dir / "bin" / "pip"
                extra_index_args: list[str] = []
                for url in spec.pip_extra_index_urls:
                    extra_index_args.extend(["--extra-index-url", url])

                # Upgrade pip itself before any other install so new resolver
                # behaviour and --extra-index-url are honoured consistently.
                await self._run_subprocess(
                    backend_type,
                    [str(pip_bin), "install", "--upgrade", "pip"],
                )

                # Seed the venv with the core oCabra runtime (FastAPI + Pydantic
                # + httpx + ...) unless the spec opts out. Workers that run as
                # FastAPI subprocesses need these in their own venv because the
                # system interpreter is not on their path.
                if spec.include_core_runtime:
                    core_runtime = [
                        "fastapi>=0.115",
                        "uvicorn[standard]>=0.32",
                        "httpx>=0.28",
                        "pydantic>=2.10",
                        # python-multipart is required by FastAPI when worker
                        # endpoints use Form/File parameters (whisper /transcribe,
                        # diffusers /generate uploads, etc.).
                        "python-multipart>=0.0.20",
                    ]
                    self._log(
                        backend_type,
                        f"pip install {' '.join(core_runtime)} (core runtime)",
                    )
                    state.install_detail = "Installing core runtime..."
                    yield self._snapshot(state)
                    async for snap in self._run_pip_install(
                        backend_type,
                        [str(pip_bin), "install", *core_runtime],
                        state,
                        progress_start=0.30,
                        progress_end=0.45,
                    ):
                        yield snap

                if spec.pip_packages:
                    if extra_index_args:
                        self._log(
                            backend_type,
                            f"pip install {' '.join(extra_index_args)} "
                            f"{' '.join(spec.pip_packages)}",
                        )
                    else:
                        self._log(
                            backend_type,
                            f"pip install {' '.join(spec.pip_packages)}",
                        )
                    state.install_detail = "Installing backend packages..."
                    yield self._snapshot(state)
                    async for snap in self._run_pip_install(
                        backend_type,
                        [
                            str(pip_bin),
                            "install",
                            *extra_index_args,
                            *spec.pip_packages,
                        ],
                        state,
                        progress_start=0.45,
                        progress_end=0.68,
                    ):
                        yield snap

            # Step 5 — post-install script (cmake builds, uv sync, ...).
            if spec.post_install_script:
                state.install_progress = 0.7
                state.install_detail = (
                    f"Running post-install script: {spec.post_install_script}"
                )
                yield self._snapshot(state)
                await self._run_post_install_script(
                    backend_type,
                    spec.post_install_script,
                    target_dir=target_dir,
                    venv_dir=venv_dir if needs_venv else None,
                    python_bin=python_bin,
                    src_dir=src_dir if spec.git_repo else None,
                    bin_dir=bin_dir,
                )

            # Step 6 — extra_bins resolution + chmod.
            resolved_bins: dict[str, str] = {}
            if spec.extra_bins:
                state.install_progress = 0.82
                state.install_detail = "Validating produced binaries..."
                yield self._snapshot(state)
                resolved_bins = await self._resolve_extra_bins(
                    backend_type, target_dir, spec.extra_bins
                )

            # Step 7 — metadata
            state.install_progress = 0.88
            state.install_detail = "Writing metadata..."
            yield self._snapshot(state)

            size_mb = await asyncio.to_thread(_compute_dir_size_mb, target_dir)
            metadata = {
                "schema_version": METADATA_SCHEMA_VERSION,
                "backend_type": backend_type,
                "version": _derive_version(spec.pip_packages, backend_type),
                "installed_at": datetime.now(UTC).isoformat(),
                "install_source": "source",
                "python_bin": str(python_bin) if python_bin else None,
                "extra_bins": resolved_bins,
                "size_mb": size_mb,
                "pip_packages": list(spec.pip_packages),
                "pip_extra_index_urls": list(spec.pip_extra_index_urls),
                "include_core_runtime": spec.include_core_runtime,
                "apt_packages": list(spec.apt_packages),
                "git_repo": spec.git_repo,
                "git_ref": spec.git_ref if spec.git_repo else None,
                "src_dir": str(src_dir) if spec.git_repo else None,
                "bin_dir": str(bin_dir),
            }
            await self._write_metadata(backend_type, metadata)

            # Step 8 — register with the worker pool
            state.install_progress = 0.95
            state.install_detail = "Registering backend..."
            yield self._snapshot(state)
            self._register(backend_type, backend)

            # Final — INSTALLED
            state.install_status = BackendInstallStatus.INSTALLED
            state.install_progress = 1.0
            state.install_detail = "Installed"
            # La versión que se enseña sale de los paquetes instalados de
            # verdad (display_version), igual que al arrancar.
            fresh = await self._installed_state(backend_type, spec, metadata)
            state.installed_version = fresh.installed_version
            state.packages = fresh.packages
            state.installed_at = _parse_iso(metadata["installed_at"])
            state.install_source = "source"
            state.actual_size_mb = size_mb
            state.error = None
            yield self._snapshot(state)

        except Exception as exc:  # noqa: BLE001
            logger.exception("backend_install_failed", backend_type=backend_type)
            self._log(backend_type, f"ERROR: {exc}")
            state.install_status = BackendInstallStatus.ERROR
            state.error = str(exc)
            state.install_detail = f"Install failed: {exc}"
            yield self._snapshot(state)

    async def uninstall(self, backend_type: str) -> BackendModuleState:
        """Uninstall a backend after verifying no models are using it.

        Args:
            backend_type: Registered backend identifier.

        Returns:
            The final :class:`BackendModuleState` after uninstall.

        Raises:
            KeyError: If the backend is unknown.
            RuntimeError: If at least one model is currently loaded with this
                backend, or if the backend is "built-in" and cannot be removed.
        """
        if backend_type not in self._states:
            raise KeyError(f"Backend '{backend_type}' is not registered")

        state = self._states[backend_type]
        if state.install_source == "built-in":
            raise RuntimeError(
                f"Backend '{backend_type}' is built-in and cannot be uninstalled"
            )
        if state.install_status != BackendInstallStatus.INSTALLED:
            raise RuntimeError(
                f"Backend '{backend_type}' is not installed (status={state.install_status})"
            )

        loaded = _count_models_for_backend(self._worker_pool, backend_type)
        if loaded > 0:
            raise RuntimeError(
                f"Cannot uninstall '{backend_type}': {loaded} model(s) still loaded"
            )

        lock = self._install_locks.setdefault(backend_type, asyncio.Lock())
        async with lock:
            state.install_status = BackendInstallStatus.UNINSTALLING
            state.install_detail = "Removing backend files..."

            target_dir = self._backends_dir / backend_type
            if target_dir.exists():
                await asyncio.to_thread(shutil.rmtree, target_dir)

            # Update worker pool: mark as disabled.
            try:
                self._worker_pool.register_disabled_backend(
                    backend_type, "not installed"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "worker_pool_disable_failed",
                    backend_type=backend_type,
                    error=str(exc),
                )

            spec = self._backends[backend_type].install_spec
            assert spec is not None  # guarded by install_source check above
            self._states[backend_type] = BackendModuleState(
                backend_type=backend_type,
                display_name=spec.display_name or backend_type,
                description=spec.description,
                tags=list(spec.tags),
                install_status=BackendInstallStatus.NOT_INSTALLED,
                estimated_size_mb=spec.estimated_size_mb,
            )
            self._install_logs[backend_type] = []
            return self._states[backend_type]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_installed(self, backend_type: str) -> bool:
        return (self._backends_dir / backend_type / METADATA_FILENAME).exists()

    def _register(self, backend_type: str, backend: BackendInterface) -> None:
        """Register *backend* with the worker pool, swallowing errors."""
        try:
            self._worker_pool.register_backend(backend_type, backend)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "worker_pool_register_failed",
                backend_type=backend_type,
                error=str(exc),
            )

    async def _load_metadata(self, backend_type: str) -> dict[str, Any] | None:
        meta_path = self._backends_dir / backend_type / METADATA_FILENAME
        if not meta_path.exists():
            return None
        try:
            raw = await asyncio.to_thread(meta_path.read_text, encoding="utf-8")
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "backend_metadata_load_failed",
                backend_type=backend_type,
                error=str(exc),
            )
            return None

    async def _write_metadata(
        self, backend_type: str, metadata: dict[str, Any]
    ) -> None:
        meta_path = self._backends_dir / backend_type / METADATA_FILENAME
        payload = json.dumps(metadata, indent=2, sort_keys=True)
        await asyncio.to_thread(meta_path.write_text, payload, encoding="utf-8")

    def _snapshot(self, state: BackendModuleState) -> BackendModuleState:
        """Return a shallow copy of *state* so callers cannot mutate our view."""
        return replace(state, tags=list(state.tags))

    def _log(self, backend_type: str, message: str) -> None:
        buf = self._install_logs.setdefault(backend_type, [])
        buf.append(f"[{datetime.now(UTC).isoformat()}] {message}")
        if len(buf) > INSTALL_LOG_MAX_LINES:
            del buf[: len(buf) - INSTALL_LOG_MAX_LINES]

    async def _run_pip_install(
        self,
        backend_type: str,
        argv: list[str],
        state: BackendModuleState,
        *,
        progress_start: float,
        progress_end: float,
    ) -> AsyncIterator[BackendModuleState]:
        """Run a ``pip install`` command, yielding snapshots per package event.

        Closes Deuda #8: instead of a single "Installing pip packages..."
        sitting on the SSE for several minutes while torch downloads, we
        parse pip's stdout for ``Collecting <pkg>`` / ``Downloading <wheel>``
        / ``Installing collected packages: ...`` lines and update
        ``state.install_detail`` so the user sees what is happening.

        Progress fraction is interpolated linearly between *progress_start*
        and *progress_end* across the wheels seen so far — pip does not
        report a real percentage and we don't know the total ahead of time,
        so we cap at *progress_end* once "Successfully installed" appears.
        """
        # Tests / callers that don't want the line-parsing path can disable it.
        if not self._pip_progress_enabled:
            await self._run_subprocess(backend_type, argv)
            state.install_progress = progress_end
            yield self._snapshot(state)
            return

        self._log(backend_type, f"$ {' '.join(argv)}")
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None

        wheels_seen = 0
        # When we see N wheels, advance progress geometrically toward
        # progress_end so the bar never hits 100% before we're really done.
        span = max(0.0, progress_end - progress_start)

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode(errors="replace").rstrip()
            self._log(backend_type, line)

            stripped = line.strip()
            new_detail: str | None = None
            if stripped.startswith("Collecting "):
                # "Collecting torch>=2.5"
                pkg = stripped[len("Collecting ") :].split(" ", 1)[0]
                new_detail = f"Collecting {pkg}"
                wheels_seen += 1
            elif stripped.startswith("Downloading "):
                # "Downloading torch-2.5.0-cp311-cp311-manylinux2014_x86_64.whl (797.5 MB)"
                rest = stripped[len("Downloading ") :]
                new_detail = f"Downloading {rest}"
                wheels_seen += 1
            elif stripped.startswith("Installing collected packages:"):
                pkgs = stripped[len("Installing collected packages:") :].strip()
                # Trim long lists for the UI.
                if len(pkgs) > 80:
                    pkgs = pkgs[:77] + "..."
                new_detail = f"Linking: {pkgs}"
            elif stripped.startswith("Successfully installed"):
                new_detail = "Pip install complete"

            if new_detail:
                if span > 0:
                    # Diminishing-returns curve: 1 - 0.7^n. Reaches ~0.97 at n=10.
                    fraction = 1.0 - (0.7 ** max(0, wheels_seen))
                    state.install_progress = progress_start + span * fraction
                state.install_detail = new_detail
                yield self._snapshot(state)

        rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"Command {' '.join(argv)!r} failed with exit code {rc}"
            )

    async def _run_subprocess(
        self,
        backend_type: str,
        argv: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Run a subprocess, capturing stdout+stderr into the install log."""
        self._log(backend_type, f"$ {' '.join(argv)}")
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            self._log(backend_type, line.decode(errors="replace").rstrip())
        rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"Command {' '.join(argv)!r} failed with exit code {rc}"
            )

    # ------------------------------------------------------------------
    # Step helpers (apt / git / post-install / extra_bins)
    # ------------------------------------------------------------------

    async def _install_apt_packages(
        self, backend_type: str, packages: list[str]
    ) -> None:
        """Install OS packages via ``apt-get``.

        Best-effort: when ``apt-get`` is missing (non-Debian base) we log a
        warning and continue rather than failing the install — the operator is
        expected to have provided the deps in the base image.  Packages that
        fail to install (typo, unavailable in repo) DO surface as install
        errors because that's a real problem the user should see.
        """
        apt_get = shutil.which("apt-get")
        if apt_get is None:
            self._log(
                backend_type,
                f"apt-get not found on PATH; skipping apt_packages={packages}. "
                "Ensure these are pre-installed in the base image.",
            )
            return

        env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
        await self._run_subprocess(
            backend_type, [apt_get, "update"], env=env
        )
        await self._run_subprocess(
            backend_type,
            [
                apt_get,
                "install",
                "-y",
                "--no-install-recommends",
                *packages,
            ],
            env=env,
        )

    async def _git_clone(
        self,
        backend_type: str,
        *,
        repo: str,
        ref: str,
        recursive: bool,
        dest: Path,
    ) -> None:
        """Clone *repo* @ *ref* into *dest*, replacing any existing checkout."""
        if await asyncio.to_thread(dest.exists):
            await asyncio.to_thread(shutil.rmtree, dest)
        argv = ["git", "clone", "--depth", "1", "--branch", ref]
        if recursive:
            argv.append("--recursive")
        argv.extend([repo, str(dest)])
        await self._run_subprocess(backend_type, argv)

    async def _run_post_install_script(
        self,
        backend_type: str,
        script: str,
        *,
        target_dir: Path,
        venv_dir: Path | None,
        python_bin: Path | None,
        src_dir: Path | None,
        bin_dir: Path,
    ) -> None:
        """Execute *script* with bash, exposing install context as env vars.

        Resolves *script* either as an absolute path or relative to the repo
        root (``backend/`` package's grandparent).  Working directory is
        ``target_dir`` so scripts can drop produced files there directly.
        """
        script_path = await asyncio.to_thread(_resolve_script_path, script)
        if not await asyncio.to_thread(script_path.is_file):
            raise FileNotFoundError(
                f"post_install_script not found: '{script_path}'"
            )

        env = {
            **os.environ,
            "BACKEND_DIR": str(target_dir),
            "VENV_DIR": str(venv_dir) if venv_dir is not None else "",
            "PYTHON_BIN": str(python_bin) if python_bin is not None else "",
            "SRC_DIR": str(src_dir) if src_dir is not None else "",
            "BIN_DIR": str(bin_dir),
            "DEBIAN_FRONTEND": "noninteractive",
        }
        await self._run_subprocess(
            backend_type,
            ["bash", str(script_path)],
            cwd=target_dir,
            env=env,
        )

    async def _resolve_extra_bins(
        self,
        backend_type: str,
        target_dir: Path,
        declared: dict[str, str],
    ) -> dict[str, str]:
        """Verify declared extra_bins exist under *target_dir*; chmod +x them."""
        resolved: dict[str, str] = {}
        target_resolved = await asyncio.to_thread(target_dir.resolve)
        for logical_name, rel_path in declared.items():
            abs_path = await asyncio.to_thread(
                lambda p=rel_path: (target_dir / p).resolve()
            )
            try:
                abs_path.relative_to(target_resolved)
            except ValueError as exc:
                raise RuntimeError(
                    f"extra_bins['{logical_name}']='{rel_path}' escapes "
                    f"backend dir '{target_dir}'"
                ) from exc
            if not await asyncio.to_thread(abs_path.is_file):
                raise FileNotFoundError(
                    f"extra_bins['{logical_name}'] not produced by install: "
                    f"expected '{abs_path}'"
                )
            await asyncio.to_thread(_chmod_exec, abs_path)
            resolved[logical_name] = str(abs_path)
        self._log(
            backend_type,
            f"extra_bins resolved: {json.dumps(resolved, sort_keys=True)}",
        )
        return resolved


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------



def _check_prebuilt_files(tree: Path, target_dir: Path, meta: dict[str, Any]) -> dict[str, Path]:
    """Filesystem side of ``_check_prebuilt``: returns the staged binaries."""
    prefix = str(target_dir)
    python_bin = meta.get("python_bin")
    if python_bin:
        if not str(python_bin).startswith(prefix + "/"):
            raise RuntimeError(f"The image's venv was built for {python_bin}, not {prefix}")
        staged = tree / Path(python_bin).relative_to(target_dir)
        cfg = staged.parent.parent / "pyvenv.cfg"
        home = None
        if cfg.is_file():
            for line in cfg.read_text().splitlines():
                if line.split("=", 1)[0].strip() == "home":
                    home = line.split("=", 1)[1].strip()
        if home and not Path(home).exists():
            raise RuntimeError(f"The image's venv needs the interpreter in {home}")
    bins: dict[str, Path] = {}
    for name, path in (meta.get("extra_bins") or {}).items():
        if not str(path).startswith(prefix + "/"):
            raise RuntimeError(f"Binary {name} points outside the backend: {path}")
        staged = tree / Path(path).relative_to(target_dir)
        if not staged.is_file():
            raise RuntimeError(f"The image lacks its binary {name} ({path})")
        bins[name] = staged
    return bins

def _count_models_for_backend(worker_pool: WorkerPool, backend_type: str) -> int:
    """Count currently loaded workers that belong to *backend_type*."""
    workers = getattr(worker_pool, "_workers", {}) or {}
    return sum(
        1 for info in workers.values() if getattr(info, "backend_type", "") == backend_type
    )


def _compute_dir_size_mb(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return int(total / (1024 * 1024))


_CORE_RUNTIME_NAMES = {
    "fastapi",
    "uvicorn",
    "httpx",
    "pydantic",
    "python-multipart",
    "pip",
    "setuptools",
    "wheel",
    "torch",
    "torchaudio",
    "torchvision",
    "numpy",
    "transformers",
    "accelerate",
}


def _req_name(req: str) -> str:
    """Extract the package name from a pip requirement spec.

    ``"torch>=2.5"`` → ``"torch"``,
    ``"nemo_toolkit[asr]>=2.2"`` → ``"nemo_toolkit"``,
    ``"chatterbox-tts==0.1.7"`` → ``"chatterbox-tts"``.
    """
    head = req.strip()
    for sep in ("==", ">=", "<=", "~=", ">", "<", "!=", "["):
        idx = head.find(sep)
        if idx != -1:
            head = head[:idx]
    return head.strip().lower()



# Backends sin instalación propia (corren aparte): qué enseñar de ellos.
_BUILTIN_INFO: dict[str, dict[str, Any]] = {
    "ollama": {
        "display_name": "Ollama",
        "description": (
            "Servidor Ollama en su propio contenedor: modelos GGUF de su registro, chat y "
            "embeddings. Un único daemon para todos sus modelos, con paralelismo y contexto "
            "máximo globales."
        ),
        "tags": ["LLM", "GGUF", "CUDA"],
    },
}

# Paquete que da la versión de cada backend cuando no se llama como él.
_MAIN_PACKAGE = {
    "whisper": "faster-whisper",
    "piper": "piper-tts",
    "upscaler": "spandrel",
    "chatterbox": "chatterbox-tts",
}

_PLAIN_VERSION = re.compile(r"^[0-9][0-9A-Za-z.+_-]*$")


def display_version(
    backend_type: str,
    declared: list[str],
    packages: dict[str, str],
    stored: str | None,
) -> str | None:
    """Versión que enseñar de un backend instalado.

    Antes se mostraba lo que ``_derive_version`` guardó al instalar, que en
    muchos backends era un requisito (``"torch>=2.5"``) o ``"source"``. Ahora:
    la versión instalada de su paquete principal (el que se llama como el
    backend, o el de ``_MAIN_PACKAGE``), con el nombre delante si no es el del
    backend; si no, lo guardado solo si es un número de versión de verdad; y si
    tampoco, ``None`` (los nativos y los clonados de git ya enseñan su rama o
    commit aparte).
    """
    def norm(name: str) -> str:
        return name.lower().replace("_", "-")

    bt = norm(backend_type)
    wanted = norm(_MAIN_PACKAGE.get(backend_type, backend_type))
    installed = {norm(k): v for k, v in (packages or {}).items()}
    candidates = [wanted, bt] + [norm(_req_name(r)) for r in declared if norm(_req_name(r)).startswith(bt)]
    for name in candidates:
        version = installed.get(name)
        if version:
            return version if name == bt else f"{name} {version}"
    if stored and _PLAIN_VERSION.match(stored):
        return stored
    return None

def _derive_version(
    pip_packages: list[str], backend_type: str | None = None
) -> str:
    """Best-effort version extraction from a ``pip`` requirement list.

    Heuristic order (avoids the cosmetic ``"torch>=2.5"`` bug — Deuda 9h):
    1. A package whose name matches *backend_type* (e.g. ``vllm`` for the
       vllm backend), pinned with ``==``.
    2. The first ``==`` pinned package that is NOT in the core/shared list.
    3. The first non-shared requirement, raw.
    4. Fall back to the first requirement.
    5. ``"source"`` if the list is empty (native backends — version comes from
       the git ref instead, which the installer persists separately).
    """
    if not pip_packages:
        return "source"

    bt = (backend_type or "").lower()
    pinned = [(p, _req_name(p)) for p in pip_packages if "==" in p]

    if bt:
        for raw, name in pinned:
            if name == bt or name.replace("-", "_") == bt or name.replace("_", "-") == bt:
                return raw.split("==", 1)[1].strip()

    for raw, name in pinned:
        if name not in _CORE_RUNTIME_NAMES:
            return raw.split("==", 1)[1].strip()

    for req in pip_packages:
        if _req_name(req) not in _CORE_RUNTIME_NAMES:
            return req

    return pip_packages[0]


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _resolve_script_path(script: str) -> Path:
    """Resolve *script* against a list of candidate base directories.

    Search order:
    1. Absolute path → returned as-is.
    2. Repo root (3 levels above this module). Works in dev where the layout
       is ``<repo>/backend/ocabra/core/backend_installer.py``.
    3. ``<package_root>/..`` (i.e. one level above the ``ocabra`` package),
       which inside the docker image resolves to ``/app``. The image
       packages ``backend/scripts/*`` as ``/app/scripts/*`` (the ``backend/``
       prefix is dropped at copy time), so we also try the path with the
       leading ``backend/`` segment stripped.
    4. Returns the first candidate that exists; if none do, returns the last
       attempted path so the caller raises a clear FileNotFoundError.
    """
    script_path = Path(script)
    if script_path.is_absolute():
        return script_path

    # Candidate bases: repo root, app root.
    repo_root = Path(__file__).resolve().parents[3]
    pkg_root = Path(__file__).resolve().parents[2]  # /app inside docker, /<repo>/backend in dev

    # Also try with the leading "backend/" segment stripped — that's the
    # convention inside the docker image where backend/* is copied to /app/*.
    parts = script_path.parts
    stripped = Path(*parts[1:]) if parts and parts[0] == "backend" else script_path

    candidates = [
        repo_root / script_path,
        repo_root / stripped,
        pkg_root / script_path,
        pkg_root / stripped,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return candidates[-1].resolve()


def _chmod_exec(path: Path) -> None:
    """Add executable bits (u+x, g+x, o+x) to *path*, preserving other perms."""
    mode = path.stat().st_mode
    path.chmod(mode | 0o111)


def venv_cuda_home(backends_dir: str | Path, backend_type: str) -> str | None:
    """Build a fake ``CUDA_HOME`` directory backed by the backend's venv wheels.

    Backends that JIT-compile CUDA kernels at runtime (sglang's tvm_ffi,
    older vllm builds without ``--enforce-eager``) call ``_find_cuda_home()``
    which mostly needs ``CUDA_HOME / {lib64, include}`` to feed ``-L`` and
    ``-I`` flags to the linker — the actual ``nvcc`` is not required for the
    inline C++ JIT path (g++ does the heavy lifting via Triton).

    The slim image has no CUDA toolkit installed; the venv has the
    ``nvidia-cuda-{runtime,cccl,nvcc}-cu12`` pip wheels scattered across
    ``site-packages/nvidia/cuda_{runtime,cccl,nvcc}/``. This helper
    materialises a single tree at ``<backend>/cuda_home/`` with symlinks:

        cuda_home/
          bin/{ptxas, ...}   <- nvidia/cuda_nvcc/bin/  (if present)
          lib64/             <- nvidia/cuda_runtime/lib/
          include/           <- nvidia/cuda_runtime/include/
          include/cccl/      <- nvidia/cuda_cccl/include/  (if present)

    Returns the absolute path of the fake ``CUDA_HOME`` or ``None`` if the
    minimum (cuda_runtime lib + include) is missing — in that case the
    caller should leave ``CUDA_HOME`` unset and let the JIT surface its own
    error so the user knows to add the wheels to the install spec.
    """
    base = Path(backends_dir) / backend_type
    site_packages = (
        base
        / "venv"
        / "lib"
        / "python3.11"
        / "site-packages"
    )
    runtime_lib = site_packages / "nvidia" / "cuda_runtime" / "lib"
    runtime_inc = site_packages / "nvidia" / "cuda_runtime" / "include"
    if not (runtime_lib.is_dir() and runtime_inc.is_dir()):
        return None

    cuda_home = base / "cuda_home"
    cuda_home.mkdir(parents=True, exist_ok=True)
    lib64 = cuda_home / "lib64"
    if not lib64.exists():
        lib64.symlink_to(runtime_lib, target_is_directory=True)
    include = cuda_home / "include"
    if not include.exists():
        include.symlink_to(runtime_inc, target_is_directory=True)

    # bin/ — only link when a REAL ``nvcc`` is present. The
    # ``nvidia-cuda-nvcc-cu12`` pip wheel ships ``ptxas`` only (no nvcc), so
    # exposing its bin/ would put a fake "nvcc" on PATH and confuse Triton /
    # tvm_ffi when they try to compile. Real nvcc requires the full CUDA
    # Toolkit (apt ``cuda-nvcc-12-4`` from NVIDIA's repo, or the runtime
    # variant ``cuda-toolkit-12-4``).
    nvcc_bin = site_packages / "nvidia" / "cuda_nvcc" / "bin"
    if nvcc_bin.is_dir() and (nvcc_bin / "nvcc").is_file():
        bin_dir = cuda_home / "bin"
        if not bin_dir.exists():
            bin_dir.symlink_to(nvcc_bin, target_is_directory=True)
    cccl_inc = site_packages / "nvidia" / "cuda_cccl" / "include"
    if cccl_inc.is_dir():
        cccl_link = cuda_home / "include" / "cccl"
        # ``include`` is itself a symlink so cccl_link writes through it; that
        # mutates the wheel dir on disk. Skip if the symlink chain doesn't
        # allow it cleanly.
        try:
            if not cccl_link.exists():
                cccl_link.symlink_to(cccl_inc, target_is_directory=True)
        except OSError:
            pass
    return str(cuda_home)


def venv_nvidia_ld_library_path(backends_dir: str | Path, backend_type: str) -> str:
    """Return a ``:``-joined list of ``nvidia/*/lib`` dirs inside the backend's venv.

    Backends running on the slim image rely on torch's bundled CUDA libs
    (``libcublas.so.12``, ``libcudnn*.so.9``, ``libcudart.so.12``, ...) which
    pip drops under ``site-packages/nvidia/<lib>/lib/``.  Native loaders
    (ctranslate2, whisper, ...) only find them through ``LD_LIBRARY_PATH``.

    Returns an empty string if the venv does not exist or has no nvidia libs
    (fat image, or backend without torch).  Callers should append it to the
    existing ``LD_LIBRARY_PATH`` so the host's CUDA driver dirs still apply.
    """
    venv_lib = (
        Path(backends_dir)
        / backend_type
        / "venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "nvidia"
    )
    if not venv_lib.is_dir():
        return ""
    paths: list[str] = []
    for entry in sorted(venv_lib.iterdir()):
        lib_dir = entry / "lib"
        if lib_dir.is_dir():
            paths.append(str(lib_dir))
    return ":".join(paths)


def read_backend_metadata(
    backends_dir: str | Path, backend_type: str
) -> dict[str, Any] | None:
    """Read the persisted ``metadata.json`` for *backend_type*, or ``None``.

    Backends use this from their ``_resolve_*_bin()`` helpers to find the
    binary or interpreter produced by the modular install before falling back
    to legacy ``settings.<name>_*_bin`` paths.  Returns ``None`` if the file
    does not exist or is not parseable.
    """
    meta_path = Path(backends_dir) / backend_type / METADATA_FILENAME
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# Re-export the module attribute for convenience (import from this module in
# downstream code: ``from ocabra.core.backend_installer import BackendInstaller``).
__all__ = [
    "BackendAlreadyInstallingError",
    "BackendInstallStatus",
    "BackendInstaller",
    "BackendModuleState",
    "METADATA_FILENAME",
    "METADATA_SCHEMA_VERSION",
    "read_backend_metadata",
    "venv_cuda_home",
    "venv_nvidia_ld_library_path",
]
