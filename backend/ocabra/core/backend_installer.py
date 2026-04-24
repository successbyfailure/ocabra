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

OCI-based installation (``method="oci"``) is deliberately out of scope and will
raise :class:`NotImplementedError`.  It will land in Fase 3.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
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
    ) -> None:
        self._backends_dir = Path(backends_dir)
        self._worker_pool = worker_pool
        # backend_type → instance (lazy-instantiated source of truth for specs
        # + a way to re-register after an install).  Fase 1 expects the caller
        # to pre-populate this from ``main.py`` using the instances that are
        # already wired into the worker pool, so we avoid re-importing every
        # backend class here (those imports are heavy).
        self._backends: dict[str, BackendInterface] = dict(backend_registry or {})
        self._states: dict[str, BackendModuleState] = {}
        self._install_locks: dict[str, asyncio.Lock] = {}
        # Last install log per backend (in-memory ring buffer) — exposed via
        # ``GET /ocabra/backends/{type}/logs``.
        self._install_logs: dict[str, list[str]] = {}

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
                self._states[backend_type] = BackendModuleState(
                    backend_type=backend_type,
                    display_name=backend_type,
                    description="External or always-available backend.",
                    tags=[],
                    install_status=BackendInstallStatus.INSTALLED,
                    install_source="built-in",
                    installed_at=None,
                )
                continue

            if meta is not None:
                self._states[backend_type] = BackendModuleState(
                    backend_type=backend_type,
                    display_name=spec.display_name or backend_type,
                    description=spec.description,
                    tags=list(spec.tags),
                    install_status=BackendInstallStatus.INSTALLED,
                    installed_version=meta.get("version"),
                    installed_at=_parse_iso(meta.get("installed_at")),
                    install_source=meta.get("install_source"),
                    estimated_size_mb=spec.estimated_size_mb,
                    actual_size_mb=meta.get("size_mb"),
                )
            else:
                # No metadata on disk — treat as built-in (pre-installed in
                # the fat image) so we don't disable currently working
                # backends.  Fase 2 migrates each backend to write its own
                # metadata on first boot.
                self._states[backend_type] = BackendModuleState(
                    backend_type=backend_type,
                    display_name=spec.display_name or backend_type,
                    description=spec.description,
                    tags=list(spec.tags),
                    install_status=BackendInstallStatus.INSTALLED,
                    install_source="built-in",
                    estimated_size_mb=spec.estimated_size_mb,
                )

        logger.info(
            "backend_installer_ready",
            backends=sorted(self._states.keys()),
            backends_dir=str(self._backends_dir),
        )

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
        self, backend_type: str, method: str = "source"
    ) -> AsyncIterator[BackendModuleState]:
        """Install *backend_type* and yield :class:`BackendModuleState` updates.

        The generator is suitable for driving an SSE stream: each yielded
        state can be serialised to JSON and forwarded to the client.

        Args:
            backend_type: Registered backend identifier.
            method: Either ``"source"`` (venv + pip) or ``"oci"``.  ``"oci"``
                raises :class:`NotImplementedError` in Fase 1.

        Yields:
            Successive :class:`BackendModuleState` snapshots with
            ``install_progress`` and ``install_detail`` set while the
            installation is running, followed by a final ``INSTALLED`` or
            ``ERROR`` state.

        Raises:
            KeyError: If *backend_type* is not registered.
            BackendAlreadyInstallingError: If another install is already in
                progress for the same backend.
            NotImplementedError: If *method* is ``"oci"``.
        """
        if backend_type not in self._backends:
            raise KeyError(f"Backend '{backend_type}' is not registered")

        lock = self._install_locks.setdefault(backend_type, asyncio.Lock())
        if lock.locked():
            raise BackendAlreadyInstallingError(
                f"Backend '{backend_type}' is already being installed"
            )

        if method not in {"source", "oci"}:
            raise ValueError(f"Unknown install method '{method}'")
        if method == "oci":
            raise NotImplementedError("OCI-based install will ship in Fase 3")

        backend = self._backends[backend_type]
        spec = backend.install_spec
        if spec is None:
            raise ValueError(
                f"Backend '{backend_type}' is always-available and cannot be installed"
            )

        async with lock:
            self._install_logs[backend_type] = []

            async for state in self._install_from_source(backend_type, backend, spec):
                yield state

    async def _install_from_source(
        self,
        backend_type: str,
        backend: BackendInterface,
        spec: BackendInstallSpec,
    ) -> AsyncIterator[BackendModuleState]:
        target_dir = self._backends_dir / backend_type
        venv_dir = target_dir / "venv"

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

            # Step 1 — venv
            state.install_progress = 0.1
            state.install_detail = "Creating venv..."
            self._log(backend_type, f"Creating venv at {venv_dir}")
            yield self._snapshot(state)
            await self._run_subprocess(
                backend_type,
                [sys.executable, "-m", "venv", str(venv_dir)],
            )

            # Step 2 — pip install
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

            # Seed the venv with the core oCabra runtime (FastAPI + Pydantic +
            # httpx + ...) unless the spec opts out. Workers that run as a
            # FastAPI subprocess need these in their own venv because the
            # system interpreter is not on their path.
            if spec.include_core_runtime:
                core_runtime = [
                    "fastapi>=0.115",
                    "uvicorn[standard]>=0.32",
                    "httpx>=0.28",
                    "pydantic>=2.10",
                ]
                self._log(
                    backend_type,
                    f"pip install {' '.join(core_runtime)} (core runtime)",
                )
                await self._run_subprocess(
                    backend_type,
                    [str(pip_bin), "install", *core_runtime],
                )

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
                await self._run_subprocess(
                    backend_type,
                    [
                        str(pip_bin),
                        "install",
                        *extra_index_args,
                        *spec.pip_packages,
                    ],
                )

            # Step 3 — post install script (reserved for Fase 2)
            if spec.post_install_script:
                state.install_progress = 0.7
                state.install_detail = (
                    f"Running post-install script: {spec.post_install_script}"
                )
                yield self._snapshot(state)
                self._log(
                    backend_type,
                    f"Post-install script hook present but not executed in Fase 1: "
                    f"{spec.post_install_script}",
                )

            # Step 4 — metadata
            state.install_progress = 0.85
            state.install_detail = "Writing metadata..."
            yield self._snapshot(state)

            size_mb = await asyncio.to_thread(_compute_dir_size_mb, target_dir)
            python_bin = venv_dir / "bin" / "python"
            metadata = {
                "schema_version": METADATA_SCHEMA_VERSION,
                "backend_type": backend_type,
                "version": _derive_version(spec.pip_packages),
                "installed_at": datetime.now(UTC).isoformat(),
                "install_source": "source",
                "python_bin": str(python_bin),
                "extra_bins": {},
                "size_mb": size_mb,
                "pip_packages": list(spec.pip_packages),
                "pip_extra_index_urls": list(spec.pip_extra_index_urls),
                "include_core_runtime": spec.include_core_runtime,
            }
            await self._write_metadata(backend_type, metadata)

            # Step 5 — register with the worker pool
            state.install_progress = 0.95
            state.install_detail = "Registering backend..."
            yield self._snapshot(state)
            self._register(backend_type, backend)

            # Final — INSTALLED
            state.install_status = BackendInstallStatus.INSTALLED
            state.install_progress = 1.0
            state.install_detail = "Installed"
            state.installed_version = metadata["version"]
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

    async def _run_subprocess(
        self, backend_type: str, argv: list[str]
    ) -> None:
        """Run a subprocess, capturing stdout+stderr into the install log."""
        self._log(backend_type, f"$ {' '.join(argv)}")
        proc = await asyncio.create_subprocess_exec(
            *argv,
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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


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


def _derive_version(pip_packages: list[str]) -> str:
    """Best-effort version extraction from a ``pip`` requirement list."""
    if not pip_packages:
        return "source"
    # Prefer the first pinned requirement in the form ``name==x.y.z``.
    for req in pip_packages:
        if "==" in req:
            return req.split("==", 1)[1].strip()
    return pip_packages[0]


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
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
]
