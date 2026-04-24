"""Tests for ``ocabra.core.backend_installer`` (Bloque 15 Fase 1)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.backends.base import BackendInstallSpec, WorkerInfo
from ocabra.core.backend_installer import (
    METADATA_FILENAME,
    BackendAlreadyInstallingError,
    BackendInstaller,
    BackendInstallStatus,
)
from ocabra.core.worker_pool import WorkerPool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _InstallableMockBackend(MockBackend):
    """MockBackend that exposes an install_spec so it is a modular backend."""

    _spec = BackendInstallSpec(
        oci_image="ghcr.io/ocabra/backend-mock",
        oci_tags={"cuda12": "latest"},
        pip_packages=["cowsay==6.1"],
        estimated_size_mb=10,
        display_name="Mock",
        description="A backend used only for tests",
        tags=["TEST"],
    )

    @property
    def install_spec(self) -> BackendInstallSpec:
        return self._spec


class _AlwaysAvailableMockBackend(MockBackend):
    """MockBackend without install_spec — represents built-in backends."""

    @property
    def install_spec(self):  # type: ignore[override]
        return None


@pytest.fixture
def worker_pool() -> WorkerPool:
    return WorkerPool()


@pytest.fixture
def backends_dir(tmp_path: Path) -> Path:
    return tmp_path / "backends"


@pytest.fixture
def installer(
    worker_pool: WorkerPool, backends_dir: Path
) -> BackendInstaller:
    return BackendInstaller(
        backends_dir=backends_dir,
        worker_pool=worker_pool,
        backend_registry={
            "mock": _InstallableMockBackend(),
            "ollama": _AlwaysAvailableMockBackend(),
        },
    )


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_empty_dir_creates_and_marks_builtins(
    installer: BackendInstaller, backends_dir: Path
) -> None:
    await installer.start()

    assert backends_dir.exists()  # noqa: ASYNC240 — sync stat is fine in tests
    states = {s.backend_type: s for s in installer.list_states()}

    # Always-available backend is INSTALLED / built-in.
    assert states["ollama"].install_status == BackendInstallStatus.INSTALLED
    assert states["ollama"].install_source == "built-in"

    # Modular backend with no metadata is treated as built-in (fat image legacy).
    assert states["mock"].install_status == BackendInstallStatus.INSTALLED
    assert states["mock"].install_source == "built-in"


@pytest.mark.asyncio
async def test_start_detects_installed_metadata(
    installer: BackendInstaller, backends_dir: Path
) -> None:
    # Pre-seed metadata.json for "mock" to simulate a previous source install.
    mock_dir = backends_dir / "mock"
    mock_dir.mkdir(parents=True)
    (mock_dir / METADATA_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_type": "mock",
                "version": "6.1",
                "installed_at": "2026-04-12T10:30:00+00:00",
                "install_source": "source",
                "python_bin": str(mock_dir / "venv" / "bin" / "python"),
                "extra_bins": {},
                "size_mb": 42,
            }
        )
    )

    await installer.start()

    state = installer.get_state("mock")
    assert state.install_status == BackendInstallStatus.INSTALLED
    assert state.install_source == "source"
    assert state.installed_version == "6.1"
    assert state.actual_size_mb == 42


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------


async def _collect_states(gen) -> list[Any]:
    return [s async for s in gen]


@pytest.mark.asyncio
async def test_install_from_source_generates_states(
    installer: BackendInstaller, backends_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await installer.start()

    # Replace the subprocess runner with a no-op that simply touches the files
    # we expect ``venv`` + ``pip`` to create.
    async def _fake_run(self, backend_type: str, argv: list[str]) -> None:  # noqa: ANN001
        self._log(backend_type, f"MOCK $ {' '.join(argv)}")
        if argv[1:3] == ["-m", "venv"]:
            venv_dir = Path(argv[-1])
            (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
            (venv_dir / "bin" / "pip").write_text("#!/bin/sh\nexit 0\n")
            (venv_dir / "bin" / "pip").chmod(0o755)
            (venv_dir / "bin" / "python").write_text("#!/bin/sh\nexit 0\n")
            (venv_dir / "bin" / "python").chmod(0o755)

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _fake_run)

    states = await _collect_states(installer.install("mock", method="source"))

    statuses = [s.install_status for s in states]
    assert BackendInstallStatus.INSTALLING in statuses
    assert statuses[-1] == BackendInstallStatus.INSTALLED

    progresses = [s.install_progress for s in states if s.install_progress is not None]
    assert progresses[0] <= progresses[-1]
    assert progresses[-1] == 1.0

    # Metadata file was written.
    meta_path = backends_dir / "mock" / METADATA_FILENAME
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["install_source"] == "source"
    assert meta["version"] == "6.1"

    # Worker pool now has the backend registered for real.
    backend = await installer._worker_pool.get_backend("mock")
    assert backend is not None


@pytest.mark.asyncio
async def test_install_oci_not_implemented(installer: BackendInstaller) -> None:
    await installer.start()
    with pytest.raises(NotImplementedError):
        async for _ in installer.install("mock", method="oci"):
            pass


@pytest.mark.asyncio
async def test_install_unknown_backend_raises(installer: BackendInstaller) -> None:
    await installer.start()
    with pytest.raises(KeyError):
        async for _ in installer.install("does_not_exist"):
            pass


@pytest.mark.asyncio
async def test_install_builtin_rejected(installer: BackendInstaller) -> None:
    await installer.start()
    with pytest.raises(ValueError):
        async for _ in installer.install("ollama", method="source"):
            pass


@pytest.mark.asyncio
async def test_concurrent_install_rejected(
    installer: BackendInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    await installer.start()

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_run(self, backend_type: str, argv: list[str]) -> None:  # noqa: ANN001
        started.set()
        await release.wait()
        # Mimic the real subprocess so the install can still complete.
        if argv[1:3] == ["-m", "venv"]:
            venv_dir = Path(argv[-1])
            (venv_dir / "bin").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _slow_run)

    async def _run_install() -> list[Any]:
        return [s async for s in installer.install("mock")]

    task = asyncio.create_task(_run_install())
    await started.wait()

    # A second install while the first is in-flight must be rejected.
    with pytest.raises(BackendAlreadyInstallingError):
        async for _ in installer.install("mock"):
            pass

    release.set()
    await task


# ---------------------------------------------------------------------------
# uninstall()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uninstall_refuses_when_models_loaded(
    installer: BackendInstaller, worker_pool: WorkerPool, backends_dir: Path
) -> None:
    # Pre-populate a metadata file so the backend is INSTALLED (source).
    mock_dir = backends_dir / "mock"
    mock_dir.mkdir(parents=True)
    (mock_dir / METADATA_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_type": "mock",
                "version": "6.1",
                "installed_at": "2026-04-12T10:30:00+00:00",
                "install_source": "source",
                "size_mb": 1,
            }
        )
    )
    await installer.start()

    # Simulate a loaded model for this backend.
    worker_pool.set_worker(
        "mock/demo",
        WorkerInfo(
            backend_type="mock",
            model_id="mock/demo",
            gpu_indices=[0],
            port=18000,
            pid=9999,
            vram_used_mb=1024,
        ),
    )

    with pytest.raises(RuntimeError, match="still loaded"):
        await installer.uninstall("mock")

    # State must remain INSTALLED and files untouched.
    assert installer.get_state("mock").install_status == BackendInstallStatus.INSTALLED
    assert (mock_dir / METADATA_FILENAME).exists()


@pytest.mark.asyncio
async def test_uninstall_removes_files_and_deregisters(
    installer: BackendInstaller, backends_dir: Path
) -> None:
    mock_dir = backends_dir / "mock"
    mock_dir.mkdir(parents=True)
    (mock_dir / METADATA_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_type": "mock",
                "version": "6.1",
                "installed_at": "2026-04-12T10:30:00+00:00",
                "install_source": "source",
                "size_mb": 1,
            }
        )
    )
    (mock_dir / "some_binary").write_text("dummy")
    await installer.start()

    state = await installer.uninstall("mock")
    assert state.install_status == BackendInstallStatus.NOT_INSTALLED
    assert not mock_dir.exists()


@pytest.mark.asyncio
async def test_uninstall_builtin_rejected(installer: BackendInstaller) -> None:
    await installer.start()
    with pytest.raises(RuntimeError, match="built-in"):
        await installer.uninstall("ollama")


# ---------------------------------------------------------------------------
# HTTP endpoints (isolated FastAPI app to avoid the main lifespan)
# ---------------------------------------------------------------------------


def _build_isolated_app(installer: BackendInstaller):
    """Create a minimal FastAPI app wired to the given installer.

    The main ``ocabra.main.app`` triggers a full lifespan with DB/Redis/GPU
    initialisation, so we spin up a stripped-down app for endpoint smoke tests.
    """
    from fastapi import FastAPI

    from ocabra.api._deps_auth import UserContext, get_current_user
    from ocabra.api.internal.backends import router as backends_router

    app = FastAPI()
    app.include_router(backends_router, prefix="/ocabra")
    app.state.backend_installer = installer

    async def _admin_user() -> UserContext:
        return UserContext(
            user_id=None,
            username="__test__",
            role="system_admin",
            group_ids=[],
            accessible_model_ids=set(),
            is_anonymous=False,
        )

    app.dependency_overrides[get_current_user] = _admin_user
    return app


@pytest.mark.asyncio
async def test_list_backends_endpoint(installer: BackendInstaller) -> None:
    from httpx import ASGITransport, AsyncClient

    await installer.start()
    app = _build_isolated_app(installer)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/ocabra/backends")

    assert resp.status_code == 200
    data = resp.json()
    backend_types = {item["backend_type"] for item in data}
    assert {"mock", "ollama"} <= backend_types
    for item in data:
        assert "install_status" in item
        assert "display_name" in item


@pytest.mark.asyncio
async def test_get_backend_unknown_returns_404(
    installer: BackendInstaller,
) -> None:
    from httpx import ASGITransport, AsyncClient

    await installer.start()
    app = _build_isolated_app(installer)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/ocabra/backends/__does_not_exist__")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_install_oci_method_rejected_at_http_layer(
    installer: BackendInstaller,
) -> None:
    from httpx import ASGITransport, AsyncClient

    await installer.start()
    app = _build_isolated_app(installer)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/ocabra/backends/mock/install", json={"method": "oci"}
        )

    assert resp.status_code == 501
