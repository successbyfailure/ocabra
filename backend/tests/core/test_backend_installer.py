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
    BackendModuleState,
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
    inst = BackendInstaller(
        backends_dir=backends_dir,
        worker_pool=worker_pool,
        backend_registry={
            "mock": _InstallableMockBackend(),
            "ollama": _AlwaysAvailableMockBackend(),
        },
    )
    # Tests in this module mock ``_run_subprocess``; the pip-progress path
    # bypasses that mock by spawning subprocesses directly to read stdout.
    # Disabling it here keeps every existing test passing with a single mock.
    inst._pip_progress_enabled = False
    return inst


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
async def test_start_slim_image_marks_unknown_backends_as_not_installed(
    worker_pool: WorkerPool, backends_dir: Path
) -> None:
    """On the slim image, backends with install_spec but no metadata.json
    should report NOT_INSTALLED, not built-in — they really aren't installed."""

    slim_installer = BackendInstaller(
        backends_dir=backends_dir,
        worker_pool=worker_pool,
        backend_registry={
            "mock": _InstallableMockBackend(),
            "ollama": _AlwaysAvailableMockBackend(),
        },
        assume_fat_image=False,
    )
    slim_installer._pip_progress_enabled = False
    await slim_installer.start()

    states = {s.backend_type: s for s in slim_installer.list_states()}

    # Always-available backend still reports installed regardless of image.
    assert states["ollama"].install_status == BackendInstallStatus.INSTALLED
    assert states["ollama"].install_source == "built-in"

    # Modular backend without metadata is NOT installed on the slim image.
    assert states["mock"].install_status == BackendInstallStatus.NOT_INSTALLED
    assert states["mock"].install_source is None


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
async def test_install_source_honours_extra_index_and_core_runtime(
    installer: BackendInstaller,
    backends_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deudas 9a + 9b: venv seeds core runtime first and honours extra-index-url."""

    await installer.start()

    # Swap in a spec with both a pip_extra_index_urls and include_core_runtime=True.
    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            pip_packages=["torch>=2.5", "cowsay==6.1"],
            pip_extra_index_urls=["https://download.pytorch.org/whl/cu124"],
            include_core_runtime=True,
            estimated_size_mb=10,
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    calls: list[list[str]] = []

    async def _capture_run(self, backend_type: str, argv: list[str]) -> None:  # noqa: ANN001
        calls.append(list(argv))
        self._log(backend_type, f"MOCK $ {' '.join(argv)}")
        if argv[1:3] == ["-m", "venv"]:
            venv_dir = Path(argv[-1])
            (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
            (venv_dir / "bin" / "pip").write_text("#!/bin/sh\nexit 0\n")
            (venv_dir / "bin" / "pip").chmod(0o755)
            (venv_dir / "bin" / "python").write_text("#!/bin/sh\nexit 0\n")
            (venv_dir / "bin" / "python").chmod(0o755)

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _capture_run)

    async for _ in installer.install("mock", method="source"):
        pass

    pip_invocations = [c for c in calls if c and c[0].endswith("/pip")]
    # Expect: upgrade pip, core runtime seed, backend deps.
    assert len(pip_invocations) >= 3, pip_invocations

    upgrade, core, deps = pip_invocations[0], pip_invocations[1], pip_invocations[2]
    assert upgrade[1:] == ["install", "--upgrade", "pip"]

    # Core runtime comes before backend deps and includes fastapi + pydantic.
    core_joined = " ".join(core)
    assert "fastapi" in core_joined and "pydantic" in core_joined, core

    # Backend deps carry the --extra-index-url and the torch requirement.
    deps_joined = " ".join(deps)
    assert "--extra-index-url" in deps and "download.pytorch.org" in deps_joined
    assert "torch>=2.5" in deps

    # Metadata captures both new fields.
    meta = json.loads((backends_dir / "mock" / METADATA_FILENAME).read_text())
    assert meta["pip_extra_index_urls"] == ["https://download.pytorch.org/whl/cu124"]
    assert meta["include_core_runtime"] is True


@pytest.mark.asyncio
async def test_install_source_skips_core_runtime_when_opted_out(
    installer: BackendInstaller,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """include_core_runtime=False keeps the venv minimal."""

    await installer.start()

    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            pip_packages=["cowsay==6.1"],
            include_core_runtime=False,
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    calls: list[list[str]] = []

    async def _capture_run(self, backend_type: str, argv: list[str]) -> None:  # noqa: ANN001
        calls.append(list(argv))
        self._log(backend_type, f"MOCK $ {' '.join(argv)}")
        if argv[1:3] == ["-m", "venv"]:
            venv_dir = Path(argv[-1])
            (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
            (venv_dir / "bin" / "pip").write_text("#!/bin/sh\nexit 0\n")
            (venv_dir / "bin" / "pip").chmod(0o755)
            (venv_dir / "bin" / "python").write_text("#!/bin/sh\nexit 0\n")
            (venv_dir / "bin" / "python").chmod(0o755)

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _capture_run)

    async for _ in installer.install("mock", method="source"):
        pass

    pip_joined = [" ".join(c) for c in calls if c and c[0].endswith("/pip")]
    assert not any("fastapi" in cmd for cmd in pip_joined), pip_joined


@pytest.mark.asyncio
async def test_install_survives_consumer_cancellation(
    installer: BackendInstaller,
    backends_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deuda 9f: cancelling the SSE consumer must NOT leave state stuck on
    'installing' — the detached task keeps running and lands the final state."""

    await installer.start()

    started = asyncio.Event()
    proceed = asyncio.Event()

    async def _slow_run(self, backend_type: str, argv: list[str]) -> None:  # noqa: ANN001
        self._log(backend_type, f"MOCK $ {' '.join(argv)}")
        if argv[1:3] == ["-m", "venv"]:
            venv_dir = Path(argv[-1])
            (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
            (venv_dir / "bin" / "pip").write_text("#!/bin/sh\nexit 0\n")
            (venv_dir / "bin" / "pip").chmod(0o755)
            (venv_dir / "bin" / "python").write_text("#!/bin/sh\nexit 0\n")
            (venv_dir / "bin" / "python").chmod(0o755)
            return
        # First pip install call: pause until the consumer is gone, then
        # let the rest of the install run to completion.
        if not started.is_set():
            started.set()
            await proceed.wait()

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _slow_run)

    gen = installer.install("mock", method="source").__aiter__()
    # Pull a couple of states until the runner is paused on the slow step.
    await gen.__anext__()
    await gen.__anext__()

    # The runner is now waiting on `proceed`. Simulate the SSE client going
    # away by closing the generator without consuming further states.
    await gen.aclose()

    # Release the runner: it should keep going and eventually mark INSTALLED.
    proceed.set()

    # Wait for the install task to finish on its own.
    task = installer._install_tasks["mock"]
    await asyncio.wait_for(task, timeout=2.0)

    final = installer.get_state("mock")
    assert final.install_status == BackendInstallStatus.INSTALLED
    assert (backends_dir / "mock" / METADATA_FILENAME).exists()


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
# Fase 2 — apt / git / post_install_script / extra_bins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_skips_venv_when_no_python_deps(
    installer: BackendInstaller,
    backends_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native backends (llama_cpp/bitnet) need no venv; the installer should skip it."""

    await installer.start()

    # Pure-native spec: no pip, no core runtime, no post-install. Just produce a binary.
    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            include_core_runtime=False,
            extra_bins={"server": "bin/mock-server"},
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    calls: list[list[str]] = []

    async def _capture_run(self, backend_type, argv, *, cwd=None, env=None):  # noqa: ANN001
        calls.append(list(argv))
        self._log(backend_type, f"MOCK $ {' '.join(argv)}")

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _capture_run)

    # Drop the expected binary so extra_bins validation passes.
    bin_dir = backends_dir / "mock" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "mock-server").write_text("#!/bin/sh\nexit 0\n")

    states = await _collect_states(installer.install("mock", method="source"))
    assert states[-1].install_status == BackendInstallStatus.INSTALLED

    venv_calls = [c for c in calls if len(c) >= 3 and c[1:3] == ["-m", "venv"]]
    assert venv_calls == [], f"unexpected venv creation: {venv_calls}"

    meta = json.loads((backends_dir / "mock" / METADATA_FILENAME).read_text())
    assert meta["python_bin"] is None
    assert meta["extra_bins"] == {"server": str(bin_dir / "mock-server")}


@pytest.mark.asyncio
async def test_install_runs_apt_when_available(
    installer: BackendInstaller,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """apt_packages → apt-get update + apt-get install with --no-install-recommends."""

    await installer.start()

    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            apt_packages=["build-essential", "cmake"],
            include_core_runtime=False,
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    calls: list[list[str]] = []

    async def _capture_run(self, backend_type, argv, *, cwd=None, env=None):  # noqa: ANN001
        calls.append(list(argv))
        self._log(backend_type, f"MOCK $ {' '.join(argv)}")

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _capture_run)
    # Pretend apt-get is on PATH at /usr/bin/apt-get.
    monkeypatch.setattr(
        "ocabra.core.backend_installer.shutil.which",
        lambda name: "/usr/bin/apt-get" if name == "apt-get" else None,
    )

    async for _ in installer.install("mock", method="source"):
        pass

    apt_calls = [c for c in calls if c and c[0].endswith("apt-get")]
    assert any(c[1:] == ["update"] for c in apt_calls), apt_calls
    install_call = next(
        c for c in apt_calls if "install" in c
    )
    assert "--no-install-recommends" in install_call
    assert "build-essential" in install_call and "cmake" in install_call


@pytest.mark.asyncio
async def test_install_skips_apt_when_unavailable(
    installer: BackendInstaller,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No apt-get on PATH → log warning, do not raise."""

    await installer.start()

    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            apt_packages=["build-essential"],
            include_core_runtime=False,
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    calls: list[list[str]] = []

    async def _capture_run(self, backend_type, argv, *, cwd=None, env=None):  # noqa: ANN001
        calls.append(list(argv))

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _capture_run)
    monkeypatch.setattr(
        "ocabra.core.backend_installer.shutil.which", lambda name: None
    )

    states = await _collect_states(installer.install("mock", method="source"))
    assert states[-1].install_status == BackendInstallStatus.INSTALLED
    assert not any(c and c[0].endswith("apt-get") for c in calls)
    log = installer.get_logs("mock")
    assert any("apt-get not found" in line for line in log)


@pytest.mark.asyncio
async def test_install_clones_git_repo_into_src_dir(
    installer: BackendInstaller,
    backends_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """git_repo → ``git clone --depth 1 --branch <ref>`` into ``<backend>/src/``."""

    await installer.start()

    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            git_repo="https://example.com/foo/bar",
            git_ref="v1.2.3",
            git_recursive=True,
            include_core_runtime=False,
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    calls: list[list[str]] = []

    async def _capture_run(self, backend_type, argv, *, cwd=None, env=None):  # noqa: ANN001
        calls.append(list(argv))

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _capture_run)

    async for _ in installer.install("mock", method="source"):
        pass

    git_call = next(c for c in calls if c and c[0] == "git")
    assert "clone" in git_call and "--depth" in git_call and "--recursive" in git_call
    assert "--branch" in git_call and "v1.2.3" in git_call
    assert git_call[-2] == "https://example.com/foo/bar"
    assert git_call[-1] == str(backends_dir / "mock" / "src")

    meta = json.loads((backends_dir / "mock" / METADATA_FILENAME).read_text())
    assert meta["git_repo"] == "https://example.com/foo/bar"
    assert meta["git_ref"] == "v1.2.3"
    assert meta["src_dir"] == str(backends_dir / "mock" / "src")


@pytest.mark.asyncio
async def test_install_runs_post_install_script_with_env(
    installer: BackendInstaller,
    backends_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """post_install_script runs in cwd=BACKEND_DIR with BACKEND_DIR/SRC_DIR/BIN_DIR env."""

    await installer.start()

    # Create a real script on disk so the absolute-path branch is exercised.
    script_path = tmp_path / "do_thing.sh"
    script_path.write_text("#!/bin/sh\necho hi\n")
    script_path.chmod(0o755)

    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            git_repo="https://example.com/foo/bar",
            post_install_script=str(script_path),
            include_core_runtime=False,
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    seen_env: dict[str, str] = {}
    seen_cwd: list[str | None] = []

    async def _capture_run(self, backend_type, argv, *, cwd=None, env=None):  # noqa: ANN001
        if argv and argv[0] == "bash":
            seen_env.update(env or {})
            seen_cwd.append(str(cwd) if cwd else None)

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _capture_run)

    async for _ in installer.install("mock", method="source"):
        pass

    assert seen_cwd == [str(backends_dir / "mock")]
    assert seen_env["BACKEND_DIR"] == str(backends_dir / "mock")
    assert seen_env["SRC_DIR"] == str(backends_dir / "mock" / "src")
    assert seen_env["BIN_DIR"] == str(backends_dir / "mock" / "bin")
    # No venv was created (no pip_packages, no core runtime), so VENV_DIR/PYTHON_BIN are empty.
    assert seen_env["VENV_DIR"] == ""
    assert seen_env["PYTHON_BIN"] == ""


@pytest.mark.asyncio
async def test_install_post_install_script_resolves_repo_relative_paths(
    installer: BackendInstaller,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relative post_install_script paths resolve against the repo root."""

    await installer.start()

    # backend/scripts/install_llama_cpp.sh ships in the repo and must resolve.
    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            post_install_script="backend/scripts/install_llama_cpp.sh",
            include_core_runtime=False,
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    seen_argv: list[list[str]] = []

    async def _capture_run(self, backend_type, argv, *, cwd=None, env=None):  # noqa: ANN001
        if argv and argv[0] == "bash":
            seen_argv.append(list(argv))

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _capture_run)

    async for _ in installer.install("mock", method="source"):
        pass

    assert len(seen_argv) == 1
    resolved = seen_argv[0][1]
    assert resolved.endswith("backend/scripts/install_llama_cpp.sh")
    assert Path(resolved).is_file()  # noqa: ASYNC240 — assertion, not IO


@pytest.mark.asyncio
async def test_install_extra_bins_missing_file_fails(
    installer: BackendInstaller,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declared extra_bins that the post-install script did not produce → ERROR."""

    await installer.start()

    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            extra_bins={"server": "bin/should-have-been-built"},
            include_core_runtime=False,
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    async def _noop(self, backend_type, argv, *, cwd=None, env=None):  # noqa: ANN001
        return None

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _noop)

    states = await _collect_states(installer.install("mock", method="source"))
    assert states[-1].install_status == BackendInstallStatus.ERROR
    assert "extra_bins" in (states[-1].error or "")


@pytest.mark.asyncio
async def test_install_extra_bins_traversal_rejected(
    installer: BackendInstaller,
    backends_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extra_bins paths that escape the backend directory are rejected."""

    await installer.start()

    backend = installer._backends["mock"]
    monkeypatch.setattr(
        type(backend),
        "_spec",
        BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-mock",
            extra_bins={"server": "../../../etc/passwd"},
            include_core_runtime=False,
            display_name="Mock",
            description="",
            tags=["TEST"],
        ),
    )

    async def _noop(self, backend_type, argv, *, cwd=None, env=None):  # noqa: ANN001
        return None

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _noop)

    states = await _collect_states(installer.install("mock", method="source"))
    assert states[-1].install_status == BackendInstallStatus.ERROR
    assert "escapes" in (states[-1].error or "")


def test_derive_version_prefers_backend_named_pin() -> None:
    """Deuda 9h fix: tts spec used to report 'torch>=2.5'; should pick qwen-tts pin."""
    from ocabra.core.backend_installer import _derive_version

    pkgs = [
        "torch>=2.5",
        "torchaudio>=2.5",
        "qwen-tts==0.1.1",
        "kokoro-tts==1.2.0",
    ]
    # When backend_type is known, prefer the matching package by name.
    assert _derive_version(pkgs, "tts") not in {"torch>=2.5", "torchaudio>=2.5"}
    # And falls back to first non-core pinned package without backend_type.
    assert _derive_version(pkgs) == "0.1.1"


def test_derive_version_native_backend_returns_source() -> None:
    """Native backends (llama_cpp, bitnet) have no pip packages."""
    from ocabra.core.backend_installer import _derive_version

    assert _derive_version([], "llama_cpp") == "source"


def test_derive_version_handles_extras_and_specifiers() -> None:
    from ocabra.core.backend_installer import _req_name

    assert _req_name("nemo_toolkit[asr]>=2.2") == "nemo_toolkit"
    assert _req_name("Chatterbox-TTS==0.1.7") == "chatterbox-tts"
    assert _req_name("torch>=2.5") == "torch"


@pytest.mark.asyncio
async def test_pip_install_parses_progress_lines(
    installer: BackendInstaller,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deuda #8: pip stdout is parsed for Collecting/Downloading/Installing lines."""

    # Re-enable the pip-progress path for this test (the fixture disables it).
    installer._pip_progress_enabled = True

    # Replace asyncio.create_subprocess_exec to inject a fake stdout stream.
    sample_lines = [
        b"Collecting torch>=2.5\n",
        b"  Downloading torch-2.5.0-cp311-cp311-manylinux2014_x86_64.whl (797.5 MB)\n",
        b"Collecting numpy\n",
        b"Installing collected packages: numpy, torch\n",
        b"Successfully installed numpy-2.0.0 torch-2.5.0\n",
    ]

    class _FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        async def readline(self):
            if not self._lines:
                return b""
            return self._lines.pop(0)

    class _FakeProc:
        def __init__(self):
            self.stdout = _FakeStream(sample_lines)

        async def wait(self):
            return 0

    async def _fake_create(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(
        "ocabra.core.backend_installer.asyncio.create_subprocess_exec", _fake_create
    )

    state = BackendModuleState(
        backend_type="mock",
        display_name="Mock",
        description="",
        tags=[],
        install_status=BackendInstallStatus.INSTALLING,
        install_progress=0.3,
    )
    snapshots = []
    async for snap in installer._run_pip_install(
        "mock",
        ["pip", "install", "torch>=2.5", "numpy"],
        state,
        progress_start=0.30,
        progress_end=0.70,
    ):
        snapshots.append(snap.install_detail)

    # We expect at least one snapshot per recognised line type.
    joined = " | ".join(s for s in snapshots if s)
    assert "Collecting torch" in joined
    assert "Downloading torch-" in joined
    assert "Linking:" in joined
    assert any(s == "Pip install complete" for s in snapshots)


def test_read_backend_metadata_helper(tmp_path: Path) -> None:
    """The public helper backends use to discover paths from metadata.json."""
    from ocabra.core.backend_installer import read_backend_metadata

    assert read_backend_metadata(tmp_path, "absent") is None

    backend_dir = tmp_path / "thing"
    backend_dir.mkdir()
    (backend_dir / METADATA_FILENAME).write_text(
        json.dumps({"extra_bins": {"server": "/abs/path/to/server"}})
    )
    meta = read_backend_metadata(tmp_path, "thing")
    assert meta is not None
    assert meta["extra_bins"]["server"] == "/abs/path/to/server"


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
