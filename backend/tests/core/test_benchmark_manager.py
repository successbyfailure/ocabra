from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ocabra.api.internal.benchmarks import BenchmarkRunCreate, _run_payload
from ocabra.benchmarks.catalog import APPS, catalog_payload
from ocabra.benchmarks.manager import (
    BenchmarkIdentity,
    BenchmarkManager,
    backend_version_observations,
    execution_observations,
    external_active_requests,
    git_authorization_header,
)
from ocabra.database import AsyncSessionLocal
from ocabra.db.benchmarks import BenchmarkAttempt, BenchmarkRun


def test_github_token_uses_basic_header_for_git_https() -> None:
    header = git_authorization_header("github-token")

    scheme, encoded = header.split(" ", 2)[1:]
    assert scheme == "Basic"
    assert base64.b64decode(encoded).decode() == "x-access-token:github-token"


def test_catalog_matches_cli_manifests() -> None:
    root = Path(__file__).resolve().parents[3]
    for name, app in APPS.items():
        raw = json.loads((root / "benchmark" / "apps" / f"{name}.json").read_text())
        assert raw["repository"] == app.repository
        assert raw["dataset_root"] == app.dataset_root
        assert raw.get("checkout_paths", []) == list(app.checkout_paths)
        assert set(raw["suites"]) == set(app.suites)
        for suite_name, suite in app.suites.items():
            if suite.local_runner:
                assert raw["suites"][suite_name]["local_runner"] == suite.local_runner
            else:
                assert raw["suites"][suite_name]["script"] == suite.script
            assert raw["suites"][suite_name]["result_glob"] == suite.result_glob


def test_catalog_exposes_resume_and_preparation() -> None:
    payload = {item["app"]: item for item in catalog_payload()}
    corrala = payload["corrala"]["suites"][0]
    escriba_stt = next(item for item in payload["escriba"]["suites"] if item["name"] == "stt")
    assert corrala["resumable"] is True
    assert corrala["needsPreparation"] is False
    assert escriba_stt["needsPreparation"] is True


def test_public_benchmarks_are_pinned_and_use_bundled_resumable_runners() -> None:
    for app_name in ("bfcl", "ifeval"):
        app = APPS[app_name]
        assert len(app.default_revision) == 40
        assert app.default_revision != "main"
        for suite in app.suites.values():
            assert suite.local_runner
            assert suite.resume_args == ("--skip-done",)


def test_external_activity_excludes_internal_benchmark_user() -> None:
    benchmark_user = uuid.uuid4()
    identity = BenchmarkIdentity(benchmark_user, "ocabra-benchmark", "secret")
    requests = [
        SimpleNamespace(
            request_id="benchmark",
            user_id=str(benchmark_user),
            profile_id="model-a",
            model_id="worker-a",
            api_key_name="ocabra-benchmark",
            username="service",
            started_at=1_700_000_000,
        ),
        SimpleNamespace(
            request_id="real",
            user_id=str(uuid.uuid4()),
            profile_id="model-b",
            model_id="worker-b",
            api_key_name="mobile",
            username="user",
            started_at=1_700_000_001,
        ),
        SimpleNamespace(
            request_id="anonymous",
            user_id=None,
            profile_id=None,
            model_id="model-c",
            api_key_name=None,
            username=None,
            started_at=1_700_000_002,
        ),
    ]
    manager = SimpleNamespace(active_requests_snapshot=lambda: requests)

    external = external_active_requests(manager, identity)

    assert [item["requestId"] for item in external] == ["real", "anonymous"]


def test_run_create_rejects_duplicate_models_and_managed_arguments() -> None:
    with pytest.raises(ValidationError):
        BenchmarkRunCreate(app="corrala", suite="llm", models=["same", "same"])
    with pytest.raises(ValidationError):
        BenchmarkRunCreate(app="corrala", suite="llm", models=["--profile"])
    with pytest.raises(ValidationError):
        BenchmarkRunCreate(
            app="corrala",
            suite="llm",
            models=["model"],
            suite_args=["--base-url=https://outside.invalid"],
        )


def test_changed_result_requires_new_or_modified_file(tmp_path: Path) -> None:
    result_dir = tmp_path / "results_today"
    result_dir.mkdir()
    result = result_dir / "results.json"
    result.write_text('{"old": true}')
    before = BenchmarkManager._result_snapshot(tmp_path, "results_*/results.json")
    assert BenchmarkManager._changed_result(tmp_path, "results_*/results.json", before) is None

    result.write_text('{"new": true, "larger": true}')

    assert BenchmarkManager._changed_result(tmp_path, "results_*/results.json", before) == result


def test_execution_observations_keep_every_gpu_backend_and_version_context() -> None:
    rows = [
        SimpleNamespace(
            runtime_telemetry={"gpu_indices": [0, 1]},
            gpu_index=0,
            backend_type="llamacpp_mks",
            runtime_preset="normal",
            effective_context=131072,
            remote_node_name=None,
            error=None,
            status_code=200,
            input_tokens=100,
            output_tokens=20,
        ),
        SimpleNamespace(
            runtime_telemetry={"gpu_indices": [1]},
            gpu_index=1,
            backend_type="llamacpp_mks",
            runtime_preset="normal",
            effective_context=131072,
            remote_node_name=None,
            error="timeout",
            status_code=500,
            input_tokens=30,
            output_tokens=0,
        ),
    ]
    states = {
        0: SimpleNamespace(name="RTX 3060", total_vram_mb=12288),
        1: SimpleNamespace(name="RTX 3090", total_vram_mb=24576),
    }

    observed = execution_observations(rows, states)

    assert observed["hardware"] == {
        "gpus": [
            {"index": 0, "name": "RTX 3060", "totalVramMb": 12288},
            {"index": 1, "name": "RTX 3090", "totalVramMb": 24576},
        ],
        "nodes": ["local"],
    }
    assert observed["model"] == {
        "backends": ["llamacpp_mks"],
        "runtimePresets": ["normal"],
        "effectiveContexts": [131072],
    }
    assert observed["requests"] == {
        "count": 2,
        "errors": 1,
        "inputTokens": 130,
        "outputTokens": 20,
    }


def test_backend_versions_include_installed_artifact_and_mark_remote_unknown() -> None:
    state = SimpleNamespace(
        installed_version="0.9.2",
        packages={"llama-cpp-python": "0.3.16"},
        oci_digest="sha256:abc",
    )
    installer = SimpleNamespace(get_state=lambda backend: state)

    assert backend_version_observations(["llamacpp_mks"], ["local"], installer) == [{
        "type": "llamacpp_mks",
        "installedVersion": "0.9.2",
        "packages": {"llama-cpp-python": "0.3.16"},
        "ociDigest": "sha256:abc",
        "node": "local",
    }]
    remote = backend_version_observations(["llamacpp_mks"], ["mapache"], None)[0]
    assert remote["installedVersion"] is None
    assert remote["node"] is None


def test_run_payload_exposes_phase_and_reproducible_attempt_environment() -> None:
    now = datetime.now(UTC)
    environment = {
        "software": {"ocabra": "0.124.0"},
        "hardware": {"gpus": [{"index": 1, "name": "RTX 3090"}]},
    }
    attempt = SimpleNamespace(
        id=uuid.uuid4(), model="model", attempt_number=1, status="completed",
        environment=environment, return_code=0, external_requests=[], error=None,
        started_at=now, finished_at=now, result={"score": 1},
    )
    run = SimpleNamespace(
        id=uuid.uuid4(), app="ifeval", suite="quick", revision="abc", models=["model"],
        suite_args=[], status="completed", phase="completed", auto_pause=True,
        pause_reason=None, current_model_index=1, current_model=None, result={"model": {}},
        contaminated_attempts=0, error=None, external_activity_at=None, created_at=now,
        started_at=now, finished_at=now, attempts=[attempt],
    )

    payload = _run_payload(run)

    assert payload["phase"] == "completed"
    assert payload["attempts"][0]["environment"] == environment


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_retry_keeps_completed_models_and_reuses_failed_checkpoint(tmp_path: Path) -> None:
    workspace = tmp_path / "attempt-1"
    workspace.mkdir()
    now = datetime.now(UTC)
    run = BenchmarkRun(
        app="ifeval",
        suite="quick",
        revision="abc",
        models=["accepted", "failed"],
        suite_args=[],
        status="failed",
        phase="failed",
        current_model_index=1,
        current_model="failed",
        workspace_path=str(tmp_path),
        result={"accepted": {"score": 1}},
        error="runner failed",
        finished_at=now,
    )
    async with AsyncSessionLocal() as session:
        session.add(run)
        await session.flush()
        session.add(BenchmarkAttempt(
            run_id=run.id,
            model="failed",
            attempt_number=1,
            status="failed",
            workspace_path=str(workspace),
            log_path=str(workspace / "runner.log"),
            environment={"software": {"ocabra": "0.124.0"}},
            error="runner failed",
            started_at=now,
            finished_at=now,
        ))
        await session.commit()

    manager = BenchmarkManager(SimpleNamespace(state=SimpleNamespace()))
    retried = await manager.request_retry(run.id)
    number, selected_workspace, resume = await manager._attempt_workspace(run.id, "failed")

    assert retried.status == "queued"
    assert retried.phase == "queued"
    assert retried.current_model_index == 1
    assert retried.current_model is None
    assert retried.result == {"accepted": {"score": 1}}
    assert retried.finished_at is None
    assert (number, selected_workspace, resume) == (2, workspace, True)

    async with AsyncSessionLocal() as session:
        persisted = await session.get(BenchmarkRun, run.id)
        assert persisted is not None
        await session.delete(persisted)
        await session.commit()


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_contaminated_attempt_requeues_same_model_without_accepting_result(
    tmp_path: Path,
) -> None:
    run = BenchmarkRun(
        app="ifeval",
        suite="quick",
        revision="abc",
        models=["model"],
        suite_args=[],
        status="running",
        phase="finalizing",
        current_model_index=0,
        current_model="model",
        workspace_path=str(tmp_path),
        result={},
    )
    async with AsyncSessionLocal() as session:
        session.add(run)
        await session.commit()

    manager = BenchmarkManager(SimpleNamespace(state=SimpleNamespace()))
    await manager._retry_contaminated(run.id)

    async with AsyncSessionLocal() as session:
        persisted = await session.get(BenchmarkRun, run.id)
        assert persisted is not None
        assert persisted.status == "queued"
        assert persisted.phase == "queued"
        assert persisted.current_model_index == 0
        assert persisted.current_model is None
        assert persisted.result == {}
        assert persisted.contaminated_attempts == 1
        await session.delete(persisted)
        await session.commit()
