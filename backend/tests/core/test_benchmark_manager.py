from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from ocabra.api.internal.benchmarks import BenchmarkRunCreate, BenchmarkScheduleCreate, _run_payload
from ocabra.benchmarks.catalog import APPS, catalog_payload
from ocabra.benchmarks.manager import (
    BenchmarkIdentity,
    BenchmarkManager,
    backend_version_observations,
    corrala_benchmark_database_url,
    execution_observations,
    external_active_requests,
    git_authorization_header,
    next_schedule_occurrence,
)
from ocabra.database import AsyncSessionLocal
from ocabra.db.benchmarks import BenchmarkAttempt, BenchmarkRun, BenchmarkSchedule


def test_github_token_uses_basic_header_for_git_https() -> None:
    header = git_authorization_header("github-token")

    scheme, encoded = header.split(" ", 2)[1:]
    assert scheme == "Basic"
    assert base64.b64decode(encoded).decode() == "x-access-token:github-token"


def test_corrala_scratch_database_keeps_the_real_password() -> None:
    derived = corrala_benchmark_database_url(
        "postgresql+asyncpg://ocabra:secret%40value@postgres:5432/ocabra"
    )
    parsed = sa.engine.make_url(derived)

    assert parsed.database == "ocabra_bench"
    assert parsed.password == "secret@value"
    assert "***" not in derived


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


def test_schedule_validates_clock_days_and_timezone() -> None:
    valid = BenchmarkScheduleCreate(
        name="Nocturno",
        app="ifeval",
        suite="quick",
        models=["model"],
        weekdays=[4, 0, 4],
        time_of_day="2:05",
        timezone="Europe/Madrid",
    )
    assert valid.weekdays == [0, 4]
    assert valid.time_of_day == "02:05"
    for values in (
        {"weekdays": [7]},
        {"time_of_day": "25:00"},
        {"timezone": "Mars/Olympus"},
    ):
        with pytest.raises(ValidationError):
            BenchmarkScheduleCreate(
                name="Inválido", app="ifeval", suite="quick", models=["model"], **values
            )


def test_next_schedule_occurrence_respects_weekday_and_timezone() -> None:
    # Friday 20 November 2026 at 01:30 UTC = 02:30 in Madrid.
    after = datetime(2026, 11, 20, 1, 30, tzinfo=UTC)
    assert next_schedule_occurrence([4], "03:00", "Europe/Madrid", after) == datetime(
        2026, 11, 20, 2, 0, tzinfo=UTC
    )
    assert next_schedule_occurrence([0], "02:00", "Europe/Madrid", after) == datetime(
        2026, 11, 23, 1, 0, tzinfo=UTC
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

    assert backend_version_observations(
        ["llamacpp_mks"],
        ["local"],
        installer,
        {"llamacpp_mks": "96fff6c0123456789"},
    ) == [
        {
            "type": "llamacpp_mks",
            "installedVersion": "0.9.2",
            "packages": {"llama-cpp-python": "0.3.16"},
            "ociDigest": "sha256:abc",
            "gitRef": None,
            "sourceCommit": "96fff6c0123456789",
            "node": "local",
        }
    ]
    remote = backend_version_observations(
        ["llamacpp_mks"], ["mapache"], installer, {"llamacpp_mks": "local-only"}
    )[0]
    assert remote["installedVersion"] is None
    assert remote["sourceCommit"] is None
    assert remote["node"] is None


def test_run_payload_exposes_phase_and_reproducible_attempt_environment() -> None:
    now = datetime.now(UTC)
    environment = {
        "software": {"ocabra": "0.124.0"},
        "hardware": {"gpus": [{"index": 1, "name": "RTX 3090"}]},
    }
    attempt = SimpleNamespace(
        id=uuid.uuid4(),
        model="model",
        attempt_number=1,
        status="completed",
        environment=environment,
        return_code=0,
        external_requests=[],
        error=None,
        started_at=now,
        finished_at=now,
        result={"score": 1},
    )
    run = SimpleNamespace(
        id=uuid.uuid4(),
        app="ifeval",
        suite="quick",
        revision="abc",
        models=["model"],
        suite_args=[],
        status="completed",
        phase="completed",
        auto_pause=True,
        schedule_id=None,
        is_baseline=True,
        pause_reason=None,
        current_model_index=1,
        current_model=None,
        result={"model": {}},
        contaminated_attempts=0,
        error=None,
        external_activity_at=None,
        created_at=now,
        started_at=now,
        finished_at=now,
        attempts=[attempt],
    )

    payload = _run_payload(run)

    assert payload["phase"] == "completed"
    assert payload["isBaseline"] is True
    assert payload["attempts"][0]["environment"] == environment


def test_run_payload_reads_bounded_progress_from_current_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now = datetime.now(UTC)
    workspace = tmp_path / "attempt-1"
    workspace.mkdir()
    (workspace / "progress.json").write_text(
        json.dumps({"completed": 17, "total": 50, "unit": "casos", "stage": "generating"})
    )
    monkeypatch.setattr("ocabra.api.internal.benchmarks.settings.benchmark_dir", str(tmp_path))
    attempt = SimpleNamespace(
        id=uuid.uuid4(),
        model="model",
        attempt_number=1,
        status="running",
        workspace_path=str(workspace),
        environment={},
        return_code=None,
        external_requests=[],
        error=None,
        started_at=now,
        finished_at=None,
        result=None,
    )
    run = SimpleNamespace(
        id=uuid.uuid4(),
        app="ifeval",
        suite="quick",
        revision="abc",
        models=["model"],
        suite_args=[],
        status="running",
        phase="running",
        auto_pause=True,
        schedule_id=None,
        is_baseline=False,
        pause_reason=None,
        current_model_index=0,
        current_model="model",
        result={},
        contaminated_attempts=0,
        error=None,
        external_activity_at=None,
        created_at=now,
        started_at=now,
        finished_at=None,
        attempts=[attempt],
    )

    payload = _run_payload(run)

    assert payload["testProgress"] == {
        "completed": 17,
        "total": 50,
        "unit": "casos",
        "stage": "generating",
        "updatedAt": payload["testProgress"]["updatedAt"],
    }


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
        session.add(
            BenchmarkAttempt(
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
            )
        )
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
async def test_due_schedule_queues_once_and_advances_next_occurrence(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    schedule = BenchmarkSchedule(
        name="Nocturno",
        app="ifeval",
        suite="quick",
        revision="abc",
        models=["model"],
        suite_args=[],
        auto_pause=True,
        enabled=True,
        weekdays=list(range(7)),
        time_of_day="00:00",
        timezone="UTC",
        next_run_at=now.replace(year=now.year - 1),
    )
    async with AsyncSessionLocal() as session:
        session.add(schedule)
        await session.commit()

    manager = BenchmarkManager(SimpleNamespace(state=SimpleNamespace()))
    await manager._enqueue_due_schedules()
    await manager._enqueue_due_schedules()
    with pytest.raises(ValueError, match="ejecución activa"):
        await manager.request_schedule_run(schedule.id)

    async with AsyncSessionLocal() as session:
        persisted = await session.get(BenchmarkSchedule, schedule.id)
        assert persisted is not None
        runs = (
            (
                await session.execute(
                    sa.select(BenchmarkRun).where(BenchmarkRun.schedule_id == schedule.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(runs) == 1
        assert runs[0].status == "queued"
        assert runs[0].models == ["model"]
        assert persisted.last_run_id == runs[0].id
        assert persisted.next_run_at is not None and persisted.next_run_at > now
        await session.delete(runs[0])
        await session.delete(persisted)
        await session.commit()


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_baseline_is_unique_per_application_suite(tmp_path: Path) -> None:
    runs = [
        BenchmarkRun(
            app="ifeval",
            suite="quick",
            revision="abc",
            models=[f"model-{index}"],
            suite_args=[],
            status="completed",
            phase="completed",
            current_model_index=1,
            workspace_path=str(tmp_path / str(index)),
            result={},
            finished_at=datetime.now(UTC),
            is_baseline=index == 0,
        )
        for index in range(2)
    ]
    async with AsyncSessionLocal() as session:
        session.add_all(runs)
        await session.commit()

    manager = BenchmarkManager(SimpleNamespace(state=SimpleNamespace()))
    selected = await manager.request_baseline(runs[1].id)
    assert selected.is_baseline is True

    async with AsyncSessionLocal() as session:
        first = await session.get(BenchmarkRun, runs[0].id)
        second = await session.get(BenchmarkRun, runs[1].id)
        assert first is not None and first.is_baseline is False
        assert second is not None and second.is_baseline is True
        await session.delete(first)
        await session.delete(second)
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


async def _run_with(tmp_path: Path, models: list[str], index: int, result: dict) -> BenchmarkRun:
    run = BenchmarkRun(
        app="fulcrum",
        suite="orquestacion",
        revision="abc",
        models=models,
        suite_args=[],
        status="running",
        phase="finalizing",
        current_model_index=index,
        current_model=models[index],
        workspace_path=str(tmp_path),
        result=result,
    )
    async with AsyncSessionLocal() as session:
        session.add(run)
        await session.commit()
    return run


async def _persisted(run_id) -> BenchmarkRun:
    async with AsyncSessionLocal() as session:
        persisted = await session.get(BenchmarkRun, run_id)
        assert persisted is not None
        await session.delete(persisted)
        await session.commit()
        return persisted


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_un_modelo_que_falla_no_tumba_a_los_siguientes(tmp_path: Path) -> None:
    # flint:40b salía con código 1 y la ejecución entera se daba por fallida:
    # los tres modelos que venían detrás nunca se medían.
    run = await _run_with(tmp_path, ["a", "flint", "c"], 1, {"a": {"score": 1}})
    manager = BenchmarkManager(SimpleNamespace(state=SimpleNamespace()))
    await manager._model_failed(run.id, "flint", "El runner terminó con código 1")

    persisted = await _persisted(run.id)
    assert persisted.status == "queued"
    assert persisted.current_model_index == 2
    assert persisted.error == "flint: El runner terminó con código 1"
    assert persisted.result == {"a": {"score": 1}}


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_si_falla_el_ultimo_con_resultados_previos_la_ejecucion_se_completa(
    tmp_path: Path,
) -> None:
    run = await _run_with(tmp_path, ["a", "b"], 1, {"a": {"score": 1}})
    manager = BenchmarkManager(SimpleNamespace(state=SimpleNamespace()))
    await manager._model_failed(run.id, "b", "boom")

    persisted = await _persisted(run.id)
    assert persisted.status == "completed"
    assert persisted.finished_at is not None
    assert persisted.error == "b: boom"


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_si_no_sale_ninguno_falla_y_reintentar_retoma_el_ultimo(tmp_path: Path) -> None:
    run = await _run_with(tmp_path, ["a", "b"], 0, {})
    manager = BenchmarkManager(SimpleNamespace(state=SimpleNamespace()))
    await manager._model_failed(run.id, "a", "primero")
    async with AsyncSessionLocal() as session:
        mid = await session.get(BenchmarkRun, run.id)
        assert mid is not None and mid.status == "queued" and mid.current_model_index == 1
        mid.status = "running"
        await session.commit()
    await manager._model_failed(run.id, "b", "segundo")

    persisted = await _persisted(run.id)
    assert persisted.status == "failed"
    assert persisted.current_model_index == 1
    assert "a: primero" in (persisted.error or "") and "segundo" in (persisted.error or "")


def test_las_notas_de_bfcl_anteriores_al_arreglo_se_marcan() -> None:
    from ocabra.benchmarks.catalog import scoring_issue

    before = datetime(2026, 11, 22, tzinfo=UTC)
    after = datetime(2026, 11, 27, tzinfo=UTC)
    assert scoring_issue("bfcl", "core", ["0.131.0"], after)["fixedIn"] == "0.137.0"
    assert scoring_issue("bfcl", "core", ["0.137.0"], before) is None
    # Sin versión registrada manda la fecha.
    assert scoring_issue("bfcl", "simple_python", [None], before) is not None
    assert scoring_issue("bfcl", "core", [], after) is None
    assert scoring_issue("ifeval", "quick", ["0.120.0"], before) is None
