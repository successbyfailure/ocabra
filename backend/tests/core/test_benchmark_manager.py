from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ocabra.api.internal.benchmarks import BenchmarkRunCreate
from ocabra.benchmarks.catalog import APPS, catalog_payload
from ocabra.benchmarks.manager import (
    BenchmarkIdentity,
    BenchmarkManager,
    external_active_requests,
    git_authorization_header,
)


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
        assert set(raw["suites"]) == set(app.suites)
        for suite_name, suite in app.suites.items():
            assert raw["suites"][suite_name]["script"] == suite.script
            assert raw["suites"][suite_name]["result_glob"] == suite.result_glob


def test_catalog_exposes_resume_and_preparation() -> None:
    payload = {item["app"]: item for item in catalog_payload()}
    corrala = payload["corrala"]["suites"][0]
    escriba_stt = next(item for item in payload["escriba"]["suites"] if item["name"] == "stt")
    assert corrala["resumable"] is True
    assert corrala["needsPreparation"] is False
    assert escriba_stt["needsPreparation"] is True


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
