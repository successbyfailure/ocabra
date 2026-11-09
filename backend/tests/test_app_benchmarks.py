"""El orquestador conserva contexto sin apropiarse de los datasets de las apps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import app_benchmarks  # noqa: E402


def _write_manifest(tmp_path: Path, *, script: str = "demo/run.py") -> Path:
    manifest = {
        "schema_version": 1,
        "app": "demo",
        "description": "Dataset de prueba",
        "repository": "https://example.invalid/demo.git",
        "default_revision": "main",
        "dataset_root": "eval/datasets",
        "suites": {
            "quality": {
                "description": "Calidad",
                "script": script,
                "result_glob": "demo/results_*/results.json",
                "models_flag": "--models",
            }
        },
    }
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_source(tmp_path: Path, script_body: str) -> Path:
    source = tmp_path / "source"
    suite_dir = source / "eval" / "datasets" / "demo"
    suite_dir.mkdir(parents=True)
    (suite_dir / "dataset.json").write_text('{"case": 1}\n', encoding="utf-8")
    (suite_dir / "run.py").write_text(script_body, encoding="utf-8")
    return source


def test_el_manifiesto_de_escriba_declara_las_cuatro_suites() -> None:
    config = app_benchmarks.load_app_config(PROJECT_ROOT / "benchmark" / "apps" / "escriba.json")

    assert config.app == "escriba"
    assert set(config.suites) == {"llm", "stt", "stt_diarization", "embeddings"}
    assert config.suites["stt"].prepare_script == Path("stt/download_dataset.py")
    assert config.suites["stt_diarization"].prepare_script == Path(
        "stt_diarization/download_dataset.py"
    )


@pytest.mark.parametrize("unsafe_path", ["../run.py", "/tmp/run.py"])
def test_un_manifiesto_no_puede_ejecutar_fuera_del_dataset(
    tmp_path: Path, unsafe_path: str
) -> None:
    path = _write_manifest(tmp_path, script=unsafe_path)

    with pytest.raises(app_benchmarks.BenchmarkError, match="dentro del repositorio"):
        app_benchmarks.load_app_config(path)


def test_archiva_resultado_y_contexto_sin_guardar_la_clave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(
        tmp_path,
        """\
import json
import os
import sys
from pathlib import Path

out = Path(__file__).parent / "results_fixed"
out.mkdir(exist_ok=True)
(out / "results.json").write_text(json.dumps({
    "argv": sys.argv[1:],
    "url": os.environ["OPENAI_COMPATIBLE_URL"],
    "has_key": bool(os.environ["OPENAI_COMPATIBLE_KEY"]),
}))
""",
    )
    config = app_benchmarks.load_app_config(_write_manifest(tmp_path))
    monkeypatch.setattr(
        app_benchmarks,
        "_server_version",
        lambda base_url, api_key: {
            "url": base_url,
            "version": "9.9.9",
            "reachable": True,
        },
    )

    exit_code, run_dir = app_benchmarks.execute_benchmark(
        config,
        "quality",
        source=source,
        revision=None,
        base_url="https://ocabra.example",
        api_key="super-secreta",
        models="modelo-a,modelo-b",
        prepare=False,
        suite_args=["--extra"],
        output_dir=tmp_path / "archive",
    )

    assert exit_code == 0
    assert run_dir is not None
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert result == {
        "argv": ["--models", "modelo-a,modelo-b", "--extra"],
        "url": "https://ocabra.example",
        "has_key": True,
    }
    metadata_text = (run_dir / "run.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert "super-secreta" not in metadata_text
    assert metadata["status"] == "completed"
    assert metadata["models"] == ["modelo-a", "modelo-b"]
    assert metadata["server"]["version"] == "9.9.9"
    assert len(metadata["dataset"]["sha256"]) == 64


def test_un_runner_que_falla_deja_un_registro_diagnosticable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path, "raise SystemExit(7)\n")
    config = app_benchmarks.load_app_config(_write_manifest(tmp_path))
    monkeypatch.setattr(
        app_benchmarks,
        "_server_version",
        lambda base_url, api_key: {"url": base_url, "reachable": False},
    )

    exit_code, run_dir = app_benchmarks.execute_benchmark(
        config,
        "quality",
        source=source,
        revision=None,
        base_url="http://localhost:8000",
        api_key="",
        models=None,
        prepare=False,
        suite_args=[],
        output_dir=tmp_path / "archive",
    )

    assert exit_code == 7
    assert run_dir is not None
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == 7
    assert metadata["result_file"] is None
    assert (run_dir / "runner.log").is_file()


def test_no_finge_una_revision_distinta_en_un_checkout_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = app_benchmarks.load_app_config(_write_manifest(tmp_path))
    monkeypatch.setattr(
        app_benchmarks,
        "_git_metadata",
        lambda root: ("commit-head", False, "https://example.invalid/demo.git"),
    )
    monkeypatch.setattr(app_benchmarks, "_run_git", lambda root, *args: "commit-pedido")

    with pytest.raises(app_benchmarks.BenchmarkError, match="no está en la revisión"):
        with app_benchmarks.acquire_source(config, source, "v1"):
            pass
