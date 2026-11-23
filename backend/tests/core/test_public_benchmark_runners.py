from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from ocabra.benchmarks.runners import bfcl, ifeval


def test_bfcl_score_parser_accepts_decimal_and_percentage() -> None:
    assert bfcl._number("0.82") == 0.82
    assert bfcl._number("82%") == 0.82
    assert bfcl._number("") is None


def test_bfcl_adds_dependency_omitted_by_the_official_project(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bfcl.py", "--models", "model"])
    command = bfcl._official_environment_command("/usr/bin/uv", Path("/bfcl"))

    assert command[:7] == [
        "/usr/bin/uv", "run", "--project", "/bfcl", "--with", "soundfile", "python"
    ]
    assert command[-2:] == ["--models", "model"]


def test_bfcl_progress_expands_collections_and_counts_checkpoint_rows(
    tmp_path: Path, monkeypatch,
) -> None:
    data = tmp_path / "bfcl_eval" / "data"
    data.mkdir(parents=True)
    (data / "BFCL_v4_simple_python.json").write_text('{"id": 1}\n{"id": 2}\n')
    result = tmp_path / "results" / "model" / "non_live"
    result.mkdir(parents=True)
    (result / "BFCL_v4_simple_python_result.json").write_text('{"id": 1}\n')
    progress = tmp_path / "progress.json"
    monkeypatch.setenv("OCABRA_BENCHMARK_PROGRESS_PATH", str(progress))
    stop = threading.Event()
    stop.set()

    bfcl._watch_generation_progress(
        tmp_path, tmp_path / "results", ["simple_python"], "BFCL_v4", stop,
    )

    assert bfcl._expanded_categories(
        ["non_live"], {"non_live": ["simple_python", "multiple"]},
    ) == ["simple_python", "multiple"]
    assert json.loads(progress.read_text()) == {
        "completed": 1,
        "total": 2,
        "unit": "casos",
        "stage": "generating",
    }


def test_ifeval_checkpoints_each_response_and_normalizes_official_scores(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "instruction_following_eval"
    data = dataset / "data"
    data.mkdir(parents=True)
    inputs = [
        {"key": 1, "prompt": "uno", "instruction_id_list": ["a"], "kwargs": [{}]},
        {"key": 2, "prompt": "dos", "instruction_id_list": ["a", "b"], "kwargs": [{}, {}]},
    ]
    (data / "input_data.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in inputs), encoding="utf-8"
    )
    calls: list[str] = []

    def fake_request(prompt: str, model: str):
        calls.append(prompt)
        return f"respuesta {prompt}", 1.5, {"input_tokens": 10, "output_tokens": 4}

    scorer_commands: list[list[str]] = []

    def fake_scorer(command, **kwargs):
        scorer_commands.append(command)
        output = dataset / "ocabra_results_model"
        strict = [
            {"follow_instruction_list": [True]},
            {"follow_instruction_list": [True, False]},
        ]
        loose = [
            {"follow_instruction_list": [True]},
            {"follow_instruction_list": [True, True]},
        ]
        for name, rows in (("eval_results_strict.jsonl", strict), ("eval_results_loose.jsonl", loose)):
            (output / name).write_text(
                "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
            )
        return None

    monkeypatch.setattr(ifeval, "_request", fake_request)
    monkeypatch.setattr(ifeval.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(ifeval.subprocess, "run", fake_scorer)
    progress_path = tmp_path / "progress.json"
    monkeypatch.setenv("OCABRA_BENCHMARK_PROGRESS_PATH", str(progress_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["ifeval.py", "--source-root", str(tmp_path), "--models", "model", "--skip-done"],
    )

    assert ifeval.main() == 0
    result_path = dataset / "ocabra_results_model" / "results.json"
    result = json.loads(result_path.read_text())
    assert calls == ["uno", "dos"]
    assert result["cases"] == 2
    assert result["metrics"]["strict_prompt_accuracy"] == 0.5
    assert result["metrics"]["strict_instruction_accuracy"] == 2 / 3
    assert result["metrics"]["loose_prompt_accuracy"] == 1
    assert "nltk.download('punkt_tab'" in scorer_commands[0][scorer_commands[0].index("-c") + 1]
    assert scorer_commands[1][-4] == "instruction_following_eval.evaluation_main"
    assert scorer_commands[1].count("--with") == 4
    assert json.loads(progress_path.read_text()) == {
        "completed": 2,
        "total": 2,
        "unit": "casos",
        "stage": "completed",
    }

    calls.clear()
    assert ifeval.main() == 0
    assert calls == []
