from __future__ import annotations

import json
import sys
from pathlib import Path

from ocabra.benchmarks.runners import bfcl, ifeval


def test_bfcl_score_parser_accepts_decimal_and_percentage() -> None:
    assert bfcl._number("0.82") == 0.82
    assert bfcl._number("82%") == 0.82
    assert bfcl._number("") is None


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

    def fake_scorer(*args, **kwargs):
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

    calls.clear()
    assert ifeval.main() == 0
    assert calls == []
