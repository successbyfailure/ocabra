#!/usr/bin/env python3
"""Run pinned BFCL code against any oCabra OpenAI-compatible profile."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


def _safe_model(model: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in model)


def _inside_official_environment(source: Path) -> None:
    if os.environ.get("OCABRA_BFCL_ENV") == "1":
        return
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("BFCL necesita uv para crear su entorno oficial aislado")
    env = dict(os.environ)
    env["OCABRA_BFCL_ENV"] = "1"
    os.execvpe(
        uv,
        _official_environment_command(uv, source),
        env,
    )


def _official_environment_command(uv: str, source: Path) -> list[str]:
    """Add dependencies used by BFCL imports but omitted from its project file."""
    return [
        uv,
        "run",
        "--project",
        str(source),
        "--with",
        "soundfile",
        "python",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]


def _number(value: str | None) -> float | None:
    if value is None or not value.strip() or value.strip().lower() == "nan":
        return None
    try:
        normalized = value.strip()
        if normalized.endswith("%"):
            return float(normalized[:-1]) / 100
        return float(normalized)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--categories", default="simple_python")
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()
    project = args.source_root / "berkeley-function-call-leaderboard"
    _inside_official_environment(project)
    sys.path.insert(0, str(project))

    base_url = os.environ["OPENAI_COMPATIBLE_URL"].rstrip("/")
    os.environ["OPENAI_BASE_URL"] = base_url if base_url.endswith("/v1") else base_url + "/v1"
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_COMPATIBLE_KEY"]

    from bfcl_eval._llm_response_generation import main as generate  # noqa: PLC0415
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig  # noqa: PLC0415
    from bfcl_eval.eval_checker.eval_runner import main as evaluate  # noqa: PLC0415
    from bfcl_eval.model_handler.api_inference.openai_completion import (  # noqa: PLC0415
        OpenAICompletionsHandler,
    )

    model = args.models
    MODEL_CONFIG_MAPPING[model] = ModelConfig(
        model_name=model,
        display_name=model,
        url="",
        org="oCabra",
        license="local",
        model_handler=OpenAICompletionsHandler,
        is_fc_model=True,
    )
    result_dir = f"ocabra_bfcl_raw_{_safe_model(model)}"
    score_dir = f"ocabra_bfcl_scores_{_safe_model(model)}"
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    generate(
        SimpleNamespace(
            model=[model], test_category=categories, temperature=0.001,
            include_input_log=False, exclude_state_log=False, num_gpus=1,
            num_threads=1, gpu_memory_utilization=0.9, backend="sglang",
            skip_server_setup=True, local_model_path=None, result_dir=result_dir,
            allow_overwrite=False, run_ids=False,
        )
    )
    evaluate([model], categories, result_dir, score_dir, partial_eval=False)

    score_path = project / score_dir / "data_non_live.csv"
    with score_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if item.get("Model") == model), rows[-1])
    metric_columns = {
        "non_live_accuracy": "Non-Live Overall Acc",
        "simple_accuracy": "Python Simple AST",
        "multiple_accuracy": "Multiple AST",
        "parallel_accuracy": "Parallel AST",
        "parallel_multiple_accuracy": "Parallel Multiple AST",
        "irrelevance_accuracy": "Irrelevance Detection",
    }
    metrics = {
        key: number
        for key, column in metric_columns.items()
        if (number := _number(row.get(column))) is not None
    }
    primary = "simple_accuracy" if categories == ["simple_python"] else "non_live_accuracy"
    output = project / f"ocabra_results_{_safe_model(model)}"
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": "bfcl_v4",
        "model": model,
        "status": "completed",
        "categories": categories,
        "primary_metric": primary,
        "metrics": metrics,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
