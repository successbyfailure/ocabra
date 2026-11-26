#!/usr/bin/env python3
"""Run pinned BFCL code against any oCabra OpenAI-compatible profile."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import threading
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


def _write_progress(completed: int, total: int, stage: str) -> None:
    raw_path = os.environ.get("OCABRA_BENCHMARK_PROGRESS_PATH")
    if not raw_path or total <= 0:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "completed": min(completed, total),
                "total": total,
                "unit": "casos",
                "stage": stage,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _line_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _expanded_categories(
    categories: list[str],
    collection_mapping: dict[str, list[str]],
) -> list[str]:
    expanded: list[str] = []
    for category in categories:
        for item in collection_mapping.get(category, [category]):
            if item not in expanded:
                expanded.append(item)
    return expanded


def _watch_generation_progress(
    project: Path,
    result_root: Path,
    categories: list[str],
    version_prefix: str,
    stop: threading.Event,
) -> None:
    total = sum(
        _line_count(project / "bfcl_eval" / "data" / f"{version_prefix}_{category}.json")
        for category in categories
    )
    while True:
        completed = sum(
            _line_count(path)
            for category in categories
            for path in result_root.rglob(f"{version_prefix}_{category}_result.json")
        )
        _write_progress(completed, total, "generating")
        if stop.wait(1):
            return


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
    from bfcl_eval.constants.category_mapping import (  # noqa: PLC0415
        TEST_COLLECTION_MAPPING,
        VERSION_PREFIX,
    )
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
        # Las APIs OpenAI no admiten «.» en los nombres de función: BFCL manda
        # math.factorial como math_factorial y solo lo deshace al puntuar con
        # esto. Sin ello, ~380 respuestas correctas por modelo contaban como
        # «nombre de función incorrecto» (simple, java y multiple).
        underscore_to_dot=True,
    )
    result_dir = f"ocabra_bfcl_raw_{_safe_model(model)}"
    score_dir = f"ocabra_bfcl_scores_{_safe_model(model)}"
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    expanded = _expanded_categories(categories, TEST_COLLECTION_MAPPING)
    total = sum(
        _line_count(project / "bfcl_eval" / "data" / f"{VERSION_PREFIX}_{category}.json")
        for category in expanded
    )
    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch_generation_progress,
        args=(project, project / result_dir, expanded, VERSION_PREFIX, stop),
        daemon=True,
    )
    watcher.start()
    try:
        generate(
            SimpleNamespace(
                model=[model],
                test_category=categories,
                temperature=0.001,
                include_input_log=False,
                exclude_state_log=False,
                num_gpus=1,
                num_threads=1,
                gpu_memory_utilization=0.9,
                backend="sglang",
                skip_server_setup=True,
                local_model_path=None,
                result_dir=result_dir,
                allow_overwrite=False,
                run_ids=False,
            )
        )
    finally:
        stop.set()
        watcher.join(timeout=2)
    _write_progress(total, total, "scoring")
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
    if categories == ["simple_python"]:
        # La media non-live de BFCL incluye categorías que esta suite no mide:
        # con solo simple_python sale ~0,08 y engaña a quien la mire.
        metrics.pop("non_live_accuracy", None)
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
    _write_progress(total, total, "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
