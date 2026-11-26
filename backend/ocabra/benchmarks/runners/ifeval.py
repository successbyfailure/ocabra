#!/usr/bin/env python3
"""Generate oCabra responses and score them with Google's official IFEval."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

_SCORER_DEPENDENCIES = ("absl-py", "langdetect", "nltk", "immutabledict")


def _safe_model(model: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in model)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_progress(completed: int, total: int, stage: str) -> None:
    raw_path = os.environ.get("OCABRA_BENCHMARK_PROGRESS_PATH")
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "completed": completed,
                "total": total,
                "unit": "casos",
                "stage": stage,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _request(prompt: str, model: str) -> tuple[str, float, dict[str, int]]:
    base_url = os.environ["OPENAI_COMPATIBLE_URL"].rstrip("/")
    endpoint = f"{base_url}/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 4096,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_COMPATIBLE_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read().decode())
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f"IFEval no pudo generar una respuesta: {exc}") from exc
    elapsed = time.monotonic() - started
    content = payload["choices"][0]["message"].get("content") or ""
    usage = payload.get("usage") or {}
    return (
        str(content),
        elapsed,
        {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    )


def _accuracy(path: Path) -> tuple[float, float, int, int]:
    prompt_total = prompt_correct = instruction_total = instruction_correct = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            followed = row["follow_instruction_list"]
            prompt_total += 1
            prompt_correct += int(all(followed))
            instruction_total += len(followed)
            instruction_correct += sum(bool(item) for item in followed)
    return (
        prompt_correct / prompt_total,
        instruction_correct / instruction_total,
        prompt_correct,
        instruction_correct,
    )


def _uv_python(uv: str) -> list[str]:
    command = [uv, "run"]
    for dependency in _SCORER_DEPENDENCIES:
        command.extend(("--with", dependency))
    return [*command, "python"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()

    dataset_root = args.source_root / "instruction_following_eval"
    official_input = dataset_root / "data" / "input_data.jsonl"
    if not official_input.is_file():
        raise SystemExit(f"No está el dataset oficial de IFEval: {official_input}")
    all_inputs = [json.loads(line) for line in official_input.read_text().splitlines() if line]
    inputs = all_inputs[: args.limit] if args.limit else all_inputs

    output = dataset_root / f"ocabra_results_{_safe_model(args.models)}"
    output.mkdir(parents=True, exist_ok=True)
    selected_input = output / "input_data.jsonl"
    responses_path = output / "responses.jsonl"
    _write_jsonl(selected_input, inputs)

    completed: dict[str, dict[str, object]] = {}
    if responses_path.is_file():
        completed = {
            row["prompt"]: row
            for row in (
                json.loads(line) for line in responses_path.read_text().splitlines() if line
            )
        }
    _write_progress(len(completed), len(inputs), "generating")
    for index, item in enumerate(inputs, 1):
        prompt = item["prompt"]
        if prompt in completed:
            continue
        response, latency, usage = _request(prompt, args.models)
        completed[prompt] = {
            "prompt": prompt,
            "response": response,
            "latency_s": latency,
            **usage,
        }
        _write_jsonl(
            responses_path,
            [completed[row["prompt"]] for row in inputs if row["prompt"] in completed],
        )
        _write_progress(len(completed), len(inputs), "generating")
        print(f"IFEval {index}/{len(inputs)}", flush=True)

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("IFEval necesita uv para instalar el scorer oficial aislado")
    cache_root = Path(os.environ.get("UV_CACHE_DIR", output / ".uv-cache")).parent
    nltk_data = cache_root / "nltk_data"
    nltk_data.mkdir(parents=True, exist_ok=True)
    scorer_env = {**os.environ, "NLTK_DATA": str(nltk_data)}
    _write_progress(len(completed), len(inputs), "scoring")
    subprocess.run(
        [
            *_uv_python(uv),
            "-c",
            (
                "import nltk,sys; "
                "ok=nltk.download('punkt_tab', download_dir=sys.argv[1], quiet=True); "
                "raise SystemExit(0 if ok else 1)"
            ),
            str(nltk_data),
        ],
        cwd=args.source_root,
        env=scorer_env,
        check=True,
    )
    subprocess.run(
        [
            *_uv_python(uv),
            "-m",
            "instruction_following_eval.evaluation_main",
            f"--input_data={selected_input}",
            f"--input_response_data={responses_path}",
            f"--output_dir={output}",
        ],
        cwd=args.source_root,
        env=scorer_env,
        check=True,
    )

    strict = _accuracy(output / "eval_results_strict.jsonl")
    loose = _accuracy(output / "eval_results_loose.jsonl")
    rows = list(completed.values())
    latencies = [float(row["latency_s"]) for row in rows]
    result = {
        "benchmark": "ifeval",
        "model": args.models,
        "status": "completed",
        "cases": len(inputs),
        "metrics": {
            "strict_prompt_accuracy": strict[0],
            "strict_instruction_accuracy": strict[1],
            "loose_prompt_accuracy": loose[0],
            "loose_instruction_accuracy": loose[1],
            "latency_s_p50": statistics.median(latencies),
            "input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_progress(len(inputs), len(inputs), "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
