"""CLI del diagnóstico de fallos de benchmark (cliente de la API, solo stdlib).

El diagnóstico corre dentro del api, que es quien ve los workers y la cola;
esto solo lo encola, espera y descarga el informe::

    OCABRA_URL=http://127.0.0.1:8484 OCABRA_API_KEY=... \\
      python -m ocabra.benchmarks.diagnosis --run <run_id> --model gemma4:12b \\
        --repeats 3 --max-cases 30 --out ./diagnostico

``--attempt`` en lugar de ``--run``/``--model`` si ya sabes el intento.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _request(base: str, key: str, method: str, path: str, body: dict | None = None) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise SystemExit(f"HTTP {exc.code} en {path}: {detail}") from exc


def _json(base: str, key: str, method: str, path: str, body: dict | None = None):
    return json.loads(_request(base, key, method, path, body))


def _attempt_for(base: str, key: str, run_id: str, model: str) -> str:
    run = _json(base, key, "GET", f"/ocabra/benchmarks/runs/{run_id}")
    attempts = [
        item
        for item in run.get("attempts") or []
        if item.get("model") == model and item.get("status") == "completed"
    ]
    if not attempts:
        raise SystemExit(f"La ejecución {run_id} no tiene un intento completado de {model}")
    return attempts[-1]["id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="¿Es el modelo o es cómo lo servimos?")
    parser.add_argument("--url", default=os.environ.get("OCABRA_URL", "http://127.0.0.1:8484"))
    parser.add_argument("--key", default=os.environ.get("OCABRA_API_KEY", ""))
    parser.add_argument("--attempt")
    parser.add_argument("--run")
    parser.add_argument("--model")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--compare-model")
    parser.add_argument("--reference-repo")
    parser.add_argument("--allow-load", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.key:
        raise SystemExit("Falta la clave: --key u OCABRA_API_KEY (rol model_manager)")
    attempt = args.attempt
    if attempt is None:
        if not (args.run and args.model):
            raise SystemExit("Indica --attempt, o --run y --model")
        attempt = _attempt_for(args.url, args.key, args.run, args.model)

    job = _json(
        args.url,
        args.key,
        "POST",
        f"/ocabra/benchmarks/attempts/{attempt}/diagnoses",
        {
            "repeats": args.repeats,
            "max_cases": args.max_cases,
            "compare_model": args.compare_model,
            "reference_repo": args.reference_repo,
            "allow_load": args.allow_load,
        },
    )
    print(f"Diagnóstico {job['id']} encolado", flush=True)
    if args.no_wait:
        return 0
    last = None
    while job["status"] not in ("completed", "failed"):
        time.sleep(10)
        job = _json(args.url, args.key, "GET", f"/ocabra/benchmarks/diagnoses/{job['id']}")
        progress = job.get("progress") or {}
        line = f"{job['phase']} {progress.get('completed', 0)}/{progress.get('total', 0)}"
        if line != last:
            print(line, flush=True)
            last = line
    if job["status"] == "failed":
        print(f"Falló: {job.get('error')}", file=sys.stderr)
        return 1
    summary = job["report"]["summary"]
    print(json.dumps(summary["percent_of_failures"], ensure_ascii=False))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name in ("report.md", "candidates.train.jsonl", "candidates.eval.jsonl"):
            content = _request(
                args.url, args.key, "GET", f"/ocabra/benchmarks/diagnoses/{job['id']}/files/{name}"
            )
            (args.out / name).write_bytes(content)
        print(f"Informe y candidatos en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
