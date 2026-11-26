"""Informe del diagnóstico y exportación de candidatos a entrenamiento.

Formato de ``candidates.train.jsonl`` / ``candidates.eval.jsonl`` (una línea por
caso clasificado como «modelo»)::

    {"id": "bfcl:simple_python_12", "app": "bfcl", "suite": "core",
     "category": "simple_python", "model": "gemma4:12b",
     "messages": [...], "tools": [...] | null,
     "response": <lo que respondió en el benchmark>,
     "expected": <respuesta esperada o restricciones; depende de la app>,
     "failure": "por qué falla", "score": 0.0,
     "replays": {"recomendado": [{"passed": false, ...}]},
     "split": "train" | "eval", "source": {"run_id", "attempt_id", "revision"}}

**Separación de evaluación.** Cada caso va a ``train`` o a ``eval`` por un
hash estable de su id (20 % a eval), así que el mismo caso cae siempre en el
mismo lado aunque se diagnostiquen otros modelos u otras ejecuciones. Los de
``eval`` no se deben usar para entrenar: son los que dirán si el fine-tune
mejora de verdad. Ojo: todos salen del benchmark; para medir sin sesgo, compara
el modelo entrenado solo sobre los casos ``eval`` (o sobre datos nuevos), no
sobre la suite completa.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ocabra.benchmarks.diagnosis.classify import CATEGORIES

EVAL_SHARE = 5  # 1 de cada 5 → eval


def split_for(case_id: str) -> str:
    digest = hashlib.sha256(case_id.encode()).digest()
    return "eval" if digest[0] % EVAL_SHARE == 0 else "train"


def write_candidates(
    directory: Path,
    model: str,
    cases: list[dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
    replays: dict[str, dict[str, list[dict[str, Any]]]],
    source: dict[str, Any],
) -> dict[str, int]:
    files = {
        "train": (directory / "candidates.train.jsonl").open("w", encoding="utf-8"),
        "eval": (directory / "candidates.eval.jsonl").open("w", encoding="utf-8"),
    }
    counts = {"train": 0, "eval": 0}
    try:
        for case in cases:
            if verdicts.get(case["case_id"], {}).get("category") != "modelo":
                continue
            split = split_for(case["case_id"])
            row = {
                "id": case["case_id"],
                "app": case["app"],
                "suite": case["suite"],
                "category": case["category"],
                "model": model,
                "messages": case.get("messages"),
                "tools": case.get("tools"),
                "response": case.get("response"),
                "expected": case.get("expected"),
                "failure": case.get("failure"),
                "score": case.get("score"),
                "replays": replays.get(case["case_id"], {}),
                "split": split,
                "source": source,
            }
            files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in files.values():
            handle.close()
    return counts


_LABELS = {
    "plantilla": "Plantilla",
    "sampling": "Sampling",
    "limites": "Límites",
    "servido": "Cuantización / servido",
    "modelo": "Modelo",
    "no_reproduce": "No se reproduce",
    "sin_verificar": "Sin verificar",
}


def _cell(text: Any, limit: int = 140) -> str:
    value = str(text if text is not None else "—").replace("|", "\\|").replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Diagnóstico de fallos — {report['model']} · {report['app']}/{report['suite']}",
        "",
        f"Ejecución `{report['run_id']}`, intento `{report['attempt_id']}`. "
        f"{summary['total']} casos fallidos; repetidos {report['replayed_cases']} "
        f"({report['repeats']} repeticiones con el muestreo recomendado).",
        "",
        "## Resumen",
        "",
        "| Categoría | Casos | % de los fallos | % de los atribuidos |",
        "|---|---|---|---|",
    ]
    for name in CATEGORIES:
        attributed = summary["percent_of_attributed"].get(name)
        lines.append(
            f"| {_LABELS[name]} | {summary['counts'][name]} | "
            f"{summary['percent_of_failures'][name]} % | "
            f"{'—' if attributed is None else f'{attributed} %'} |"
        )
    findings = report.get("findings") or []
    if findings:
        lines += ["", "## Hallazgos de servido", ""]
        lines += [f"- {item}" for item in findings]
    params = report.get("params") or {}
    if params:
        lines += [
            "",
            "## Parámetros efectivos vs recomendados",
            "",
            f"Referencia: `{(report.get('reference') or {}).get('repo') or '—'}`.",
            "",
            "| Parámetro | Efectivo (benchmark) | Recomendado |",
            "|---|---|---|",
        ]
        for key, value in (params.get("differences") or {}).items():
            lines.append(f"| `{key}` | {value['efectivo']} | {value['recomendado']} |")
    template = report.get("template") or {}
    if template:
        lines += ["", "## Plantilla", ""]
        for name, probe in (template.get("probes") or {}).items():
            lines.append(f"- **{name}**: {probe.get('verdict')} — {probe.get('detail', '')}")
    lines += [
        "",
        "## Casos",
        "",
        "| Caso | Categoría | Fallo | Evidencia |",
        "|---|---|---|---|",
    ]
    for verdict in report["cases"]:
        lines.append(
            f"| `{_cell(verdict['case_id'], 60)}` | {_LABELS[verdict['category']]} | "
            f"{_cell(verdict['failure'])} | {_cell('; '.join(verdict['evidence']), 200)} |"
        )
    candidates = report.get("candidates") or {}
    lines += [
        "",
        "## Candidatos a entrenamiento",
        "",
        f"{candidates.get('train', 0)} en `candidates.train.jsonl` y "
        f"{candidates.get('eval', 0)} en `candidates.eval.jsonl` (reservados para medir; "
        "no entrenar con ellos).",
        "",
    ]
    return "\n".join(lines)
