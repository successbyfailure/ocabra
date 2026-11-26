"""Casos fallidos de un intento de benchmark, leídos de su espacio de trabajo.

Cada app deja su evidencia por caso en un sitio distinto. Aquí se normaliza a
:class:`FailedCase`: los mensajes que vio el modelo (si la app los guarda), el
cuerpo que mandó el runner, la respuesta, por qué falló y las señales que ya
apuntan a un límite (salida cortada) o al arnés.

Solo IFEval y BFCL guardan lo bastante para repetir y volver a puntuar un caso;
Corrala y Fulcrum guardan el resultado pero no los mensajes, así que sus casos
solo admiten las comprobaciones que no necesitan repetir.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Lo que manda cada runner, tal cual (ver benchmarks/runners/*.py y los
# runners oficiales). Es la línea base de los «parámetros efectivos».
IFEVAL_CLIENT_BODY: dict[str, Any] = {"temperature": 0, "max_tokens": 4096}
BFCL_CLIENT_BODY: dict[str, Any] = {"temperature": 0.001}

REPLAYABLE_APPS = frozenset({"ifeval", "bfcl"})


@dataclass
class FailedCase:
    """Un caso que el benchmark dio por fallido."""

    case_id: str
    app: str
    suite: str
    category: str
    failure: str
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    client_body: dict[str, Any] = field(default_factory=dict)
    response: Any = None
    expected: Any = None
    score: float | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    # Lo que necesita el script de repetición de la app (fila del dataset).
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def replayable(self) -> bool:
        return self.app in REPLAYABLE_APPS and bool(self.payload)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["replayable"] = self.replayable
        return data


def safe_model(model: str) -> str:
    """El mismo nombre de carpeta que usan los runners de oCabra."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in model)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _newest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


# ── IFEval ────────────────────────────────────────────────────────────────


def ifeval_cases(source: Path, model: str, suite: str) -> list[FailedCase]:
    output = source / "instruction_following_eval" / f"ocabra_results_{safe_model(model)}"
    strict = output / "eval_results_strict.jsonl"
    if not strict.is_file():
        raise FileNotFoundError(f"No está la evaluación de IFEval: {strict}")
    inputs = {row["prompt"]: row for row in _read_jsonl(output / "input_data.jsonl")}
    responses = {}
    if (output / "responses.jsonl").is_file():
        responses = {row["prompt"]: row for row in _read_jsonl(output / "responses.jsonl")}
    cases: list[FailedCase] = []
    for row in _read_jsonl(strict):
        if row.get("follow_all_instructions"):
            continue
        prompt = row["prompt"]
        source_row = inputs.get(prompt, {})
        followed = [bool(item) for item in row.get("follow_instruction_list") or []]
        ids = list(row.get("instruction_id_list") or [])
        missed = [name for name, ok in zip(ids, followed, strict=False) if not ok]
        generated = responses.get(prompt, {})
        output_tokens = int(generated.get("output_tokens") or 0)
        cases.append(
            FailedCase(
                case_id=f"ifeval:{source_row.get('key', len(cases))}",
                app="ifeval",
                suite=suite,
                category=",".join(sorted({name.split(":")[0] for name in missed})) or "ifeval",
                failure="no cumple: " + ", ".join(missed),
                messages=[{"role": "user", "content": prompt}],
                client_body=dict(IFEVAL_CLIENT_BODY),
                response=row.get("response"),
                expected={"instruction_id_list": ids, "kwargs": source_row.get("kwargs")},
                score=(sum(followed) / len(followed)) if followed else 0.0,
                signals={
                    "output_tokens": output_tokens,
                    # El runner no guarda finish_reason: llegar justo al tope
                    # es la señal de que la salida se cortó.
                    "hit_output_cap": output_tokens >= IFEVAL_CLIENT_BODY["max_tokens"],
                    "input_tokens": int(generated.get("input_tokens") or 0),
                },
                payload={"input": source_row} if source_row else {},
            )
        )
    return cases


# ── BFCL ──────────────────────────────────────────────────────────────────


def _bfcl_type(schema: Any) -> Any:
    """Tipos de Gorilla (``dict``, ``float``…) a JSON Schema, como hace BFCL."""
    mapping = {
        "dict": "object",
        "float": "number",
        "tuple": "array",
        "list": "array",
        "any": "string",
        "char": "string",
        "byte": "integer",
        "short": "integer",
        "long": "integer",
        "double": "number",
        "Boolean": "boolean",
        "String": "string",
        "ArrayList": "array",
        "HashMap": "object",
        "Hashtable": "object",
        "Queue": "array",
        "Stack": "array",
        "Any": "string",
        "bigint": "integer",
        "Array": "array",
    }
    if isinstance(schema, dict):
        converted = {key: _bfcl_type(value) for key, value in schema.items()}
        if isinstance(converted.get("type"), str):
            converted["type"] = mapping.get(converted["type"], converted["type"])
        return converted
    if isinstance(schema, list):
        return [_bfcl_type(item) for item in schema]
    return schema


def bfcl_openai_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aproximación de ``convert_to_tool`` de BFCL para renderizar la plantilla.

    La repetición usa la conversión oficial dentro del entorno de BFCL; esta
    solo sirve para pedir al worker el prompt renderizado sin cargar BFCL.
    """
    tools = []
    for function in functions:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": re.sub(r"\.", "_", str(function.get("name", ""))),
                    "description": function.get("description", ""),
                    "parameters": _bfcl_type(function.get("parameters") or {}),
                },
            }
        )
    return tools


def bfcl_dot_artifact(entry: dict[str, Any]) -> bool:
    """El modelo llamó a la función correcta, pero con ``_`` en vez de ``.``.

    BFCL cambia ``math.factorial`` por ``math_factorial`` al mandar las tools a
    una API OpenAI (no admite puntos) y solo lo deshace al puntuar si el modelo
    se registró con ``underscore_to_dot``. Si el runner no lo hace, cada una de
    esas respuestas correctas cuenta como «nombre de función incorrecto».
    """
    if "wrong_func_name" not in str(
        entry.get("error_type") or ""
    ) and "cannot_find_match" not in str(entry.get("error_type") or ""):
        return False
    expected = {
        name.replace(".", "_")
        for answer in entry.get("possible_answer") or []
        if isinstance(answer, dict)
        for name in answer
    }
    has_dot = any(
        "." in name
        for answer in entry.get("possible_answer") or []
        if isinstance(answer, dict)
        for name in answer
    )
    produced = {
        name
        for call in entry.get("model_result_decoded") or []
        if isinstance(call, dict)
        for name in call
    }
    return has_dot and bool(produced) and produced <= expected


def bfcl_cases(source: Path, model: str, suite: str) -> list[FailedCase]:
    project = source / "berkeley-function-call-leaderboard"
    scores = project / f"ocabra_bfcl_scores_{safe_model(model)}"
    files = sorted(scores.glob("*/non_live/*_score.json"))
    if not files:
        raise FileNotFoundError(f"No están las puntuaciones de BFCL: {scores}")
    cases: list[FailedCase] = []
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("valid", False):
                continue
            prompt = entry.get("prompt") or {}
            question = prompt.get("question") or []
            first_turn = question[0] if question and isinstance(question[0], list) else question
            functions = prompt.get("function") or []
            category = str(entry.get("test_category") or path.stem)
            errors = entry.get("error") or []
            cases.append(
                FailedCase(
                    case_id=f"bfcl:{entry.get('id')}",
                    app="bfcl",
                    suite=suite,
                    category=category,
                    failure=f"{entry.get('error_type')}: {'; '.join(map(str, errors))[:300]}",
                    messages=list(first_turn) if isinstance(first_turn, list) else None,
                    tools=bfcl_openai_tools(functions) if functions else None,
                    client_body=dict(BFCL_CLIENT_BODY),
                    response=entry.get("model_result_raw"),
                    expected=entry.get("possible_answer"),
                    score=0.0,
                    signals={"harness_dot_to_underscore": bfcl_dot_artifact(entry)},
                    payload={
                        "id": entry.get("id"),
                        "test_category": category,
                        "question": question,
                        "function": functions,
                        "possible_answer": entry.get("possible_answer"),
                        "model_result_decoded": entry.get("model_result_decoded"),
                    },
                )
            )
    return cases


# ── Corrala ───────────────────────────────────────────────────────────────


def corrala_cases(source: Path, model: str, suite: str) -> list[FailedCase]:
    root = source / "backend" / "scripts" / "llm_benchmark"
    runs = _newest(list(root.glob("results_*/runs.jsonl")))
    if runs is None:
        raise FileNotFoundError(f"No está runs.jsonl de Corrala en {root}")
    cases: list[FailedCase] = []
    for row in _read_jsonl(runs):
        if row.get("model") not in (model, None) or row.get("passed", True):
            continue
        if row.get("infra_error"):
            continue
        checks = row.get("checks") or {}
        failed_checks = sorted(name for name, ok in checks.items() if ok is False)
        cases.append(
            FailedCase(
                case_id=f"corrala:{row.get('suite')}:{row.get('case')}:{row.get('run', 0)}",
                app="corrala",
                suite=suite,
                category=str(row.get("suite") or "corrala"),
                failure="falla: " + (", ".join(failed_checks) or str(row.get("error") or "?")),
                response=row.get("final_text"),
                expected={
                    "primary": row.get("primary"),
                    "expected_queue": row.get("expected_queue"),
                },
                score=None,
                client_body={
                    key: row.get(key)
                    for key in ("reasoning_effort", "generation_overrides")
                    if row.get(key) is not None
                },
                signals={
                    "length_finishes": int(row.get("chain_length_finishes") or 0),
                    "completion_tokens": int(row.get("completion_tokens") or 0),
                    "prompt_tokens": int(row.get("prompt_tokens") or 0),
                    "tool_loop": bool((row.get("loops") or {}).get("tool_loop")),
                    "iteration_cap_hit": bool((row.get("loops") or {}).get("iteration_cap_hit")),
                },
            )
        )
    return cases


# ── Fulcrum ───────────────────────────────────────────────────────────────


def fulcrum_cases(source: Path, model: str, suite: str) -> list[FailedCase]:
    root = source / "benchmarks" / "orchestration"
    runs = _newest(list(root.glob("results_*/runs.jsonl")))
    if runs is None:
        raise FileNotFoundError(f"No está runs.jsonl de Fulcrum en {root}")
    cases: list[FailedCase] = []
    for row in _read_jsonl(runs):
        if row.get("model") not in (model, None) or row.get("error"):
            continue
        caps = row.get("caps") or {}
        failed = sorted(name for name, ok in caps.items() if ok is False)
        if not failed:
            continue
        detail = row.get("detail") or {}
        cutoffs = detail.get("cutoffs") or []
        for capability in failed:
            cases.append(
                FailedCase(
                    case_id=f"fulcrum:{row.get('scenario', 'fase_round')}:{row.get('rep', 0)}:{capability}",
                    app="fulcrum",
                    suite=suite,
                    category=capability,
                    failure=f"capacidad {capability} en falso",
                    score=None,
                    signals={
                        "length_finishes": sum(1 for item in cutoffs if item == "truncated"),
                        "tool_loop": any(item == "loop" for item in cutoffs),
                        "truncated_calls": int(detail.get("truncated_calls") or 0),
                    },
                )
            )
    return cases


EXTRACTORS = {
    "ifeval": ifeval_cases,
    "bfcl": bfcl_cases,
    "corrala": corrala_cases,
    "fulcrum": fulcrum_cases,
}


def failed_cases(app: str, suite: str, workspace: Path, model: str) -> list[FailedCase]:
    """Casos fallidos del intento cuyo espacio de trabajo es ``workspace``."""
    extractor = EXTRACTORS.get(app)
    if extractor is None:
        raise ValueError(f"El diagnóstico aún no sabe leer los casos de '{app}'")
    return extractor(workspace / "source", model, suite)
