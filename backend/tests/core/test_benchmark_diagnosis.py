"""Diagnóstico de fallos de benchmark: extracción, reglas, informe y trabajos."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ocabra.benchmarks.diagnosis import cases as case_module
from ocabra.benchmarks.diagnosis.classify import classify_case, summarize
from ocabra.benchmarks.diagnosis.jobs import DiagnosisRunner, DiagnosisStore
from ocabra.benchmarks.diagnosis.reference import recommended_sampling, resolve_reference_repo
from ocabra.benchmarks.diagnosis.report import render_markdown, split_for, write_candidates
from ocabra.benchmarks.diagnosis.serving import (
    compare_prompts,
    effective_params,
    sampling_differences,
)

SCRIPTS = Path(case_module.__file__).resolve().parent / "scripts"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# ── Extracción de casos ───────────────────────────────────────────────────


def test_ifeval_solo_devuelve_los_fallidos_con_su_evidencia(tmp_path: Path) -> None:
    out = tmp_path / "source" / "instruction_following_eval" / "ocabra_results_gemma4_12b"
    _jsonl(
        out / "input_data.jsonl",
        [
            {"key": 1, "prompt": "A", "instruction_id_list": ["length:max"], "kwargs": [{}]},
            {
                "key": 2,
                "prompt": "B",
                "instruction_id_list": ["case:lower", "x:y"],
                "kwargs": [{}, {}],
            },
        ],
    )
    _jsonl(
        out / "responses.jsonl",
        [{"prompt": "A", "output_tokens": 10}, {"prompt": "B", "output_tokens": 4096}],
    )
    _jsonl(
        out / "eval_results_strict.jsonl",
        [
            {"prompt": "A", "follow_all_instructions": True, "follow_instruction_list": [True],
             "instruction_id_list": ["length:max"], "response": "ok"},
            {"prompt": "B", "follow_all_instructions": False, "follow_instruction_list": [False, True],
             "instruction_id_list": ["case:lower", "x:y"], "response": "NO"},
        ],
    )  # fmt: skip
    found = case_module.failed_cases("ifeval", "quick", tmp_path, "gemma4:12b")
    assert [case.case_id for case in found] == ["ifeval:2"]
    case = found[0]
    assert case.category == "case"
    assert case.score == 0.5
    assert case.signals["hit_output_cap"] is True
    assert case.messages == [{"role": "user", "content": "B"}]
    assert case.replayable


def _bfcl_entry(case_id: str, decoded: list, error_type: str) -> dict:
    return {
        "id": case_id,
        "test_category": "simple_python",
        "valid": False,
        "error": ["Function name 'math.factorial' not found in model output."],
        "error_type": error_type,
        "prompt": {
            "question": [[{"role": "user", "content": "factorial de 5"}]],
            "function": [{"name": "math.factorial", "description": "f", "parameters": {
                "type": "dict", "properties": {"number": {"type": "integer"}}}}],
        },
        "model_result_raw": [],
        "model_result_decoded": decoded,
        "possible_answer": [{"math.factorial": {"number": [5]}}],
    }  # fmt: skip


def test_bfcl_detecta_el_artefacto_punto_guion_bajo(tmp_path: Path) -> None:
    scores = (
        tmp_path / "source" / "berkeley-function-call-leaderboard"
        / "ocabra_bfcl_scores_gemma4_12b" / "gemma4:12b" / "non_live"
    )  # fmt: skip
    scores.mkdir(parents=True)
    rows = [
        {"accuracy": 0.5},
        _bfcl_entry("simple_python_1", [{"math_factorial": {"number": 5}}], "simple_function_checker:wrong_func_name"),
        _bfcl_entry("simple_python_2", [{"other": {}}], "simple_function_checker:wrong_func_name"),
    ]  # fmt: skip
    (scores / "BFCL_v4_simple_python_score.json").write_text(
        "\n".join(json.dumps(row) for row in rows)
    )
    found = {
        case.case_id: case
        for case in case_module.failed_cases("bfcl", "core", tmp_path, "gemma4:12b")
    }
    assert found["bfcl:simple_python_1"].signals["harness_dot_to_underscore"] is True
    assert found["bfcl:simple_python_2"].signals["harness_dot_to_underscore"] is False
    tool = found["bfcl:simple_python_1"].tools[0]["function"]
    assert tool["name"] == "math_factorial"
    assert tool["parameters"]["type"] == "object"


def test_corrala_y_fulcrum_se_leen_sin_mensajes(tmp_path: Path) -> None:
    _jsonl(
        tmp_path / "source/backend/scripts/llm_benchmark/results_2026-11-23/runs.jsonl",
        [
            {"model": "m", "suite": "zammad", "case": "c1", "run": 0, "passed": False,
             "checks": {"queue": False, "tags": True}, "chain_length_finishes": 1},
            {"model": "m", "suite": "zammad", "case": "c2", "run": 0, "passed": True},
        ],
    )  # fmt: skip
    corrala = case_module.failed_cases("corrala", "llm", tmp_path, "m")
    assert [case.case_id for case in corrala] == ["corrala:zammad:c1:0"]
    assert corrala[0].signals["length_finishes"] == 1 and not corrala[0].replayable
    _jsonl(
        tmp_path / "source/benchmarks/orchestration/results_2026-11-26/runs.jsonl",
        [
            {
                "model": "m",
                "rep": 0,
                "caps": {"cola": False, "despacho": True},
                "detail": {"cutoffs": ["loop"]},
            }
        ],
    )
    fulcrum = case_module.failed_cases("fulcrum", "orquestacion", tmp_path, "m")
    assert [case.category for case in fulcrum] == ["cola"]
    assert fulcrum[0].signals["tool_loop"] is True


# ── Referencia y parámetros ───────────────────────────────────────────────


def test_el_muestreo_recomendado_neutraliza_lo_que_la_ficha_no_fija() -> None:
    sampling = recommended_sampling({"temperature": 1.0, "top_k": 64, "top_p": 0.95})
    assert sampling["top_k"] == 64 and sampling["top_p"] == 0.95
    assert sampling["min_p"] == 0.0 and sampling["repeat_penalty"] == 1.0
    assert sampling["dry_multiplier"] == 0.0


def test_el_repo_de_referencia_se_resuelve_por_orden() -> None:
    assert (
        resolve_reference_repo("gemma4:12b", {"source": "ollama"}, {}, None)
        == "google/gemma-4-12B-it"
    )
    assert resolve_reference_repo("x", {"source": "hf", "repo": "org/m"}, {}, None) == "org/m"
    assert resolve_reference_repo("x", {}, {"diagnosis": {"reference_repo": "a/b"}}, None) == "a/b"
    assert resolve_reference_repo("gemma4:12b", {}, {}, "c/d") == "c/d"
    assert resolve_reference_repo("desconocido", {}, {}, None) is None


def test_lo_efectivo_son_los_defaults_del_servidor_con_el_cuerpo_encima() -> None:
    effective = effective_params(
        {"temperature": 0.8, "top_k": 20, "min_p": 0.05, "max_tokens": -1},
        {"temperature": 0, "max_tokens": 4096, "messages": []},
    )
    assert effective == {"temperature": 0, "top_k": 20, "min_p": 0.05, "max_tokens": 4096}
    diffs = sampling_differences(effective, {"temperature": 1.0, "top_k": 64, "min_p": 0.0})
    assert set(diffs) == {"temperature", "top_k", "min_p"}


def test_la_comparacion_de_prompts_ignora_el_bos_y_separa_espacios() -> None:
    assert (
        compare_prompts("<bos>hola", "hola", live_bos="<bos>", official_bos="<bos>")["status"]
        == "identical"
    )
    assert (
        compare_prompts("a  b", "a b", live_bos=None, official_bos=None)["status"] == "whitespace"
    )
    different = compare_prompts("<u>hola", "<user>hola", live_bos=None, official_bos=None)
    assert different["status"] == "different" and different["diff"]


# ── Clasificación ─────────────────────────────────────────────────────────


def _classify(case=None, **kwargs):
    base = {"app": "ifeval", "signals": {}}
    defaults = {
        "template": None,
        "sampling_diffs": {},
        "replays": {},
        "context_per_slot": 0,
        "prompt_tokens": None,
    }
    return classify_case({**base, **(case or {})}, **{**defaults, **kwargs})[0]


def test_reglas_de_clasificacion() -> None:
    passing = [{"passed": True, "finish_reason": "stop"}] * 3
    failing = [{"passed": False, "finish_reason": "stop"}] * 3
    assert _classify({"signals": {"harness_dot_to_underscore": True}}) == "servido"
    assert _classify({"signals": {"hit_output_cap": True}}) == "limites"
    assert _classify(prompt_tokens=9000, context_per_slot=8192) == "limites"
    assert (
        _classify(replays={"original": [{"passed": False, "finish_reason": "length"}]}) == "limites"
    )
    assert _classify(template={"status": "different"}) == "plantilla"
    assert (
        _classify(replays={"recomendado": passing}, sampling_diffs={"temperature": {}})
        == "sampling"
    )
    assert _classify(replays={"recomendado": passing}) == "no_reproduce"
    assert _classify(replays={"recomendado": failing, "original": [{"passed": False}]}) == "modelo"
    assert _classify(replays={"recomendado": failing, "comparado": passing}) == "servido"
    assert _classify({"app": "corrala"}) == "sin_verificar"


def test_el_resumen_da_porcentajes_sobre_fallos_y_sobre_atribuidos() -> None:
    summary = summarize(
        [
            {"category": "modelo"},
            {"category": "modelo"},
            {"category": "sampling"},
            {"category": "sin_verificar"},
        ]
    )
    assert summary["total"] == 4
    assert summary["percent_of_failures"]["modelo"] == 50.0
    assert summary["percent_of_attributed"]["modelo"] == pytest.approx(66.7)


# ── Informe y candidatos ──────────────────────────────────────────────────


def test_el_split_es_estable_y_solo_exporta_casos_modelo(tmp_path: Path) -> None:
    assert split_for("bfcl:simple_python_1") == split_for("bfcl:simple_python_1")
    ids = [f"ifeval:{index}" for index in range(500)]
    share = sum(split_for(item) == "eval" for item in ids) / len(ids)
    assert 0.12 < share < 0.28
    cases = [
        {
            "case_id": item,
            "app": "ifeval",
            "suite": "quick",
            "category": "c",
            "messages": [],
            "failure": "f",
        }
        for item in ids[:20]
    ]
    verdicts = {
        item: {"category": "modelo" if index % 2 else "sampling"}
        for index, item in enumerate(ids[:20])
    }
    counts = write_candidates(tmp_path, "m", cases, verdicts, {}, {"run_id": "r"})
    assert counts["train"] + counts["eval"] == 10
    rows = [
        json.loads(line)
        for name in ("candidates.train.jsonl", "candidates.eval.jsonl")
        for line in (tmp_path / name).read_text().splitlines()
    ]
    assert {row["split"] for row in rows} <= {"train", "eval"}
    assert all(split_for(row["id"]) == row["split"] for row in rows)


def test_el_markdown_lleva_resumen_y_casos() -> None:
    report = {
        "model": "gemma4:12b", "app": "ifeval", "suite": "quick", "run_id": "r", "attempt_id": "a",
        "repeats": 3, "replayed_cases": 1,
        "summary": summarize([{"category": "modelo"}]),
        "cases": [{"case_id": "ifeval:1", "category": "modelo", "failure": "x|y", "evidence": ["e"]}],
        "candidates": {"train": 1, "eval": 0},
        "params": {"differences": {"top_k": {"efectivo": 20, "recomendado": 64}}},
        "findings": ["hallazgo"],
    }  # fmt: skip
    text = render_markdown(report)
    assert "| Modelo | 1 | 100.0 % |" in text
    assert "x\\|y" in text and "`top_k`" in text and "hallazgo" in text


# ── Trabajos en disco ─────────────────────────────────────────────────────


def test_el_almacen_encola_en_orden_y_recupera_los_interrumpidos(tmp_path: Path) -> None:
    store = DiagnosisStore(tmp_path)
    first = store.create({"attempt_id": "a"})
    second = store.create({"attempt_id": "b"})
    assert store.next_queued()["id"] == first["id"]
    first["status"] = "running"
    store.save(first)
    store.recover_interrupted()
    assert store.load(first["id"])["status"] == "failed"
    assert store.next_queued()["id"] == second["id"]
    with pytest.raises(ValueError):
        store.path("../../etc")


def test_la_muestra_es_estratificada_y_salta_el_arnes() -> None:
    runner = DiagnosisRunner.__new__(DiagnosisRunner)
    cases = [{"case_id": f"a{i}", "category": "a", "signals": {}} for i in range(10)]
    cases += [{"case_id": f"b{i}", "category": "b", "signals": {}} for i in range(10)]
    cases += [{"case_id": "h", "category": "a", "signals": {"harness_dot_to_underscore": True}}]
    sample = runner._sample(cases, 6)
    assert len(sample) == 6 and "h" not in sample
    assert sum(item.startswith("a") for item in sample) == 3


# ── Scripts de repetición ─────────────────────────────────────────────────


def test_el_bucle_de_repeticion_anota_cada_intento_y_los_errores(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("_replay_common", SCRIPTS / "_replay_common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    spec_path = tmp_path / "in.json"
    spec_path.write_text(
        json.dumps(
            {
                "base_url": "http://x",
                "cases": [{"case_id": "c1", "payload": {}}, {"case_id": "c2", "payload": {}}],
                "settings": [{"name": "original", "model": "m", "params": {}, "repeats": 2}],
            }
        )
    )
    out = tmp_path / "out.jsonl"

    def score(base_url, case, setting):
        if case["case_id"] == "c2":
            raise RuntimeError("HTTP 500")
        return {"passed": True}

    common.run(str(spec_path), str(out), score)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 4
    assert all(row["passed"] for row in rows if row["case_id"] == "c1")
    assert all(
        row["passed"] is None and "HTTP 500" in row["error"]
        for row in rows
        if row["case_id"] == "c2"
    )


def test_la_plantilla_oficial_se_renderiza_como_transformers() -> None:
    pytest.importorskip("jinja2")
    import subprocess
    import sys

    template = (
        "{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>\n"
        "{% endfor %}{% if tools %}{{ tools | tojson }}{% endif %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    payload = {
        "template": template,
        "messages": [{"role": "user", "content": "año"}],
        "tools": [{"name": "f"}],
        "bos_token": "<bos>",
        "eos_token": "",
        "kwargs": {},
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "render_template.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    prompt = json.loads(result.stdout)["prompt"]
    assert prompt == '<bos><user>año</user>\n[{"name": "f"}]<assistant>'
