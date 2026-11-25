"""Trusted benchmark runners exposed by the administration UI.

The scoring code and datasets remain in each application's repository.  This
catalog contains only the checkout and invocation contract required by oCabra.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    description: str
    script: str | None
    result_glob: str
    models_flag: str
    prepare_script: str | None = None
    resume_args: tuple[str, ...] = ("--skip-done",)
    local_runner: str | None = None
    default_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppSpec:
    name: str
    description: str
    repository: str
    default_revision: str
    dataset_root: str
    suites: dict[str, SuiteSpec]
    checkout_paths: tuple[str, ...] = ()


APPS: dict[str, AppSpec] = {
    "bfcl": AppSpec(
        name="bfcl",
        description="Berkeley Function Calling Leaderboard: selección y argumentos de herramientas.",
        repository="https://github.com/ShishirPatil/gorilla.git",
        default_revision="f7cf7359b7ac615a0b294831c5ba2bc95ee4a000",
        dataset_root="berkeley-function-call-leaderboard",
        checkout_paths=("berkeley-function-call-leaderboard",),
        suites={
            "simple_python": SuiteSpec(
                name="simple_python",
                description="BFCL v4: llamadas simples a funciones Python mediante tools OpenAI.",
                script=None,
                local_runner="bfcl.py",
                default_args=("--categories", "simple_python"),
                result_glob="ocabra_results_*/results.json",
                models_flag="--models",
            ),
            "core": SuiteSpec(
                name="core",
                description="BFCL v4 no-live: simple, múltiple, paralela e irrelevancia.",
                script=None,
                local_runner="bfcl.py",
                default_args=("--categories", "non_live"),
                result_glob="ocabra_results_*/results.json",
                models_flag="--models",
            ),
        },
    ),
    "corrala": AppSpec(
        name="corrala",
        description=(
            "Casos reales anonimizados de Corrala: triage Zammad, routing de GLaDOS, "
            "subagentes y detección de bucles."
        ),
        repository="https://github.com/successbyfailure/Corrala.git",
        default_revision="main",
        dataset_root="backend/scripts/llm_benchmark",
        suites={
            "llm": SuiteSpec(
                name="llm",
                description="Calidad y velocidad por el stack agéntico real con tools en replay.",
                script="ocabra_suite.py",
                result_glob="results_*/results.json",
                models_flag="--models",
            )
        },
    ),
    "escriba": AppSpec(
        name="escriba",
        description="Casos reales de transcripción, reuniones y búsqueda de Escriba.",
        repository="https://github.com/makespacemadrid/escriba.git",
        default_revision="main",
        dataset_root="eval/datasets",
        suites={
            "llm": SuiteSpec(
                name="llm",
                description="Extracción de claims y velocidad sobre transcripciones.",
                script="llm/benchmark_llm.py",
                result_glob="llm/results_*/results.json",
                models_flag="--models",
            ),
            "stt": SuiteSpec(
                name="stt",
                description="WER y velocidad sobre audio español de VoxPopuli.",
                script="stt/benchmark_stt.py",
                prepare_script="stt/download_dataset.py",
                result_glob="stt/results_*/results.json",
                models_flag="--models",
            ),
            "stt_diarization": SuiteSpec(
                name="stt_diarization",
                description="DER, JER y velocidad sobre reuniones AMI SDM.",
                script="stt_diarization/benchmark_diarization.py",
                prepare_script="stt_diarization/download_dataset.py",
                result_glob="stt_diarization/results_*/results.json",
                models_flag="--models",
            ),
            "embeddings": SuiteSpec(
                name="embeddings",
                description="Recuperación, clustering y reranking sobre reuniones.",
                script="embeddings/benchmark_embeddings.py",
                result_glob="embeddings/results_*/results.json",
                models_flag="--models",
            ),
        },
    ),
    "fulcrum": AppSpec(
        name="fulcrum",
        description=(
            "Orquestación de Fulcrum: llevar varias sesiones de agentes en paralelo "
            "durante muchos turnos sobre un mundo simulado, con el prompt y las tools reales."
        ),
        repository="https://github.com/successbyfailure/project-assistant.git",
        default_revision="main",
        dataset_root="benchmarks/orchestration",
        suites={
            "orquestacion": SuiteSpec(
                name="orquestacion",
                description=(
                    "Ronda por fases (5 runs, 3 fases, máx. 3 a la vez): despacho, textos fijos, "
                    "máximo, orden, cola, honestidad. Sin juez LLM; 5 repeticiones (~12 min cada una con un 27B MoE)."
                ),
                script="ocabra_suite.py",
                result_glob="results_*/results.json",
                models_flag="--models",
            )
        },
    ),
    "ifeval": AppSpec(
        name="ifeval",
        description="IFEval oficial de Google: cumplimiento verificable de instrucciones.",
        repository="https://github.com/google-research/google-research.git",
        default_revision="c878a53ec1764bcb91a6178dc5c315202a3b5fa2",
        dataset_root="instruction_following_eval",
        checkout_paths=("instruction_following_eval",),
        suites={
            "quick": SuiteSpec(
                name="quick",
                description="Primeros 50 prompts de IFEval para una comparación rápida.",
                script=None,
                local_runner="ifeval.py",
                default_args=("--limit", "50"),
                result_glob="ocabra_results_*/results.json",
                models_flag="--models",
            ),
            "full": SuiteSpec(
                name="full",
                description="Los 541 prompts del dataset oficial IFEval.",
                script=None,
                local_runner="ifeval.py",
                result_glob="ocabra_results_*/results.json",
                models_flag="--models",
            ),
        },
    ),
}


def catalog_payload() -> list[dict[str, object]]:
    """Return the stable JSON shape consumed by the dashboard."""
    return [
        {
            "app": app.name,
            "description": app.description,
            "defaultRevision": app.default_revision,
            "suites": [
                {
                    "name": suite.name,
                    "description": suite.description,
                    "resumable": bool(suite.resume_args),
                    "needsPreparation": suite.prepare_script is not None,
                }
                for suite in app.suites.values()
            ],
        }
        for app in APPS.values()
    ]
