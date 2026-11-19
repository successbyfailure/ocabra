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
    script: str
    result_glob: str
    models_flag: str
    prepare_script: str | None = None
    resume_args: tuple[str, ...] = ("--skip-done",)


@dataclass(frozen=True)
class AppSpec:
    name: str
    description: str
    repository: str
    default_revision: str
    dataset_root: str
    suites: dict[str, SuiteSpec]


APPS: dict[str, AppSpec] = {
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
