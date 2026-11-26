"""Ficha oficial del modelo en Hugging Face: plantilla y muestreo recomendado.

La mayoría de nuestros GGUF vienen de Ollama y no apuntan a ningún repo, así
que el repo de referencia se resuelve, por orden: el que pida el diagnóstico,
``extra_config.diagnosis.reference_repo``, el ``artifact.repo`` de un GGUF de
HF y, por último, una tabla de modelos conocidos.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from ocabra.config import settings

# Modelos conocidos sin repo en su artefacto (vienen de Ollama). Solo lo que
# se ha comprobado a mano contra la ficha del fabricante.
KNOWN_REFERENCE_REPOS: dict[str, str] = {
    "gemma4:12b": "google/gemma-4-12B-it",
    "gemma4:26b": "google/gemma-4-26B-A4B-it",
}

# Campos de generation_config.json → nombre que entiende llama-server.
_SAMPLING_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repeat_penalty",
}


@dataclass
class Reference:
    repo: str | None
    sampling: dict[str, Any] = field(default_factory=dict)
    chat_template: str | None = None
    bos_token: str | None = None
    eos_token: str | None = None
    suppress_tokens: list[int] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "sampling": self.sampling,
            "has_chat_template": bool(self.chat_template),
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "suppress_tokens": self.suppress_tokens,
            "error": self.error,
        }


def resolve_reference_repo(
    model_key: str | None,
    artifact: dict[str, Any] | None,
    extra_config: dict[str, Any] | None,
    override: str | None = None,
) -> str | None:
    if override:
        return override.strip()
    diagnosis = (extra_config or {}).get("diagnosis")
    if isinstance(diagnosis, dict) and diagnosis.get("reference_repo"):
        return str(diagnosis["reference_repo"])
    artifact = artifact or {}
    if artifact.get("source") == "hf":
        repo = artifact.get("repo") or str(artifact.get("ref") or "").split("::")[0]
        if "/" in str(repo):
            return str(repo)
    return KNOWN_REFERENCE_REPOS.get(model_key or "")


def recommended_sampling(generation_config: dict[str, Any] | None) -> dict[str, Any]:
    """Muestreo del fabricante, con las penalizaciones extra neutralizadas.

    Lo que la ficha no fija se pone al valor neutro (sin ``min_p``, sin
    repetición, sin DRY ni presence/frequency): si no, se cuelan los valores por
    defecto de llama-server, que no son los del fabricante.
    """
    config = generation_config or {}
    sampling: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "dry_multiplier": 0.0,
    }
    if config.get("do_sample") is False:
        sampling["temperature"] = 0.0
    for source, target in _SAMPLING_FIELDS.items():
        if config.get(source) is not None:
            sampling[target] = config[source]
    return sampling


def _download(repo: str, filename: str) -> Path | None:
    try:
        return Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                token=settings.hf_token or None,
                cache_dir=settings.hf_cache_dir or None,
            )
        )
    except Exception:  # noqa: BLE001 — un fichero que no existe no es un error
        return None


def _special(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("content")
    return str(value) if value else None


def load_reference_sync(repo: str | None) -> Reference:
    if not repo:
        return Reference(repo=None, error="sin repo de referencia en Hugging Face")
    generation_path = _download(repo, "generation_config.json")
    tokenizer_path = _download(repo, "tokenizer_config.json")
    template_path = _download(repo, "chat_template.jinja")
    if generation_path is None and tokenizer_path is None:
        return Reference(repo=repo, error=f"no se pudo leer la ficha de {repo}")
    generation = json.loads(generation_path.read_text()) if generation_path else {}
    tokenizer = json.loads(tokenizer_path.read_text()) if tokenizer_path else {}
    template = template_path.read_text(encoding="utf-8") if template_path else None
    if template is None:
        raw = tokenizer.get("chat_template")
        if isinstance(raw, list):  # varias plantillas con nombre: la «default»
            raw = next(
                (item.get("template") for item in raw if item.get("name") == "default"),
                None,
            )
        template = raw if isinstance(raw, str) else None
    suppress = generation.get("suppress_tokens") or []
    return Reference(
        repo=repo,
        sampling=recommended_sampling(generation),
        chat_template=template,
        bos_token=_special(tokenizer.get("bos_token")),
        eos_token=_special(tokenizer.get("eos_token")),
        suppress_tokens=[int(item) for item in suppress if isinstance(item, int)],
    )


async def load_reference(repo: str | None) -> Reference:
    return await asyncio.to_thread(load_reference_sync, repo)
