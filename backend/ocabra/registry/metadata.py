"""Helpers to fetch registry-side metadata (release / last-modified dates) for
already-installed models.

The result is stored under ``model_configs.extra_config["registry_metadata"]``
so we don't need an Alembic migration. Returned shape::

    {
        "release_date": "2024-09-21T15:00:00+00:00" | None,
        "last_updated": "2025-04-02T12:30:00+00:00" | None,
        "fetched_at":   "2026-05-09T20:11:00+00:00",
        "source":       "huggingface" | "ollama",
    }
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from huggingface_hub import model_info

from ocabra.config import settings
from ocabra.registry.ollama_registry import OllamaRegistry

logger = logging.getLogger(__name__)


def _to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, datetime):
        return value.isoformat()
    return None


async def fetch_huggingface_metadata(repo_id: str) -> dict | None:
    """Return ``{release_date, last_updated, fetched_at, source}`` or ``None``."""
    try:
        info = await asyncio.to_thread(
            lambda: model_info(repo_id=repo_id, token=settings.hf_token or None),
        )
    except Exception as exc:
        logger.info("hf_metadata_unavailable repo=%s err=%s", repo_id, exc)
        return None

    return {
        "release_date": _to_iso(getattr(info, "created_at", None)),
        "last_updated": _to_iso(getattr(info, "last_modified", None)),
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "huggingface",
    }


async def fetch_ollama_metadata(model_ref: str, registry: OllamaRegistry | None = None) -> dict | None:
    """Pull the per-tag ``modified_at`` from the local Ollama daemon.

    Ollama doesn't expose an "original release" date, so ``release_date`` is
    always ``None`` here. ``last_updated`` is ``modified_at`` from
    ``GET /api/tags`` which equals "when did we pull this version". For models
    that are re-pulled regularly that approximates upstream's last update.
    """
    reg = registry or OllamaRegistry()
    try:
        details = await reg.list_installed_details()
    except Exception as exc:
        logger.info("ollama_metadata_unavailable model=%s err=%s", model_ref, exc)
        return None

    needle = model_ref.strip().lower()
    last_updated: str | None = None
    for item in details:
        name = str(item.get("name") or "").strip().lower()
        if name == needle or name.startswith(f"{needle}:") or needle.startswith(f"{name}:"):
            last_updated = _to_iso(item.get("modified_at"))
            if last_updated:
                break

    return {
        "release_date": None,
        "last_updated": last_updated,
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "ollama",
    }


async def fetch_registry_metadata(
    backend_type: str,
    backend_model_id: str,
) -> dict | None:
    """Dispatch to the appropriate registry. Returns ``None`` for local-only.

    ``backend_model_id`` is the id without the ``backend/`` prefix; for HF
    artifacts of form ``repo_id::stem`` only the repo_id portion is queried.
    """
    backend_type = (backend_type or "").strip().lower()
    raw = (backend_model_id or "").strip()
    if not raw:
        return None

    if backend_type == "ollama":
        return await fetch_ollama_metadata(raw)

    repo_id = raw.split("::", 1)[0]
    if backend_type in {"vllm", "sglang", "diffusers", "whisper", "tts", "chatterbox", "voxtral", "transformers", "bitnet", "llama_cpp"}:
        if "/" not in repo_id:
            return None
        return await fetch_huggingface_metadata(repo_id)

    return None
