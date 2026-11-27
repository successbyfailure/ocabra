"""Qué modelo mide un benchmark: un perfil o un backend concreto de un modelo.

El selector de Benchmarks deja elegir, para cada modelo, cualquier backend
capaz de servir su fichero, esté desplegado aquí o no. Un backend concreto se
pide con el id del despliegue, ``<backend>/<modelo>`` (``ollama/gemma4:12b``):
es la forma pública de fijar el motor; los nombres derivados
(``gemma4:12b-ollama``) la resolución los rechaza a propósito, así que se
traducen al crear la ejecución.

Un despliegue que aún no existe se crea al medir en este servidor
(:func:`ensure_deployment`) y se replica con ese backend en un pod
(``core/cloud/preload.py``). En un nodo de la federación solo valen los que
él anuncia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from ocabra import database
from ocabra.core.model_ref import build_model_ref


@dataclass(frozen=True)
class PendingDeployment:
    """Un backend compatible de un modelo, sin desplegar todavía aquí."""

    model_id: str
    model_key: str
    backend: str
    source: Any  # ModelConfig del despliegue del que sale el fichero


def split_deployment_id(name: str) -> tuple[str, str] | None:
    """``(backend, model_key)`` si *name* tiene forma de id de despliegue."""
    from ocabra.core.model_ref import KNOWN_BACKEND_TYPES

    backend, sep, key = name.partition("/")
    if not sep or not key or backend not in KNOWN_BACKEND_TYPES:
        return None
    return backend, key


async def bench_profile(registry: Any, model: str) -> Any | None:
    """El perfil con el que se medirá *model* (perfil o id de despliegue)."""
    profile = await registry.get(model)
    if profile is not None or "/" not in model:
        return profile
    profiles = await registry.list_by_model(model)
    return next((p for p in profiles if p.is_default and p.enabled), None) or next(
        (p for p in profiles if p.enabled), None
    )


async def normalize_models(registry: Any, models: list[str]) -> list[str]:
    """Traduce los nombres derivados (``gemma4:12b-ollama``) a ids de despliegue."""
    suffixed = getattr(registry, "engine_suffixed_name", None)
    out: list[str] = []
    for model in models:
        alternative = await suffixed(model) if suffixed is not None else None
        name = alternative[1] if alternative is not None else model
        if name not in out:
            out.append(name)
    return out


async def pending_deployment(model: str) -> PendingDeployment | None:
    """Si *model* es ``<backend>/<modelo>`` sin desplegar pero compatible, cuál.

    Raises:
        ValueError: es un id de despliegue de un modelo desconocido o de un
            backend que no puede leer su fichero.
    """
    from ocabra.core.compatibility import backends_for_artifact
    from ocabra.db.model_config import LogicalModel, ModelConfig

    parts = split_deployment_id(model)
    if parts is None:
        return None
    backend, key = parts
    async with database.AsyncSessionLocal() as session:
        exists = await session.scalar(
            select(ModelConfig.id).where(ModelConfig.model_id == model).limit(1)
        )
        if exists is not None:
            return None
        logical = await session.get(LogicalModel, key)
        if logical is None:
            return None
        rows = list(
            (
                await session.execute(select(ModelConfig).where(ModelConfig.model_key == key))
            ).scalars()
        )
    source = next((r for r in rows if r.backend_type == logical.default_backend), None) or (
        rows[0] if rows else None
    )
    if source is None or not isinstance(source.artifact, dict):
        raise ValueError(f"{key} no tiene un despliegue con fichero conocido")
    allowed = {entry["backend"] for entry in backends_for_artifact(source.artifact)}
    if backend not in allowed:
        raise ValueError(
            f"{backend} no puede servir {key} (compatibles: {', '.join(sorted(allowed)) or '-'})"
        )
    return PendingDeployment(build_model_ref(backend, key), key, backend, source)


async def validate_models(models: list[str]) -> None:
    """Falla (``ValueError``) si algún backend pedido no puede servir su modelo."""
    for model in models:
        await pending_deployment(model)


async def ensure_deployment(app_state: Any, model: str) -> bool:
    """Crea el despliegue ``<backend>/<modelo>`` si hace falta. ``True`` si lo creó."""
    from ocabra.core import model_deployments as md

    pending = await pending_deployment(model)
    if pending is None:
        return False
    await md.add_deployment(
        pending.model_key,
        pending.backend,
        model_manager=app_state.model_manager,
        profile_registry=app_state.profile_registry,
        source_backend=pending.source.backend_type,
    )
    return True
