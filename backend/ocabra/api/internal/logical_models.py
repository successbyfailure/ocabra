"""Logical models and their per-backend deployments (model/backends unification, Fase 3).

``/ocabra/models`` keeps listing *deployments* (one row per backend, the shape
the dashboard and external consumers already use). This router exposes the
model-centric view: one entry per public model with its deployments, the
backend that serves its public name, and which other backends could serve the
same file.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from ocabra import database
from ocabra.api._deps_auth import UserContext, require_role
from ocabra.core import model_deployments as md
from ocabra.core.compatibility import backends_for_artifact
from ocabra.db.model_config import LogicalModel, ModelConfig

router = APIRouter(tags=["logical-models"])


class LogicalModelPatch(BaseModel):
    """Editable fields of a logical model."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    default_backend: str | None = Field(default=None, alias="defaultBackend")
    display_name: str | None = Field(default=None, alias="displayName")


class DeploymentCreate(BaseModel):
    """Add one more backend to a model, serving the same file."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    backend: str
    source_backend: str | None = Field(default=None, alias="sourceBackend")
    extra_config: dict | None = Field(default=None, alias="extraConfig")


def _services(request: Request) -> tuple[Any, Any]:
    mm = getattr(request.app.state, "model_manager", None)
    pr = getattr(request.app.state, "profile_registry", None)
    if mm is None or pr is None:
        raise HTTPException(status_code=503, detail="Model manager not ready", headers={"Retry-After": "5"})
    return mm, pr


def _artifact_summary(artifact: dict | None) -> dict | None:
    if not isinstance(artifact, dict):
        return None
    out = {k: artifact.get(k) for k in ("kind", "source", "ref", "repo", "file") if artifact.get(k)}
    for label in ("mmproj", "draft"):
        if isinstance(artifact.get(label), dict):
            out[label] = True
    return out


async def _serialize(request: Request, model: LogicalModel, configs: list[ModelConfig]) -> dict[str, Any]:
    mm, pr = _services(request)
    installed = md.installed_backends(request.app.state)
    profiles = [p for p in await pr.list_all() if p.base_model_id in {c.model_id for c in configs}]
    deployments = []
    default_cfg = None
    for cfg in sorted(configs, key=lambda c: (c.backend_type != model.default_backend, c.backend_type)):
        state = await mm.get_state(cfg.model_id)
        is_default = cfg.backend_type == model.default_backend
        if is_default:
            default_cfg = cfg
        deployments.append(
            {
                "modelId": cfg.model_id,
                "backend": cfg.backend_type,
                "isDefault": is_default,
                "installed": True if installed is None else cfg.backend_type in installed,
                "status": getattr(getattr(state, "status", None), "value", None),
                "loadPolicy": cfg.load_policy,
                "pinPermanent": bool(cfg.pin_permanent),
                "preferredGpu": cfg.preferred_gpu,
                "artifact": _artifact_summary(cfg.artifact),
                "profiles": sorted(p.profile_id for p in profiles if p.base_model_id == cfg.model_id),
            }
        )
    reference = default_cfg or (configs[0] if configs else None)
    present = {c.backend_type for c in configs}
    compatible = [
        {**entry, "deployed": entry["backend"] in present}
        for entry in backends_for_artifact(reference.artifact if reference else None, installed)
    ]
    public = sorted(p.profile_id for p in profiles if getattr(p, "backend", None) is None)
    category = next(
        (p.category for p in profiles if p.profile_id == model.model_key),
        next((p.category for p in profiles), None),
    )
    return {
        "modelKey": model.model_key,
        "displayName": model.display_name,
        "category": category,
        "defaultBackend": model.default_backend,
        "deployments": deployments,
        "compatibleBackends": compatible,
        "publicProfiles": public,
        "hiddenProfiles": sorted(p.profile_id for p in profiles if getattr(p, "backend", None)),
        "registryMetadata": model.registry_metadata,
    }


async def _load(model_key: str) -> tuple[LogicalModel, list[ModelConfig]]:
    async with database.AsyncSessionLocal() as session:
        model = await session.get(LogicalModel, model_key)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
        configs = list(
            (await session.execute(select(ModelConfig).where(ModelConfig.model_key == model_key))).scalars()
        )
    return model, configs


@router.get(
    "/logical-models",
    summary="List logical models",
    description="One entry per public model with its deployments (one per backend), the default backend and compatible backends.",
)
async def list_logical_models(
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict[str, Any]]:
    """List every logical model.

    Returns:
        A list of ``{modelKey, displayName, category, defaultBackend,
        deployments[], compatibleBackends[], publicProfiles[], hiddenProfiles[]}``
        sorted by ``modelKey``. ``deployments[].installed`` is false when the
        backend module is not installed on this node; ``compatibleBackends``
        lists the backends that can serve the default deployment's file, with
        ``deployed`` marking the ones already configured.
    """
    async with database.AsyncSessionLocal() as session:
        models = list((await session.execute(select(LogicalModel).order_by(LogicalModel.model_key))).scalars())
        configs = list((await session.execute(select(ModelConfig))).scalars())
    by_key: dict[str, list[ModelConfig]] = {}
    for cfg in configs:
        by_key.setdefault(cfg.model_key, []).append(cfg)
    return [await _serialize(request, m, by_key.get(m.model_key, [])) for m in models]


@router.get(
    "/logical-models/{model_key:path}/compatibility",
    summary="Backends that can serve a model's file",
)
async def get_model_compatibility(
    model_key: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict[str, Any]]:
    """Return the backends able to serve the model's default deployment file.

    Args:
        model_key: Public model name (``gemma4:26b``).

    Returns:
        ``[{backend, status, installed, reason, deployed}]``, preferred first.
    """
    entry = await _serialize(request, *(await _load(model_key)))
    return entry["compatibleBackends"]


@router.post(
    "/logical-models/{model_key:path}/deployments",
    summary="Add a deployment (backend) to a model",
    status_code=201,
    responses={400: {"description": "Backend cannot serve this file"}, 404: {"description": "Model not found"}},
)
async def create_deployment(
    model_key: str,
    body: DeploymentCreate,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    """Serve the model's file from one more backend.

    Args:
        model_key: Public model name.
        body: ``backend`` to add; optional ``sourceBackend`` (deployment whose
            file is reused, default = the default backend) and ``extraConfig``.

    Returns:
        The updated logical model. The new deployment is not the default and
        is reachable as ``<backend>/<model_key>`` through a hidden variant.
    """
    mm, pr = _services(request)
    installed = md.installed_backends(request.app.state)
    if installed is not None and body.backend not in installed:
        raise HTTPException(status_code=400, detail=f"Backend '{body.backend}' is not installed on this node")
    try:
        await md.add_deployment(
            model_key,
            body.backend,
            model_manager=mm,
            profile_registry=pr,
            source_backend=body.source_backend,
            extra_config=body.extra_config,
        )
    except md.DeploymentError as exc:
        status = 404 if "not found" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return await _serialize(request, *(await _load(model_key)))


@router.delete(
    "/logical-models/{model_key:path}/deployments/{backend}",
    summary="Remove a deployment from a model",
    responses={404: {"description": "Model or deployment not found"}},
)
async def delete_deployment(
    model_key: str,
    backend: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    """Remove one backend from a model (configuration only; files are kept).

    Args:
        model_key: Public model name.
        backend: Deployment to remove.

    Returns:
        ``{modelKey, removed, newDefaultBackend}``. Removing the default moves
        the public name to the next deployment first; removing the last one
        removes the model.
    """
    mm, pr = _services(request)
    try:
        result = await md.remove_deployment(model_key, backend, model_manager=mm, profile_registry=pr)
    except md.DeploymentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "modelKey": result["model_key"],
        "removed": result["removed"],
        "newDefaultBackend": result["new_default_backend"],
    }


@router.get(
    "/logical-models/{model_key:path}",
    summary="Get a logical model",
    responses={404: {"description": "Model not found"}},
)
async def get_logical_model(
    model_key: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    """Return one logical model with its deployments.

    Args:
        model_key: Public model name (may contain ``:`` and ``/``).

    Returns:
        Same shape as each entry of ``GET /logical-models``.
    """
    return await _serialize(request, *(await _load(model_key)))


@router.patch(
    "/logical-models/{model_key:path}",
    summary="Update a logical model (default backend, display name)",
    responses={400: {"description": "No deployment on that backend"}, 404: {"description": "Model not found"}},
)
async def patch_logical_model(
    model_key: str,
    body: LogicalModelPatch,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    """Change which backend serves the model's public name, or its display name.

    Args:
        model_key: Public model name.
        body: ``defaultBackend`` (must have a deployment) and/or ``displayName``.

    Returns:
        The updated logical model. Switching the backend re-points the public
        profiles, moves the load policy to the new default and gives the old
        default a hidden variant; the old worker is left to idle eviction.
    """
    mm, pr = _services(request)
    if body.display_name is not None:
        async with database.AsyncSessionLocal() as session:
            model = await session.get(LogicalModel, model_key)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
            model.display_name = body.display_name
            await session.commit()
    if body.default_backend:
        try:
            await md.set_default_backend(model_key, body.default_backend, model_manager=mm, profile_registry=pr)
        except md.DeploymentError as exc:
            status = 404 if "not found" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
    return await _serialize(request, *(await _load(model_key)))
