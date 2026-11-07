"""Operations on a logical model's deployments (model/backends unification, Fase 2).

A logical model (``models`` row, key = public name) has one deployment per
backend (``model_configs`` row, id ``<backend>/<model_key>``). Its public
profiles — ``backend IS NULL`` — point at the deployment of the model's
``default_backend``; variants pinned to another backend (``backend`` set) are
hidden from listings but stay resolvable. Switching the default backend is
therefore a re-point of the public profiles plus moving the load policy, done
here in one place so the invariants hold:

- every public profile of the model points at the default deployment;
- every deployment is reachable through at least one profile;
- only the default deployment carries the model's load policy (pin/warm);
- the validated capacity of each preset (``runtime_policy.validated_*``)
  always describes the backend the profile currently points at — a value
  measured on llama.cpp must never admit a request on Ollama.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from ocabra import database
from ocabra.db.model_config import LogicalModel, ModelConfig, ModelProfile

if TYPE_CHECKING:
    from ocabra.core.model_manager import ModelManager
    from ocabra.core.profile_registry import ProfileRegistry

logger = structlog.get_logger(__name__)

# runtime_policy keys whose values are only true for the backend they were
# measured on (keyed by preset).
VALIDATED_POLICY_KEYS = ("validated_vram_mb", "validated_context_tokens")
VALIDATED_BY_BACKEND_KEY = "validated_by_backend"


class DeploymentError(ValueError):
    """The requested change is not possible (unknown model/backend, ...)."""


def swap_validated_capacity(policy: dict | None, old_backend: str, new_backend: str) -> dict:
    """Park the old backend's validated capacity and restore the new one's.

    Readers (runtime coordinator, capacity checks) keep reading the flat
    ``validated_*`` maps, which always describe the current backend; values for
    other backends wait under ``validated_by_backend`` until switched back.
    """
    out = dict(policy or {})
    parked: dict[str, Any] = dict(out.get(VALIDATED_BY_BACKEND_KEY) or {})
    parked[old_backend] = {key: out.get(key) or {} for key in VALIDATED_POLICY_KEYS}
    restored = parked.pop(new_backend, None) or {}
    for key in VALIDATED_POLICY_KEYS:
        out[key] = dict(restored.get(key) or {})
    # Drop empty parked entries so the policy does not grow with noise.
    parked = {b: v for b, v in parked.items() if any(v.get(k) for k in VALIDATED_POLICY_KEYS)}
    if parked:
        out[VALIDATED_BY_BACKEND_KEY] = parked
    else:
        out.pop(VALIDATED_BY_BACKEND_KEY, None)
    return out


async def set_default_backend(
    model_key: str,
    backend: str,
    *,
    model_manager: ModelManager,
    profile_registry: ProfileRegistry,
) -> dict[str, Any]:
    """Make ``backend`` the deployment that serves ``model_key``'s public name.

    Returns a summary ``{model_key, old_backend, new_backend, moved_profiles,
    pinned_variant}``. A no-op (same backend) returns ``moved_profiles=[]``.
    Raises :class:`DeploymentError` for an unknown model or a backend the
    model has no deployment on.
    """
    async with database.AsyncSessionLocal() as session:
        model = await session.get(LogicalModel, model_key)
        if model is None:
            raise DeploymentError(f"Model '{model_key}' not found.")
        deployments = {
            cfg.backend_type: cfg
            for cfg in (
                await session.execute(select(ModelConfig).where(ModelConfig.model_key == model_key))
            ).scalars()
        }
        if backend not in deployments:
            raise DeploymentError(
                f"Model '{model_key}' has no deployment on backend '{backend}'. "
                f"Available: {', '.join(sorted(deployments)) or 'none'}."
            )
        old_backend = model.default_backend
        summary: dict[str, Any] = {
            "model_key": model_key,
            "old_backend": old_backend,
            "new_backend": backend,
            "moved_profiles": [],
            "pinned_variant": None,
        }
        if old_backend == backend:
            return summary

        old_cfg = deployments.get(old_backend)
        new_cfg = deployments[backend]
        old_id = old_cfg.model_id if old_cfg is not None else None

        public = []
        if old_id is not None:
            public = list(
                (
                    await session.execute(
                        select(ModelProfile).where(
                            ModelProfile.base_model_id == old_id,
                            ModelProfile.backend.is_(None),
                        )
                    )
                ).scalars()
            )
        for profile in public:
            profile.base_model_id = new_cfg.model_id
            profile.runtime_policy = swap_validated_capacity(
                profile.runtime_policy, old_backend, backend
            )
            summary["moved_profiles"].append(profile.profile_id)
        # A variant pinned to the new default is now redundant with the public
        # profile but harmless (hidden, own presets): keep it, clients may use it.

        policy = old_cfg.load_policy if old_cfg is not None else new_cfg.load_policy
        pin = bool(old_cfg.pin_permanent) if old_cfg is not None else bool(new_cfg.pin_permanent)
        model.default_backend = backend
        await session.commit()

    # Load policy follows the default deployment (in-memory state + DB).
    if old_cfg is not None and old_id is not None:
        await model_manager.update_config(old_id, {"load_policy": "on_demand", "pin_permanent": False})
    await model_manager.update_config(new_cfg.model_id, {"load_policy": policy, "pin_permanent": pin})

    model_manager.note_default_backend(model_key, backend)
    async with database.AsyncSessionLocal() as session:
        for profile_id in summary["moved_profiles"]:
            await profile_registry._reload_profile(session, profile_id)
        # The old default may have lost its only profile: give it a hidden one.
        if old_id is not None:
            summary["pinned_variant"] = await profile_registry.on_model_added(old_id)

    logger.info("model_default_backend_changed", **summary)
    return summary


def installed_backends(app_state: Any) -> set[str] | None:
    """Backends installed on this node, or None when the installer is unavailable."""
    installer = getattr(app_state, "backend_installer", None)
    if installer is None:
        return None
    try:
        from ocabra.core.backend_installer import BackendInstallStatus

        return {
            s.backend_type
            for s in installer.list_states()
            if s.install_status == BackendInstallStatus.INSTALLED
        }
    except Exception:  # noqa: BLE001 — compatibility info is advisory
        return None


async def add_deployment(
    model_key: str,
    backend: str,
    *,
    model_manager: ModelManager,
    profile_registry: ProfileRegistry,
    source_backend: str | None = None,
    extra_config: dict | None = None,
) -> dict[str, Any]:
    """Serve an existing model's file from one more backend.

    The artifact is derived from an existing deployment (``source_backend`` or
    the default one) by :func:`ocabra.core.compatibility.derive_artifact`, so
    only backends that can read that very file are accepted. The new
    deployment is not the default; it gets a hidden pinned variant so it is
    reachable as ``<backend>/<model_key>`` straight away.
    """
    from ocabra.core.compatibility import derive_artifact

    async with database.AsyncSessionLocal() as session:
        model = await session.get(LogicalModel, model_key)
        if model is None:
            raise DeploymentError(f"Model '{model_key}' not found.")
        deployments = {
            cfg.backend_type: cfg
            for cfg in (
                await session.execute(select(ModelConfig).where(ModelConfig.model_key == model_key))
            ).scalars()
        }
    if backend in deployments:
        raise DeploymentError(f"Model '{model_key}' already has a '{backend}' deployment.")
    source = deployments.get(source_backend or model.default_backend)
    if source is None:
        source = next(iter(deployments.values()), None)
    if source is None or not isinstance(source.artifact, dict):
        raise DeploymentError(f"Model '{model_key}' has no deployment with a known artifact.")
    try:
        artifact = derive_artifact(source.artifact, backend)
    except ValueError as exc:
        raise DeploymentError(str(exc)) from exc

    model_id = f"{backend}/{model_key}"
    await model_manager.add_model(
        model_id=model_id,
        backend_type=backend,
        display_name=model.display_name or model_key,
        load_policy="on_demand",
        extra_config=extra_config or None,
        model_key=model_key,
        artifact=artifact,
    )
    pinned = await profile_registry.on_model_added(model_id)
    logger.info("model_deployment_added", model_key=model_key, backend=backend, pinned_variant=pinned)
    return {"model_key": model_key, "model_id": model_id, "backend": backend, "pinned_variant": pinned}


async def remove_deployment(
    model_key: str,
    backend: str,
    *,
    model_manager: ModelManager,
    profile_registry: ProfileRegistry,
) -> dict[str, Any]:
    """Drop one deployment of a model (configuration only, never files).

    Removing the default deployment of a model that has others first moves
    the public name to the next one — otherwise the public profiles (and their
    presets) would go with it, since profiles cascade with their deployment.
    Removing the last deployment removes the model.
    """
    from ocabra.core.model_unification import BACKEND_PRIORITY, DEFAULT_PRIORITY

    async with database.AsyncSessionLocal() as session:
        model = await session.get(LogicalModel, model_key)
        if model is None:
            raise DeploymentError(f"Model '{model_key}' not found.")
        backends = list(
            (
                await session.execute(
                    select(ModelConfig.backend_type).where(ModelConfig.model_key == model_key)
                )
            ).scalars()
        )
        default_backend = model.default_backend
    if backend not in backends:
        raise DeploymentError(f"Model '{model_key}' has no '{backend}' deployment.")
    switched_to = None
    others = [b for b in backends if b != backend]
    if backend == default_backend and others:
        switched_to = min(others, key=lambda b: (BACKEND_PRIORITY.get(b, DEFAULT_PRIORITY), b))
        await set_default_backend(
            model_key, switched_to, model_manager=model_manager, profile_registry=profile_registry
        )
    model_id = f"{backend}/{model_key}"
    await model_manager.delete_model(model_id)
    profile_registry.forget_model_profiles(model_id)
    logger.info("model_deployment_removed", model_key=model_key, backend=backend, new_default=switched_to)
    return {"model_key": model_key, "removed": model_id, "new_default_backend": switched_to}
