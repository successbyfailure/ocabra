"""
GET /v1/models — list available models in OpenAI format.

With the profile system enabled, this endpoint exposes **profiles** rather
than raw model entries.  Each profile is presented as an OpenAI-compatible
"model" object whose ``id`` is the ``profile_id``.

When federation is enabled and peers are online, remote models from federated
nodes are merged into the listing with deduplication.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from ocabra.api._deps_auth import UserContext

from ._deps import (
    _openai_error,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    resolve_profile,
)

router = APIRouter()


async def _context_metadata(
    profile: Any, model_manager: Any, base_state: Any
) -> dict[str, Any] | None:
    """Resolve the public context contract against current hardware."""

    enabled = [
        preset
        for preset in getattr(profile, "runtime_presets", [])
        if preset.enabled
    ]
    if not enabled:
        return None

    from ocabra.core.runtime_preset_capacity import runtime_capacity_cache

    gpu_states = (
        await model_manager.gpu_manager.get_all_states()
        if model_manager.gpu_manager is not None
        else []
    )
    rows = await runtime_capacity_cache.resolve(profile, base_state, gpu_states)
    enabled_keys = {preset.preset_key for preset in enabled}
    usable = [
        row
        for row in rows
        if row.get("preset_key") in enabled_keys
        and row.get("resolved")
        and row.get("fits")
        and int(row.get("effective_context") or 0) > 0
    ]
    if not usable:
        return None

    default_key = next(
        (preset.preset_key for preset in enabled if preset.is_default), None
    )
    default_row = next(
        (row for row in usable if row["preset_key"] == default_key),
        min(usable, key=lambda row: int(row["effective_context"])),
    )
    active_context = None
    active_preset = None
    active_loaded_at = None
    for runtime_state in await model_manager.list_states(include_derived=True):
        if runtime_state.status.value != "loaded":
            continue
        extra = (
            runtime_state.extra_config
            if isinstance(runtime_state.extra_config, dict)
            else {}
        )
        runtime = extra.get("_runtime")
        if (
            not isinstance(runtime, dict)
            or runtime.get("profile_id") != profile.profile_id
        ):
            continue
        loaded_at = runtime_state.loaded_at
        if active_loaded_at is None or (
            loaded_at is not None and loaded_at > active_loaded_at
        ):
            active_loaded_at = loaded_at
            active_context = runtime.get("effective_context")
            active_preset = runtime.get("preset_key")

    return {
        "model_max": max(int(row.get("model_max") or 0) for row in usable) or None,
        "effective_max": max(int(row["effective_context"]) for row in usable),
        "default": int(default_row["effective_context"]),
        "default_preset": default_row["preset_key"],
        "active": int(active_context) if active_context else None,
        "active_preset": active_preset,
    }


def _apply_context_metadata(
    entry: dict[str, Any], metadata: dict[str, Any] | None
) -> None:
    if metadata is None:
        return
    effective_max = int(metadata["effective_max"])
    entry["context_length"] = effective_max
    entry["ocabra"]["context"] = metadata
    capabilities = dict(entry["ocabra"].get("capabilities") or {})
    capabilities["context_length"] = effective_max
    entry["ocabra"]["capabilities"] = capabilities


def _get_federation_manager(request: Request):
    """Return the FederationManager if enabled, or None."""
    return getattr(request.app.state, "federation_manager", None)


def _build_federation_nodes_metadata(
    peers: list,
) -> list[dict[str, Any]]:
    """Build a list of node metadata dicts from PeerState objects."""
    return [
        {
            "node_name": p.name,
            "node_id": p.peer_id,
        }
        for p in peers
    ]


def _ollama_cap_list(caps: dict) -> list[str]:
    """Convert an oCabra capabilities dict to an Ollama-style capability array.

    e.g. ``["completion", "vision", "tools", "thinking", "embedding"]``. Exposed
    at the top level of each /v1/models entry so clients that look for a
    ``capabilities`` list (rather than our nested ``ocabra.capabilities``) can
    still detect vision/tool support on models served through the OpenAI path.
    """
    caps = caps or {}
    out: list[str] = []
    if caps.get("completion") or caps.get("chat"):
        out.append("completion")
    if caps.get("embeddings"):
        out.append("embedding")
    if caps.get("vision"):
        out.append("vision")
    if caps.get("tools"):
        out.append("tools")
    if caps.get("reasoning"):
        out.append("thinking")
    return out


@router.get("/models", summary="List models")
async def list_models(
    request: Request,
    user: UserContext = Depends(get_openai_user),
) -> dict:
    """
    List all enabled profiles as OpenAI-compatible model objects.

    Filters profiles by the caller's group membership unless the caller is
    an admin.  Includes an ``ocabra`` extension field with category, status,
    capabilities, and display name.

    When federation is enabled, remote models from online peers are merged
    into the listing. Duplicate models (same model_id available both locally
    and remotely) appear once with metadata about all available nodes.
    """
    from ocabra.core.model_manager import ModelStatus, resolve_display_capabilities

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)
    enabled_profiles = await profile_registry.list_enabled()
    execution_profiles = [
        profile
        for profile in enabled_profiles
        if getattr(profile, "profile_kind", "execution") == "execution"
    ]
    router_profiles = [
        profile
        for profile in enabled_profiles
        if getattr(profile, "profile_kind", "execution") == "router"
    ]

    visible_statuses = {
        ModelStatus.LOADED,
        ModelStatus.CONFIGURED,
        ModelStatus.LOADING,
        ModelStatus.UNLOADED,
    }

    data = []
    now_ts = int(time.time())

    # Track local model_ids and profile_ids for deduplication with remote
    local_model_ids: set[str] = set()
    local_profile_ids: set[str] = set()
    # Map from model entry index to model_id for federation annotation
    entry_index_by_model_id: dict[str, int] = {}
    entries_by_profile_id: dict[str, dict[str, Any]] = {}

    for profile in execution_profiles:
        # Access control: non-admin users only see profiles in their set.
        # Check both profile_id and base_model_id since group_models stores
        # canonical model_ids while clients use profile_ids.
        if (
            not user.is_admin
            and profile.profile_id not in user.accessible_model_ids
            and profile.base_model_id not in user.accessible_model_ids
        ):
            continue

        base_state = await model_manager.get_state(profile.base_model_id)
        if base_state is None:
            continue
        if base_state.status not in visible_statuses:
            continue

        local_model_ids.add(profile.base_model_id)
        local_profile_ids.add(profile.profile_id)

        capabilities, capabilities_source = resolve_display_capabilities(base_state)
        entry: dict[str, Any] = {
            "id": profile.profile_id,
            "object": "model",
            "created": now_ts,
            "owned_by": "ocabra",
            # Top-level Ollama-style capabilities array (["vision","tools",...]).
            # OpenAI has no standard capability field, so clients (OpenWebUI)
            # can't read our nested ``ocabra.capabilities``. Mirroring the
            # Ollama array here at the top level gives parsers that look for a
            # ``capabilities`` list a chance to auto-detect vision/tools.
            "capabilities": _ollama_cap_list(capabilities),
            "ocabra": {
                "category": profile.category,
                "status": base_state.status.value,
                "capabilities": capabilities,
                "capabilities_source": capabilities_source,
                "display_name": profile.display_name or base_state.display_name,
                "base_model_id": profile.base_model_id,
                "load_policy": base_state.load_policy.value,
                "gpu": base_state.current_gpu,
                "vram_used_mb": base_state.vram_used_mb,
            },
        }
        try:
            _apply_context_metadata(
                entry,
                await _context_metadata(profile, model_manager, base_state),
            )
        except Exception as exc:  # noqa: BLE001 -- listing must remain available
            import structlog

            structlog.get_logger(__name__).warning(
                "openai_model_context_resolution_failed",
                profile_id=profile.profile_id,
                error=str(exc),
            )

        entry_index_by_model_id[profile.base_model_id] = len(data)
        entries_by_profile_id[profile.profile_id] = entry
        data.append(entry)

    # Routers are first-class public entries but own no physical model. Their
    # advertised context is the greatest request size any visible target can
    # route successfully; status/capabilities come from the first target.
    for profile in router_profiles:
        if not user.is_admin and profile.profile_id not in user.accessible_model_ids:
            continue
        target_entries = [
            entries_by_profile_id[target_id]
            for target_id in (profile.routing_targets or [])
            if target_id in entries_by_profile_id
        ]
        if not target_entries:
            continue
        primary = target_entries[0]
        primary_meta = dict(primary.get("ocabra") or {})
        contexts = [
            entry.get("ocabra", {}).get("context")
            for entry in target_entries
            if isinstance(entry.get("ocabra", {}).get("context"), dict)
        ]
        entry = {
            "id": profile.profile_id,
            "object": "model",
            "created": now_ts,
            "owned_by": "ocabra",
            "capabilities": list(primary.get("capabilities") or []),
            "ocabra": {
                **primary_meta,
                "kind": "router",
                "category": profile.category,
                "display_name": profile.display_name or profile.profile_id,
                "base_model_id": None,
                "routing_targets": list(profile.routing_targets or []),
                "status": (
                    "loaded"
                    if any(e.get("ocabra", {}).get("status") == "loaded" for e in target_entries)
                    else primary_meta.get("status", "configured")
                ),
            },
        }
        if contexts:
            best = max(contexts, key=lambda item: int(item.get("effective_max") or 0))
            context = {
                **best,
                "effective_max": max(int(item.get("effective_max") or 0) for item in contexts),
                "model_max": max(int(item.get("model_max") or 0) for item in contexts) or None,
                "active": max(
                    (int(item.get("active") or 0) for item in contexts),
                    default=0,
                ) or None,
            }
            _apply_context_metadata(entry, context)
        local_profile_ids.add(profile.profile_id)
        entries_by_profile_id[profile.profile_id] = entry
        data.append(entry)

    # ── Federation: merge remote models ─────────────────────────
    fm = _get_federation_manager(request)
    if fm is not None:
        remote_models = fm.get_remote_models()
        for model_id, peers in remote_models.items():
            if model_id in local_model_ids:
                # Model exists locally — annotate with remote availability
                idx = entry_index_by_model_id.get(model_id)
                if idx is not None and idx < len(data):
                    data[idx]["federation"] = {
                        "remote": False,
                        "also_available_on": _build_federation_nodes_metadata(peers),
                    }
            else:
                # Check if any profile_id from the remote model matches a local one
                remote_profiles = set()
                for peer in peers:
                    for m in peer.models:
                        if m.get("model_id") == model_id:
                            remote_profiles.update(m.get("profiles", []))
                if remote_profiles & local_profile_ids:
                    # Already listed under a local profile — skip
                    continue

                # Remote-only model: add it with federation metadata
                # Use the first profile_id if available, otherwise model_id
                first_peer = peers[0]
                display_id = model_id
                for m in first_peer.models:
                    if m.get("model_id") == model_id:
                        model_profiles = m.get("profiles", [])
                        if model_profiles:
                            display_id = model_profiles[0]
                        break

                data.append(
                    {
                        "id": display_id,
                        "object": "model",
                        "created": now_ts,
                        "owned_by": first_peer.name,
                        "federation": {
                            "remote": True,
                            "node_name": first_peer.name,
                            "node_id": first_peer.peer_id,
                            "available_on": _build_federation_nodes_metadata(peers),
                        },
                    }
                )

    # ── Agents (plan: docs/tasks/agents-mcp-plan.md — "Inventario de modelos")
    try:
        agent_entries = await _list_agent_model_entries(user, now_ts)
    except Exception as exc:  # noqa: BLE001 — never block /v1/models on agent lookup
        import structlog

        structlog.get_logger(__name__).warning(
            "openai_models_agent_listing_failed", error=str(exc)
        )
        agent_entries = []
    data.extend(agent_entries)

    return {"object": "list", "data": data}


async def _list_agent_model_entries(
    user: UserContext, now_ts: int
) -> list[dict[str, Any]]:
    """Return ``[{id: 'agent/<slug>', owned_by: 'ocabra-agent', ...}, ...]``.

    Filtering mirrors ``/v1/models``: admins see every agent; other callers
    only see agents whose ``group_id`` is NULL (public) or in their group set.
    """
    import sqlalchemy as sa
    from sqlalchemy.orm import selectinload

    from ocabra.database import AsyncSessionLocal
    from ocabra.db.agents import Agent

    async with AsyncSessionLocal() as session:
        rows = (
            (
                await session.execute(
                    sa.select(Agent)
                    .options(selectinload(Agent.mcp_links))
                    .order_by(Agent.slug)
                )
            )
            .scalars()
            .all()
        )

    result: list[dict[str, Any]] = []
    group_set = set(user.group_ids or [])
    for row in rows:
        if not user.is_admin:
            if row.group_id is not None and str(row.group_id) not in group_set:
                continue
        result.append(
            {
                "id": f"agent/{row.slug}",
                "object": "model",
                "created": now_ts,
                "owned_by": "ocabra-agent",
                "ocabra": {
                    "kind": "agent",
                    "display_name": row.display_name,
                    "description": row.description,
                    "base_model_id": row.base_model_id,
                    "profile_id": row.profile_id,
                    "tool_choice_default": row.tool_choice_default,
                    "require_approval": row.require_approval,
                    "max_tool_hops": row.max_tool_hops,
                    "mcp_server_count": len(row.mcp_links or []),
                },
            }
        )
    return result


@router.get("/models/{model_id:path}", summary="Retrieve a model")
async def get_model(
    model_id: str,
    request: Request,
    user: UserContext = Depends(get_openai_user),
) -> dict:
    """Retrieve a single model by profile_id (or legacy model_id with fallback)."""
    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    direct_profile = await profile_registry.get(model_id)
    if direct_profile is not None and getattr(direct_profile, "profile_kind", "execution") == "router":
        listing = await list_models(request, user)
        entry = next(
            (item for item in listing["data"] if item.get("id") == model_id),
            None,
        )
        if entry is None:
            raise _openai_error(
                f"The model '{model_id}' does not exist.",
                "invalid_request_error",
                param="model",
                code="model_not_found",
                status_code=404,
            )
        return entry

    try:
        profile, state = await resolve_profile(
            model_id,
            model_manager,
            profile_registry,
            user=user,
        )
    except Exception as exc:
        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        ) from exc

    status = state.status.value
    expected_wait = None
    if status != "loaded":
        expected_wait = await model_manager.get_expected_load_seconds(state.model_id)

    entry = {
        "id": profile.profile_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ocabra",
        "ocabra": {
            "category": profile.category,
            "status": status,
            "loaded": status == "loaded",
            "expected_wait_seconds": expected_wait,
            "capabilities": state.capabilities.to_dict(),
            "display_name": profile.display_name or state.display_name,
            "base_model_id": profile.base_model_id,
            "load_policy": state.load_policy.value,
            "gpu": state.current_gpu,
            "vram_used_mb": state.vram_used_mb,
        },
    }
    try:
        base_state = await model_manager.get_state(profile.base_model_id)
        if base_state is not None:
            _apply_context_metadata(
                entry,
                await _context_metadata(profile, model_manager, base_state),
            )
    except Exception as exc:  # noqa: BLE001 -- preserve OpenAI model retrieval
        import structlog

        structlog.get_logger(__name__).warning(
            "openai_model_context_resolution_failed",
            profile_id=profile.profile_id,
            error=str(exc),
        )
    return entry
