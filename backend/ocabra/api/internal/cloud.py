"""Cloud GPU nodes: rent a pod, it joins the federation, kill it when done.

- GET    /cloud/runpod                 — key configured?, defaults, month spend, orphans
- PUT    /cloud/runpod                 — set API key / defaults
- GET    /cloud/runpod/gpu-types       — GPU catalog with price and stock
- GET    /cloud/nodes                  — nodes (live and recent)
- POST   /cloud/nodes                  — launch one
- DELETE /cloud/nodes/{id}             — terminate it
- POST   /cloud/orphans/{pod_id}/terminate — kill an untracked ocabra- pod
- GET    /cloud/replicable-models      — our models a node can carry (and why not)
- POST   /cloud/nodes/{id}/models      — add models to a running node
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.schemas.cloud import (
    CloudNodeImageUpdate,
    CloudNodeLaunch,
    CloudNodeModels,
    CloudProviderUpdate,
    CloudVolumeCreate,
    CloudVolumeResize,
)

router = APIRouter(tags=["cloud"])


def _cloud(request: Request):
    fm = getattr(request.app.state, "federation_manager", None)
    if fm is None or fm.cloud is None:
        raise HTTPException(
            status_code=409,
            detail="Activa la federación: los nodos en la nube se unen a ella",
        )
    return fm.cloud


def _err(exc: Exception) -> HTTPException:
    from ocabra.core.cloud.manager import CloudError

    if isinstance(exc, CloudError):
        return HTTPException(status_code=exc.status, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/cloud/runpod",
    summary="RunPod configuration",
    description="Whether an API key is stored, the launch defaults, this month's spend and orphan pods.",
)
async def get_runpod(
    request: Request, _user: UserContext = Depends(require_role("system_admin"))
) -> dict:
    """Return RunPod config and spend.

    Returns:
        ``{provider, has_api_key, settings, month_spend_usd, orphans, last_reconcile_at, last_error}``.
    """
    cloud = _cloud(request)
    out = await cloud.get_provider("runpod")
    out["month_spend_usd"] = await cloud.month_spend()
    out["orphans"] = cloud.orphans
    out["last_reconcile_at"] = cloud.last_reconcile_at
    out["last_error"] = cloud.last_error
    return out


@router.put(
    "/cloud/runpod",
    summary="Update RunPod configuration",
    description="Store the API key (checked against RunPod first; empty string deletes it) and/or defaults.",
    responses={401: {"description": "Invalid API key"}},
)
async def put_runpod(
    body: CloudProviderUpdate,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Update RunPod configuration.

    Returns:
        The same shape as ``GET /cloud/runpod`` minus spend/orphans.
    """
    cloud = _cloud(request)
    try:
        return await cloud.update_provider(
            "runpod",
            api_key=body.api_key,
            settings=body.settings.model_dump(exclude_none=True) if body.settings else None,
        )
    except Exception as exc:
        raise _err(exc) from exc


@router.get(
    "/cloud/runpod/gpu-types",
    summary="RunPod GPU catalog",
    description="GPU types with community/secure hourly price per GPU and stock. Cached 5 minutes.",
)
async def runpod_gpu_types(
    request: Request, _user: UserContext = Depends(require_role("system_admin"))
) -> list[dict]:
    """List RunPod GPU types, cheapest first.

    Returns:
        ``[{id, name, memory_gb, community, secure, price_community, price_secure,
        max_count, availability}]``.
    """
    try:
        return await _cloud(request).gpu_types("runpod")
    except Exception as exc:
        raise _err(exc) from exc


@router.get(
    "/cloud/nodes",
    summary="Cloud nodes",
    description="Nodes we launched (live and the last terminated ones) with cost so far and deadlines.",
)
async def list_nodes(
    request: Request, _user: UserContext = Depends(require_role("system_admin"))
) -> list[dict]:
    """List cloud nodes.

    Returns:
        Nodes with status, cost_per_hr, cost_so_far, terminates_at, peer link and
        terminate_reason.
    """
    return await _cloud(request).list_nodes()


@router.post(
    "/cloud/nodes",
    summary="Launch a cloud node",
    description=(
        "Create a pod running the ocabra-node image with a single-use invite: it "
        "joins this federation as a swarm peer on its own. Refused when the "
        "monthly budget is spent or there is no public https URL."
    ),
    responses={
        402: {"description": "Insufficient RunPod balance"},
        409: {"description": "No API key / budget spent / no capacity / no public URL"},
    },
)
async def launch_node(
    body: CloudNodeLaunch,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Launch a cloud node.

    Returns:
        The new node (status ``starting``).
    """
    try:
        return await _cloud(request).launch(**body.model_dump())
    except Exception as exc:
        raise _err(exc) from exc


@router.delete(
    "/cloud/nodes/{node_id}",
    summary="Terminate a cloud node",
    description="Terminate the pod (stops billing), remove its peer and revoke its invite.",
    responses={404: {"description": "Not found"}},
)
async def terminate_node(
    node_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Terminate a cloud node.

    Returns:
        The node, now ``terminated``.
    """
    try:
        return await _cloud(request).terminate(node_id, reason="manual")
    except Exception as exc:
        raise _err(exc) from exc


@router.post(
    "/cloud/nodes/{node_id}/update-image",
    summary="Restart a cloud node on another ocabra-node image",
    description=(
        "Same pod and GPU, new image. The container disk is wiped, so the node "
        "joins again as a new peer (fresh invite); models on its network volume "
        "stay and its preload list is queued again."
    ),
    responses={404: {"description": "Not found"}, 409: {"description": "Not running"}},
)
async def update_node_image(
    node_id: str,
    body: CloudNodeImageUpdate,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Restart a cloud node on ``ocabra-node:<version>`` (or the configured image).

    Returns:
        The node, ``starting`` until it joins again.
    """
    try:
        return await _cloud(request).update_image(node_id, body.version)
    except Exception as exc:
        raise _err(exc) from exc


@router.post(
    "/cloud/orphans/{pod_id}/terminate",
    summary="Terminate an orphan pod",
    description="Terminate a pod named ocabra-… that RunPod has but this node does not track.",
    responses={404: {"description": "Not an orphan"}},
)
async def terminate_orphan(
    pod_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Terminate an orphan pod.

    Returns:
        ``{"ok": True}``.
    """
    try:
        await _cloud(request).terminate_orphan(pod_id)
    except Exception as exc:
        raise _err(exc) from exc
    return {"ok": True}


@router.get(
    "/cloud/replicable-models",
    summary="Models a cloud node can carry",
    description=(
        "Our llama.cpp and diffusers (image) models with size, whether they are loaded here, and whether "
        "they can be replicated on a node (with the reason when not: e.g. files "
        "that only exist on this server). Loaded ones first."
    ),
)
async def replicable_models(
    request: Request, _user: UserContext = Depends(require_role("system_admin"))
) -> list[dict]:
    """List our models for the cloud launch picker.

    Returns:
        ``[{model_id, display_name, size_bytes, loaded, replicable, reason}]``.
    """
    from ocabra.core.cloud.preload import replicable_models as _list

    cloud = _cloud(request)
    return await _list(cloud.fm.session_factory, getattr(request.app.state, "model_manager", None))


@router.post(
    "/cloud/nodes/{node_id}/models",
    summary="Add models to a running cloud node",
    description="Queue more of our models to download and load on a node that is already running.",
    responses={
        400: {"description": "Unknown or non-replicable model"},
        404: {"description": "Not found"},
        409: {"description": "Node not running"},
    },
)
async def add_node_models(
    node_id: str,
    body: CloudNodeModels,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Add models to a running node.

    Returns:
        The node with its updated ``preload_models``.
    """
    try:
        return await _cloud(request).add_models(node_id, body.models)
    except Exception as exc:
        raise _err(exc) from exc


# ── network volumes ──────────────────────────────────────────────


@router.get(
    "/cloud/volumes",
    summary="Network volumes",
    description=(
        "RunPod network volumes on the account (model/backend cache for cloud nodes): "
        "size, data center, monthly price and the live node using each, if any."
    ),
)
async def list_volumes(
    request: Request, _user: UserContext = Depends(require_role("system_admin"))
) -> list[dict]:
    """List network volumes.

    Returns:
        ``[{id, name, size_gb, data_center, type, usd_per_month, in_use_by}]``.
    """
    try:
        return await _cloud(request).volumes()
    except Exception as exc:
        raise _err(exc) from exc


@router.get(
    "/cloud/datacenters",
    summary="Data centers for volumes",
    description=(
        "Data centers that take network volumes, with the GPUs in stock in each — "
        "a node using a volume can only run in the volume's data center."
    ),
)
async def list_data_centers(
    request: Request, _user: UserContext = Depends(require_role("system_admin"))
) -> list[dict]:
    """List data centers.

    Returns:
        ``[{id, name, region, gpus: [{id, name, availability}]}]``, most GPUs first.
    """
    try:
        return await _cloud(request).data_centers()
    except Exception as exc:
        raise _err(exc) from exc


@router.post(
    "/cloud/volumes",
    summary="Create a network volume",
    description="Create a network volume (10–4096 GB). It is billed until deleted.",
    responses={400: {"description": "Invalid size or data center"}},
)
async def create_volume(
    body: CloudVolumeCreate,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Create a network volume.

    Returns:
        The new volume.
    """
    try:
        return await _cloud(request).create_volume(body.name, body.size_gb, body.data_center)
    except Exception as exc:
        raise _err(exc) from exc


@router.patch(
    "/cloud/volumes/{volume_id}",
    summary="Grow a network volume",
    description="Network volumes can only grow.",
    responses={400: {"description": "Smaller than the current size"}},
)
async def resize_volume(
    volume_id: str,
    body: CloudVolumeResize,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Grow a network volume.

    Returns:
        The volume with its new size.
    """
    try:
        return await _cloud(request).resize_volume(volume_id, body.size_gb)
    except Exception as exc:
        raise _err(exc) from exc


@router.delete(
    "/cloud/volumes/{volume_id}",
    summary="Delete a network volume",
    description="Delete a network volume and everything on it. Refused while a node uses it.",
    responses={409: {"description": "A live node is using it"}},
)
async def delete_volume(
    volume_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Delete a network volume.

    Returns:
        ``{"deleted": volume_id}``.
    """
    try:
        await _cloud(request).delete_volume(volume_id)
    except Exception as exc:
        raise _err(exc) from exc
    return {"deleted": volume_id}
