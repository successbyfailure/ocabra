"""RunPod REST API **v2** client (``https://api.runpod.io/v2``).

v1 (``rest.runpod.io/v1``) was retired on 2026-11-15 and GraphQL is going away
in early 2027, so only v2 is used. Field names differ a lot from v1: ``image``
(not ``imageName``), ``gpu: {id, count}``, ``cloud``, ``disk``, ``env`` as an
object, status = actual state (``PROVISIONING|STARTING|RUNNING|EXITED|ERROR|
TERMINATED``), ``cost`` = USD/hour. Errors are RFC 9457 problem+json.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from ocabra.core.cloud.manager import CloudError, PodInfo, PodSpec

logger = structlog.get_logger(__name__)

BASE_URL = "https://api.runpod.io/v2"
# Must match the CUDA the ocabra-node image is built with (Dockerfile.node).
MIN_CUDA_VERSION = "12.8"

_STATUS_MAP = {
    "PROVISIONING": "starting",
    "STARTING": "starting",
    "RUNNING": "running",
    "EXITED": "exited",
    "ERROR": "error",
    "TERMINATED": "terminated",
}


def _problem(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return resp.text[:300] or resp.reason_phrase
    if isinstance(data, dict):
        detail = data.get("detail") or data.get("title") or data.get("message")
        errors = data.get("errors")
        if errors:
            detail = f"{detail}: {errors}"
        return str(detail)[:500]
    return str(data)[:300]


def _status_for(resp: httpx.Response) -> int:
    """Map RunPod errors to what we report to our own client."""
    if resp.status_code in (401, 403):
        return 401 if resp.status_code == 401 else 403
    if resp.status_code == 402:
        return 402  # insufficient balance
    if resp.status_code == 429:
        return 429
    if resp.status_code in (400, 404, 409, 422):
        return 400
    return 502


def pod_info(data: dict[str, Any]) -> PodInfo:
    gpu = data.get("gpu") if isinstance(data.get("gpu"), dict) else {}
    cost = data.get("cost")
    return PodInfo(
        pod_id=str(data.get("id") or ""),
        name=str(data.get("name") or ""),
        status=_STATUS_MAP.get(str(data.get("status") or "").upper(), "starting"),
        cost_per_hr=float(cost) if isinstance(cost, (int, float)) else None,
        gpu_type_id=gpu.get("id"),
        raw=data,
    )


class RunPodClient:
    """Thin async client for the parts of RunPod v2 we use."""

    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=httpx.Timeout(30.0, connect=10.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            resp = await self._client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise CloudError(f"No se pudo contactar con RunPod: {exc}", 502) from exc
        if resp.status_code >= 400:
            message = _problem(resp)
            if resp.status_code == 402:
                message = f"Saldo insuficiente en RunPod: {message}"
            elif resp.status_code == 401:
                message = f"API key de RunPod no válida: {message}"
            raise CloudError(f"RunPod {resp.status_code}: {message}", _status_for(resp))
        return resp

    async def validate(self) -> None:
        """Cheap authenticated call: fails on a bad key."""
        await self._request("GET", "/pods")

    async def gpu_types(self) -> list[dict[str, Any]]:
        """GPU catalog with per-GPU hourly prices and stock, cheapest first."""
        resp = await self._request(
            "GET", "/catalog/gpus", params={"include": "AVAILABILITY", "product": "POD"}
        )
        out = []
        for g in resp.json().get("gpus", []):
            price = g.get("price") or {}
            avail = g.get("availability")
            out.append(
                {
                    "id": g.get("id"),
                    "name": g.get("name") or g.get("id"),
                    "memory_gb": g.get("memory"),
                    "community": bool(g.get("community")),
                    "secure": bool(g.get("secure")),
                    "price_community": price.get("community"),
                    "price_secure": price.get("secure"),
                    "max_count": g.get("maxCount") or {},
                    "availability": (avail or "NONE").upper() if isinstance(avail, str) else "NONE",
                }
            )
        out.sort(key=lambda g: g["price_community"] or g["price_secure"] or 1e9)
        return out

    async def create_pod(self, spec: PodSpec) -> PodInfo:
        body: dict[str, Any] = {
            "name": spec.name,
            "image": spec.image,
            # The ocabra-node image is built with CUDA 12.8: only machines whose
            # driver supports it (otherwise llama.cpp can't use the GPU and the
            # pod bills while useless).
            "gpu": {
                "id": spec.gpu_type_id,
                "count": spec.gpu_count,
                "minCudaVersion": MIN_CUDA_VERSION,
            },
            "cloud": spec.cloud_type,
            "disk": spec.container_disk_gb,
            "env": spec.env,
            # The node needs no inbound port: it dials us. No SSH either.
            "ports": [],
        }
        try:
            resp = await self._request("POST", "/pods", json=body)
        except CloudError as exc:
            # RunPod answers 400 both for a bad request and for "no machine
            # free right now" (its catalog stock is approximate). The latter
            # is the common one and needs a different action from the user.
            if "does not have the resources" in str(exc) or "no longer any instances" in str(exc):
                cloud = "Community" if spec.cloud_type == "COMMUNITY" else "Secure"
                raise CloudError(
                    f"RunPod no tiene ahora mismo ninguna máquina libre con {spec.gpu_type_id} "
                    f"en {cloud} Cloud (el stock del catálogo es aproximado). Prueba otra GPU "
                    f"o la otra nube, o vuelve a intentarlo en unos minutos.",
                    409,
                ) from exc
            raise
        if resp.status_code == 202:
            raise CloudError("RunPod no tiene capacidad ahora mismo para esa GPU", 409)
        return pod_info(resp.json())

    async def list_pods(self) -> list[PodInfo]:
        pods: list[PodInfo] = []
        cursor: str | None = None
        for _ in range(20):  # 20 pages × 1000 is far beyond anything we launch
            params = {"cursor": cursor} if cursor else None
            data = (await self._request("GET", "/pods", params=params)).json()
            pods.extend(pod_info(p) for p in data.get("pods", []) if isinstance(p, dict))
            pagination = data.get("pagination") or {}
            if not pagination.get("hasNextPage"):
                break
            cursor = pagination.get("nextCursor")
            if not cursor:
                break
        return pods

    async def terminate_pod(self, pod_id: str) -> None:
        try:
            await self._request("DELETE", f"/pods/{pod_id}")
        except CloudError as exc:
            # Already gone is fine: that's what we wanted.
            if "404" not in str(exc):
                raise
