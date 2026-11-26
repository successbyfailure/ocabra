"""Temporary federation nodes on rented GPUs (RunPod first).

Launching a node = create a single-use federation **invite**, then create a pod
running the ``ocabra-node`` image with ``FEDERATION_JOIN_URL`` (us) and
``FEDERATION_JOIN_INVITE`` in its environment. The pod boots, dials us through
the reverse tunnel and appears as a swarm peer — it needs no open port.

Money is the risk, so a reconcile loop runs every ``RECONCILE_INTERVAL_S``:

* syncs every live node with the provider (status, real cost/hour, gone pods);
* links a node to the peer it became (via its invite's join request);
* terminates on **TTL**, **idle** (no real traffic routed to it), **join
  timeout** (never managed to join) and **monthly budget**;
* lists **orphans**: pods named ``ocabra-…`` the provider has but we don't
  track (e.g. the DB row was lost) so the UI can kill them.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import structlog

if TYPE_CHECKING:
    from ocabra.core.federation import FederationManager

from ocabra.core.cloud import timeline as cloud_timeline  # noqa: E402

logger = structlog.get_logger(__name__)

RECONCILE_INTERVAL_S = 30
# Recipes of what a live node has (core/cloud/recipes.py): captured this often,
# plus right before terminating or restarting it.
RECIPE_SNAPSHOT_EVERY_S = 600
RECIPE_SNAPSHOT_TIMEOUT_S = 30
STARTUP_DELAY_S = 15
POD_NAME_PREFIX = "ocabra-"
# A pod that has been running this long without joining us is broken (bad
# image, can't reach us): stop paying for it.
JOIN_TIMEOUT = timedelta(minutes=30)
INVITE_TTL_HOURS = 2
LIVE_STATUSES = ("creating", "starting", "running", "stopping")

DEFAULT_SETTINGS: dict[str, Any] = {
    "image": "ghcr.io/successbyfailure/ocabra-node:latest",
    "cloud_type": "COMMUNITY",
    "default_ttl_minutes": 240,
    "default_idle_minutes": 30,
    "monthly_budget_usd": 20.0,
    "container_disk_gb": 80,
    "pass_hf_token": False,
    "hub_url": "",  # empty = PUBLIC_URL
}


class CloudError(Exception):
    """A cloud operation failed; ``status`` is the HTTP status to report."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class CloudCapacityError(CloudError):
    """El proveedor no tiene ahora mismo máquinas libres de ese tipo (reintentable).

    Se separa del resto de 409 (presupuesto agotado, sin API key, volumen en uso)
    porque es lo único que tiene sentido reintentar solo.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, 409)


@dataclass
class PodSpec:
    name: str
    image: str
    gpu_type_id: str
    gpu_count: int
    cloud_type: str
    container_disk_gb: int
    env: dict[str, str]
    # Newest CUDA the backends to install need (None = the node image's own).
    min_cuda_version: str | None = None
    # Network volume holding models/backends (forces its data center + Secure).
    network_volume_id: str | None = None
    data_center_id: str | None = None


@dataclass
class PodInfo:
    pod_id: str
    name: str
    status: str  # our vocabulary: starting | running | stopping | terminated | error
    cost_per_hr: float | None
    gpu_type_id: str | None = None
    raw: dict[str, Any] | None = None


class CloudProviderClient(Protocol):
    """What the manager needs from a provider."""

    async def validate(self) -> None: ...

    async def gpu_types(self) -> list[dict[str, Any]]: ...

    async def create_pod(self, spec: PodSpec) -> PodInfo: ...

    async def list_pods(self) -> list[PodInfo]: ...

    async def list_volumes(self) -> list[dict[str, Any]]: ...

    async def create_volume(self, name: str, size_gb: int, data_center: str) -> dict[str, Any]: ...

    async def resize_volume(self, volume_id: str, size_gb: int) -> dict[str, Any]: ...

    async def delete_volume(self, volume_id: str) -> None: ...

    async def data_centers(self) -> list[dict[str, Any]]: ...

    async def terminate_pod(self, pod_id: str) -> None: ...

    async def get_pod(self, pod_id: str) -> dict[str, Any]: ...

    async def update_pod(self, pod_id: str, *, image: str, env: dict[str, str]) -> PodInfo: ...

    async def restart_pod(self, pod_id: str) -> None: ...

    async def aclose(self) -> None: ...


def _now() -> datetime:
    return datetime.now(UTC)


def node_cost(row: Any, *, since: datetime | None = None, now: datetime | None = None) -> float:
    """Approximate spend of a node (cost/hour × billed time), optionally from *since*."""
    if not row.cost_per_hr or row.started_at is None and row.created_at is None:
        return 0.0
    start = row.started_at or row.created_at
    end = row.terminated_at or now or _now()
    if since is not None and start < since:
        start = since
    if end <= start:
        return 0.0
    return float(row.cost_per_hr) * (end - start).total_seconds() / 3600.0


def _month_start(now: datetime) -> datetime:
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


class CloudNodeManager:
    """Launches, watches and kills cloud nodes for one FederationManager."""

    def __init__(
        self,
        fm: FederationManager,
        *,
        client_factory: Any = None,
    ) -> None:
        self.fm = fm
        self._client_factory = client_factory or self._default_client_factory
        self._task: asyncio.Task | None = None
        from ocabra.core.cloud.requests import LaunchRequestQueue

        # Peticiones de máquina que esperan capacidad (core/cloud/requests.py).
        self.requests = LaunchRequestQueue(self)
        self.orphans: list[dict[str, Any]] = []
        self.last_reconcile_at: float | None = None
        self.last_error: str | None = None
        self._gpu_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._volume_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._dc_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._lock = asyncio.Lock()
        # Background model loads started by the preload driver.
        self._preload_tasks: dict = {}
        # node id → when it was restarted with a new image and has to join
        # again (update_image): the join timeout counts from here, not from
        # the pod's start. In memory: after a hub restart it counts from the
        # start again, which at worst terminates a node that failed to rejoin.
        self._rejoin_since: dict[str, datetime] = {}
        # node id → monotonic time of its last recipe capture.
        self._recipes_at: dict[str, float] = {}
        self._recipe_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _default_client_factory(provider: str, api_key: str) -> CloudProviderClient:
        if provider == "runpod":
            from ocabra.core.cloud.runpod import RunPodClient

            return RunPodClient(api_key)
        raise CloudError(f"Proveedor no soportado: {provider}", 400)

    # ── lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="cloud-node-reconcile")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        # Let the API finish booting (and the tunnels reconnect) first.
        await asyncio.sleep(STARTUP_DELAY_S)
        while True:
            try:
                await self.reconcile()
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — keep the watchdog alive
                self.last_error = str(exc) or type(exc).__name__
                logger.warning("cloud_reconcile_error", error=self.last_error)
            try:
                await self.requests.drive()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — una petición rota no para el bucle
                logger.warning("cloud_requests_error", error=str(exc) or type(exc).__name__)
            await asyncio.sleep(RECONCILE_INTERVAL_S)

    # ── provider config ──────────────────────────────────────────

    async def get_provider(self, provider: str = "runpod") -> dict[str, Any]:
        """Settings (merged with defaults) and whether an API key is stored."""
        from ocabra.db.cloud import CloudProvider

        async with self.fm.session_factory() as session:
            row = await session.get(CloudProvider, provider)
        settings = {**DEFAULT_SETTINGS, **((row.settings or {}) if row else {})}
        return {
            "provider": provider,
            "has_api_key": bool(row and row.api_key_encrypted),
            "settings": settings,
        }

    async def update_provider(
        self,
        provider: str,
        *,
        api_key: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store the API key (validated against the provider first) and/or settings."""
        from ocabra.db.cloud import CloudProvider

        if api_key:
            client = self._client_factory(provider, api_key)
            try:
                await client.validate()
            finally:
                await client.aclose()
        async with self.fm.session_factory() as session:
            row = await session.get(CloudProvider, provider)
            if row is None:
                row = CloudProvider(provider=provider, settings={})
                session.add(row)
            if api_key is not None:
                row.api_key_encrypted = self.fm.encrypt_api_key(api_key) if api_key else None
            if settings:
                clean = {k: v for k, v in settings.items() if k in DEFAULT_SETTINGS}
                row.settings = {**(row.settings or {}), **clean}
            await session.commit()
        self._gpu_cache = None
        return await self.get_provider(provider)

    async def _client(self, provider: str = "runpod") -> CloudProviderClient:
        from ocabra.db.cloud import CloudProvider

        async with self.fm.session_factory() as session:
            row = await session.get(CloudProvider, provider)
        if row is None or not row.api_key_encrypted:
            raise CloudError("Configura primero la API key de RunPod", 409)
        return self._client_factory(provider, self.fm.decrypt_api_key(row.api_key_encrypted))

    async def gpu_types(self, provider: str = "runpod") -> list[dict[str, Any]]:
        """GPU types with prices/availability (cached 5 min)."""
        if self._gpu_cache is not None and time.monotonic() - self._gpu_cache[0] < 300:
            return self._gpu_cache[1]
        client = await self._client(provider)
        try:
            types = await client.gpu_types()
        finally:
            await client.aclose()
        self._gpu_cache = (time.monotonic(), types)
        return types

    # ── network volumes (model/backend cache) ─────────────────────

    async def _volume_users(self) -> dict[str, str]:
        """volume id -> name of the live node using it."""
        import sqlalchemy as sa

        from ocabra.db.cloud import CloudNode

        async with self.fm.session_factory() as session:
            rows = (
                await session.execute(
                    sa.select(CloudNode.network_volume_id, CloudNode.name).where(
                        CloudNode.network_volume_id.is_not(None),
                        CloudNode.status.in_(LIVE_STATUSES),
                    )
                )
            ).all()
        return {vid: name for vid, name in rows}

    async def volumes(self, provider: str = "runpod") -> list[dict[str, Any]]:
        """Network volumes on the account, with the live node using each."""
        client = await self._client(provider)
        try:
            vols = await client.list_volumes()
        finally:
            await client.aclose()
        users = await self._volume_users()
        for v in vols:
            v["in_use_by"] = users.get(v["id"])
        self._volume_cache = (time.monotonic(), vols)
        return vols

    async def data_centers(self, provider: str = "runpod") -> list[dict[str, Any]]:
        """Data centers that take volumes, with GPU stock (cached 5 min)."""
        if self._dc_cache is not None and time.monotonic() - self._dc_cache[0] < 300:
            return self._dc_cache[1]
        client = await self._client(provider)
        try:
            dcs = await client.data_centers()
        finally:
            await client.aclose()
        self._dc_cache = (time.monotonic(), dcs)
        return dcs

    async def create_volume(
        self, name: str, size_gb: int, data_center: str, provider: str = "runpod"
    ) -> dict[str, Any]:
        if not 10 <= int(size_gb) <= 4096:
            raise CloudError("El tamaño de un disco va de 10 a 4096 GB", 400)
        client = await self._client(provider)
        try:
            vol = await client.create_volume(name.strip() or "ocabra-models", size_gb, data_center)
        finally:
            await client.aclose()
        self._volume_cache = None
        logger.info("cloud_volume_created", id=vol["id"], size_gb=size_gb, dc=data_center)
        return vol

    async def resize_volume(
        self, volume_id: str, size_gb: int, provider: str = "runpod"
    ) -> dict[str, Any]:
        client = await self._client(provider)
        try:
            vol = await client.resize_volume(volume_id, size_gb)
        finally:
            await client.aclose()
        self._volume_cache = None
        return vol

    async def delete_volume(self, volume_id: str, provider: str = "runpod") -> None:
        user = (await self._volume_users()).get(volume_id)
        if user:
            raise CloudError(f"El disco lo está usando {user}: termínalo antes de borrarlo", 409)
        client = await self._client(provider)
        try:
            await client.delete_volume(volume_id)
        finally:
            await client.aclose()
        self._volume_cache = None
        logger.info("cloud_volume_deleted", id=volume_id)

    def volumes_month_cost(self, now: datetime | None = None) -> float:
        """This month's share of the known volumes' monthly price (estimate).

        RunPod bills volumes until they are deleted; we only know them from
        the last listing, and not their creation date, so this prorates the
        whole month so far for every volume seen.
        """
        if self._volume_cache is None:
            return 0.0
        now = now or _now()
        start = _month_start(now)
        next_month = (start + timedelta(days=32)).replace(day=1)
        fraction = (now - start) / (next_month - start)
        return round(sum(v.get("usd_per_month") or 0 for v in self._volume_cache[1]) * fraction, 4)

    # ── spend ────────────────────────────────────────────────────

    async def month_spend(self) -> float:
        import sqlalchemy as sa

        from ocabra.db.cloud import CloudNode

        now = _now()
        start = _month_start(now)
        async with self.fm.session_factory() as session:
            rows = (
                (
                    await session.execute(
                        sa.select(CloudNode).where(
                            sa.or_(
                                CloudNode.terminated_at.is_(None), CloudNode.terminated_at >= start
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )
        pods = sum(node_cost(r, since=start, now=now) for r in rows)
        return round(pods + self.volumes_month_cost(now), 4)

    # ── nodes ────────────────────────────────────────────────────

    def _node_out(self, row: Any) -> dict[str, Any]:
        peer = None
        if row.peer_id is not None:
            peer = next((p for p in self.fm.get_all_peers() if p.peer_id == str(row.peer_id)), None)
        now = _now()
        deadline = None
        if row.max_runtime_minutes and row.status in LIVE_STATUSES:
            deadline = (row.started_at or row.created_at) + timedelta(
                minutes=row.max_runtime_minutes
            )
        return {
            "id": str(row.id),
            "provider": row.provider,
            "provider_pod_id": row.provider_pod_id,
            "name": row.name,
            "status": row.status,
            "gpu_type_id": row.gpu_type_id,
            "gpu_count": row.gpu_count,
            "cloud_type": row.cloud_type,
            "image": row.image,
            "cost_per_hr": row.cost_per_hr,
            "cost_so_far": round(node_cost(row, now=now), 4),
            "max_runtime_minutes": row.max_runtime_minutes,
            "idle_shutdown_minutes": row.idle_shutdown_minutes,
            "terminates_at": deadline,
            "peer_id": str(row.peer_id) if row.peer_id else None,
            "peer_name": peer.name if peer else None,
            "peer_online": bool(peer and peer.online),
            "requests_proxied": peer.requests_proxied if peer else 0,
            "preload_models": row.preload_models or [],
            "network_volume_id": row.network_volume_id,
            "data_center_id": row.data_center_id,
            "error": row.error,
            "created_at": row.created_at,
            "started_at": row.started_at,
            "last_active_at": row.last_active_at,
            "timeline": row.timeline,
            "durations": cloud_timeline.summary(row.timeline) if row.timeline else None,
            "terminated_at": row.terminated_at,
            "terminate_reason": row.terminate_reason,
            "privacy_mode": row.privacy_mode,
        }

    async def list_nodes(self, *, include_terminated: bool = True) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        from ocabra.db.cloud import CloudNode

        async with self.fm.session_factory() as session:
            q = sa.select(CloudNode).order_by(CloudNode.created_at.desc()).limit(50)
            if not include_terminated:
                q = q.where(CloudNode.status.in_(LIVE_STATUSES))
            rows = (await session.execute(q)).scalars().all()
        return [self._node_out(r) for r in rows]

    async def launch(
        self,
        *,
        gpu_type_id: str,
        gpu_count: int = 1,
        cloud_type: str | None = None,
        max_runtime_minutes: int | None = None,
        idle_shutdown_minutes: int | None = None,
        name: str | None = None,
        models: list[str] | None = None,
        provider: str = "runpod",
        volume_id: str | None = None,
        privacy_mode: str = "functional",
        recipes: list[str] | None = None,
        download_only: list[str] | None = None,
        backends: list[str] | None = None,
        benchmark_prefetch: bool = False,
    ) -> dict[str, Any]:
        """Create a pod that joins us as a swarm peer. Returns the node.

        *models* are our model ids to replicate there once it joins
        (see core/cloud/preload.py); they are validated before renting anything.
        *recipes* are models captured from earlier pods (core/cloud/recipes.py).
        *download_only*: ids (of either) to download but not load.
        """
        from ocabra.core.cloud import preload
        from ocabra.db.cloud import CloudNode

        models = list(dict.fromkeys(models or []))
        await preload.validate_models(self.fm.session_factory, models)
        preload_items = await self._preload_items(
            models, recipes or [], download_only or [], backends or []
        )
        if benchmark_prefetch:
            for item in preload_items:
                if item.get("kind") != "backend":
                    item.update(benchmark_prefetch=True, load=False)

        cfg = (await self.get_provider(provider))["settings"]
        hub_url = (cfg.get("hub_url") or getattr(self.fm.settings, "public_url", "") or "").rstrip(
            "/"
        )
        if not hub_url.startswith("https://"):
            raise CloudError(
                "Hace falta una URL pública https para que el nodo se conecte "
                "(PUBLIC_URL o hub_url en la configuración de la nube)",
                409,
            )
        budget = float(cfg.get("monthly_budget_usd") or 0)
        if budget > 0 and await self.month_spend() >= budget:
            raise CloudError(f"Presupuesto del mes agotado (${budget:.2f})", 409)

        ttl = max_runtime_minutes if max_runtime_minutes is not None else cfg["default_ttl_minutes"]
        idle = (
            idle_shutdown_minutes
            if idle_shutdown_minutes is not None
            else cfg["default_idle_minutes"]
        )
        node_id = uuid.uuid4()
        pod_name = f"{POD_NAME_PREFIX}{(name or gpu_type_id).lower().replace(' ', '-')[:40]}-{node_id.hex[:6]}"
        invite = await self.fm.join_service.create_invite(
            label=f"cloud:{pod_name}", max_uses=1, expires_in_hours=INVITE_TTL_HOURS
        )
        env = {
            "FEDERATION_JOIN_URL": hub_url,
            "FEDERATION_JOIN_INVITE": invite["code"],
            "FEDERATION_NODE_NAME": pod_name,
            "FEDERATION_ENABLED": "true",
            "OCABRA_CLOUD_NODE_ID": str(node_id),
            # Fresh secrets per pod: nothing from this node is reused there.
            "JWT_SECRET": secrets.token_urlsafe(48),
            "OCABRA_ADMIN_PASS": secrets.token_urlsafe(24),
            # The pod kills itself if it can't keep its tunnel (see node watchdog).
            "OCABRA_CLOUD_SELF_DESTRUCT_MINUTES": "20",
        }
        if benchmark_prefetch:
            env["DOWNLOAD_WORKERS"] = "2"
        zero_data = privacy_mode == "zero_data"
        if zero_data:
            # Nothing about requests is kept on the pod; Secure Cloud (vetted
            # data centers, not someone's spare machine) and no secrets it
            # doesn't strictly need.
            env["PRIVACY_MODE"] = "zero_data"
            cloud_type = "SECURE"
        elif cfg.get("pass_hf_token"):
            hf = getattr(self.fm.settings, "hf_token", "") or ""
            if hf:
                env["HF_TOKEN"] = hf

        data_center = None
        if volume_id:
            vol = next((v for v in await self.volumes(provider) if v["id"] == volume_id), None)
            if vol is None:
                raise CloudError("Ese disco no existe en la cuenta de RunPod", 404)
            if vol.get("in_use_by"):
                raise CloudError(f"El disco ya lo está usando {vol['in_use_by']}", 409)
            data_center = vol["data_center"]
            cloud_type = "SECURE"  # network volumes only exist in Secure Cloud

        spec = PodSpec(
            name=pod_name,
            image=str(cfg["image"]),
            gpu_type_id=gpu_type_id,
            gpu_count=max(1, int(gpu_count)),
            cloud_type=(cloud_type or cfg["cloud_type"]).upper(),
            container_disk_gb=int(cfg["container_disk_gb"]),
            env=env,
            min_cuda_version=await preload.required_cuda(self.fm.session_factory, models),
            network_volume_id=volume_id,
            data_center_id=data_center,
        )
        client = await self._client(provider)
        try:
            pod = await client.create_pod(spec)
        except Exception:
            await self.fm.join_service.revoke_invite(invite["id"])
            raise
        finally:
            await client.aclose()

        async with self.fm.session_factory() as session:
            row = CloudNode(
                id=node_id,
                provider=provider,
                provider_pod_id=pod.pod_id,
                name=pod_name,
                status=pod.status if pod.status in LIVE_STATUSES else "starting",
                gpu_type_id=gpu_type_id,
                gpu_count=spec.gpu_count,
                cloud_type=spec.cloud_type,
                image=spec.image,
                cost_per_hr=pod.cost_per_hr,
                max_runtime_minutes=ttl or None,
                idle_shutdown_minutes=idle or None,
                invite_id=uuid.UUID(invite["id"]),
                started_at=_now(),
                timeline={"launched_at": _now().isoformat()},
                preload_models=preload_items or None,
                network_volume_id=volume_id,
                data_center_id=data_center,
                privacy_mode=privacy_mode,
            )
            session.add(row)
            await session.commit()
            out = self._node_out(row)
        logger.info(
            "cloud_node_launched",
            name=pod_name,
            pod_id=pod.pod_id,
            gpu=gpu_type_id,
            cost_per_hr=pod.cost_per_hr,
        )
        return out

    async def _preload_items(
        self,
        models: list[str],
        recipe_ids: list[str],
        download_only: list[str],
        backends: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Preload entries: the backends first (the chosen ones plus those the
        models need), then our *models* and captured *recipe_ids*."""
        from ocabra.core.cloud import preload
        from ocabra.core.cloud import recipes as recipes_mod

        skip_load = set(download_only)
        items: list[dict[str, Any]] = [
            {"model_id": m, "status": "pending", "load": m not in skip_load} for m in models
        ]
        rows = await recipes_mod.load_recipes(
            self.fm.session_factory, list(dict.fromkeys(recipe_ids))
        )
        missing = [r for r in recipe_ids if r not in rows]
        if missing:
            raise CloudError("Esa receta de modelo ya no existe", 400)
        taken = {m for m in models}
        for rid in dict.fromkeys(recipe_ids):
            row = rows[rid]
            if row.node_model_id in taken:
                continue
            taken.add(row.node_model_id)
            items.append(recipes_mod.preload_item(row, load=rid not in skip_load))

        needed = list(dict.fromkeys(backends or []))
        if models:
            ours = await preload.load_model_rows(self.fm.session_factory, models)
            needed += [str(r.backend_type) for r in ours.values()]
        needed += [str(rows[rid].backend_type) for rid in dict.fromkeys(recipe_ids)]
        backend_items = [
            recipes_mod.backend_item(b)
            for b in dict.fromkeys(needed)
            if b and b != "ollama" and b not in recipes_mod.IMAGE_BACKENDS
        ]
        return backend_items + items

    async def add_models(
        self,
        node_id: str,
        models: list[str],
        recipes: list[str] | None = None,
        download_only: list[str] | None = None,
        backends: list[str] | None = None,
    ) -> dict[str, Any]:
        """Queue more of our models (or captured recipes) for a running node."""
        from ocabra.core.cloud import preload
        from ocabra.db.cloud import CloudNode

        models = list(dict.fromkeys(models))
        await preload.validate_models(self.fm.session_factory, models)
        new_items = await self._preload_items(
            models, recipes or [], download_only or [], backends or []
        )
        async with self.fm.session_factory() as session:
            row = await session.get(CloudNode, uuid.UUID(node_id))
            if row is None:
                raise CloudError("Nodo no encontrado", 404)
            if row.status not in LIVE_STATUSES:
                raise CloudError("El nodo ya no está en marcha", 409)
            items = list(row.preload_models or [])
            present = {i["model_id"] for i in items if i.get("status") != "error"}
            wanted = {i["model_id"] for i in new_items}
            # A model that failed before can be retried by adding it again.
            items = [
                i for i in items if not (i["model_id"] in wanted and i.get("status") == "error")
            ]
            items += [i for i in new_items if i["model_id"] not in present]
            row.preload_models = items
            await session.commit()
            out = self._node_out(row)
        logger.info("cloud_node_models_added", node=out["name"], models=sorted(wanted))
        return out

    async def _capture_recipes(
        self, *, peer_id: Any, node_name: str, gpu_type_id: str | None
    ) -> None:
        """Keep the recipes of the models a live node has (core/cloud/recipes.py)."""
        from ocabra.core.cloud import recipes

        peer = (
            next((p for p in self.fm.get_all_peers() if p.peer_id == str(peer_id)), None)
            if peer_id is not None
            else None
        )
        if peer is None or not peer.online:
            return
        try:
            await asyncio.wait_for(
                recipes.snapshot(self.fm, peer, node_name=node_name, gpu_type_id=gpu_type_id),
                timeout=RECIPE_SNAPSHOT_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 — a recipe is never worth failing for
            logger.warning("cloud_recipes_capture_failed", node=node_name, error=str(exc))

    async def update_image(self, node_id: str, version: str | None = None) -> dict[str, Any]:
        """Point a running pod at another ``ocabra-node`` image and restart it.

        Same machine and GPU (no new rental), but the container disk is wiped:
        the node comes back with an empty database, so it joins again as a new
        peer with a fresh single-use invite. The pod's environment is read
        back from RunPod and only the invite changes (privacy mode, HF token,
        everything else stays). Models on its network volume survive; the ones
        queued for preload are queued again. *version* is an ``X.Y.Z`` tag of
        the image's repository (default: the tag of the configured image).
        """
        from ocabra.db.cloud import CloudNode

        async with self.fm.session_factory() as session:
            row = await session.get(CloudNode, uuid.UUID(node_id))
            if row is None:
                raise CloudError("Nodo no encontrado", 404)
            if row.status not in LIVE_STATUSES or not row.provider_pod_id:
                raise CloudError("El nodo no está en marcha", 409)
            pod_id, provider, name = row.provider_pod_id, row.provider, row.name
            old_peer, old_invite = row.peer_id, row.invite_id
            gpu_type_id = row.gpu_type_id
        # Its disk is about to be wiped: keep what it has so it comes back.
        await self._capture_recipes(peer_id=old_peer, node_name=name, gpu_type_id=gpu_type_id)
        cfg = (await self.get_provider(provider))["settings"]
        repo = (
            str(cfg["image"]).rsplit(":", 1)[0]
            if ":" in str(cfg["image"]).rsplit("/", 1)[-1]
            else str(cfg["image"])
        )
        image = f"{repo}:{version}" if version else str(cfg["image"])

        client = await self._client(provider)
        invite = await self.fm.join_service.create_invite(
            label=f"cloud:{name}:update", max_uses=1, expires_in_hours=INVITE_TTL_HOURS
        )
        try:
            current = await client.get_pod(pod_id)
            env = current.get("env")
            if not isinstance(env, dict) or "FEDERATION_JOIN_URL" not in env:
                raise CloudError(
                    "RunPod no devolvió el entorno del pod: no se puede cambiar la imagen "
                    "sin perder su configuración",
                    502,
                )
            env = {str(k): str(v) for k, v in env.items()}
            env["FEDERATION_JOIN_INVITE"] = invite["code"]
            await client.update_pod(pod_id, image=image, env=env)
            with contextlib.suppress(CloudError):
                # Changing the image may already restart it; then this is a no-op.
                await client.restart_pod(pod_id)
        except Exception:
            await self.fm.join_service.revoke_invite(invite["id"])
            raise
        finally:
            await client.aclose()

        if old_peer is not None:
            with contextlib.suppress(Exception):
                await self.fm.remove_peer(str(old_peer))
        if old_invite is not None:
            with contextlib.suppress(Exception):
                await self.fm.join_service.revoke_invite(str(old_invite))
        self._rejoin_since[node_id] = _now()
        async with self.fm.session_factory() as session:
            row = await session.get(CloudNode, uuid.UUID(node_id))
            row.image = image
            row.invite_id = uuid.UUID(invite["id"])
            row.peer_id = None
            row.status = "starting"
            row.error = None
            row.preload_models = (
                await self._requeue_after_wipe(row.preload_models or [], node_name=name) or None
            )
            await session.commit()
            out = self._node_out(row)
        logger.info("cloud_node_image_updated", node=name, image=image)
        return out

    async def _requeue_after_wipe(
        self, items: list[dict[str, Any]], *, node_name: str
    ) -> list[dict[str, Any]]:
        """Preload after the node's disk was wiped: what it had queued, plus the
        models captured from it (set up by hand in its panel) that weren't."""
        import sqlalchemy as sa

        from ocabra.core.cloud import recipes as recipes_mod
        from ocabra.db.cloud import CloudModelRecipe

        keep = ("model_id", "kind", "recipe_id", "recipe_steps", "load", "benchmark_prefetch")
        out = [{**{k: i[k] for k in keep if k in i}, "status": "pending"} for i in items]
        have = {i["model_id"] for i in out}
        async with self.fm.session_factory() as session:
            rows = (
                (
                    await session.execute(
                        sa.select(CloudModelRecipe).where(
                            CloudModelRecipe.last_node_name == node_name
                        )
                    )
                )
                .scalars()
                .all()
            )
        for row in rows:
            if row.node_model_id not in have:
                out.append(recipes_mod.preload_item(row, load=True))
        return out

    async def terminate(self, node_id: str, *, reason: str = "manual") -> dict[str, Any]:
        """Terminate the pod, remove its peer and revoke its invite."""
        from ocabra.db.cloud import CloudNode

        async with self.fm.session_factory() as session:
            row = await session.get(CloudNode, uuid.UUID(node_id))
            if row is None:
                raise CloudError("Nodo no encontrado", 404)
            pod_id, provider = row.provider_pod_id, row.provider
            peer_id, invite_id = row.peer_id, row.invite_id
            already = row.status == "terminated"
            name, gpu_type_id = row.name, row.gpu_type_id
        if not already:
            # Last chance to learn what it had (models set up by hand in it).
            await self._capture_recipes(peer_id=peer_id, node_name=name, gpu_type_id=gpu_type_id)
        if not already and pod_id:
            client = await self._client(provider)
            try:
                await client.terminate_pod(pod_id)
            finally:
                await client.aclose()
        if peer_id is not None:
            with contextlib.suppress(Exception):
                await self.fm.remove_peer(str(peer_id))
        if invite_id is not None:
            with contextlib.suppress(Exception):
                await self.fm.join_service.revoke_invite(str(invite_id))
        async with self.fm.session_factory() as session:
            row = await session.get(CloudNode, uuid.UUID(node_id))
            if row is not None and row.status != "terminated":
                row.status = "terminated"
                row.terminated_at = _now()
                row.terminate_reason = reason
            await session.commit()
            out = self._node_out(row)
        logger.info("cloud_node_terminated", node=out["name"], reason=reason)
        return out

    async def terminate_orphan(self, pod_id: str, provider: str = "runpod") -> None:
        if not any(o["pod_id"] == pod_id for o in self.orphans):
            raise CloudError("Ese pod no figura como huérfano", 404)
        client = await self._client(provider)
        try:
            await client.terminate_pod(pod_id)
        finally:
            await client.aclose()
        self.orphans = [o for o in self.orphans if o["pod_id"] != pod_id]

    # ── watchdog ─────────────────────────────────────────────────

    async def reconcile(self, provider: str = "runpod") -> None:
        """One pass of the watchdog (see module docstring)."""
        import sqlalchemy as sa

        from ocabra.db.cloud import CloudNode
        from ocabra.db.federation import FederationJoinRequest

        async with self._lock:
            async with self.fm.session_factory() as session:
                live = (
                    (
                        await session.execute(
                            sa.select(CloudNode).where(CloudNode.status.in_(LIVE_STATUSES))
                        )
                    )
                    .scalars()
                    .all()
                )
            cfg = await self.get_provider(provider)
            if not cfg["has_api_key"]:
                self.orphans = []
                return

            client = await self._client(provider)
            try:
                pods = await client.list_pods()
            finally:
                await client.aclose()
            self.last_reconcile_at = time.time()
            by_id = {p.pod_id: p for p in pods}
            tracked = {r.provider_pod_id for r in live}
            # Orphans: ours by name, unknown to the DB (live rows only: a
            # terminated row whose pod still exists is also an orphan).
            self.orphans = [
                {
                    "pod_id": p.pod_id,
                    "name": p.name,
                    "status": p.status,
                    "cost_per_hr": p.cost_per_hr,
                }
                for p in pods
                if p.name.startswith(POD_NAME_PREFIX)
                and p.pod_id not in tracked
                and p.status != "terminated"
            ]

            now = _now()
            to_kill: list[tuple[str, str]] = []
            budget = float(cfg["settings"].get("monthly_budget_usd") or 0)
            over_budget = budget > 0 and await self.month_spend() >= budget

            async with self.fm.session_factory() as session:
                for row in live:
                    row = await session.get(CloudNode, row.id)
                    if row is None:
                        continue
                    pod = by_id.get(row.provider_pod_id or "")
                    if pod is None:
                        if row.status != "creating":
                            row.status = "terminated"
                            row.terminated_at = row.terminated_at or now
                            row.terminate_reason = row.terminate_reason or "gone"
                        continue
                    if pod.status == "terminated":
                        row.status = "terminated"
                        row.terminated_at = row.terminated_at or now
                        row.terminate_reason = row.terminate_reason or "provider"
                    elif pod.status in LIVE_STATUSES:
                        row.status = pod.status
                        if pod.status == "running":
                            row.timeline = cloud_timeline.stamp(row.timeline, "running_at", now)
                    elif pod.status in ("exited", "error"):
                        # We never stop pods on purpose, and a stopped one still
                        # bills its disk: an exited/errored node is finished.
                        row.error = row.error or f"El pod pasó a {pod.status} en el proveedor"
                        to_kill.append((str(row.id), pod.status))
                        continue
                    if pod.cost_per_hr:
                        row.cost_per_hr = pod.cost_per_hr
                    if row.status == "terminated":
                        continue

                    # Link to the peer it became.
                    if row.peer_id is None and row.invite_id is not None:
                        req = (
                            await session.execute(
                                sa.select(FederationJoinRequest).where(
                                    FederationJoinRequest.invite_id == row.invite_id,
                                    FederationJoinRequest.peer_id.is_not(None),
                                )
                            )
                        ).scalar_one_or_none()
                        if req is not None:
                            row.peer_id = req.peer_id
                            row.timeline = cloud_timeline.stamp(row.timeline, "joined_at", now)
                    peer = None
                    if row.peer_id is not None:
                        peer = next(
                            (p for p in self.fm.get_all_peers() if p.peer_id == str(row.peer_id)),
                            None,
                        )
                    from ocabra.core.cloud import preload

                    if (
                        peer is not None
                        and peer.online
                        and row.preload_models
                        and any(
                            i.get("status") in ("pending", *preload.ACTIVE)
                            for i in row.preload_models
                        )
                    ):
                        row.preload_models = await preload.drive(
                            self.fm, peer, row.preload_models, self._preload_tasks
                        )
                        # Getting models ready is not idling: a big diffusers
                        # copy (backend install + tens of GB) outlasts the
                        # idle timeout.
                        row.last_active_at = now
                    if peer is not None and peer.online and row.timeline is not None:
                        if row.preload_models:
                            row.timeline = cloud_timeline.track_items(
                                row.timeline, row.preload_models, now
                            )
                        elif row.timeline.get("joined_at"):
                            row.timeline = cloud_timeline.stamp(row.timeline, "ready_at", now)
                    if (
                        peer is not None
                        and peer.online
                        and time.monotonic() - self._recipes_at.get(str(row.id), 0.0)
                        >= RECIPE_SNAPSHOT_EVERY_S
                    ):
                        self._recipes_at[str(row.id)] = time.monotonic()
                        task = asyncio.create_task(
                            self._capture_recipes(
                                peer_id=row.peer_id, node_name=row.name, gpu_type_id=row.gpu_type_id
                            ),
                            name=f"cloud-recipes-{row.name}",
                        )
                        self._recipe_tasks.add(task)
                        task.add_done_callback(self._recipe_tasks.discard)
                    if peer is not None and peer.last_proxied_at:
                        active = datetime.fromtimestamp(peer.last_proxied_at, UTC)
                        if row.last_active_at is None or active > row.last_active_at:
                            row.last_active_at = active

                    started = row.started_at or row.created_at
                    join_from = max(started, self._rejoin_since.get(str(row.id), started))
                    if row.peer_id is not None:
                        self._rejoin_since.pop(str(row.id), None)
                    if over_budget:
                        to_kill.append((str(row.id), "budget"))
                    elif row.max_runtime_minutes and now - started >= timedelta(
                        minutes=row.max_runtime_minutes
                    ):
                        to_kill.append((str(row.id), "ttl"))
                    elif peer is not None and (peer.cuda or {}).get("ok") is False:
                        # The GPU is visible but CUDA doesn't work (e.g. a host
                        # whose GPU needs a reboot): models would silently run
                        # on the CPU while we pay for the GPU.
                        row.error = (
                            "La GPU del pod no funciona para cómputo "
                            f"({(peer.cuda or {}).get('error')}); se termina para no pagar en vano"
                        )
                        to_kill.append((str(row.id), "gpu_broken"))
                    elif row.peer_id is None and now - join_from >= JOIN_TIMEOUT:
                        row.error = "No llegó a unirse a la federación (¿imagen o red?)"
                        to_kill.append((str(row.id), "join_timeout"))
                    elif (
                        row.idle_shutdown_minutes
                        and peer is not None
                        and peer.online
                        and now - (row.last_active_at or started)
                        >= timedelta(minutes=row.idle_shutdown_minutes)
                    ):
                        to_kill.append((str(row.id), "idle"))
                await session.commit()

        for node_id, reason in to_kill:
            try:
                await self.terminate(node_id, reason=reason)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cloud_auto_terminate_failed", node=node_id, error=str(exc))
