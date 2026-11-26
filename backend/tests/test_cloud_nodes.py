"""Cloud GPU nodes: RunPod v2 client (mocked HTTP) and the manager (Postgres)."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import sqlalchemy as sa

from ocabra.core.cloud.manager import (
    CloudError,
    PodInfo,
    PodSpec,
    node_cost,
)
from ocabra.core.cloud.runpod import RunPodClient

# ── RunPod client ────────────────────────────────────────────────


def _rp(handler) -> RunPodClient:
    return RunPodClient("rp_key", transport=httpx.MockTransport(handler))


async def test_create_pod_sends_v2_shape():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": "pod1",
                "name": "ocabra-x",
                "status": "PROVISIONING",
                "cost": 0.22,
                "gpu": {"id": "NVIDIA GeForce RTX 3090", "count": 1},
            },
        )

    client = _rp(handler)
    pod = await client.create_pod(
        PodSpec(
            name="ocabra-x",
            image="ghcr.io/a/ocabra-node:latest",
            gpu_type_id="NVIDIA GeForce RTX 3090",
            gpu_count=1,
            cloud_type="COMMUNITY",
            container_disk_gb=80,
            env={"FEDERATION_JOIN_URL": "https://hub"},
        )
    )
    await client.aclose()
    assert seen["url"] == "https://api.runpod.io/v2/pods"
    assert seen["auth"] == "Bearer rp_key"
    assert seen["body"] == {
        "name": "ocabra-x",
        "image": "ghcr.io/a/ocabra-node:latest",
        "gpu": {"id": "NVIDIA GeForce RTX 3090", "count": 1, "minCudaVersion": "12.8"},
        "cloud": "COMMUNITY",
        "disk": 80,
        "env": {"FEDERATION_JOIN_URL": "https://hub"},
        "ports": [],
    }
    assert pod == PodInfo(
        pod_id="pod1",
        name="ocabra-x",
        status="starting",
        cost_per_hr=0.22,
        gpu_type_id="NVIDIA GeForce RTX 3090",
        raw=pod.raw,
    )


async def test_list_pods_follows_pagination_and_maps_status():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("cursor") == "c2":
            return httpx.Response(
                200,
                json={
                    "pods": [{"id": "b", "name": "n", "status": "EXITED"}],
                    "pagination": {"hasNextPage": False},
                },
            )
        return httpx.Response(
            200,
            json={
                "pods": [{"id": "a", "name": "n", "status": "RUNNING", "cost": 1.5}],
                "pagination": {"hasNextPage": True, "nextCursor": "c2"},
            },
        )

    client = _rp(handler)
    pods = await client.list_pods()
    await client.aclose()
    assert [(p.pod_id, p.status, p.cost_per_hr) for p in pods] == [
        ("a", "running", 1.5),
        ("b", "exited", None),
    ]


async def test_gpu_catalog_sorted_by_price():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/catalog/gpus"
        assert request.url.params["include"] == "AVAILABILITY"
        return httpx.Response(
            200,
            json={
                "gpus": [
                    {
                        "id": "H100",
                        "name": "H100",
                        "memory": 80,
                        "community": True,
                        "secure": True,
                        "price": {"community": 1.99, "secure": 2.69},
                        "availability": "LOW",
                    },
                    {
                        "id": "3090",
                        "name": "RTX 3090",
                        "memory": 24,
                        "community": True,
                        "secure": True,
                        "price": {"community": 0.22, "secure": 0.5},
                        "availability": "HIGH",
                    },
                ]
            },
        )

    client = _rp(handler)
    gpus = await client.gpu_types()
    await client.aclose()
    assert [g["id"] for g in gpus] == ["3090", "H100"]
    assert gpus[0]["price_community"] == 0.22 and gpus[0]["availability"] == "HIGH"


@pytest.mark.parametrize(
    ("status", "expected", "fragment"),
    [
        (401, 401, "no válida"),
        (402, 402, "Saldo insuficiente"),
        (422, 400, "422"),
        (503, 502, "503"),
    ],
)
async def test_errors_are_mapped(status, expected, fragment):
    async def handler(request):
        return httpx.Response(status, json={"title": "Nope", "status": status, "detail": "detalle"})

    client = _rp(handler)
    with pytest.raises(CloudError) as exc:
        await client.validate()
    await client.aclose()
    assert exc.value.status == expected and fragment in str(exc.value)


async def test_no_capacity_202_is_a_clear_error():
    async def handler(request):
        return httpx.Response(202, json={"id": "q1"})

    client = _rp(handler)
    with pytest.raises(CloudError) as exc:
        await client.create_pod(PodSpec("n", "i", "g", 1, "COMMUNITY", 20, {}))
    await client.aclose()
    assert exc.value.status == 409


async def test_terminating_a_gone_pod_is_ok():
    async def handler(request):
        assert request.method == "DELETE"
        return httpx.Response(404, json={"title": "Not Found", "status": 404})

    client = _rp(handler)
    await client.terminate_pod("gone")
    await client.aclose()


def test_node_cost():
    class Row:
        cost_per_hr = 0.5
        created_at = datetime(2026, 11, 1, 10, tzinfo=UTC)
        started_at = datetime(2026, 11, 1, 10, tzinfo=UTC)
        terminated_at = datetime(2026, 11, 1, 12, tzinfo=UTC)

    assert node_cost(Row()) == pytest.approx(1.0)
    assert node_cost(Row(), since=datetime(2026, 11, 1, 11, tzinfo=UTC)) == pytest.approx(0.5)


# ── Manager over Postgres with a fake provider ───────────────────

from ocabra.core.federation import FederationManager  # noqa: E402
from ocabra.database import AsyncSessionLocal  # noqa: E402
from tests.conftest import requires_db  # noqa: E402
from tests.test_federation import _make_settings  # noqa: E402


class FakeProvider:
    def __init__(self) -> None:
        self.pods: dict[str, PodInfo] = {}
        self.created: list[PodSpec] = []
        self.terminated: list[str] = []
        self.volumes: dict[str, dict] = {}
        self.updated: list[tuple] = []
        self.restarted: list[str] = []
        self.envs: dict[str, dict] = {}
        # GPUs sin máquinas libres (para la cola de peticiones).
        self.no_capacity: set[str] = set()
        self.fatal: str | None = None

    def factory(self, provider, api_key):
        return self

    async def validate(self):
        pass

    async def gpu_types(self):
        return [{"id": "3090", "price_community": 0.22}]

    async def create_pod(self, spec):
        if spec.gpu_type_id in self.no_capacity:
            from ocabra.core.cloud.manager import CloudCapacityError

            raise CloudCapacityError(f"sin máquinas {spec.gpu_type_id}")
        if self.fatal:
            from ocabra.core.cloud.manager import CloudError

            raise CloudError(self.fatal, 402)
        pod = PodInfo(
            pod_id=f"pod-{len(self.created)}", name=spec.name, status="starting", cost_per_hr=0.25
        )
        self.created.append(spec)
        self.pods[pod.pod_id] = pod
        return pod

    async def list_pods(self):
        return list(self.pods.values())

    async def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        if pod_id in self.pods:
            self.pods[pod_id].status = "terminated"

    async def list_volumes(self):
        return list(self.volumes.values())

    async def create_volume(self, name, size_gb, data_center):
        vid = f"vol-{len(self.volumes)}"
        self.volumes[vid] = {
            "id": vid,
            "name": name,
            "size_gb": size_gb,
            "data_center": data_center,
            "usd_per_month": round(size_gb * 0.07, 2),
        }
        return self.volumes[vid]

    async def resize_volume(self, volume_id, size_gb):
        self.volumes[volume_id]["size_gb"] = size_gb
        return self.volumes[volume_id]

    async def delete_volume(self, volume_id):
        self.volumes.pop(volume_id, None)

    async def data_centers(self):
        return [{"id": "EU-RO-1", "name": "EU-RO-1", "region": "EUROPE", "gpus": []}]

    async def get_pod(self, pod_id):
        spec = next(s for s in self.created if self.pods[pod_id].name == s.name)
        return {"id": pod_id, "env": dict(self.envs.get(pod_id, spec.env))}

    async def update_pod(self, pod_id, *, image, env):
        self.updated.append((pod_id, image, dict(env)))
        self.envs[pod_id] = dict(env)
        return self.pods[pod_id]

    async def restart_pod(self, pod_id):
        self.restarted.append(pod_id)

    async def aclose(self):
        pass


async def _clean():
    from ocabra.db.auth import User
    from ocabra.db.cloud import CloudLaunchRequest, CloudNode, CloudProvider
    from ocabra.db.federation import FederationInvite, FederationJoinRequest, FederationPeer

    async with AsyncSessionLocal() as s:
        for model in (
            CloudLaunchRequest,
            CloudNode,
            CloudProvider,
            FederationJoinRequest,
            FederationInvite,
            FederationPeer,
        ):
            await s.execute(sa.delete(model))
        await s.execute(sa.delete(User).where(User.username.like("federation:%")))
        await s.commit()


@pytest.fixture
async def hub():
    await _clean()
    fm = FederationManager(
        _make_settings(
            federation_node_id="hub", public_url="https://hub.example", hf_token="hf_secret"
        ),
        AsyncSessionLocal,
    )
    await fm.start()
    await fm.cloud.stop()  # drive reconcile by hand
    fake = FakeProvider()
    fm.cloud._client_factory = fake.factory
    await fm.cloud.update_provider("runpod", api_key="rp_key")
    try:
        yield fm, fake
    finally:
        await fm.stop()
        await _clean()


async def _set(node_id: str, **fields):
    from ocabra.db.cloud import CloudNode

    async with AsyncSessionLocal() as s:
        row = await s.get(CloudNode, uuid.UUID(node_id))
        for k, v in fields.items():
            setattr(row, k, v)
        await s.commit()


async def _node(fm, node_id):
    return next(n for n in await fm.cloud.list_nodes() if n["id"] == node_id)


@requires_db
async def test_launch_creates_pod_with_single_use_invite(hub):
    fm, fake = hub
    node = await fm.cloud.launch(gpu_type_id="3090")
    assert node["status"] == "starting" and node["cost_per_hr"] == 0.25
    assert node["max_runtime_minutes"] == 240 and node["idle_shutdown_minutes"] == 30
    spec = fake.created[0]
    assert spec.name.startswith("ocabra-3090-") and spec.cloud_type == "COMMUNITY"
    assert spec.env["FEDERATION_JOIN_URL"] == "https://hub.example"
    assert spec.env["FEDERATION_JOIN_INVITE"].startswith("ocinv_")
    assert spec.env["OCABRA_CLOUD_SELF_DESTRUCT_MINUTES"] == "20"
    assert "HF_TOKEN" not in spec.env  # opt-in only
    invites = await fm.join_service.list_invites()
    assert invites[0]["max_uses"] == 1 and invites[0]["label"] == f"cloud:{spec.name}"


@requires_db
async def test_hf_token_only_when_enabled(hub):
    fm, fake = hub
    await fm.cloud.update_provider("runpod", settings={"pass_hf_token": True})
    await fm.cloud.launch(gpu_type_id="3090")
    assert fake.created[0].env["HF_TOKEN"] == "hf_secret"


@requires_db
async def test_ttl_terminates_and_cleans_up(hub):
    fm, fake = hub
    node = await fm.cloud.launch(gpu_type_id="3090", max_runtime_minutes=60)
    await _set(node["id"], started_at=datetime.now(UTC) - timedelta(minutes=61))
    await fm.cloud.reconcile()
    after = await _node(fm, node["id"])
    assert after["status"] == "terminated" and after["terminate_reason"] == "ttl"
    assert fake.terminated == [node["provider_pod_id"]]
    assert (await fm.join_service.list_invites())[0]["revoked"] is True


@requires_db
async def test_join_timeout(hub):
    fm, _ = hub
    node = await fm.cloud.launch(gpu_type_id="3090", max_runtime_minutes=0)
    await _set(node["id"], started_at=datetime.now(UTC) - timedelta(minutes=31))
    await fm.cloud.reconcile()
    after = await _node(fm, node["id"])
    assert after["terminate_reason"] == "join_timeout" and after["error"]


@requires_db
async def test_links_peer_and_kills_when_idle(hub):
    fm, fake = hub
    node = await fm.cloud.launch(gpu_type_id="3090", idle_shutdown_minutes=30)
    fake.pods[node["provider_pod_id"]].status = "running"
    # The pod joins with its invite (what the node does on boot).
    spec = fake.created[0]
    res = await fm.join_service.create_request(
        {
            "node_id": "pod-node",
            "node_name": spec.name,
            "invite_code": spec.env["FEDERATION_JOIN_INVITE"],
        },
        source_ip="1.2.3.4",
    )
    assert res["status"] == "accepted"
    await fm.cloud.reconcile()
    linked = await _node(fm, node["id"])
    assert linked["status"] == "running" and linked["peer_name"] == spec.name

    peer = next(p for p in fm.get_all_peers() if p.peer_id == linked["peer_id"])
    peer.online = True
    peer.last_proxied_at = time.time()  # just used
    await fm.cloud.reconcile()
    assert (await _node(fm, node["id"]))["status"] == "running"

    peer.last_proxied_at = time.time() - 31 * 60
    await _set(
        node["id"], last_active_at=None, started_at=datetime.now(UTC) - timedelta(minutes=40)
    )
    await fm.cloud.reconcile()
    after = await _node(fm, node["id"])
    assert after["terminate_reason"] == "idle"
    assert all(p.peer_id != linked["peer_id"] for p in fm.get_all_peers())  # peer removed


@requires_db
async def test_pod_that_exits_on_its_own_is_terminated(hub):
    fm, fake = hub
    node = await fm.cloud.launch(gpu_type_id="3090")
    fake.pods[node["provider_pod_id"]].status = "exited"
    await fm.cloud.reconcile()
    after = await _node(fm, node["id"])
    assert after["status"] == "terminated" and after["terminate_reason"] == "exited"


@requires_db
async def test_budget_blocks_launch_and_kills_running(hub):
    fm, _ = hub
    await fm.cloud.update_provider("runpod", settings={"monthly_budget_usd": 1.0})
    node = await fm.cloud.launch(gpu_type_id="3090")
    await _set(node["id"], cost_per_hr=10.0, started_at=datetime.now(UTC) - timedelta(minutes=30))
    with pytest.raises(CloudError) as exc:
        await fm.cloud.launch(gpu_type_id="3090")
    assert exc.value.status == 409 and "Presupuesto" in str(exc.value)
    await fm.cloud.reconcile()
    assert (await _node(fm, node["id"]))["terminate_reason"] == "budget"


@requires_db
async def test_orphans_detected_and_terminable(hub):
    fm, fake = hub
    fake.pods["stray"] = PodInfo(
        pod_id="stray", name="ocabra-lost-1", status="running", cost_per_hr=0.3
    )
    fake.pods["mine-not"] = PodInfo(
        pod_id="mine-not", name="jupyter-de-otro", status="running", cost_per_hr=1
    )
    await fm.cloud.reconcile()
    assert [o["pod_id"] for o in fm.cloud.orphans] == ["stray"]
    await fm.cloud.terminate_orphan("stray")
    assert fake.terminated == ["stray"]
    with pytest.raises(CloudError):
        await fm.cloud.terminate_orphan("mine-not")  # never touch pods that aren't ours


@requires_db
async def test_launch_needs_public_https_url(hub):
    fm, _ = hub
    fm.settings.public_url = ""
    with pytest.raises(CloudError) as exc:
        await fm.cloud.launch(gpu_type_id="3090")
    assert exc.value.status == 409


@requires_db
async def test_failed_create_revokes_invite(hub):
    fm, fake = hub

    async def boom(spec):
        raise CloudError("RunPod 402: Saldo insuficiente", 402)

    fake.create_pod = boom
    with pytest.raises(CloudError):
        await fm.cloud.launch(gpu_type_id="3090")
    assert (await fm.join_service.list_invites())[0]["revoked"] is True
    assert await fm.cloud.list_nodes() == []


# ── Node-side self-destruct ──────────────────────────────────────


async def test_self_destruct_only_armed_on_a_runpod_pod(monkeypatch):
    from unittest.mock import MagicMock

    from ocabra.core.cloud.node_watchdog import NodeSelfDestruct

    fm = MagicMock()
    fm.settings.ocabra_cloud_self_destruct_minutes = 20
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert NodeSelfDestruct.from_env(fm) is None
    monkeypatch.setenv("RUNPOD_POD_ID", "p1")
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    assert NodeSelfDestruct.from_env(fm) is not None
    fm.settings.ocabra_cloud_self_destruct_minutes = 0
    assert NodeSelfDestruct.from_env(fm) is None


@requires_db
async def test_node_with_broken_cuda_is_terminated(hub):
    fm, fake = hub
    node = await fm.cloud.launch(gpu_type_id="3090")
    fake.pods[node["provider_pod_id"]].status = "running"
    spec = fake.created[0]
    await fm.join_service.create_request(
        {
            "node_id": "pod-node",
            "node_name": spec.name,
            "invite_code": spec.env["FEDERATION_JOIN_INVITE"],
        },
        source_ip="1.2.3.4",
    )
    await fm.cloud.reconcile()
    linked = await _node(fm, node["id"])
    peer = next(p for p in fm.get_all_peers() if p.peer_id == linked["peer_id"])
    peer.online = True
    peer.cuda = {"ok": False, "devices": 0, "error": "cuInit failed with CUDA error 999"}
    await fm.cloud.reconcile()
    after = await _node(fm, node["id"])
    assert after["terminate_reason"] == "gpu_broken" and "999" in after["error"]


def test_cuda_probe_outcomes(monkeypatch):
    import ctypes

    from ocabra.core import cuda_probe

    class Lib:
        def __init__(self, init_rc, count):
            self.init_rc, self.count = init_rc, count

        def cuInit(self, flags):
            return self.init_rc

        def cuDeviceGetCount(self, ref):
            ref._obj.value = self.count
            return 0

        def cuDriverGetVersion(self, ref):
            ref._obj.value = 12080  # a driver that stops at CUDA 12.8
            return 0

    def run(factory):
        monkeypatch.setattr(cuda_probe, "_RESULT", None)
        monkeypatch.setattr(ctypes, "CDLL", factory)
        return cuda_probe.probe_cuda()

    assert run(lambda name: Lib(0, 1)) == {
        "ok": True,
        "devices": 1,
        "error": None,
        "driver_cuda": "12.8",
    }
    broken = run(lambda name: Lib(999, 0))
    assert broken["ok"] is False and "999" in broken["error"]

    def missing(name):
        raise OSError("not found")

    assert run(missing)["ok"] is None


async def test_no_free_machine_is_a_clear_409():
    async def handler(request):
        return httpx.Response(
            400,
            json={
                "title": "Bad Request",
                "status": 400,
                "detail": "This machine does not have the resources to deploy your pod. Please try a different machine",
            },
        )

    client = _rp(handler)
    with pytest.raises(CloudError) as exc:
        await client.create_pod(
            PodSpec("n", "i", "NVIDIA GeForce RTX 4090", 1, "COMMUNITY", 20, {})
        )
    await client.aclose()
    assert exc.value.status == 409
    assert "ninguna máquina libre" in str(exc.value) and "Community" in str(exc.value)


# ── network volumes ──────────────────────────────────────────────


async def test_pod_with_a_volume_is_pinned_to_its_data_center():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "p", "name": "n", "status": "PROVISIONING"})

    client = _rp(handler)
    await client.create_pod(
        PodSpec(
            name="n",
            image="img",
            gpu_type_id="g",
            gpu_count=1,
            cloud_type="COMMUNITY",
            container_disk_gb=40,
            env={"A": "1"},
            network_volume_id="vol1",
            data_center_id="EU-RO-1",
        )
    )
    await client.aclose()
    body = seen["body"]
    assert body["mounts"] == {"network": [{"volumeId": "vol1", "path": "/persist"}]}
    assert body["dataCenterIds"] == ["EU-RO-1"]
    assert body["cloud"] == "SECURE"  # network volumes only exist in Secure Cloud
    assert body["env"] == {"A": "1", "OCABRA_PERSIST_DIR": "/persist"}


async def test_volume_api_shapes():
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            (request.method, request.url.path, request.content and json.loads(request.content))
        )
        if request.url.path.endswith("/catalog/datacenters"):
            return httpx.Response(
                200,
                json={
                    "dataCenters": [
                        {
                            "id": "US-KS-2",
                            "networkVolumeTypes": ["STANDARD"],
                            "gpuAvailability": [
                                {"id": "4090", "name": "RTX 4090", "availability": "HIGH"},
                                {"id": "h100", "availability": "NONE"},
                            ],
                        },
                        {"id": "XX-NO-1", "networkVolumeTypes": [], "gpuAvailability": []},
                    ]
                },
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "networkVolumes": [
                        {"id": "v1", "name": "m", "size": 100, "dataCenter": "US-KS-2"}
                    ],
                    "pagination": {"hasNextPage": False},
                },
            )
        if request.method == "DELETE":
            return httpx.Response(204)
        body = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": "v2",
                "name": body.get("name", "m"),
                "size": body["size"],
                "dataCenter": "US-KS-2",
            },
        )

    client = _rp(handler)
    vols = await client.list_volumes()
    assert vols[0]["size_gb"] == 100 and vols[0]["usd_per_month"] == 7.0
    vol = await client.create_volume("models", 80, "US-KS-2")
    assert vol["id"] == "v2" and vol["data_center"] == "US-KS-2"
    await client.resize_volume("v2", 120)
    await client.delete_volume("v2")
    dcs = await client.data_centers()
    await client.aclose()
    assert [d["id"] for d in dcs] == ["US-KS-2"]  # no volumes there → left out
    assert dcs[0]["gpus"] == [{"id": "4090", "name": "RTX 4090", "availability": "HIGH"}]
    assert (
        "POST",
        "/v2/network-volumes",
        {"name": "models", "size": 80, "dataCenter": "US-KS-2"},
    ) in calls
    assert ("PATCH", "/v2/network-volumes/v2", {"size": 120}) in calls
    assert any(m == "DELETE" and pth == "/v2/network-volumes/v2" for m, pth, _ in calls)


@requires_db
async def test_volume_is_used_by_one_live_node_and_not_deleted_under_it(hub):  # noqa: F811
    fm, fake = hub
    vol = await fm.cloud.create_volume("models", 100, "EU-RO-1")
    node = await fm.cloud.launch(gpu_type_id="4090", cloud_type="COMMUNITY", volume_id=vol["id"])
    spec = fake.created[-1]
    assert spec.network_volume_id == vol["id"] and spec.data_center_id == "EU-RO-1"
    assert spec.cloud_type == "SECURE"
    assert node["network_volume_id"] == vol["id"] and node["data_center_id"] == "EU-RO-1"

    listed = {v["id"]: v for v in await fm.cloud.volumes()}
    assert listed[vol["id"]]["in_use_by"] == node["name"]
    with pytest.raises(CloudError) as exc:
        await fm.cloud.launch(gpu_type_id="4090", volume_id=vol["id"])
    assert exc.value.status == 409  # one pod per volume
    with pytest.raises(CloudError) as exc:
        await fm.cloud.delete_volume(vol["id"])
    assert exc.value.status == 409

    await fm.cloud.terminate(node["id"])
    await fm.cloud.delete_volume(vol["id"])
    assert vol["id"] not in fake.volumes
    with pytest.raises(CloudError):
        await fm.cloud.create_volume("x", 5, "EU-RO-1")  # under the 10 GB minimum


def test_volume_cost_counts_towards_the_month():
    from datetime import UTC, datetime

    from ocabra.core.cloud.manager import CloudNodeManager

    mgr = object.__new__(CloudNodeManager)
    mgr._volume_cache = (0.0, [{"usd_per_month": 7.0}, {"usd_per_month": 3.0}])
    # Halfway through a 30-day month: half of $10.
    assert mgr.volumes_month_cost(datetime(2026, 11, 16, 0, 0, tzinfo=UTC)) == pytest.approx(
        5.0, rel=0.01
    )


@requires_db
async def test_zero_data_pod_is_secure_and_gets_no_hf_token(hub):  # noqa: F811
    fm, fake = hub
    await fm.cloud.update_provider("runpod", settings={"pass_hf_token": True})
    await fm.cloud.launch(gpu_type_id="3090", cloud_type="COMMUNITY", privacy_mode="zero_data")
    spec = fake.created[-1]
    assert spec.env["PRIVACY_MODE"] == "zero_data"
    assert spec.cloud_type == "SECURE"
    assert "HF_TOKEN" not in spec.env


async def test_update_pod_sends_v2_patch_and_restart():
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(200, json={"id": "pod1", "status": "RUNNING"})

    client = _rp(handler)
    await client.update_pod("pod1", image="ghcr.io/a/ocabra-node:0.96.0", env={"A": "1"})
    await client.restart_pod("pod1")
    await client.aclose()
    assert seen == [
        ("PATCH", "/v2/pods/pod1", {"image": "ghcr.io/a/ocabra-node:0.96.0", "env": {"A": "1"}}),
        ("POST", "/v2/pods/pod1/action", {"action": "restart"}),
    ]


@requires_db
async def test_update_image_restarts_the_pod_and_it_joins_again(hub):
    fm, fake = hub
    await fm.cloud.update_provider("runpod", settings={"pass_hf_token": True})
    node = await fm.cloud.launch(gpu_type_id="3090", max_runtime_minutes=0)
    pod_id = node["provider_pod_id"]
    fake.pods[pod_id].status = "running"
    spec = fake.created[0]
    first = spec.env["FEDERATION_JOIN_INVITE"]
    await fm.join_service.create_request(
        {"node_id": "pod-node", "node_name": spec.name, "invite_code": first}, source_ip="1.2.3.4"
    )
    await fm.cloud.reconcile()
    old_peer = (await _node(fm, node["id"]))["peer_id"]
    assert old_peer

    out = await fm.cloud.update_image(node["id"], "0.96.0")
    assert out["status"] == "starting" and out["peer_id"] is None
    ((pid, image, env),) = fake.updated
    assert pid == pod_id and image == "ghcr.io/successbyfailure/ocabra-node:0.96.0"
    # Everything stays but the invite: a new single-use one.
    assert env["HF_TOKEN"] == "hf_secret" and env["FEDERATION_JOIN_URL"] == "https://hub.example"
    assert env["FEDERATION_JOIN_INVITE"] != first
    assert fake.restarted == [pod_id]
    assert all(p.peer_id != old_peer for p in fm.get_all_peers())
    revoked = {i["label"]: i["revoked"] for i in await fm.join_service.list_invites()}
    assert revoked[f"cloud:{spec.name}"] is True

    # Started long ago: the join timeout counts from the update, not the start.
    await _set(node["id"], started_at=datetime.now(UTC) - timedelta(minutes=45))
    await fm.cloud.reconcile()
    assert (await _node(fm, node["id"]))["status"] != "terminated"

    await fm.join_service.create_request(
        {
            "node_id": "pod-node-2",
            "node_name": spec.name,
            "invite_code": env["FEDERATION_JOIN_INVITE"],
        },
        source_ip="1.2.3.4",
    )
    await fm.cloud.reconcile()
    after = await _node(fm, node["id"])
    assert after["peer_id"] and after["peer_id"] != old_peer


@requires_db
async def test_update_image_needs_a_running_node(hub):
    fm, _ = hub
    node = await fm.cloud.launch(gpu_type_id="3090")
    await fm.cloud.terminate(node["id"])
    with pytest.raises(CloudError, match="no está en marcha"):
        await fm.cloud.update_image(node["id"])


# ── Peticiones de máquina que esperan capacidad ──────────────────


async def _age_request(request_id: str, seconds: int) -> None:
    from datetime import UTC, datetime, timedelta

    from ocabra.db.cloud import CloudLaunchRequest

    async with AsyncSessionLocal() as s:
        row = await s.get(CloudLaunchRequest, uuid.UUID(request_id))
        row.last_attempt_at = datetime.now(UTC) - timedelta(seconds=seconds)
        await s.commit()


@requires_db
async def test_la_peticion_prueba_las_gpu_por_orden(hub):
    fm, fake = hub
    fake.no_capacity = {"4090"}
    req = await fm.cloud.requests.create(
        gpu_type_ids=["4090", "3090"], launch={}, wait_minutes=60, retry_interval_s=120
    )
    assert req["status"] == "launched" and req["launchedGpuTypeId"] == "3090"
    assert req["cloudNodeId"] and [spec.gpu_type_id for spec in fake.created] == ["3090"]


@requires_db
async def test_sin_capacidad_espera_y_reintenta_cuando_toca(hub):
    fm, fake = hub
    fake.no_capacity = {"4090"}
    req = await fm.cloud.requests.create(
        gpu_type_ids=["4090"], launch={"gpu_count": 1}, wait_minutes=60, retry_interval_s=120
    )
    assert (
        req["status"] == "waiting" and req["attempts"] == 1 and "sin capacidad" in req["lastError"]
    )
    await fm.cloud.requests.drive()  # aún no toca
    assert (await fm.cloud.requests.get(req["id"]))["attempts"] == 1
    fake.no_capacity = set()
    await _age_request(req["id"], 200)
    await fm.cloud.requests.drive()
    done = await fm.cloud.requests.get(req["id"])
    assert done["status"] == "launched" and done["attempts"] == 2


@requires_db
async def test_un_fallo_que_no_es_de_capacidad_cierra_la_peticion(hub):
    fm, fake = hub
    fake.no_capacity = {"4090"}
    fake.fatal = "Saldo insuficiente en RunPod"
    req = await fm.cloud.requests.create(
        gpu_type_ids=["4090", "3090"], launch={}, wait_minutes=60, retry_interval_s=120
    )
    assert req["status"] == "failed" and req["lastError"] == "Saldo insuficiente en RunPod"


@requires_db
async def test_caduca_y_se_cancela(hub):
    fm, fake = hub
    fake.no_capacity = {"4090"}
    req = await fm.cloud.requests.create(
        gpu_type_ids=["4090"], launch={}, wait_minutes=1, retry_interval_s=60
    )
    from datetime import UTC, datetime, timedelta

    from ocabra.db.cloud import CloudLaunchRequest

    async with AsyncSessionLocal() as s:
        row = await s.get(CloudLaunchRequest, uuid.UUID(req["id"]))
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await s.commit()
    await fm.cloud.requests.drive()
    assert (await fm.cloud.requests.get(req["id"]))["status"] == "expired"

    other = await fm.cloud.requests.create(
        gpu_type_ids=["4090"], launch={}, wait_minutes=60, retry_interval_s=60
    )
    assert (await fm.cloud.requests.cancel(other["id"]))["status"] == "cancelled"
    assert [item["id"] for item in await fm.cloud.requests.list(include_finished=False)] == []
