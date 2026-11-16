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

    def factory(self, provider, api_key):
        return self

    async def validate(self):
        pass

    async def gpu_types(self):
        return [{"id": "3090", "price_community": 0.22}]

    async def create_pod(self, spec):
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

    async def aclose(self):
        pass


async def _clean():
    from ocabra.db.auth import User
    from ocabra.db.cloud import CloudNode, CloudProvider
    from ocabra.db.federation import FederationInvite, FederationJoinRequest, FederationPeer

    async with AsyncSessionLocal() as s:
        for model in (
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

    def run(factory):
        monkeypatch.setattr(cuda_probe, "_RESULT", None)
        monkeypatch.setattr(ctypes, "CDLL", factory)
        return cuda_probe.probe_cuda()

    assert run(lambda name: Lib(0, 1)) == {"ok": True, "devices": 1, "error": None}
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
