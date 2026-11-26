"""Benchmarks con pod propio: pedir la máquina, esperarla, medir y apagarla."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ocabra.api.internal.benchmarks import BenchmarkRunCreate
from ocabra.benchmarks.manager import BenchmarkManager
from ocabra.database import AsyncSessionLocal
from ocabra.db.benchmarks import BenchmarkRun

SPEC = {"gpu_type_ids": ["4090", "3090"], "terminate_after": True, "ready_timeout_minutes": 90}


def test_la_api_valida_el_pod_propio() -> None:
    run = BenchmarkRunCreate(app="ifeval", suite="quick", models=["m"], cloud=SPEC)
    assert run.cloud is not None and run.cloud.gpu_type_ids == ["4090", "3090"]
    assert run.cloud.terminate_after is True and run.cloud.wait_minutes == 360
    with pytest.raises(ValidationError):
        BenchmarkRunCreate(app="ifeval", suite="quick", models=["m"], cloud=SPEC, node="Mapache")
    with pytest.raises(ValidationError):
        BenchmarkRunCreate(
            app="ifeval", suite="quick", models=["m"], cloud=SPEC, sampling="recommended"
        )
    with pytest.raises(ValidationError):
        BenchmarkRunCreate(app="ifeval", suite="quick", models=["m"], cloud={"gpu_type_ids": []})


class FakeRequests:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.created: list[dict] = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        rid = str(uuid.uuid4())
        self.rows[rid] = {"id": rid, "status": "waiting", "cloudNodeId": None, "lastError": None}
        return dict(self.rows[rid])

    async def get(self, rid):
        return dict(self.rows[rid])

    async def cancel(self, rid):
        self.rows[rid]["status"] = "cancelled"
        return dict(self.rows[rid])


class FakeCloud:
    def __init__(self) -> None:
        self.requests = FakeRequests()
        self.nodes: list[dict] = []
        self.terminated: list[str] = []

    async def list_nodes(self):
        return [dict(node) for node in self.nodes]

    async def terminate(self, node_id, *, reason="manual"):
        self.terminated.append((node_id, reason))
        for node in self.nodes:
            if node["id"] == node_id:
                node["status"] = "terminated"


def _manager(cloud: FakeCloud) -> BenchmarkManager:
    registry = SimpleNamespace(
        get=lambda model: _async(SimpleNamespace(base_model_id=f"llamacpp_mks/{model}"))
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            federation_manager=SimpleNamespace(cloud=cloud), profile_registry=registry
        )
    )
    return BenchmarkManager(app)


async def _async(value):
    return value


async def _new_run(tmp_path: Path, **fields) -> BenchmarkRun:
    run = BenchmarkRun(
        app="ifeval",
        suite="quick",
        revision="abc",
        models=["gemma4:12b"],
        suite_args=[],
        status="queued",
        phase="queued",
        workspace_path=str(tmp_path),
        result={},
        cloud=dict(SPEC),
        **fields,
    )
    async with AsyncSessionLocal() as session:
        session.add(run)
        await session.commit()
    return run


async def _load(run_id) -> BenchmarkRun:
    async with AsyncSessionLocal() as session:
        return await session.get(BenchmarkRun, run_id)


async def _delete(run_id) -> None:
    async with AsyncSessionLocal() as session:
        row = await session.get(BenchmarkRun, run_id)
        if row is not None:
            await session.delete(row)
            await session.commit()


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_el_pod_se_pide_se_espera_y_solo_entonces_ocupa_la_cola(tmp_path: Path) -> None:
    cloud = FakeCloud()
    manager = _manager(cloud)
    run = await _new_run(tmp_path)
    try:
        assert await manager._next_queued_run() != run.id
        await manager._advance_provisioning()
        created = cloud.requests.created[0]
        assert created["gpu_type_ids"] == ["4090", "3090"]
        assert created["launch"]["models"] == ["llamacpp_mks/gemma4:12b"]
        assert created["purpose"] == "benchmark"
        assert (await _load(run.id)).phase == "waiting_capacity"

        # Sale el pod, aún preparando sus modelos.
        rid = str((await _load(run.id)).cloud_request_id)
        node_id = str(uuid.uuid4())
        cloud.requests.rows[rid].update(
            status="launched", cloudNodeId=node_id, finishedAt=datetime.now(UTC).isoformat()
        )
        cloud.nodes.append(
            {"id": node_id, "status": "running", "peer_online": True, "peer_name": "ocabra-bench-1",
             "preload_models": [{"model_id": "llamacpp_mks/gemma4:12b", "status": "loading"}]}
        )  # fmt: skip
        await manager._advance_provisioning()
        assert (await _load(run.id)).phase == "preparing_pod"
        assert await manager._next_queued_run() != run.id

        cloud.nodes[0]["preload_models"][0]["status"] = "loaded"
        await manager._advance_provisioning()
        ready = await _load(run.id)
        assert ready.node == "ocabra-bench-1" and ready.phase == "queued"
        assert await manager._next_queued_run() == run.id

        # Al terminar la medida se apaga el pod.
        await manager._advance_run(run.id, "gemma4:12b", {"metrics": {}})
        assert cloud.terminated == [(node_id, "benchmark_done")]
    finally:
        await _delete(run.id)


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_si_falla_la_preparacion_la_ejecucion_falla_y_se_apaga(tmp_path: Path) -> None:
    cloud = FakeCloud()
    manager = _manager(cloud)
    run = await _new_run(tmp_path)
    try:
        await manager._advance_provisioning()
        rid = str((await _load(run.id)).cloud_request_id)
        node_id = str(uuid.uuid4())
        cloud.requests.rows[rid].update(status="launched", cloudNodeId=node_id)
        cloud.nodes.append(
            {"id": node_id, "status": "running", "peer_online": True, "peer_name": "p",
             "preload_models": [{"model_id": "x", "status": "error", "error": "OOM"}]}
        )  # fmt: skip
        await manager._advance_provisioning()
        failed = await _load(run.id)
        assert failed.status == "failed" and "OOM" in failed.error
        assert cloud.terminated == [(node_id, "benchmark_done")]
    finally:
        await _delete(run.id)


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_cancelar_mientras_espera_suelta_la_peticion(tmp_path: Path) -> None:
    cloud = FakeCloud()
    manager = _manager(cloud)
    run = await _new_run(tmp_path)
    try:
        await manager._advance_provisioning()
        rid = str((await _load(run.id)).cloud_request_id)
        await manager.request_cancel(run.id)
        assert cloud.requests.rows[rid]["status"] == "cancelled"
        assert (await _load(run.id)).status == "cancelled"
    finally:
        await _delete(run.id)


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_sin_capacidad_en_plazo_la_ejecucion_falla(tmp_path: Path) -> None:
    cloud = FakeCloud()
    manager = _manager(cloud)
    run = await _new_run(tmp_path)
    try:
        await manager._advance_provisioning()
        rid = str((await _load(run.id)).cloud_request_id)
        cloud.requests.rows[rid].update(status="expired", lastError="no hubo capacidad")
        await manager._advance_provisioning()
        failed = await _load(run.id)
        assert failed.status == "failed" and "no hubo capacidad" in failed.error
        assert cloud.terminated == []
    finally:
        await _delete(run.id)
