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
    both = BenchmarkRunCreate(
        app="ifeval", suite="quick", models=["m"], cloud={**SPEC, "cloud_type": "BOTH"}
    )
    assert both.cloud.cloud_type == "BOTH"
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
        models=fields.pop("models", ["gemma4:12b"]),
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
async def test_mide_el_primero_disponible_y_espera_solo_los_restantes(
    tmp_path, monkeypatch
) -> None:
    cloud = FakeCloud()
    manager = _manager(cloud)
    run = await _new_run(tmp_path, models=["lento", "rapido", "tercero"])
    measured = []

    async def measure(run_id, app, suite, model):
        measured.append(model)
        await manager._advance_run(run_id, model, {"score": 1})

    monkeypatch.setattr(manager, "_execute_model", measure)
    try:
        await manager._provision(run.id)
        assert cloud.requests.created[0]["launch"]["benchmark_prefetch"]
        rid = str((await _load(run.id)).cloud_request_id)
        node_id = str(uuid.uuid4())
        cloud.requests.rows[rid].update(status="launched", cloudNodeId=node_id)
        items = [
            {"model_id": f"llamacpp_mks/{model}", "status": state, "benchmark_prefetch": True}
            for model, state in zip(run.models, ["downloading", "ready", "pending"], strict=True)
        ]
        cloud.nodes.append(
            dict(
                id=node_id,
                status="running",
                peer_online=True,
                peer_name="pod",
                preload_models=items,
            )
        )
        await manager._provision(run.id)
        first = await _load(run.id)
        assert first.node == "pod" and first.models == ["rapido", "lento", "tercero"]
        await manager._execute_run(run.id)
        await manager._provision(run.id)
        assert (await _load(run.id)).node is None
        assert await manager._next_queued_run() != run.id
        items[2]["status"] = "ready"
        await manager._provision(run.id)
        second = await _load(run.id)
        assert second.node == "pod" and second.models[second.current_model_index] == "tercero"
        assert second.result == {"rapido": {"score": 1}}
        assert not cloud.terminated
        await manager._execute_run(run.id)
        items[0]["status"] = "ready"
        await manager._provision(run.id)
        await manager._execute_run(run.id)
        assert measured == ["rapido", "tercero", "lento"]
        assert (await _load(run.id)).status == "completed"
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


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_dos_pasadas_a_la_vez_no_piden_dos_pods(tmp_path: Path) -> None:
    import asyncio

    cloud = FakeCloud()
    manager = _manager(cloud)
    run = await _new_run(tmp_path)
    try:
        await asyncio.gather(manager._advance_provisioning(), manager._advance_provisioning())
        assert len(cloud.requests.created) == 1
    finally:
        await _delete(run.id)


# ── Baterías ─────────────────────────────────────────────────────────────


def test_la_bateria_no_repite_pruebas_y_valida_argumentos() -> None:
    run = BenchmarkRunCreate(
        app="escriba",
        suite="llm",
        models=["m"],
        battery=[{"app": "corrala", "suite": "llm"}, {"app": "escriba", "suite": "llm"}],
    )
    assert run.battery_items() == [("escriba", "llm", []), ("corrala", "llm", [])]
    with pytest.raises(ValidationError):
        BenchmarkRunCreate(
            app="escriba", suite="llm", models=["m"],
            battery=[{"app": "corrala", "suite": "llm", "suite_args": ["--models=x"]}],
        )  # fmt: skip


def test_un_plan_con_bateria_crea_todas_sus_ejecuciones() -> None:
    from ocabra.benchmarks.manager import runs_from_schedule
    from ocabra.db.benchmarks import BenchmarkSchedule

    schedule = BenchmarkSchedule(
        id=uuid.uuid4(), name="noche", app="escriba", suite="llm", revision="abc",
        models=["gemma4:12b"], suite_args=[], auto_pause=True, node=None, sampling=None,
        cloud=dict(SPEC), battery=[{"app": "corrala", "suite": "llm", "suite_args": ["--profile", "quick"]}],
        weekdays=[0], time_of_day="02:00", timezone="Europe/Madrid",
    )  # fmt: skip
    runs = runs_from_schedule(schedule)
    assert [(r.app, r.suite) for r in runs] == [("escriba", "llm"), ("corrala", "llm")]
    assert runs[0].batch_id is not None and runs[0].batch_id == runs[1].batch_id
    assert runs[1].suite_args == ["--profile", "quick"] and runs[1].revision is None
    assert runs[1].cloud == SPEC


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_una_bateria_con_pod_comparte_maquina_y_se_apaga_con_la_ultima(
    tmp_path: Path,
) -> None:
    cloud = FakeCloud()
    manager = _manager(cloud)
    batch = uuid.uuid4()
    first = await _new_run(tmp_path, batch_id=batch)
    second = await _new_run(tmp_path, batch_id=batch)
    try:
        await manager._advance_provisioning()
        await manager._advance_provisioning()
        assert len(cloud.requests.created) == 1
        a, b = await _load(first.id), await _load(second.id)
        assert a.cloud_request_id == b.cloud_request_id
        rid = str(a.cloud_request_id)
        node_id = str(uuid.uuid4())
        cloud.requests.rows[rid].update(
            status="launched", cloudNodeId=node_id, finishedAt=datetime.now(UTC).isoformat()
        )
        cloud.nodes.append(
            {"id": node_id, "status": "running", "peer_online": True, "peer_name": "pod",
             "preload_models": [{"model_id": "x", "status": "loaded"}]}
        )  # fmt: skip
        await manager._advance_provisioning()
        assert (await _load(first.id)).node == "pod" and (await _load(second.id)).node == "pod"
        await manager._advance_run(first.id, "gemma4:12b", {"metrics": {}})
        assert cloud.terminated == []  # queda la otra prueba
        await manager._advance_run(second.id, "gemma4:12b", {"metrics": {}})
        assert cloud.terminated == [(node_id, "benchmark_done")]
    finally:
        await _delete(first.id)
        await _delete(second.id)


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_un_modelo_que_el_pod_no_pudo_bajar_falla_solo_y_se_sigue(
    tmp_path, monkeypatch
) -> None:
    cloud = FakeCloud()
    manager = _manager(cloud)
    run = await _new_run(tmp_path, models=["bueno", "sin-disco", "otro"])
    measured = []

    async def measure(run_id, app, suite, model):
        measured.append(model)
        await manager._advance_run(run_id, model, {"score": 1})

    monkeypatch.setattr(manager, "_execute_model", measure)
    try:
        await manager._provision(run.id)
        rid = str((await _load(run.id)).cloud_request_id)
        node_id = str(uuid.uuid4())
        cloud.requests.rows[rid].update(status="launched", cloudNodeId=node_id)
        items = [
            {"model_id": "llamacpp_mks/bueno", "status": "ready", "benchmark_prefetch": True},
            {"model_id": "llamacpp_mks/sin-disco", "status": "error",
             "error": "La descarga falló: [Errno 28] No space left on device",
             "benchmark_prefetch": True},
            {"model_id": "llamacpp_mks/otro", "status": "pending", "benchmark_prefetch": True},
        ]  # fmt: skip
        cloud.nodes.append(
            dict(id=node_id, status="running", peer_online=True, peer_name="pod",
                 preload_models=items)
        )  # fmt: skip
        await manager._provision(run.id)
        await manager._execute_run(run.id)
        # Al modelo sin disco le toca pero «otro» aún se prepara: esperar a otro,
        # no fallar la ejecución.
        await manager._provision(run.id)
        waiting = await _load(run.id)
        assert waiting.status == "queued" and waiting.node is None
        # Ya no queda nada en preparación: falla solo ese modelo, con su motivo.
        items[2]["status"] = "error"
        items[2]["error"] = "Ollama tag not found in the registry"
        await manager._provision(run.id)
        after = await _load(run.id)
        assert after.status == "queued" and after.current_model_index == 2
        assert "sin-disco: no se pudo preparar en el pod" in after.error
        assert "No space left" in after.error
        await manager._provision(run.id)
        done = await _load(run.id)
        assert done.status == "completed"
        assert "otro: no se pudo preparar en el pod" in done.error
        assert measured == ["bueno"]
    finally:
        await _delete(run.id)


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_si_toca_una_gpu_averiada_se_espera_y_a_la_tercera_se_deja(
    tmp_path: Path, monkeypatch
) -> None:
    import ocabra.benchmarks.manager as manager_module

    cloud = FakeCloud()
    manager = _manager(cloud)
    run = await _new_run(tmp_path)
    broken = "La GPU del pod no funciona para cómputo (cuInit failed with CUDA error 999)"

    async def launch_broken(gpu: str) -> None:
        rid = str((await _load(run.id)).cloud_request_id)
        node_id = str(uuid.uuid4())
        cloud.requests.rows[rid].update(status="launched", cloudNodeId=node_id)
        cloud.nodes.append(
            {"id": node_id, "status": "terminated", "gpu_type_id": gpu, "error": broken}
        )
        await manager._advance_provisioning()

    try:
        await manager._advance_provisioning()
        await launch_broken("4090")
        waiting = await _load(run.id)
        assert waiting.status == "queued" and waiting.cloud_request_id is None
        assert waiting.cloud["broken_gpus"] == {"4090": 1}
        # Se espera antes de pedir otra: nada nuevo de momento.
        await manager._advance_provisioning()
        assert len(cloud.requests.created) == 1
        # Pasado el tiempo, otra vez con las mismas GPUs, hasta el tercer fallo.
        monkeypatch.setattr(manager_module, "BROKEN_POD_RETRY_S", -1)
        for strike in (2, 3):
            async with AsyncSessionLocal() as session:
                row = await session.get(BenchmarkRun, run.id)
                row.cloud = {**row.cloud, "retry_after": None}
                await session.commit()
            await manager._advance_provisioning()
            assert cloud.requests.created[-1]["gpu_type_ids"] == ["4090", "3090"]
            await launch_broken("4090")
            assert (await _load(run.id)).cloud["broken_gpus"] == {"4090": strike}
        # A la tercera, la 4090 sale de la lista.
        await manager._advance_provisioning()
        assert cloud.requests.created[-1]["gpu_type_ids"] == ["3090"]
        assert len(cloud.requests.created) == 4
    finally:
        await _delete(run.id)
