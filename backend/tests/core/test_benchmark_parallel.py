"""Una ejecución a la vez por máquina y baterías de pod con modelos que no se copian."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ocabra.backends import ollama_daemon
from ocabra.benchmarks.models import normalize_models, split_deployment_id
from ocabra.database import AsyncSessionLocal
from ocabra.db.benchmarks import BenchmarkRun
from tests.core.test_benchmark_cloud import FakeCloud, _delete, _load, _manager, _new_run


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_un_modelo_que_no_se_puede_copiar_no_tumba_la_bateria(tmp_path: Path) -> None:
    cloud = FakeCloud()

    async def problems(models):
        return {m: "sus ficheros solo existen en este servidor" for m in models if "local" in m}

    cloud.replication_problems = problems
    manager = _manager(cloud)
    run = await _new_run(tmp_path)
    async with AsyncSessionLocal() as session:
        row = await session.get(BenchmarkRun, run.id)
        row.models = ["solo-local:1b", "gemma4:12b"]
        await session.commit()
    try:
        await manager._advance_provisioning()
        assert cloud.requests.created[0]["launch"]["models"] == ["llamacpp_mks/gemma4:12b"]
        loaded = await _load(run.id)
        assert loaded.status == "queued"
        assert "solo existen" in loaded.cloud["skipped"]["solo-local:1b"]

        # Al llegarle el turno se da por fallido con su motivo y se sigue.
        async with AsyncSessionLocal() as session:
            row = await session.get(BenchmarkRun, run.id)
            row.node = "ocabra-bench-1"
            await session.commit()
        await manager._execute_run(run.id)
        after = await _load(run.id)
        assert after.current_model_index == 1 and after.status == "queued"
        assert "no se pudo copiar al pod" in after.error
    finally:
        await _delete(run.id)


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_si_no_se_puede_copiar_ninguno_no_se_alquila_nada(tmp_path: Path) -> None:
    cloud = FakeCloud()

    async def problems(models):
        return {m: "no se puede" for m in models}

    cloud.replication_problems = problems
    manager = _manager(cloud)
    run = await _new_run(tmp_path)
    try:
        await manager._advance_provisioning()
        assert cloud.requests.created == []
        failed = await _load(run.id)
        assert failed.status == "failed" and "no se puede" in failed.error
    finally:
        await _delete(run.id)


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_cada_maquina_mide_lo_suyo_a_la_vez(tmp_path: Path) -> None:
    manager = _manager(FakeCloud())
    local = await _new_run(tmp_path)
    mapache = await _new_run(tmp_path, node="Mapache")
    otra_local = await _new_run(tmp_path)
    for run in (local, mapache, otra_local):
        async with AsyncSessionLocal() as session:
            row = await session.get(BenchmarkRun, run.id)
            row.cloud = None
            await session.commit()
    try:
        assert await manager._next_queued(set()) == (local.id, "")
        # Con este servidor midiendo, la siguiente es la de Mapache, no otra local.
        assert await manager._next_queued({""}) == (mapache.id, "Mapache")
        assert await manager._next_queued({"", "Mapache"}) is None
        assert await manager._next_queued({"Mapache"}) == (local.id, "")
    finally:
        for run in (local, mapache, otra_local):
            await _delete(run.id)


def test_overrides_de_varias_ejecuciones_y_el_del_diagnostico() -> None:
    manager = _manager(FakeCloud())
    manager._sampling_overrides["run-1"] = {"profile": "a"}
    manager.sampling_override = {"profile": "b"}
    assert manager.sampling_override == {"profile": "b"}
    assert [item["profile"] for item in manager.sampling_overrides] == ["a", "b"]
    manager.sampling_override = None
    assert manager.sampling_overrides == [{"profile": "a"}]


@pytest.mark.asyncio
async def test_los_nombres_derivados_pasan_a_id_de_despliegue() -> None:
    async def suffixed(name):
        return ("gemma4:12b", "ollama/gemma4:12b") if name == "gemma4:12b-ollama" else None

    registry = SimpleNamespace(engine_suffixed_name=suffixed)
    models = ["gemma4:12b", "gemma4:12b-ollama", "ollama/gemma4:12b"]
    assert await normalize_models(registry, models) == ["gemma4:12b", "ollama/gemma4:12b"]
    assert split_deployment_id("ollama/gemma4:12b") == ("ollama", "gemma4:12b")
    assert split_deployment_id("org/repo") is None


@pytest.mark.asyncio
async def test_el_hub_no_arranca_ollama_propio(monkeypatch) -> None:
    monkeypatch.setattr(ollama_daemon.settings, "ollama_node_autostart", False)
    called = []

    async def reachable(url=None):
        called.append(url)
        return False

    monkeypatch.setattr(ollama_daemon, "reachable", reachable)
    assert await ollama_daemon.ensure_daemon() is False
    assert called == [None]
    assert ollama_daemon.installed_version() is None


def test_el_ollama_del_pod_copia_los_ajustes_del_hub(monkeypatch) -> None:
    monkeypatch.setattr(ollama_daemon.settings, "ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(ollama_daemon.settings, "ollama_num_parallel", 1)
    monkeypatch.setattr(ollama_daemon.settings, "ollama_max_context", 65536)
    env = ollama_daemon.daemon_env()
    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert env["OLLAMA_NUM_PARALLEL"] == "1" and env["OLLAMA_CONTEXT_LENGTH"] == "65536"
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0" and env["OLLAMA_FLASH_ATTENTION"] == "1"


def test_un_pod_reservado_no_entra_en_el_reparto_automatico() -> None:
    from ocabra.core.federation import FederationManager, federation_request_info
    from tests.test_federation import _make_peer_state, _make_session_factory, _make_settings

    fm = FederationManager(_make_settings(), _make_session_factory())
    loaded = [{"model_id": "llamacpp_mks/ornith-1.5:35b", "status": "loaded"}]
    pod = _make_peer_state(name="ocabra-bench-bfcl-528c2e", models=loaded)
    mapache = _make_peer_state(name="Mapache", models=loaded)
    fm._peers = {pod.peer_id: pod, mapache.peer_id: mapache}
    assert len(fm.get_remote_models()["llamacpp_mks/ornith-1.5:35b"]) == 2

    fm.reserved_peers = {"ocabra-bench-bfcl-528c2e"}
    assert fm.get_remote_models()["llamacpp_mks/ornith-1.5:35b"] == [mapache]
    assert [p for p, _ in fm.get_remote_catalog()["llamacpp_mks/ornith-1.5:35b"]] == [mapache]
    # El runner del benchmark, que lo pide por /_node/<pod>/, sí llega.
    marker = federation_request_info.set({"force_node": "ocabra-bench-bfcl-528c2e"})
    try:
        assert pod in fm.get_remote_models()["llamacpp_mks/ornith-1.5:35b"]
    finally:
        federation_request_info.reset(marker)


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_los_pods_de_benchmarks_en_curso_quedan_reservados(tmp_path: Path) -> None:
    import uuid

    cloud = FakeCloud()
    manager = _manager(cloud)
    fm = manager._app.state.federation_manager
    fm.reserved_peers = set()
    node_id = uuid.uuid4()
    cloud.nodes.append({"id": str(node_id), "status": "running", "peer_name": "Ocabra-Bench-1"})
    cloud.nodes.append({"id": str(uuid.uuid4()), "status": "running", "peer_name": "manual-pod"})
    run = await _new_run(tmp_path, cloud_node_id=node_id)
    try:
        await manager._sync_reserved_pods()
        assert fm.reserved_peers == {"ocabra-bench-1"}
        async with AsyncSessionLocal() as session:
            row = await session.get(BenchmarkRun, run.id)
            row.status = "completed"
            await session.commit()
        await manager._sync_reserved_pods()
        assert fm.reserved_peers == set()
    finally:
        await _delete(run.id)


def test_ifeval_espera_a_que_vuelva_el_nodo(monkeypatch) -> None:
    import io
    import json
    import urllib.error

    from ocabra.benchmarks.runners import ifeval

    monkeypatch.setenv("OPENAI_COMPATIBLE_URL", "http://hub/_node/pod")
    monkeypatch.setenv("OPENAI_COMPATIBLE_KEY", "sk")
    monkeypatch.setattr(ifeval.time, "sleep", lambda _s: None)
    calls = []

    class Ok(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "x", {"Retry-After": "15"}, None)
        if len(calls) == 2:
            raise urllib.error.URLError(ConnectionRefusedError())
        body = {"choices": [{"message": {"content": "hola"}}], "usage": {}}
        return Ok(json.dumps(body).encode())

    monkeypatch.setattr(ifeval.urllib.request, "urlopen", urlopen)
    assert ifeval._request("p", "m")[0] == "hola" and len(calls) == 3

    def bad(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "x", {}, None)

    monkeypatch.setattr(ifeval.urllib.request, "urlopen", bad)
    with pytest.raises(RuntimeError):
        ifeval._request("p", "m")
