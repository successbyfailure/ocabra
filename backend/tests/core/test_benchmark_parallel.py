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


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_se_borra_del_pod_lo_que_ya_no_necesita_ninguna_prueba(tmp_path: Path) -> None:
    import uuid

    import httpx

    cloud = FakeCloud()
    manager = _manager(cloud)
    node_id = uuid.uuid4()
    batch = uuid.uuid4()
    cloud.nodes.append({"id": str(node_id), "status": "running", "peer_name": "pod-1"})
    deleted: list[str] = []

    async def proxy_raw(peer, method, path, content, headers=None):
        assert method == "DELETE"
        deleted.append(path)
        return httpx.Response(200, json={"ok": True})

    peer = SimpleNamespace(name="pod-1")
    manager._app.state.federation_manager.get_online_peers = lambda: [peer]
    manager._app.state.federation_manager.proxy_raw = proxy_raw
    models = ["a:1b", "b:1b", "c:1b"]
    ifeval = await _new_run(tmp_path, cloud_node_id=node_id, batch_id=batch)
    bfcl = await _new_run(tmp_path, cloud_node_id=node_id, batch_id=batch)
    async with AsyncSessionLocal() as session:
        for run_id, index in ((ifeval.id, 2), (bfcl.id, 1)):
            row = await session.get(BenchmarkRun, run_id)
            row.models = models
            row.current_model_index = index
            await session.commit()
    try:
        # IFEval ya hizo a y b; BFCL solo a: se puede borrar a, b todavía no.
        await manager._free_pod_models(ifeval.id)
        assert deleted == ["/ocabra/models/llamacpp_mks/a:1b?delete_files=true"]
        await manager._free_pod_models(ifeval.id)
        assert len(deleted) == 1  # no se pide dos veces
        async with AsyncSessionLocal() as session:
            row = await session.get(BenchmarkRun, bfcl.id)
            row.current_model_index = 2
            await session.commit()
        await manager._free_pod_models(bfcl.id)
        assert deleted[-1] == "/ocabra/models/llamacpp_mks/b:1b?delete_files=true"
    finally:
        await _delete(ifeval.id)
        await _delete(bfcl.id)


@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_un_modelo_que_no_responde_no_da_un_resultado_valido() -> None:
    import uuid
    from datetime import UTC, datetime, timedelta

    import sqlalchemy as sa

    from ocabra.benchmarks.manager import BenchmarkIdentity
    from ocabra.db.auth import User
    from ocabra.db.stats import RequestStat

    manager = _manager(FakeCloud())
    user = User(username=f"zz-bench-{uuid.uuid4().hex[:8]}", hashed_password="x")
    async with AsyncSessionLocal() as session:
        session.add(user)
        await session.commit()
    manager._identity = BenchmarkIdentity(user.id, "ocabra-benchmark", "secret")
    start = datetime.now(UTC)
    error = "GPU 0: need 5366 MB, only 4327 MB available after eviction [insufficient_vram]"

    async def add(model, code, n, err=None):
        async with AsyncSessionLocal() as session:
            for i in range(n):
                session.add(
                    RequestStat(
                        model_id=f"llamacpp_mks/{model}",
                        requested_model=model,
                        user_id=user.id,
                        started_at=start + timedelta(seconds=i),
                        status_code=code,
                        error=err,
                    )
                )
            await session.commit()

    try:
        # Escriba: los embeddings van bien, el modelo medido no carga nunca.
        await add("qwen3-embedding:0.6b", 200, 7)
        await add("bonsai-2:9b", 409, 5, error)
        end = start + timedelta(minutes=1)
        why = await manager._measured_failures(start, end, "bonsai-2:9b")
        assert why is not None and "5 de 5" in why and "insufficient_vram" in why
        assert await manager._measured_failures(start, end, "qwen3-embedding:0.6b") is None

        # Algún fallo suelto o reintentos pasajeros no invalidan la medida.
        await add("gemma4:12b", 200, 9)
        await add("gemma4:12b", 409, 1, "x")
        await add("gemma4:12b", 503, 20)
        assert await manager._measured_failures(start, end, "gemma4:12b") is None
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(sa.delete(RequestStat).where(RequestStat.user_id == user.id))
            await session.execute(sa.delete(User).where(User.id == user.id))
            await session.commit()


@pytest.mark.asyncio
async def test_listar_modelos_de_ollama_en_un_pod_no_arranca_su_ollama(monkeypatch) -> None:
    import time

    from ocabra.backends import ollama_backend

    monkeypatch.setattr(ollama_backend.settings, "ollama_node_autostart", True)
    started = []

    async def ensure():
        started.append(1)
        return True

    async def not_running(url=None):
        return False

    monkeypatch.setattr(ollama_backend, "ensure_daemon", ensure)
    monkeypatch.setattr(ollama_backend, "daemon_reachable", not_running)
    backend = ollama_backend.OllamaBackend()
    t0 = time.monotonic()
    caps = await backend.get_capabilities("gemma4:12b")
    assert started == [] and time.monotonic() - t0 < 1
    assert caps.chat is True
    # Sin respuesta de Ollama no se recuerda la heurística: se pregunta otra vez.
    assert "gemma4:12b" not in backend._caps_cache


def test_el_panel_remoto_puede_leer_la_cpu_del_nodo() -> None:
    from ocabra.core.federation_tunnel import path_allowed

    assert path_allowed("/ocabra/host/stats", "full", "GET")
    assert not path_allowed("/ocabra/host/stats", "full", "POST")
    assert not path_allowed("/ocabra/host/stats", "inference", "GET")


def test_las_peticiones_forzadas_a_otro_nodo_son_de_otra_ejecucion() -> None:
    from ocabra.benchmarks.manager import belongs_to_run

    def row(route, node):
        return SimpleNamespace(federation_route=route, remote_node_name=node)

    pod = "ocabra-bench-bfcl-df2426"
    # IFEval en el pod de BFCL mientras Fulcrum mide el mismo modelo en el suyo.
    assert belongs_to_run(row("forced", pod), pod)
    assert not belongs_to_run(row("forced", "ocabra-bench-fulcrum-af0679"), pod)
    # Sin forzar (el juez de otra suite, servido aquí): no es de esta medida.
    assert not belongs_to_run(row("no_peer_has_model", None), pod)
    # Una medida local no ve las de los pods, sí las suyas.
    assert not belongs_to_run(row("forced", "Mapache"), None)
    assert belongs_to_run(row(None, None), None)
    assert belongs_to_run(row("no_peer_has_model", None), None)


@pytest.mark.asyncio
async def test_la_bateria_no_baja_mas_modelos_de_los_que_caben_a_la_vez(monkeypatch) -> None:
    from ocabra.core.cloud import preload

    started: list[str] = []

    async def drive_item(fm, peer, item, tasks):
        started.append(item["model_id"])
        item["status"] = "downloading"

    monkeypatch.setattr(preload, "_drive_item", drive_item)
    items = [
        {"model_id": "a", "status": "loaded", "benchmark_prefetch": True},
        {"model_id": "b", "status": "ready", "benchmark_prefetch": True},
        {"model_id": "c", "status": "pending", "benchmark_prefetch": True},
        {"model_id": "d", "status": "pending", "benchmark_prefetch": True},
    ]
    out = await preload.drive(None, SimpleNamespace(peer_id="p"), items, {})
    assert started == ["c"]  # el medido + 2 en disco: d espera
    # Al borrarse el ya medido entra el siguiente.
    out[0]["status"] = "removed"
    started.clear()
    await preload.drive(None, SimpleNamespace(peer_id="p"), out, {})
    assert started == ["c", "d"]


@pytest.mark.asyncio
async def test_si_el_disco_se_llena_a_mitad_la_descarga_espera() -> None:
    import httpx

    from ocabra.core.cloud import preload

    item = {
        "model_id": "llamacpp_mks/m",
        "status": "downloading",
        "benchmark_prefetch": True,
        "steps": [{"kind": "download", "body": {}}, {"kind": "load"}],
        "step": 0,
        "job_id": "j1",
        "node_model_id": "llamacpp_mks/m",
    }

    async def proxy_get(peer, path, headers=None, timeout=None):  # noqa: ASYNC109
        return httpx.Response(
            200, json={"status": "failed", "error": "[Errno 28] No space left on device"}
        )

    await preload._drive_item(
        SimpleNamespace(proxy_get=proxy_get), SimpleNamespace(peer_id="p"), item, {}
    )
    assert item["status"] == "pending" and item["waiting_disk"] is True
