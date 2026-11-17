"""Replicating our models onto cloud nodes (core/cloud/preload.py)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import httpx
import pytest
import sqlalchemy as sa

from ocabra.core.cloud.manager import CloudError
from ocabra.core.cloud.preload import (
    download_source,
    fit_llama_cpp_config,
    replication_request,
)

# ── rules ────────────────────────────────────────────────────────


def test_download_source_ollama_and_huggingface():
    assert download_source({"source": "ollama", "ref": "gemma4:26b", "kind": "gguf"}) == {
        "source": "ollama",
        "model_ref": "gemma4:26b",
    }
    assert download_source(
        {"ref": "org/Repo-GGUF::Model-Q4_K_M", "file": "Model-Q4_K_M.gguf", "kind": "gguf"}
    ) == {"source": "huggingface", "model_ref": "org/Repo-GGUF", "artifact": "Model-Q4_K_M.gguf"}


def test_local_only_files_are_not_replicable():
    with pytest.raises(CloudError) as exc:
        download_source({"ref": "x:iq1_m", "kind": "gguf", "path": "/data/models/local.gguf"})
    assert exc.value.status == 400


def test_our_3090_reservation_is_fitted_to_a_4090():
    """The real case: gemma4:26b reserves 23600 MB, a 4090 reports 24564 MB."""
    ours = {
        "ctx_size": 262144,
        "vram_estimate_mb": 23600,
        "disabled_gpus": [0],
        "main_gpu": None,
        "tensor_split": None,
        "cache_type_k": "q8_0",
        "chat_template": "{# t #}",
    }
    fitted = fit_llama_cpp_config(ours, 24564)
    assert fitted["vram_estimate_mb"] == 24564 - 1500
    assert fitted["ctx_size"] == 131072
    assert "disabled_gpus" not in fitted and "main_gpu" not in fitted
    assert fitted["cache_type_k"] == "q8_0" and fitted["chat_template"] == "{# t #}"
    # A big GPU keeps ours untouched (minus host keys).
    assert fit_llama_cpp_config(ours, 81920)["ctx_size"] == 262144
    assert ours["disabled_gpus"] == [0]  # input not mutated


def test_only_llama_cpp_family_for_now():
    row = SimpleNamespace(
        backend_type="whisper",
        artifact={"source": "ollama", "ref": "x"},
        model_key="x",
        model_id="whisper/x",
        extra_config={},
    )
    with pytest.raises(CloudError):
        replication_request(row, None)


def test_replication_request_keeps_our_id():
    row = SimpleNamespace(
        backend_type="llamacpp_mks",
        artifact={"source": "ollama", "ref": "gemma4:26b"},
        model_key="gemma4:26b",
        model_id="llamacpp_mks/gemma4:26b",
        extra_config={"llama_cpp": {"ctx_size": 8192}},
    )
    node_id, body = replication_request(row, 24564)
    assert node_id == "llamacpp_mks/gemma4:26b"
    assert body["register_config"] == {
        "backend_type": "llamacpp_mks",
        "model_key": "gemma4:26b",
        "extra_config": {"llama_cpp": {"ctx_size": 8192}},
    }


# ── full sequence over Postgres with a simulated node ────────────

from ocabra.database import AsyncSessionLocal  # noqa: E402
from tests.conftest import requires_db  # noqa: E402
from tests.test_cloud_nodes import _node, hub  # noqa: E402,F401

KEY = "zz-test-preload:1b"
MID = f"llamacpp_mks/{KEY}"


@pytest.fixture
async def our_model():
    from ocabra.db.model_config import LogicalModel, ModelConfig

    async def _clean():
        async with AsyncSessionLocal() as s:
            await s.execute(sa.delete(ModelConfig).where(ModelConfig.model_key == KEY))
            await s.execute(sa.delete(LogicalModel).where(LogicalModel.model_key == KEY))
            await s.commit()

    await _clean()
    async with AsyncSessionLocal() as s:
        s.add(LogicalModel(model_key=KEY, default_backend="llamacpp_mks"))
        await s.flush()
        s.add(
            ModelConfig(
                model_id=MID,
                model_key=KEY,
                backend_type="llamacpp_mks",
                artifact={"source": "ollama", "ref": "zz-test-preload:1b", "kind": "gguf"},
                extra_config={"llama_cpp": {"ctx_size": 8192, "disabled_gpus": [0]}},
            )
        )
        await s.commit()
    yield
    await _clean()


@requires_db
async def test_launch_rejects_unknown_models_before_renting(hub):  # noqa: F811
    fm, fake = hub
    with pytest.raises(CloudError) as exc:
        await fm.cloud.launch(gpu_type_id="3090", models=["llamacpp_mks/does-not-exist"])
    assert exc.value.status == 400
    assert fake.created == []  # nothing rented


@requires_db
async def test_preload_downloads_then_loads_on_the_node(hub, our_model):  # noqa: F811
    fm, fake = hub
    node = await fm.cloud.launch(gpu_type_id="3090", models=[MID])
    assert node["preload_models"] == [{"model_id": MID, "status": "pending", "load": True}]
    fake.pods[node["provider_pod_id"]].status = "running"
    spec = fake.created[0]
    await fm.join_service.create_request(
        {
            "node_id": "pod",
            "node_name": spec.name,
            "invite_code": spec.env["FEDERATION_JOIN_INVITE"],
        },
        source_ip="1.2.3.4",
    )
    await fm.cloud.reconcile()  # links the peer
    linked = await _node(fm, node["id"])
    peer = next(p for p in fm.get_all_peers() if p.peer_id == linked["peer_id"])
    peer.online = True
    peer.gpus = [{"index": 0, "name": "RTX 4090", "total_vram_mb": 24564, "free_vram_mb": 24000}]
    peer.last_proxied_at = time.time()

    calls = []
    job = {"status": "downloading", "progress_pct": 40.0}

    async def proxy_request(p, path, body, headers, timeout=None):  # noqa: ASYNC109 — mirrors the real signature
        calls.append(("POST", path, body))
        if path == "/ocabra/downloads":
            return httpx.Response(200, json={"job_id": "j1", "status": "queued"})
        return httpx.Response(200, json={"model_id": MID, "status": "loaded"})

    async def proxy_get(p, path, headers=None, timeout=None):  # noqa: ASYNC109 — mirrors the real signature
        calls.append(("GET", path, None))
        return httpx.Response(200, json={"job_id": "j1", **job})

    fm.proxy_request = proxy_request
    fm.proxy_get = proxy_get

    await fm.cloud.reconcile()  # pending → downloading
    item = (await _node(fm, node["id"]))["preload_models"][0]
    assert item["status"] == "downloading" and item["job_id"] == "j1"
    body = calls[0][2]
    assert body["source"] == "ollama" and body["model_ref"] == "zz-test-preload:1b"
    assert body["register_config"]["extra_config"] == {"llama_cpp": {"ctx_size": 8192}}

    await fm.cloud.reconcile()  # still downloading: progress
    assert (await _node(fm, node["id"]))["preload_models"][0]["progress"] == 40.0

    job.update(status="completed", progress_pct=100.0)
    await fm.cloud.reconcile()  # → loading (load runs in the background)
    assert (await _node(fm, node["id"]))["preload_models"][0]["status"] == "loading"
    import asyncio

    await asyncio.sleep(0)
    await fm.cloud.reconcile()  # load finished → loaded
    assert (await _node(fm, node["id"]))["preload_models"][0]["status"] == "loaded"
    assert ("POST", f"/ocabra/models/{MID}/load", {}) in calls


@requires_db
async def test_preload_failure_is_reported_not_fatal(hub, our_model):  # noqa: F811
    fm, fake = hub
    node = await fm.cloud.launch(gpu_type_id="3090", models=[MID])
    fake.pods[node["provider_pod_id"]].status = "running"
    spec = fake.created[0]
    await fm.join_service.create_request(
        {
            "node_id": "pod",
            "node_name": spec.name,
            "invite_code": spec.env["FEDERATION_JOIN_INVITE"],
        },
        source_ip="1.2.3.4",
    )
    await fm.cloud.reconcile()
    peer = next(p for p in fm.get_all_peers() if p.name == spec.name)
    peer.online = True
    peer.last_proxied_at = time.time()

    async def proxy_request(p, path, body, headers, timeout=None):  # noqa: ASYNC109 — mirrors the real signature
        return httpx.Response(500, text="disk full")

    fm.proxy_request = proxy_request
    await fm.cloud.reconcile()
    after = await _node(fm, node["id"])
    assert after["status"] != "terminated"
    assert after["preload_models"][0]["status"] == "error"
    assert "disk full" in after["preload_models"][0]["error"]


async def test_models_storage_works_before_the_models_dir_exists(monkeypatch, tmp_path):
    """A fresh node has no /data/models yet: the panel showed an error."""
    from ocabra.api.internal import models as api
    from ocabra.config import settings

    monkeypatch.setattr(settings, "models_dir", str(tmp_path / "not" / "yet"))
    out = await api.get_models_storage(_user=None)
    assert out["path"].endswith("not/yet") and out["total_bytes"] > 0


@requires_db
async def test_replicable_models_marks_local_only_files(our_model):
    from ocabra.core.cloud.preload import replicable_models
    from ocabra.db.model_config import LogicalModel, ModelConfig

    local_key = "zz-test-preload-local:1b"
    async with AsyncSessionLocal() as s:
        s.add(LogicalModel(model_key=local_key, default_backend="llamacpp_mks"))
        await s.flush()
        s.add(
            ModelConfig(
                model_id=f"llamacpp_mks/{local_key}",
                model_key=local_key,
                backend_type="llamacpp_mks",
                artifact={"ref": "x", "kind": "gguf", "path": "/data/models/only-here.gguf"},
            )
        )
        await s.commit()
    try:
        listed = {m["model_id"]: m for m in await replicable_models(AsyncSessionLocal)}
        assert listed[MID]["replicable"] is True and listed[MID]["reason"] is None
        local = listed[f"llamacpp_mks/{local_key}"]
        assert local["replicable"] is False and "solo existen en este servidor" in local["reason"]
        ids = [m for m in listed]
        assert ids.index(MID) < ids.index(f"llamacpp_mks/{local_key}")  # replicable first
    finally:
        async with AsyncSessionLocal() as s:
            await s.execute(sa.delete(ModelConfig).where(ModelConfig.model_key == local_key))
            await s.execute(sa.delete(LogicalModel).where(LogicalModel.model_key == local_key))
            await s.commit()


@requires_db
async def test_models_can_be_added_to_a_running_node(hub, our_model):  # noqa: F811
    fm, fake = hub
    node = await fm.cloud.launch(gpu_type_id="3090")
    assert node["preload_models"] == []
    out = await fm.cloud.add_models(node["id"], [MID])
    assert out["preload_models"] == [{"model_id": MID, "status": "pending", "load": True}]
    again = await fm.cloud.add_models(node["id"], [MID])  # no duplicates
    assert len(again["preload_models"]) == 1
    with pytest.raises(CloudError):
        await fm.cloud.add_models(node["id"], ["llamacpp_mks/nope"])
    await fm.cloud.terminate(node["id"])
    with pytest.raises(CloudError) as exc:
        await fm.cloud.add_models(node["id"], [MID])
    assert exc.value.status == 409


def test_profile_with_parked_backend_capacity_serializes():
    """Deleting a model's llama.cpp deployment parks its validated capacity
    under runtime_policy.validated_by_backend; the schema rejected that key and
    GET /ocabra/models answered 500 for every model."""
    from ocabra.core.model_deployments import swap_validated_capacity
    from ocabra.schemas.profiles import RuntimePolicy

    policy = swap_validated_capacity(
        {"validated_context_tokens": {"normal": 131072, "dual-gpu": 40960}},
        "llamacpp_mks",
        "ollama",
    )
    parsed = RuntimePolicy.model_validate(policy)
    assert (
        parsed.validated_by_backend["llamacpp_mks"]["validated_context_tokens"]["normal"] == 131072
    )
    # Round-trip keeps it, so switching back can restore it.
    back = swap_validated_capacity(parsed.model_dump(), "ollama", "llamacpp_mks")
    assert back["validated_context_tokens"] == {"normal": 131072, "dual-gpu": 40960}


# ── diffusers: rebuilt on the node from the same Hugging Face files ──


def _fake_hf_cache(tmp_path):
    """A models dir laid out like ours: the tree is a symlink into a HF cache
    snapshot whose files are symlinks into ``blobs/``, and the GGUF lives in
    another repo's snapshot."""
    import os

    cache = tmp_path / "hf_cache" / "hub"
    blobs = cache / "models--Qwen--Qwen-Image-2.1" / "blobs"
    snap = cache / "models--Qwen--Qwen-Image-2.1" / "snapshots" / "rev1"
    blobs.mkdir(parents=True)
    for rel, size in {
        "model_index.json": 10,
        "vae/w.safetensors": 300,
        "transformer/config.json": 5,
    }.items():
        blob = blobs / rel.replace("/", "_")
        blob.write_bytes(b"x" * size)
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(blob, snap / rel)
    g_blobs = cache / "models--Abiray--Qwen-Image-2.1-GGUF" / "blobs"
    g_snap = cache / "models--Abiray--Qwen-Image-2.1-GGUF" / "snapshots" / "rev2"
    g_blobs.mkdir(parents=True)
    g_snap.mkdir(parents=True)
    (g_blobs / "sha").write_bytes(b"g" * 1000)
    os.symlink(g_blobs / "sha", g_snap / "q5.gguf")
    models = tmp_path / "models"
    models.mkdir()
    os.symlink(snap, models / "qwen-image-2.1")
    return str(models), str(g_snap / "q5.gguf")


def _diffusers_row(gguf):
    return SimpleNamespace(
        backend_type="diffusers",
        model_id="diffusers/qwen-image-2.1",
        model_key="qwen-image-2.1",
        display_name="Qwen-Image 2.1",
        artifact={"ref": "qwen-image-2.1", "kind": "diffusers_tree", "source": "local"},
        extra_config={
            "diffusers": {
                "pipeline_class": "QwenImage21Pipeline",
                "text_encoder_quant": "nf4",
                "gguf_transformer_path": gguf,
            },
            "preferred_gpus": [1],
        },
    )


def test_diffusers_plan_copies_exactly_our_files(tmp_path):
    from ocabra.core.cloud.preload import plan_replication

    models_dir, gguf = _fake_hf_cache(tmp_path)
    plan = plan_replication(_diffusers_row(gguf), 24564, models_dir)
    kinds = [s["kind"] for s in plan["steps"]]
    assert kinds == ["backend", "download", "download", "register", "load"]
    base, weights = plan["steps"][1]["body"], plan["steps"][2]["body"]
    # The pipeline repo minus the transformer weights we never downloaded.
    assert base == {
        "source": "huggingface",
        "model_ref": "Qwen/Qwen-Image-2.1",
        "revision": "rev1",
        "allow_patterns": ["model_index.json", "transformer/config.json", "vae/w.safetensors"],
        "local_dir": "qwen-image-2.1",
        "register": False,
    }
    assert weights["model_ref"] == "Abiray/Qwen-Image-2.1-GGUF"
    assert weights["allow_patterns"] == ["q5.gguf"] and weights["revision"] == "rev2"
    reg = plan["steps"][3]["body"]
    assert reg["model_id"] == "diffusers/qwen-image-2.1"
    # Relative to the node's models dir; our GPU choice stays here.
    cfg = reg["extra_config"]
    assert (
        cfg["diffusers"]["gguf_transformer_path"]
        == "huggingface/Abiray--Qwen-Image-2.1-GGUF/q5.gguf"
    )
    assert "preferred_gpus" not in cfg and cfg["diffusers"]["text_encoder_quant"] == "nf4"
    assert plan["size_bytes"] == 10 + 300 + 5 + 1000


def test_diffusers_with_files_only_here_is_not_replicable(tmp_path):
    from ocabra.core.cloud.preload import plan_replication

    models_dir, gguf = _fake_hf_cache(tmp_path)
    local = tmp_path / "mine.gguf"
    local.write_bytes(b"1")
    with pytest.raises(CloudError) as exc:
        plan_replication(_diffusers_row(str(local)), None, models_dir)
    assert "GGUF" in str(exc.value)
    plain = tmp_path / "models" / "plain"
    plain.mkdir()
    row = _diffusers_row(gguf)
    row.artifact = {"ref": "plain", "kind": "diffusers_tree", "source": "local"}
    with pytest.raises(CloudError):
        plan_replication(row, None, models_dir)


def test_relative_gguf_path_resolves_on_the_node(monkeypatch, tmp_path):
    from ocabra.backends.diffusers_backend import _models_path
    from ocabra.config import settings

    monkeypatch.setattr(settings, "models_dir", str(tmp_path))
    assert _models_path("huggingface/x/q.gguf") == tmp_path / "huggingface/x/q.gguf"
    assert str(_models_path("/abs/q.gguf")) == "/abs/q.gguf"


class _Resp:
    """Streamed response stand-in for ``proxy_raw``."""

    status_code = 200

    async def aiter_bytes(self):
        yield b'data: {"install_status": "installing"}\n\n'

    async def aclose(self):
        self.closed = True


async def test_drive_walks_install_downloads_register_load(tmp_path, monkeypatch):
    import asyncio

    from ocabra.core.cloud import preload

    models_dir, gguf = _fake_hf_cache(tmp_path)
    row = _diffusers_row(gguf)

    async def rows(_sf, ids):
        return {row.model_id: row}

    monkeypatch.setattr(preload, "load_model_rows", rows)
    real_plan = preload.plan_replication
    monkeypatch.setattr(preload, "plan_replication", lambda r, g: real_plan(r, g, models_dir))

    node = {"backend": "not_installed", "jobs": {}, "registered": False}
    calls = []

    async def proxy_get(peer, path, headers=None, timeout=None):  # noqa: ASYNC109
        calls.append(("GET", path))
        if path == "/ocabra/backends/diffusers":
            return httpx.Response(200, json={"install_status": node["backend"]})
        if path.startswith("/ocabra/downloads/"):
            return httpx.Response(200, json=node["jobs"][path.rsplit("/", 1)[1]])
        if path == "/ocabra/models/diffusers/qwen-image-2.1":
            return httpx.Response(200 if node["registered"] else 404, json={})
        raise AssertionError(path)

    async def proxy_raw(peer, method, path, content, headers=None):
        calls.append((method, path))
        node["backend"] = "installing"
        return _Resp()

    async def proxy_request(peer, path, body, headers, timeout=None):  # noqa: ASYNC109
        calls.append(("POST", path))
        if path == "/ocabra/downloads":
            jid = f"j{len(node['jobs'])}"
            node["jobs"][jid] = {"status": "downloading", "progress_pct": 10}
            return httpx.Response(200, json={"job_id": jid})
        if path == "/ocabra/models":
            node["registered"] = True
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"status": "loaded"})

    fm = SimpleNamespace(
        session_factory=None, proxy_get=proxy_get, proxy_raw=proxy_raw, proxy_request=proxy_request
    )
    session = SimpleNamespace(remote_info={"version": "0.87.0"})
    peer = SimpleNamespace(peer_id="p", gpus=[], tunnel_session=session)
    tasks: dict = {}
    items = [{"model_id": row.model_id, "status": "pending"}]

    items = await preload.drive(fm, peer, items, tasks)
    assert items[0]["status"] == "installing"
    items = await preload.drive(fm, peer, items, tasks)
    assert items[0]["status"] == "installing"  # still going
    node["backend"] = "installed"
    items = await preload.drive(fm, peer, items, tasks)
    assert items[0]["status"] == "downloading" and items[0]["job_id"] == "j0"
    node["jobs"]["j0"] = {"status": "completed", "progress_pct": 100}
    items = await preload.drive(fm, peer, items, tasks)
    assert items[0]["job_id"] == "j1" and items[0]["step"] == 2
    node["jobs"]["j1"] = {"status": "completed", "progress_pct": 100}
    items = await preload.drive(fm, peer, items, tasks)  # register (instant) → load
    assert items[0]["status"] == "loading" and node["registered"]
    await asyncio.sleep(0)
    items = await preload.drive(fm, peer, items, tasks)
    assert items[0]["status"] == "loaded", items[0]
    assert ("POST", "/ocabra/models/diffusers/qwen-image-2.1/load") in calls


async def test_drive_refuses_diffusers_on_an_old_node(tmp_path, monkeypatch):
    from ocabra.core.cloud import preload

    models_dir, gguf = _fake_hf_cache(tmp_path)
    row = _diffusers_row(gguf)

    async def rows(_sf, ids):
        return {row.model_id: row}

    monkeypatch.setattr(preload, "load_model_rows", rows)
    real_plan = preload.plan_replication
    monkeypatch.setattr(preload, "plan_replication", lambda r, g: real_plan(r, g, models_dir))
    fm = SimpleNamespace(session_factory=None)
    peer = SimpleNamespace(
        peer_id="p", gpus=[], tunnel_session=SimpleNamespace(remote_info={"version": "0.86.1"})
    )
    items = await preload.drive(fm, peer, [{"model_id": row.model_id, "status": "pending"}], {})
    assert items[0]["status"] == "error" and "0.86.1" in items[0]["error"]


def test_exact_files_download_stays_inside_the_models_dir(monkeypatch, tmp_path):
    from ocabra.api.internal.downloads import _safe_local_dir
    from ocabra.config import settings

    monkeypatch.setattr(settings, "models_dir", str(tmp_path))
    assert _safe_local_dir("huggingface/a--b") == tmp_path / "huggingface/a--b"
    for bad in ("../etc", "/etc", "", "a/../../x", "."):
        with pytest.raises(ValueError):
            _safe_local_dir(bad)


async def test_model_loaded_by_hand_meanwhile_counts_as_loaded():
    import asyncio

    from ocabra.core.cloud import preload

    async def already():
        return httpx.Response(409, json={"detail": "Model is already loaded"})

    task = asyncio.ensure_future(already())
    await task
    item = {
        "model_id": "m",
        "status": "loading",
        "node_model_id": "m",
        "steps": [{"kind": "load"}],
        "step": 0,
    }
    assert await preload._poll_step(
        SimpleNamespace(), SimpleNamespace(), item, {("p", "m"): task}, ("p", "m")
    )


# ── recipes: models captured from pods (core/cloud/recipes.py) ───

NEMO = "llamacpp_mks/nvidia-nemotron-3.5-super-vl-120b-a12b-gguf"
NEMO_JOB = {
    "job_id": "j1",
    "status": "completed",
    "source": "huggingface",
    "model_ref": "ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF",
    "artifact": "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf",
    "register": True,
    "register_config": {
        "display_name": "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF",
        "load_policy": "on_demand",
        "backend_type": "llamacpp_mks",
        "model_key": "nvidia-nemotron-3.5-super-vl-120b-a12b-gguf",
        "placement": {"gpus": [0], "cpu": False},
    },
}
NEMO_MODEL = {
    "model_id": NEMO,
    "backend_type": "llamacpp_mks",
    "display_name": "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF",
    "load_policy": "pin",
    "disk_size_bytes": 76918795904,
    "last_request_at": "2026-11-17T02:16:36+00:00",
    "extra_config": {
        "llama_cpp": {
            "ctx_size": 262144,
            "parallel_slots": 4,
            "model_path": "/data/models/huggingface/x/NVIDIA-NVFP4.gguf",
            "tensor_split": [1],
        },
        "_runtime": {"preset_key": "normal"},
        "registry_metadata": {"source": "huggingface"},
    },
}


def test_recipe_keeps_the_source_and_the_config_tuned_on_the_pod():
    """The 120B nemotron set up by hand on the PRO 6000: where it came from
    plus the context/slots/pin it was given, without that machine's paths."""
    from ocabra.core.cloud.recipes import recipe_from_node

    r = recipe_from_node(NEMO_JOB, NEMO_MODEL)
    assert r["node_model_id"] == NEMO
    download, load = r["steps"]
    assert load == {"kind": "load"}
    body = download["body"]
    assert body["model_ref"] == NEMO_JOB["model_ref"] and body["artifact"] == NEMO_JOB["artifact"]
    rc = body["register_config"]
    assert rc["load_policy"] == "pin" and "placement" not in rc
    assert rc["extra_config"] == {"llama_cpp": {"ctx_size": 262144, "parallel_slots": 4}}
    assert r["source"].startswith("huggingface: ggml-org/")


async def test_drive_runs_a_recipe_and_can_download_without_loading():
    """A recipe item uses its own steps (no model of ours behind); with
    ``load: False`` it ends «ready», downloaded but not loaded."""
    from ocabra.core.cloud import preload
    from ocabra.core.cloud.recipes import recipe_from_node

    steps = recipe_from_node(NEMO_JOB, NEMO_MODEL)["steps"]
    jobs: dict = {}
    calls = []

    async def proxy_get(peer, path, headers=None, timeout=None):  # noqa: ASYNC109
        return httpx.Response(200, json=jobs[path.rsplit("/", 1)[1]])

    async def proxy_request(peer, path, body, headers, timeout=None):  # noqa: ASYNC109
        calls.append((path, body))
        jobs["j0"] = {"status": "completed", "progress_pct": 100}
        return httpx.Response(200, json={"job_id": "j0"})

    async def no_rows(*_a):  # a recipe never looks for one of our models
        raise AssertionError("looked up our models")

    fm = SimpleNamespace(session_factory=None, proxy_get=proxy_get, proxy_request=proxy_request)
    peer = SimpleNamespace(peer_id="p", gpus=[], tunnel_session=None)
    item = {"model_id": NEMO, "status": "pending", "recipe_steps": steps, "load": False}
    mp = pytest.MonkeyPatch()
    mp.setattr(preload, "load_model_rows", no_rows)
    try:
        items = await preload.drive(fm, peer, [item], {})
        assert items[0]["status"] == "downloading"
        items = await preload.drive(fm, peer, items, {})
    finally:
        mp.undo()
    assert items[0]["status"] == "ready", items[0]
    assert [p for p, _ in calls] == ["/ocabra/downloads"]  # never /load
    assert calls[0][1]["register_config"]["extra_config"]["llama_cpp"]["ctx_size"] == 262144


async def test_snapshot_reads_the_pod_and_skips_our_own_models(monkeypatch):
    from ocabra.core.cloud import recipes

    async def proxy_get(peer, path, headers=None, timeout=None):  # noqa: ASYNC109
        if path == "/ocabra/downloads":
            other = {**NEMO_JOB, "status": "failed"}
            return httpx.Response(200, json=[NEMO_JOB, other])
        if path.startswith("/ocabra/models/"):
            return httpx.Response(200, json=NEMO_MODEL)
        raise AssertionError(path)

    got = await recipes._node_recipes(SimpleNamespace(proxy_get=proxy_get), SimpleNamespace())
    assert [r["node_model_id"] for r in got] == [NEMO]


async def test_relaunch_and_image_update_items(monkeypatch):
    """_preload_items: backends first (chosen + what the models need), then
    ours + recipes, «solo descargar» per id, no duplicates."""
    from ocabra.core.cloud import preload
    from ocabra.core.cloud import recipes as recipes_mod
    from ocabra.core.cloud.manager import CloudNodeManager

    row = SimpleNamespace(
        id="r1", node_model_id=NEMO, steps=[{"kind": "load"}], backend_type="llamacpp_mks"
    )

    async def load_recipes(_sf, ids):
        return {"r1": row} if "r1" in ids else {}

    async def load_rows(_sf, ids):
        return {i: SimpleNamespace(backend_type="llamacpp_mks") for i in ids}

    monkeypatch.setattr(recipes_mod, "load_recipes", load_recipes)
    monkeypatch.setattr(preload, "load_model_rows", load_rows)
    mgr = object.__new__(CloudNodeManager)
    mgr.fm = SimpleNamespace(session_factory=None)
    rows_backend = {"llamacpp_mks/gemma4:26b": "llamacpp_mks"}

    async def load_rows2(_sf, ids):
        return {i: SimpleNamespace(backend_type=rows_backend.get(i, "diffusers")) for i in ids}

    monkeypatch.setattr(preload, "load_model_rows", load_rows2)
    items = await mgr._preload_items(
        ["llamacpp_mks/gemma4:26b", "diffusers/qwen-image"],
        ["r1"],
        ["r1"],
        ["whisper", "ollama", "whisper"],
    )
    backends = [i for i in items if i.get("kind") == "backend"]
    # Chosen first, then what the models need (diffusers); once each; never
    # Ollama nor llama.cpp, which the node image already has.
    assert [b["model_id"] for b in backends] == ["backend:whisper", "backend:diffusers"]
    assert items[: len(backends)] == backends
    models = items[len(backends) :]
    assert models[0] == {"model_id": "llamacpp_mks/gemma4:26b", "status": "pending", "load": True}
    assert models[2]["recipe_id"] == "r1" and models[2]["load"] is False
    with pytest.raises(CloudError):
        await mgr._preload_items([], ["gone"], [])


async def test_a_backend_item_only_installs_and_ends_ready():
    from ocabra.core.cloud import preload
    from ocabra.core.cloud.recipes import backend_item

    state = {"s": "not_installed"}
    calls = []

    class _Stream:
        status_code = 200

        async def aiter_bytes(self):
            yield b"data: {}\n\n"

        async def aclose(self):
            pass

    async def proxy_get(peer, path, headers=None, timeout=None):  # noqa: ASYNC109
        return httpx.Response(200, json={"install_status": state["s"]})

    async def proxy_raw(peer, method, path, content, headers=None):
        calls.append(path)
        state["s"] = "installing"
        return _Stream()

    fm = SimpleNamespace(session_factory=None, proxy_get=proxy_get, proxy_raw=proxy_raw)
    peer = SimpleNamespace(peer_id="p", gpus=[], tunnel_session=None)
    items = await preload.drive(fm, peer, [backend_item("whisper")], {})
    assert items[0]["status"] == "installing" and calls == ["/ocabra/backends/whisper/install"]
    state["s"] = "installed"
    items = await preload.drive(fm, peer, items, {})
    assert items[0]["status"] == "ready"


@requires_db
async def test_snapshot_stores_and_updates_one_recipe_per_model():
    from ocabra.core.cloud import recipes
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.cloud import CloudModelRecipe

    async def proxy_get(peer, path, headers=None, timeout=None):  # noqa: ASYNC109
        if path == "/ocabra/downloads":
            return httpx.Response(200, json=[NEMO_JOB])
        return httpx.Response(200, json=NEMO_MODEL)

    fm = SimpleNamespace(session_factory=AsyncSessionLocal, proxy_get=proxy_get)
    try:
        assert (
            await recipes.snapshot(fm, SimpleNamespace(), node_name="pod-a", gpu_type_id="PRO 6000")
            == 1
        )
        assert (
            await recipes.snapshot(fm, SimpleNamespace(), node_name="pod-b", gpu_type_id="PRO 6000")
            == 1
        )
        listed = [
            r for r in await recipes.list_recipes(AsyncSessionLocal) if r["node_model_id"] == NEMO
        ]
        assert len(listed) == 1
        assert listed[0]["last_node_name"] == "pod-b" and listed[0]["context"] == 262144
        assert listed[0]["last_used_at"] is not None
    finally:
        async with AsyncSessionLocal() as s:
            await s.execute(
                sa.delete(CloudModelRecipe).where(CloudModelRecipe.node_model_id == NEMO)
            )
            await s.commit()


def test_recipe_from_search_hf_gguf_and_ollama_tag():
    from ocabra.core.cloud.recipes import new_recipe

    hf = new_recipe(
        source="huggingface",
        model_ref="ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF",
        artifact="NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf",
        display_name=None,
        size_bytes=77 * 1024**3,
        context=262144,
        parallel=4,
        load_policy="pin",
    )
    # One repo, several quantizations: the file names the model.
    assert hf["node_model_id"] == "llamacpp_mks/nvidia-nemotron-3.5-super-vl-120b-a12b-nvfp4"
    body = hf["steps"][0]["body"]
    assert body["artifact"].endswith("NVFP4.gguf") and body["register"] is True
    rc = body["register_config"]
    assert rc["backend_type"] == "llamacpp_mks" and rc["load_policy"] == "pin"
    assert rc["extra_config"] == {"llama_cpp": {"ctx_size": 262144, "parallel_slots": 4}}
    assert hf["steps"][1] == {"kind": "load"}

    ol = new_recipe(
        source="ollama",
        model_ref="qwen3:8b",
        artifact=None,
        display_name=None,
        size_bytes=None,
        context=None,
        parallel=None,
        load_policy="on_demand",
    )
    assert ol["node_model_id"] == "llamacpp_mks/qwen3:8b"
    assert ol["steps"][0]["body"] == {
        "source": "ollama",
        "model_ref": "qwen3:8b",
        "register": True,
        "register_config": {
            "backend_type": "llamacpp_mks",
            "model_key": "qwen3:8b",
            "extra_config": {},
            "load_policy": "on_demand",
            "display_name": "qwen3:8b",
        },
    }


def test_editing_a_recipe_drops_the_stale_reservation():
    from ocabra.core.cloud.recipes import _apply_config

    rc = {"extra_config": {"llama_cpp": {"ctx_size": 65536, "vram_estimate_mb": 24576}}}
    _apply_config(rc, context=131072, load_policy="pin")
    assert rc["extra_config"]["llama_cpp"] == {"ctx_size": 131072}
    assert rc["load_policy"] == "pin"
    _apply_config(rc, context=0)  # back to the node's default
    assert rc["extra_config"] == {}


def test_a_hf_recipe_needs_a_gguf_file():
    from pydantic import ValidationError

    from ocabra.schemas.cloud import CloudRecipeCreate

    with pytest.raises(ValidationError):
        CloudRecipeCreate(source="huggingface", model_ref="org/repo")
    CloudRecipeCreate(source="huggingface", model_ref="org/repo", artifact="m-Q4_K_M.gguf")
    CloudRecipeCreate(source="ollama", model_ref="qwen3:8b")


@requires_db
async def test_create_then_edit_a_recipe():
    from ocabra.core.cloud import recipes
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.cloud import CloudModelRecipe

    try:
        made = await recipes.create_recipe(
            AsyncSessionLocal,
            source="ollama",
            model_ref="zz-test:1b",
            artifact=None,
            display_name=None,
            size_bytes=None,
            context=32768,
            parallel=2,
            load_policy="on_demand",
        )
        assert made["context"] == 32768 and made["parallel"] == 2
        out = await recipes.update_recipe(
            AsyncSessionLocal, made["id"], context=65536, load_policy="pin"
        )
        assert out["context"] == 65536 and out["parallel"] == 2 and out["load_policy"] == "pin"
        assert (
            await recipes.update_recipe(AsyncSessionLocal, "00000000-0000-0000-0000-000000000000")
            is None
        )
    finally:
        async with AsyncSessionLocal() as s:
            await s.execute(
                sa.delete(CloudModelRecipe).where(
                    CloudModelRecipe.node_model_id == "llamacpp_mks/zz-test:1b"
                )
            )
            await s.commit()


def test_recipe_for_another_backend_registers_the_whole_repo():
    """A Whisper (or diffusers, vLLM…) repo from HF: the node downloads the
    repo and registers it for that backend; context/slots don't apply."""
    from ocabra.core.cloud.recipes import new_recipe

    r = new_recipe(
        source="huggingface",
        model_ref="Systran/faster-whisper-large-v3",
        artifact=None,
        display_name=None,
        size_bytes=3 * 1024**3,
        context=65536,
        parallel=4,
        load_policy="on_demand",
        backend_type="whisper",
    )
    assert r["node_model_id"] == "whisper/faster-whisper-large-v3"
    rc = r["steps"][0]["body"]["register_config"]
    assert rc["backend_type"] == "whisper" and rc["extra_config"] == {}
    assert "artifact" not in r["steps"][0]["body"]


def test_recipe_backend_rules():
    from pydantic import ValidationError

    from ocabra.schemas.cloud import CloudRecipeCreate

    CloudRecipeCreate(source="huggingface", model_ref="org/whisper", backend_type="whisper")
    with pytest.raises(ValidationError):  # nodes have no Ollama daemon
        CloudRecipeCreate(source="ollama", model_ref="qwen3:8b", backend_type="ollama")
    with pytest.raises(ValidationError):  # an Ollama tag is a GGUF for llama.cpp
        CloudRecipeCreate(source="ollama", model_ref="qwen3:8b", backend_type="vllm")
