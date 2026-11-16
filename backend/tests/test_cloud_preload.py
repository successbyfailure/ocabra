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
    assert node["preload_models"] == [{"model_id": MID, "status": "pending"}]
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
