"""Fase 2/3: default-backend switch, add/remove deployment, grants, compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa

from ocabra.api._deps_auth import _expand_model_grants
from ocabra.core import artifacts
from ocabra.core import model_deployments as md
from ocabra.core.compatibility import (
    backends_for_artifact,
    backends_for_files,
    backends_for_ollama_tag,
    derive_artifact,
)
from ocabra.core.model_manager import LoadPolicy, ModelManager
from ocabra.core.profile_registry import ProfileRegistry
from ocabra.database import AsyncSessionLocal
from ocabra.db.model_config import LogicalModel, ModelConfig, ModelProfile, ModelProfileAlias
from tests.conftest import requires_db

KEY = "zz-test-switch:1b"
LLAMA = f"llamacpp_mks/{KEY}"
OLLAMA = f"ollama/{KEY}"


# ── pure ─────────────────────────────────────────────────────────────────────


def test_swap_validated_capacity_parks_old_and_restores_new():
    policy = {
        "selection": "prefer_loaded",
        "validated_vram_mb": {"normal": 23772},
        "validated_context_tokens": {"normal": 40960},
    }
    to_ollama = md.swap_validated_capacity(policy, "llamacpp_mks", "ollama")
    assert to_ollama["validated_vram_mb"] == {} and to_ollama["validated_context_tokens"] == {}
    assert to_ollama["selection"] == "prefer_loaded"
    assert to_ollama["validated_by_backend"]["llamacpp_mks"]["validated_vram_mb"] == {"normal": 23772}

    back = md.swap_validated_capacity(to_ollama, "ollama", "llamacpp_mks")
    assert back["validated_vram_mb"] == {"normal": 23772}
    assert back["validated_context_tokens"] == {"normal": 40960}
    assert "validated_by_backend" not in back  # nothing measured on ollama → nothing parked


def test_gguf_artifacts_are_shared_by_the_llama_cpp_builds_and_ollama_only_from_ollama():
    from_ollama = {"kind": "gguf", "source": "ollama", "ref": "gemma4:26b"}
    from_hf = {"kind": "gguf", "source": "hf", "path": "/m/x.gguf", "file": "x-Q4_K_M.gguf"}
    assert [e["backend"] for e in backends_for_artifact(from_ollama)] == ["llamacpp_mks", "llama_cpp", "ollama"]
    assert [e["backend"] for e in backends_for_artifact(from_hf)] == ["llamacpp_mks", "llama_cpp"]
    assert [e["backend"] for e in backends_for_artifact({"kind": "gguf", "ref": "Bonsai-27B-Q1_0"})] == [
        "bitnet"
    ]
    assert backends_for_artifact({"kind": "ctranslate2", "ref": "openai/whisper"}) == []


def test_derive_artifact_reads_the_ollama_tag_blob_with_optional_layers():
    derived = derive_artifact({"kind": "ollama_tag", "source": "ollama", "ref": "mistral:7b"}, "llamacpp_mks")
    assert derived["kind"] == "gguf" and derived["source"] == "ollama" and derived["ref"] == "mistral:7b"
    assert derived["mmproj"]["optional"] and derived["draft"]["optional"]
    assert derive_artifact({"kind": "gguf", "source": "ollama", "ref": "mistral:7b"}, "ollama") == {
        "kind": "ollama_tag",
        "source": "ollama",
        "ref": "mistral:7b",
    }
    with pytest.raises(ValueError):
        derive_artifact({"kind": "gguf", "source": "hf", "path": "/m/x.gguf"}, "ollama")


def test_optional_missing_layer_does_not_drop_the_model_path(tmp_path: Path):
    manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "mistral" / "7b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:m"}]}))
    art = derive_artifact({"kind": "ollama_tag", "source": "ollama", "ref": "mistral:7b"}, "llamacpp_mks")
    assert artifacts.resolve_paths(art, tmp_path) == {"model": str(tmp_path / "blobs" / "sha256-m")}


def test_search_icons_for_repos_and_tags():
    gguf_repo = backends_for_files(["README.md", "x-Q4_K_M.gguf", "mmproj-x.gguf"], pipeline_tag="text-generation")
    assert [e["backend"] for e in gguf_repo] == ["llamacpp_mks", "llama_cpp"]
    hf_repo = backends_for_files(["config.json", "model.safetensors"], pipeline_tag="text-generation", suggested="vllm")
    assert [e["backend"] for e in hf_repo] == ["vllm", "sglang"]
    both = backends_for_files(["model.safetensors", "x-Q8_0.gguf"], pipeline_tag="text-generation", suggested="vllm")
    assert [e["backend"] for e in both] == ["llamacpp_mks", "llama_cpp", "vllm", "sglang"]
    tts = backends_for_files(["kokoro.pth", "config.json"], pipeline_tag="text-to-speech", library_name="kokoro", suggested="tts")
    assert [e["backend"] for e in tts] == ["tts"]
    ternary = backends_for_files(["ggml-model-i2_s.gguf"])
    assert [e["backend"] for e in ternary] == ["bitnet"]
    installed = {"llamacpp_mks", "ollama"}
    tag = backends_for_ollama_tag(installed)
    assert [(e["backend"], e["installed"]) for e in tag] == [
        ("llamacpp_mks", True),
        ("llama_cpp", False),
        ("ollama", True),
    ]


# ── DB ───────────────────────────────────────────────────────────────────────


async def _cleanup() -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(sa.delete(ModelConfig).where(ModelConfig.model_key == KEY))
        await session.execute(sa.delete(LogicalModel).where(LogicalModel.model_key == KEY))
        await session.commit()


@pytest.fixture
async def model_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A model served by llamacpp_mks (default, pin) with an Ollama copy."""
    manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "zz-test-switch" / "1b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc"}]}))
    monkeypatch.setattr(artifacts, "ollama_root", lambda: str(tmp_path))
    await _cleanup()
    async with AsyncSessionLocal() as session:
        session.add(LogicalModel(model_key=KEY, default_backend="llamacpp_mks"))
        await session.flush()
        session.add_all(
            [
                ModelConfig(
                    model_id=LLAMA,
                    model_key=KEY,
                    backend_type="llamacpp_mks",
                    load_policy="pin",
                    artifact={"kind": "gguf", "source": "ollama", "ref": "zz-test-switch:1b"},
                ),
                ModelConfig(
                    model_id=OLLAMA,
                    model_key=KEY,
                    backend_type="ollama",
                    artifact={"kind": "ollama_tag", "source": "ollama", "ref": "zz-test-switch:1b"},
                ),
            ]
        )
        await session.flush()
        session.add_all(
            [
                ModelProfile(
                    profile_id=KEY,
                    base_model_id=LLAMA,
                    is_default=True,
                    runtime_policy={"validated_vram_mb": {"normal": 1000}},
                ),
                ModelProfile(profile_id=f"{KEY}-ollama", base_model_id=OLLAMA, is_default=True, backend="ollama"),
            ]
        )
        await session.flush()
        session.add(ModelProfileAlias(alias_id=f"{KEY}-old", profile_id=KEY))
        await session.commit()
    mm = ModelManager(worker_pool=None)
    await mm._load_configs_from_db()
    pr = ProfileRegistry()
    async with AsyncSessionLocal() as session:
        await pr.load_all(session)
    yield mm, pr
    await _cleanup()


async def _profile(profile_id: str) -> ModelProfile | None:
    async with AsyncSessionLocal() as session:
        return await session.get(ModelProfile, profile_id)


@requires_db
async def test_switch_default_backend_repoints_public_profile_and_moves_policy(model_pair):
    mm, pr = model_pair
    summary = await md.set_default_backend(KEY, "ollama", model_manager=mm, profile_registry=pr)

    assert summary["moved_profiles"] == [KEY]
    public = await _profile(KEY)
    assert public.base_model_id == OLLAMA
    assert public.runtime_policy["validated_vram_mb"] == {}
    assert public.runtime_policy["validated_by_backend"]["llamacpp_mks"]["validated_vram_mb"] == {"normal": 1000}
    assert (await pr.get(KEY)).base_model_id == OLLAMA  # cache refreshed
    assert mm._states[OLLAMA].load_policy == LoadPolicy.PIN
    assert mm._states[LLAMA].load_policy == LoadPolicy.ON_DEMAND
    assert mm.default_backend_of(KEY) == "ollama"
    # the old default stays reachable through a hidden pinned variant
    variant = await _profile(summary["pinned_variant"])
    assert variant.base_model_id == LLAMA and variant.backend == "llamacpp_mks"
    async with AsyncSessionLocal() as session:
        assert (await session.get(LogicalModel, KEY)).default_backend == "ollama"

    # switching back restores the measured capacity
    await md.set_default_backend(KEY, "llamacpp_mks", model_manager=mm, profile_registry=pr)
    assert (await _profile(KEY)).runtime_policy["validated_vram_mb"] == {"normal": 1000}


@requires_db
async def test_switch_to_backend_without_deployment_is_rejected(model_pair):
    mm, pr = model_pair
    with pytest.raises(md.DeploymentError, match="no deployment on backend 'vllm'"):
        await md.set_default_backend(KEY, "vllm", model_manager=mm, profile_registry=pr)


@requires_db
async def test_removing_the_default_deployment_keeps_the_public_profile(model_pair):
    mm, pr = model_pair
    result = await md.remove_deployment(KEY, "llamacpp_mks", model_manager=mm, profile_registry=pr)

    assert result["new_default_backend"] == "ollama"
    public = await _profile(KEY)
    assert public is not None and public.base_model_id == OLLAMA
    assert LLAMA not in mm._states
    assert all(p.base_model_id != LLAMA for p in await pr.list_all())


@requires_db
async def test_add_deployment_derives_artifact_and_gets_hidden_variant(model_pair):
    mm, pr = model_pair
    await md.remove_deployment(KEY, "ollama", model_manager=mm, profile_registry=pr)
    out = await md.add_deployment(KEY, "llama_cpp", model_manager=mm, profile_registry=pr)

    async with AsyncSessionLocal() as session:
        cfg = await session.get(ModelConfig, None) if False else (
            await session.execute(sa.select(ModelConfig).where(ModelConfig.model_id == f"llama_cpp/{KEY}"))
        ).scalar_one()
    assert cfg.artifact["source"] == "ollama" and cfg.artifact["ref"] == "zz-test-switch:1b"
    assert cfg.load_policy == "on_demand"
    variant = await _profile(out["pinned_variant"])
    assert variant.backend == "llama_cpp"
    with pytest.raises(md.DeploymentError, match="already has"):
        await md.add_deployment(KEY, "llama_cpp", model_manager=mm, profile_registry=pr)
    with pytest.raises(md.DeploymentError, match="cannot serve"):
        await md.add_deployment(KEY, "vllm", model_manager=mm, profile_registry=pr)


@requires_db
async def test_granting_a_model_grants_its_deployments_variants_and_aliases(model_pair):
    async with AsyncSessionLocal() as session:
        expanded = await _expand_model_grants({KEY, "recomendado-x"}, session)
        assert {KEY, LLAMA, OLLAMA, f"{KEY}-ollama", f"{KEY}-old", "recomendado-x"} <= expanded
        # a single variant grant stays a single-profile grant
        only_variant = await _expand_model_grants({f"{KEY}-ollama"}, session)
        assert only_variant == {f"{KEY}-ollama"}
        # a deployment-id grant (pre-0034 style) widens to the model
        assert KEY in await _expand_model_grants({OLLAMA}, session)
