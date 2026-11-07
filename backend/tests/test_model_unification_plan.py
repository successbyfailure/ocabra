"""Tests for the pure planning logic of scripts/model_unification_plan.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "model_unification_plan",
    Path(__file__).resolve().parent.parent / "scripts" / "model_unification_plan.py",
)
plan_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plan_mod)

BLOBS = "/data/ollama_models/blobs"


def _cfg(model_id, backend, extra=None, load_policy="on_demand"):
    return {
        "model_id": model_id,
        "backend_type": backend,
        "display_name": None,
        "load_policy": load_policy,
        "pin_permanent": False,
        "preferred_gpu": None,
        "extra_config": extra,
    }


def _profile(pid, base, is_default=True, **kw):
    return {
        "profile_id": pid,
        "base_model_id": base,
        "profile_kind": "execution",
        "is_default": is_default,
        "routing_targets": None,
        "load_overrides": kw.get("load_overrides"),
        "request_defaults": kw.get("request_defaults"),
        "runtime_policy": None,
    }


def _router(pid, targets):
    return {**_profile(pid, None), "profile_kind": "router", "routing_targets": targets}


def _snapshot(**over):
    snap = {
        "model_configs": [
            _cfg(
                "llamacpp_mks/gemma-x",
                "llamacpp_mks",
                {
                    "llama_cpp": {
                        "model_path": f"{BLOBS}/sha256-model",
                        "mmproj_model_id": f"{BLOBS}/sha256-proj",
                        "ctxSize": 1,
                        "ctx_size": 2,
                    }
                },
            ),
            _cfg("ollama/gemma:x", "ollama", load_policy="warm"),
            _cfg("whisper/openai/whisper-large-v3-turbo", "whisper"),
            _cfg("whisper_cpp/large-v3-turbo-q8_0-cpu", "whisper_cpp"),
        ],
        "model_profiles": [
            _profile("gemma:x", "llamacpp_mks/gemma-x"),
            _profile("gemma:x-ollama", "ollama/gemma:x"),
            _profile("whisper-large-v3-turbo", "whisper/openai/whisper-large-v3-turbo"),
            _profile(
                "whisper-large-v3-turbo-diarized",
                "whisper/openai/whisper-large-v3-turbo",
                is_default=False,
                request_defaults={"diarize": True},
            ),
            _profile("whisper-cpp-cpu", "whisper_cpp/large-v3-turbo-q8_0-cpu", is_default=False),
            _router("recomendado", ["gemma:x-ollama", "gemma:x", "whisper-cpp-cpu", "ghost"]),
        ],
        "model_profile_aliases": [],
        "profile_runtime_presets": [{"profile_id": "gemma:x-ollama", "preset_key": "32k"}],
        "group_models": [
            {"group_name": "g", "model_id": "gemma:x-ollama"},
            {"group_name": "g", "model_id": "whisper/openai/whisper-large-v3-turbo"},
            {"group_name": "g", "model_id": "recomendado"},
            {"group_name": "g", "model_id": "gone"},
        ],
        "agents": [],
        "stats_counts": {"request_stats": {"ollama/gemma:x": 5, "llama_cpp/old": 9}},
        # one GGUF blob published under two tags: the Ollama deployment uses the friendly one
        "ollama_manifests": {
            "hf.co/someone/gemma-gguf:Q4": {"model": "sha256-model", "projector": "sha256-proj"},
            "gemma:x": {"model": "sha256-model", "projector": "sha256-proj"},
        },
        "ollama_root": "/data/ollama_models",
    }
    snap.update(over)
    return snap


def _model(plan, key):
    return next(m for m in plan["models"] if m["model_key"] == key)


def test_same_blob_under_two_tags_is_one_model_with_llamacpp_default():
    plan = plan_mod.build_plan(_snapshot())

    gemma = _model(plan, "gemma:x")
    assert gemma["default_backend"] == "llamacpp_mks"
    ids = {d["new_model_id"]: d for d in gemma["deployments"]}
    assert set(ids) == {"llamacpp_mks/gemma:x", "ollama/gemma:x"}
    art = ids["llamacpp_mks/gemma:x"]["artifact"]
    # re-pointed from the hf.co tag to the tag the Ollama deployment uses
    assert art["source"] == "ollama" and art["ref"] == "gemma:x"
    assert art["mmproj"] == {"source": "ollama", "layer": "projector"}


def test_plain_profile_of_non_default_deployment_becomes_alias_and_loses_presets():
    plan = plan_mod.build_plan(_snapshot())

    assert plan["profiles"]["gemma:x-ollama"]["action"] == "alias"
    assert plan["profiles"]["gemma:x-ollama"]["backend"] == "ollama"
    assert plan["dropped_presets"] == {"gemma:x-ollama": ["32k"]}
    assert plan["profiles"]["whisper-large-v3-turbo-diarized"]["action"] == "variant"


def test_manual_group_puts_whisper_cpp_under_the_whisper_model():
    plan = plan_mod.build_plan(_snapshot())

    whisper = _model(plan, "whisper-large-v3-turbo")
    assert whisper["default_backend"] == "whisper"
    assert {d["new_model_id"] for d in whisper["deployments"]} == {
        "whisper/whisper-large-v3-turbo",
        "whisper_cpp/whisper-large-v3-turbo",
    }


def test_router_targets_keep_backend_choice_and_drop_ghosts():
    plan = plan_mod.build_plan(_snapshot())

    router = next(r for r in plan["routers"] if r["router"] == "recomendado")
    assert router["new"] == ["ollama/gemma:x", "gemma:x", "whisper_cpp/whisper-large-v3-turbo"]
    assert any("ghost" in w for w in plan["warnings"])


def test_acl_maps_everything_to_model_keys_and_reports_widening():
    plan = plan_mod.build_plan(_snapshot())

    acl = plan["acl"]["g"]
    assert acl["new"] == ["gemma:x", "recomendado", "whisper-large-v3-turbo"]
    assert any("gemma:x-ollama" in n and "modelo entero" in n for n in acl["notes"])
    assert any("gone" in n for n in acl["notes"])


def test_conflicting_camel_snake_values_and_policy_mismatch_are_warned():
    plan = plan_mod.build_plan(_snapshot())

    assert any("ctxSize=1" in w and "ctx_size=2" in w for w in plan["warnings"])
    assert any("load_policy distinta" in w for w in plan["warnings"])


def test_dangling_ollama_blob_is_reported_and_not_grouped():
    snap = _snapshot(ollama_manifests={"gemma:x": {"model": "sha256-other"}})
    plan = plan_mod.build_plan(snap)

    assert any("blob de Ollama que ningún manifest" in w for w in plan["warnings"])
    # the llama.cpp deployment no longer shares a file with the Ollama tag → 2 models, same key
    assert any("COLISIÓN" in w and "gemma:x" in w for w in plan["warnings"])


def test_stats_map_every_known_name_to_its_model_key():
    snap = _snapshot(
        stats_counts={
            "request_stats": {
                "ollama/gemma:x": 5,
                "gemma:x-ollama": 2,
                "gemma:x": 7,
                "llama_cpp/old": 9,
                "llamacpp_mks/gemma-x::02e08fbd7161": 4,
            }
        }
    )
    plan = plan_mod.build_plan(snap)

    rs = plan["stats"]["request_stats"]
    assert rs["renames"] == {
        "ollama/gemma:x": {"to": "gemma:x", "rows": 5},
        "gemma:x-ollama": {"to": "gemma:x", "rows": 2},
        "llamacpp_mks/gemma-x::02e08fbd7161": {"to": "gemma:x", "rows": 4},
    }
    assert rs["unchanged_rows"] == 7
    assert rs["orphans"] == {"llama_cpp/old": 9}


def test_ollama_manifest_path_roundtrip():
    rel = plan_mod.ollama_manifest_relpath("gemma4:26b")
    assert rel == "registry.ollama.ai/library/gemma4/26b"
    assert plan_mod.ollama_tag_from_relpath(rel) == "gemma4:26b"
    assert (
        plan_mod.ollama_tag_from_relpath("hf.co/huihui-ai/X-GGUF/Q5") == "hf.co/huihui-ai/X-GGUF:Q5"
    )
