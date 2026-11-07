"""Tests for the model/backends unification planner (ocabra.core.model_unification)."""

from __future__ import annotations

from ocabra.core import artifacts
from ocabra.core import model_unification as plan_mod

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
            _router("recomendado", ["gemma:x-ollama", "whisper-cpp-cpu", "ghost"]),
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


def test_profile_of_non_default_deployment_stays_a_pinned_variant_with_its_presets():
    """``<name>-ollama`` profiles are live client entry points (embeddings among
    them): until aliases can force a backend they keep serving from Ollama."""
    plan = plan_mod.build_plan(_snapshot())

    act = plan["profiles"]["gemma:x-ollama"]
    assert act["action"] == "variant" and act["backend"] == "ollama" and act["door"] is True
    assert act["new_base_model_id"] == "ollama/gemma:x"
    assert plan["dropped_presets"] == {}
    assert plan["profiles"]["whisper-large-v3-turbo-diarized"]["action"] == "variant"


def test_manual_group_puts_whisper_cpp_under_the_whisper_model():
    plan = plan_mod.build_plan(_snapshot())

    whisper = _model(plan, "whisper-large-v3-turbo")
    assert whisper["default_backend"] == "whisper"
    assert {d["new_model_id"] for d in whisper["deployments"]} == {
        "whisper/whisper-large-v3-turbo",
        "whisper_cpp/whisper-large-v3-turbo",
    }


def test_router_alias_targets_become_the_generic_model():
    plan = plan_mod.build_plan(_snapshot())

    router = next(r for r in plan["routers"] if r["router"] == "recomendado")
    assert router["new"] == ["gemma:x", "whisper-large-v3-turbo"]
    assert any("ghost" in w for w in plan["warnings"])


def test_router_cpu_fallback_next_to_its_gpu_model_stays_a_pinned_variant():
    """Routers only take profile ids, so the CPU fallback keeps its profile,
    pinned to whisper_cpp, in its original position."""
    snap = _snapshot()
    snap["model_profiles"] = [p for p in snap["model_profiles"] if p["profile_id"] != "recomendado"]
    snap["model_profiles"].append(_router("stt", ["whisper-large-v3-turbo", "whisper-cpp-cpu"]))
    plan = plan_mod.build_plan(snap)

    router = next(r for r in plan["routers"] if r["router"] == "stt")
    assert router["new"] == ["whisper-large-v3-turbo", "whisper-cpp-cpu"]
    act = plan["profiles"]["whisper-cpp-cpu"]
    assert act["action"] == "variant" and act["backend"] == "whisper_cpp"
    assert act["new_base_model_id"] == "whisper_cpp/whisper-large-v3-turbo"


def test_acl_maps_ids_and_aliases_to_model_keys_but_keeps_variants():
    snap = _snapshot()
    snap["group_models"].append({"group_name": "g", "model_id": "whisper-large-v3-turbo-diarized"})
    plan = plan_mod.build_plan(snap)

    acl = plan["acl"]["g"]
    # alias profile and canonical id → model keys; the diarized variant keeps its own
    # entry (granting a variant must not widen into the whole model); ghosts dropped
    assert acl["new"] == [
        "gemma:x-ollama",  # variants (the -ollama door, -diarized) keep their own entry
        "recomendado",
        "whisper-large-v3-turbo",
        "whisper-large-v3-turbo-diarized",
    ]
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
    rel = artifacts.ollama_manifest_relpath("gemma4:26b")
    assert rel == "registry.ollama.ai/library/gemma4/26b"
    assert artifacts.ollama_tag_from_relpath(rel) == "gemma4:26b"
    assert (
        artifacts.ollama_tag_from_relpath("hf.co/huihui-ai/X-GGUF/Q5")
        == "hf.co/huihui-ai/X-GGUF:Q5"
    )


def test_normalize_case_snake_wins_and_deep_dicts_are_left_alone():
    norm, dropped = plan_mod.normalize_case(
        {
            "llama_cpp": {
                "ctxSize": 229376,
                "ctx_size": 40960,
                "gpuLayers": 99,
                "hf_overrides": {"ropeTheta": 1},
            }
        }
    )
    assert norm == {
        "llama_cpp": {"ctx_size": 40960, "gpu_layers": 99, "hf_overrides": {"ropeTheta": 1}}
    }
    assert dropped == ["llama_cpp.ctxSize=229376 (gana ctx_size=40960)"]


def test_deployment_extra_config_is_normalised_without_artifact_paths():
    plan = plan_mod.build_plan(_snapshot())

    dep = next(d for d in _model(plan, "gemma:x")["deployments"] if d["backend"] == "llamacpp_mks")
    assert dep["extra_config"] == {"llama_cpp": {"ctx_size": 2}}


def test_diarized_profile_overrides_are_normalised_to_snake_case():
    """The base whisper section becomes ``diarization_enabled: false``; a
    camelCase ``diarizationEnabled: true`` override would then lose against it
    (readers check snake_case first) and diarization would silently turn off."""
    snap = _snapshot()
    snap["model_configs"][2]["extra_config"] = {"whisper": {"diarizationEnabled": False}}
    snap["model_profiles"][3]["load_overrides"] = {
        "whisper": {"diarizationEnabled": True},
        "diarization_enabled": True,
    }
    plan = plan_mod.build_plan(snap)

    dep = next(
        d
        for d in _model(plan, "whisper-large-v3-turbo")["deployments"]
        if d["backend"] == "whisper"
    )
    assert dep["extra_config"] == {"whisper": {"diarization_enabled": False}}
    assert plan["profiles"]["whisper-large-v3-turbo-diarized"]["load_overrides"] == {
        "whisper": {"diarization_enabled": True},
        "diarization_enabled": True,
    }


def test_strongest_load_policy_of_the_deployments_wins():
    plan = plan_mod.build_plan(_snapshot())

    # llama.cpp default is on_demand, the Ollama copy was warm
    assert _model(plan, "gemma:x")["load_policy"] == "warm"


def test_incoming_edit_prefers_camel_case_and_foreign_dicts_are_untouched():
    from ocabra.core.config_keys import normalize_case

    norm, _ = normalize_case(
        {
            "llama_cpp": {"ctx_size": 40960, "ctxSize": 229376, "speculative": {"draftN": 4}},
            "hfOverrides": {"ropeTheta": 1},
        },
        prefer="camel",
    )
    assert norm == {
        "llama_cpp": {"ctx_size": 229376, "speculative": {"draft_n": 4}},
        "hf_overrides": {"ropeTheta": 1},
    }


def test_renamed_deployment_ids_stay_resolvable_through_hidden_aliases():
    plan = plan_mod.build_plan(_snapshot())

    legacy = {a["alias_id"]: a for a in plan["legacy_aliases"]}
    # the default deployment's old id → the model's default profile
    assert legacy["llamacpp_mks/gemma-x"] == {
        "alias_id": "llamacpp_mks/gemma-x",
        "profile_id": "gemma:x",
        "backend": "llamacpp_mks",
    }
    # a non-default deployment's old id → its own pinned profile, not the GPU default
    assert legacy["whisper_cpp/large-v3-turbo-q8_0-cpu"]["profile_id"] == "whisper-cpp-cpu"
    # unchanged ids get no alias
    assert "ollama/gemma:x" not in legacy
