"""Backend pieces of the new model editor (2026-11-12): the per-backend config
schema, the draft preview and the fixes the editor's inventory uncovered."""

from __future__ import annotations

import ast
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ocabra.core import config_schema
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus
from ocabra.core.vram_capacity import kv_dtype_for

BACKENDS_DIR = Path(__file__).resolve().parents[2] / "ocabra" / "backends"

# ── Schema vs what the backends actually read ───────────────────────────────

_GETTER = re.compile(r"_?(get|resolve)_\w*option$")
_SECTION_GET_BACKENDS = {
    # backend -> (file, name of the dict variable holding its section)
    "whisper_cpp": ("whispercpp_backend.py", {"cfg"}),
    "vibeasr": ("vibeasr_backend.py", {"cfg"}),
    "piper": ("piper_backend.py", {"piper_cfg"}),
    "diffusers": ("diffusers_backend.py", {"diffusers_cfg"}),
    "strata": ("strata_backend.py", {"options"}),
}
_GETTER_BACKENDS = {
    "llama_cpp": "llama_cpp_backend.py",
    "bitnet": "bitnet_backend.py",
    "vllm": "vllm_backend.py",
    "sglang": "sglang_backend.py",
    "tensorrt_llm": "tensorrt_llm_backend.py",
    "whisper": "whisper_backend.py",
}


def _getter_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if not _GETTER.match(name or ""):
            continue
        for arg in node.args[:3]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                keys.add(arg.value)
                break
    return keys


def _section_get_keys(path: Path, variables: set[str]) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in variables
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
    return keys


def _schema_keys(backend: str) -> set[str]:
    return {f.path[-1] if len(f.path) == 2 else f.path[0] for f in config_schema.fields_for(backend)}


@pytest.mark.parametrize("backend", sorted(_GETTER_BACKENDS))
def test_every_option_a_backend_reads_is_in_the_schema(backend):
    """A backend option without a schema field is invisible in the editor and
    unvalidated on save. Add the field (``hidden=True`` if it must not be a
    control) when this fails."""
    read = _getter_keys(BACKENDS_DIR / _GETTER_BACKENDS[backend])
    assert read, "the scan found nothing: the getter naming changed"
    missing = read - _schema_keys(backend)
    assert not missing, f"{backend} reads options the schema lacks: {sorted(missing)}"


@pytest.mark.parametrize("backend", sorted(_SECTION_GET_BACKENDS))
def test_every_section_key_a_backend_reads_is_in_the_schema(backend):
    filename, variables = _SECTION_GET_BACKENDS[backend]
    read = _section_get_keys(BACKENDS_DIR / filename, variables)
    assert read
    # strata's ``options`` also carries its run config's own keys.
    read -= {"args", "gpu", "cwd", "model_name", "parallel", "vision"}
    missing = read - _schema_keys(backend)
    assert not missing, f"{backend} reads options the schema lacks: {sorted(missing)}"


def test_llamacpp_mks_shares_the_llama_cpp_section_and_fields():
    schema = config_schema.schema_for("llamacpp_mks")
    assert schema["section"] == "llama_cpp"
    keys = {f["key"] for f in schema["fields"]}
    assert "llama_cpp.ctx_size" in keys and "llama_cpp.speculative" in keys
    assert "keep_alive_seconds" in keys  # universal, top level


def test_hidden_fields_are_not_offered():
    keys = {f["key"] for f in config_schema.schema_for("llama_cpp")["fields"]}
    assert "llama_cpp.model_path" not in keys
    assert "llama_cpp.keep_alive_seconds" not in keys  # the top-level one is offered


def test_global_defaults_are_resolved(monkeypatch):
    from ocabra.config import settings

    monkeypatch.setattr(settings, "llama_cpp_ubatch_size", 256)
    fields = {f["key"]: f for f in config_schema.schema_for("llama_cpp")["fields"]}
    assert fields["llama_cpp.ubatch_size"]["default"] == 256


def test_llama_runtime_is_offered_only_with_an_installed_binary(tmp_path, monkeypatch):
    from ocabra.config import settings

    monkeypatch.setattr(settings, "backends_dir", str(tmp_path))
    keys = {f["key"] for f in config_schema.schema_for("llama_cpp")["fields"]}
    assert "llama_cpp.runtime" not in keys
    (tmp_path / "llama_cpp_vulkan").mkdir()
    runtime = next(
        f for f in config_schema.schema_for("llama_cpp")["fields"] if f["key"] == "llama_cpp.runtime"
    )
    assert [o["value"] for o in runtime["options"]] == ["cuda", "vulkan"]


def test_split_mode_tensor_is_valid_in_schema_and_load_config():
    from ocabra.schemas.backend_load import LlamaCppLoadConfig

    assert config_schema.validate_extra_config(
        "llamacpp_mks", {"llama_cpp": {"split_mode": "tensor"}}
    ) == []
    assert LlamaCppLoadConfig(split_mode="tensor").split_mode == "tensor"


# ── Validation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("extra", "key"),
    [
        ({"llama_cpp": {"ctx_size": "32k"}}, "llama_cpp.ctx_size"),
        ({"llama_cpp": {"parallel_slots": 0}}, "llama_cpp.parallel_slots"),
        ({"llama_cpp": {"flash_attn": "yes"}}, "llama_cpp.flash_attn"),
        ({"llama_cpp": {"cache_type_k": "q3"}}, "llama_cpp.cache_type_k"),
        ({"llama_cpp": {"tensor_split": [0, 0]}}, "llama_cpp.tensor_split"),
        ({"llama_cpp": {"disabled_gpus": [0.5]}}, "llama_cpp.disabled_gpus"),
        ({"keep_alive_seconds": -5}, "keep_alive_seconds"),
        ({"default_reasoning_effort": "max"}, "default_reasoning_effort"),
    ],
)
def test_invalid_options_are_reported(extra, key):
    errors = config_schema.validate_extra_config("llamacpp_mks", extra)
    assert [e["key"] for e in errors] == [key]


def test_valid_and_null_options_pass():
    extra = {
        "keep_alive_seconds": 0,
        "llama_cpp": {
            "ctx_size": 98304,
            "parallel_slots": 4,
            "cache_type_k": "q8_0",
            "tensor_split": [1, 2.5],
            "main_gpu": None,
            "speculative": {"spec_type": "draft-mtp"},
        },
    }
    assert config_schema.validate_extra_config("llamacpp_mks", extra) == []


def test_unknown_keys_flag_typos_but_not_internal_keys():
    extra = {
        "_runtime": {"parallelism": 2},
        "use_case": {"slots": 1},
        "llama_cpp": {"ctx_size": 1, "ctx_sise": 2},
        "keep_alive": 3,
    }
    assert config_schema.unknown_keys("llamacpp_mks", extra) == ["keep_alive", "llama_cpp.ctx_sise"]


def test_vllm_recipe_and_probe_keys_are_known():
    extra = {"vllm": {"recipe_id": "x", "probe_status": "ok", "suggested_tuning": {}}}
    assert config_schema.unknown_keys("vllm", extra) == []


# ── keep_alive for llamacpp_mks ─────────────────────────────────────────────


def _state(**kw):
    defaults = {"model_id": "m", "display_name": "m", "backend_type": "llamacpp_mks"}
    return ModelState(**{**defaults, **kw})


def test_keep_alive_in_the_llama_cpp_section_applies_to_mks():
    state = _state(extra_config={"llama_cpp": {"keep_alive_seconds": 42}})
    assert ModelManager._resolve_keep_alive_seconds(state, 300) == 42


def test_keep_alive_top_level_and_global_fallback():
    assert ModelManager._resolve_keep_alive_seconds(_state(extra_config={"keep_alive_seconds": 0}), 300) == 0
    assert ModelManager._resolve_keep_alive_seconds(_state(extra_config={}), 300) == 300


# ── pin_permanent only protects PIN models ──────────────────────────────────


@pytest.mark.asyncio
async def test_hard_pin_flag_on_a_warm_model_does_not_shield_it():
    mm = ModelManager(MagicMock())
    for model_id, policy, hard in (
        ("t/warm-hard", LoadPolicy.WARM, True),
        ("t/pin-hard", LoadPolicy.PIN, True),
    ):
        mm._states[model_id] = _state(
            model_id=model_id,
            backend_type="vllm",
            load_policy=policy,
            pin_permanent=hard,
            status=ModelStatus.LOADED,
            current_gpu=[0],
            last_request_at=datetime.now(UTC),
        )
    assert mm._get_pressure_eviction_candidates("t/other") == ["t/warm-hard"]


# ── Per-model schedules act whatever the load policy ────────────────────────


def _scheduler(state, action):
    from ocabra.core.scheduler import GPUScheduler

    model_manager = AsyncMock()
    model_manager.get_state = AsyncMock(return_value=state)
    scheduler = GPUScheduler(MagicMock(), model_manager=model_manager)
    scheduler._load_enabled_schedules = AsyncMock(
        return_value=[SimpleNamespace(id="s", model_id=state.model_id, action=action, cron_expr="* * * * *")]
    )
    scheduler._is_schedule_due = MagicMock(return_value=True)
    return scheduler, model_manager


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [LoadPolicy.PIN, LoadPolicy.ON_DEMAND, LoadPolicy.WARM])
async def test_per_model_window_evicts_any_policy(policy):
    state = _state(model_id="vllm/x", backend_type="vllm", load_policy=policy, status=ModelStatus.LOADED)
    scheduler, mm = _scheduler(state, "evict_warm")
    await scheduler.check_schedule_evictions()
    mm.unload.assert_awaited_once_with("vllm/x", reason="schedule:evict_warm")
    # A PIN model's watcher must not reload it inside the window...
    assert (state.reload_suppressed_until is not None) == (policy == LoadPolicy.PIN)


@pytest.mark.asyncio
async def test_window_end_lifts_the_pin_hold_and_reloads():
    state = _state(
        model_id="vllm/x",
        backend_type="vllm",
        load_policy=LoadPolicy.PIN,
        status=ModelStatus.UNLOADED,
        reload_suppressed_until=datetime(2100, 1, 1, tzinfo=UTC),
    )
    scheduler, mm = _scheduler(state, "reload")
    await scheduler.check_schedule_reloads()
    assert state.reload_suppressed_until is None
    mm.load.assert_awaited_once()


def test_malformed_schedule_is_a_value_error_not_a_key_error():
    import asyncio

    from ocabra.db.model_config import replace_model_schedules

    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: [])))
    with pytest.raises(ValueError, match="missing: days"):
        asyncio.run(replace_model_schedules(session, "m", [{"id": "a", "start": "09:00", "end": "10:00"}]))


# ── KV dtype as the backend launches it ─────────────────────────────────────


@pytest.mark.parametrize(
    ("backend", "label", "expected"),
    [
        ("vllm", "q8", (1.0, "fp8")),
        ("vllm", "q4", (1.0, "fp8")),
        ("vllm", "fp16", (2.0, None)),
        ("sglang", "q4", (2.0, None)),
        ("llamacpp_mks", "q4", (0.5, None)),
    ],
)
def test_kv_dtype_matches_the_launch(backend, label, expected):
    assert kv_dtype_for(backend, label)[:2] == expected


def test_sglang_use_case_writes_its_own_context_key():
    from ocabra.core.vram_capacity import apply_use_case_flags

    merged = apply_use_case_flags({}, "sglang", {"applied": True, "effective_context": 8192, "slots": 2})
    assert merged == {"sglang": {"context_length": 8192}}


def test_bitnet_slots_come_from_its_parallel_key():
    from ocabra.core.model_manager_helpers import resolve_worker_slots

    state = _state(backend_type="bitnet", extra_config={"bitnet": {"parallel": 3}})
    assert resolve_worker_slots(state) == 3


# ── Router request_defaults ─────────────────────────────────────────────────


def _profile(pid, defaults):
    return SimpleNamespace(profile_id=pid, request_defaults=defaults, assets=None)


def test_router_defaults_apply_to_its_target_between_target_and_client():
    import contextvars

    from ocabra.api.openai import _deps

    def run():
        router = _profile("recomendado", {"temperature": 0.2, "top_p": 0.5})
        target = _profile("qwen", {"temperature": 0.9, "max_tokens": 100})
        _deps.note_router_resolution(router, target)
        merged = _deps.merge_profile_defaults(target, {"top_p": 0.7})
        other = _deps.merge_profile_defaults(_profile("tts", {}), {})
        return merged, other

    merged, other = contextvars.copy_context().run(run)
    assert merged == {"temperature": 0.2, "max_tokens": 100, "top_p": 0.7}
    assert other == {}  # another profile in the same request is untouched


# ── load_overrides deep merge ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_profile_override_keeps_the_rest_of_the_backend_section():
    """A profile overriding one vLLM option used to replace the whole base
    ``vllm`` section (shallow merge)."""
    from ocabra.api.openai._deps import _ensure_worker_loaded

    base = _state(
        model_id="vllm/m",
        backend_type="vllm",
        extra_config={"vllm": {"max_model_len": 32768, "gpu_memory_utilization": 0.8}},
    )
    mm = MagicMock()
    mm.get_state = AsyncMock(side_effect=lambda key: base if key == "vllm/m" else None)
    mm.add_model = AsyncMock(side_effect=RuntimeError("stop after add_model"))
    with pytest.raises(Exception):  # noqa: B017 — only the merged config matters
        await _ensure_worker_loaded(mm, "vllm/m", "vllm/m::abc", {"vllm": {"max_model_len": 8192}})
    extra = mm.add_model.await_args.kwargs["extra_config"]
    assert extra["vllm"] == {"max_model_len": 8192, "gpu_memory_utilization": 0.8}


def test_deep_merge_used_for_overrides_keeps_siblings_and_drops_nulls():
    from ocabra.core.model_manager import _deep_merge_dicts

    merged = _deep_merge_dicts(
        {"vllm": {"a": 1, "b": 2}, "top": 1}, {"vllm": {"a": 3, "b": None}}
    )
    assert merged == {"vllm": {"a": 3}, "top": 1}


# ── Preview helpers ─────────────────────────────────────────────────────────


def test_split_estimate_follows_the_tensor_split():
    from ocabra.api.internal.models import _split_estimate

    estimate = {
        "model_loading_memory_mb": 1400,
        "breakdown_mb": {"weights": 900, "kv": 300, "overhead": 200},
    }
    extra = {"llama_cpp": {"tensor_split": [1, 2], "main_gpu": 1}}
    gpus = _split_estimate(estimate, extra, [0, 1], {0: 12288, 1: 24576})
    assert [g["index"] for g in gpus] == [0, 1]
    assert gpus[0]["breakdown"] == {"weights": 300, "kv": 100}
    assert gpus[1]["breakdown"] == {"weights": 600, "kv": 200, "overhead": 200}


def test_split_estimate_evenly_is_proportional_to_vram():
    from ocabra.api.internal.models import _split_estimate

    estimate = {"model_loading_memory_mb": 300, "breakdown_mb": {"weights": 300}}
    gpus = _split_estimate(estimate, {"llama_cpp": {}}, [0, 1], {0: 12288, 1: 24576})
    assert [g["mb"] for g in gpus] == [100, 200]


def test_preview_gpu_indices():
    from ocabra.api.internal.models import _preview_gpu_indices

    state = _state(backend_type="llamacpp_mks")
    assert _preview_gpu_indices(state, {"llama_cpp": {}}, 1, 2) == [0, 1]
    assert _preview_gpu_indices(state, {"llama_cpp": {"disabled_gpus": [0]}}, None, 2) == [1]
    assert _preview_gpu_indices(state, {"llama_cpp": {"split_mode": "none"}}, 1, 2) == [1]
    assert _preview_gpu_indices(_state(backend_type="vllm"), {}, 0, 2) == [0]


@pytest.mark.asyncio
async def test_llama_preview_launch_builds_flags_without_starting(monkeypatch, tmp_path):
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend

    backend = LlamaCppBackend()
    monkeypatch.setattr(backend, "_resolve_server_bin", lambda: "/bin/llama-server")
    extra = {"llama_cpp": {"model_path": str(tmp_path / "missing.gguf"), "ctx_size": 8192,
                           "gpu_layers": 99, "parallel_slots": 2, "flash_attn": True,
                           "kv_unified": True, "split_mode": "none"}}
    out = await backend.preview_launch("m", [1], extra)
    argv = out["argv"]
    assert argv[argv.index("--ctx-size") + 1] == "8192"
    assert "--flash-attn" in argv and "--kv-unified" in argv
    assert argv[argv.index("--parallel") + 1] == "2"
    assert out["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert backend._processes == {}


# ── PATCH ───────────────────────────────────────────────────────────────────


def _patch_request(state):
    mm = MagicMock()
    mm.get_state = AsyncMock(return_value=state)
    mm.update_config = AsyncMock(return_value=state)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model_manager=mm))), mm


@pytest.mark.asyncio
async def test_patch_stores_the_vram_estimate_where_the_backend_reads_it(monkeypatch):
    from ocabra.api.internal import models as api

    monkeypatch.setattr(api, "_serialize_state_with_profiles", AsyncMock(return_value={}), raising=False)
    state = _state(backend_type="llamacpp_mks")
    request, mm = _patch_request(state)
    body = api.ModelPatch(vramEstimateMb=20000)
    try:
        await api.update_model("m", body, request, _user=None)
    except Exception:  # noqa: BLE001 — serialization after the update is not under test
        pass
    patch = mm.update_config.await_args.args[1]
    assert patch == {"extra_config": {"llama_cpp": {"vram_estimate_mb": 20000}}}


@pytest.mark.asyncio
async def test_patch_rejects_backend_change_and_invalid_options():
    from fastapi import HTTPException

    from ocabra.api.internal import models as api

    state = _state(backend_type="llamacpp_mks")
    request, mm = _patch_request(state)
    with pytest.raises(HTTPException) as exc:
        await api.update_model("m", api.ModelPatch(backendType="vllm"), request, _user=None)
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        await api.update_model(
            "m", api.ModelPatch(extraConfig={"llama_cpp": {"ctx_size": "big"}}), request, _user=None
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["errors"][0]["key"] == "llama_cpp.ctx_size"
    mm.update_config.assert_not_awaited()


def test_patch_requires_model_manager():
    import inspect

    from ocabra.api.internal import models as api

    dependency = inspect.signature(api.update_model).parameters["_user"].default.dependency
    closure = {c.cell_contents for c in dependency.__closure__ or []}
    assert "model_manager" in closure


def test_model_subroutes_are_registered_before_the_catch_all():
    """``GET /models/{model_id:path}`` swallows any later ``GET
    /models/{id}/<sub>``: speculative-candidates answered 404 ("model
    '<id>/speculative-candidates' not found") from day one."""
    from ocabra.api.internal.models import router

    gets = [r.path for r in router.routes if "GET" in getattr(r, "methods", set())]
    catch_all = gets.index("/models/{model_id:path}")
    late = [p for p in gets[catch_all + 1 :] if p.startswith("/models/{model_id:path}/")]
    assert not late, f"shadowed by the catch-all: {late}"


def test_no_model_subroute_is_shadowed_app_wide():
    """Other routers also mount ``/models/{id}/<sub>`` GETs; the profiles one
    (``GET /ocabra/models/{id}/profiles``) answered 404 because the models
    router, included first, matched it with its catch-all."""
    from ocabra.main import app

    gets = [r.path for r in app.routes if "GET" in getattr(r, "methods", set())]
    catch_all = gets.index("/ocabra/models/{model_id:path}")
    late = [p for p in gets[catch_all + 1 :] if p.startswith("/ocabra/models/{model_id:path}/")]
    assert not late, f"shadowed by the catch-all: {late}"
