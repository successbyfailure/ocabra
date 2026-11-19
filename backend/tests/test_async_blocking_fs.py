"""Filesystem work of model listings and image saves stays off the event loop.

ASY-02/03/05 (docs/REVISION-2026-11-16.md): ``/api/tags``, ``/api/ps`` and
``/ocabra/models`` walked the models directory once per model, on the loop, on
every request; ``/v1/images/*`` wrote the PNG on the loop. These tests pin
both halves of the fix: the blocking work runs in a worker thread, and a walk
of an unchanged directory is not repeated.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
from types import SimpleNamespace

import pytest

from ocabra.api._deps_auth import UserContext
from ocabra.config import settings


def _on_event_loop() -> bool:
    """True when called from a thread that is running an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _admin() -> UserContext:
    return UserContext(user_id="u", username="admin", role="system_admin")


def _state(model_id: str, *, vram_mb: int = 0, status: str = "loaded", backend_model_id=None):
    return SimpleNamespace(
        model_id=model_id,
        backend_model_id=backend_model_id or model_id,
        backend_type="vllm",
        loaded_at=None,
        last_request_at=None,
        vram_used_mb=vram_mb,
        status=SimpleNamespace(value=status),
        extra_config={},
    )


class _FakeModelManager:
    def __init__(self, states):
        self._states = {s.model_id: s for s in states}

    async def get_state(self, model_id):
        return self._states.get(model_id)

    async def list_states(self, include_derived: bool = False):
        return list(self._states.values())

    async def get_expected_load_seconds(self, model_id):
        return None


class _FakeProfiles:
    def __init__(self, profiles):
        self._profiles = profiles

    async def list_enabled(self):
        return list(self._profiles)


def _profile(profile_id, base_model_id, *, kind="execution", targets=None):
    return SimpleNamespace(
        profile_id=profile_id,
        base_model_id=base_model_id,
        profile_kind=kind,
        backend=None,
        category="llm",
        runtime_presets=[],
        routing_targets=targets or [],
        router_context=None,
    )


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(settings, "models_dir", str(root))
    return root


def _write(path: pathlib.Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def _record_threads(monkeypatch, module, name):
    """Wrap ``module.name`` so each call records whether it ran on the loop thread."""
    original = getattr(module, name)
    on_loop: list[bool] = []

    def wrapper(*args, **kwargs):
        on_loop.append(_on_event_loop())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return on_loop


# ── /api/tags ────────────────────────────────────────────────────


async def test_tags_sizes_are_computed_off_the_loop_and_unchanged(models_dir, monkeypatch):
    from ocabra.api.ollama import tags

    _write(models_dir / "org" / "a" / "model.safetensors", 1000)
    _write(models_dir / "org" / "a" / "notes.txt", 50)  # not a weight file
    on_loop = _record_threads(monkeypatch, tags, "_estimate_size_bytes")

    manager = _FakeModelManager(
        [_state("org/a", status="configured"), _state("missing/b", vram_mb=3)]
    )
    profiles = _FakeProfiles(
        [
            _profile("a", "org/a"),
            _profile("b", "missing/b"),
            _profile("router", "", kind="router", targets=["b", "a"]),
        ]
    )
    result = await tags._list_tags_from_profiles(manager, profiles, _admin())

    sizes = {m["name"]: m["size"] for m in result["models"]}
    # On disk: weights only. Not on disk: the VRAM in use. Router: max of its
    # targets, which needs the sizes to be filled before routers are built.
    assert sizes == {"a": 1000, "b": 3 * 1024 * 1024, "router": 3 * 1024 * 1024}
    assert on_loop and not any(on_loop)


async def test_tags_legacy_sizes_are_computed_off_the_loop(models_dir, monkeypatch):
    from ocabra.api.ollama import tags

    async def installed():
        return [{"name": "foo:1b", "size": 0}, {"name": "bar:2b", "size": 77}]

    async def loaded():
        return []

    monkeypatch.setattr(tags._registry, "list_installed_details", installed)
    monkeypatch.setattr(tags._registry, "list_loaded", loaded)
    on_loop = _record_threads(monkeypatch, tags, "_estimate_size_bytes")

    result = await tags._list_tags_legacy(_FakeModelManager([]), _admin())

    assert [m["size"] for m in result["models"]] == [0, 77]
    assert on_loop == [False]  # only the entry Ollama gave no size for


# ── /api/ps ──────────────────────────────────────────────────────


async def test_ps_sizes_are_computed_off_the_loop(models_dir, monkeypatch):
    from ocabra.api.ollama import ps

    _write(models_dir / "org" / "a" / "w.gguf", 4096)
    on_loop = _record_threads(monkeypatch, ps, "_estimate_size_bytes")
    manager = _FakeModelManager([_state("org/a", vram_mb=1), _state("solo", vram_mb=2)])
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                model_manager=manager, profile_registry=_FakeProfiles([_profile("a", "org/a")])
            )
        )
    )

    result = await ps.list_running(request, _admin())

    assert {m["name"]: m["size"] for m in result["models"]} == {
        "a": 4096,
        "solo": 2 * 1024 * 1024,
    }
    assert on_loop and not any(on_loop)


# ── /ocabra/models ───────────────────────────────────────────────


def _count_rglob(monkeypatch, pattern: str) -> list[pathlib.Path]:
    calls: list[pathlib.Path] = []
    original = pathlib.Path.rglob

    def rglob(self, pat, *args, **kwargs):
        if pat == pattern:
            calls.append(self)
        return original(self, pat, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "rglob", rglob)
    return calls


def test_local_model_lookup_walks_models_dir_once(models_dir, monkeypatch):
    from ocabra.api.internal import models as models_api

    gguf = models_dir / "bitnet" / "foo-q4.gguf"
    _write(gguf, 10)
    calls = _count_rglob(monkeypatch, "*.gguf")

    first = models_api._resolve_local_model_path("foo-q4")
    second = models_api._resolve_local_model_path("foo-q4")

    assert first == second == gguf
    assert len(calls) == 1


def test_local_model_lookup_sees_new_and_deleted_files(models_dir, monkeypatch):
    from ocabra.api.internal import models as models_api
    from ocabra.core import fs_scan_cache

    assert models_api._resolve_local_model_path("bar-q4") is None

    # A file added directly in the directory changes its mtime: seen at once.
    added = models_dir / "bar-q4.gguf"
    _write(added, 10)
    os.utime(models_dir, ns=(1, os.stat(models_dir).st_mtime_ns + 10**9))
    assert models_api._resolve_local_model_path("bar-q4") == added

    # One nested deeper does not touch it; the TTL bounds how long it hides.
    models_api._resolve_local_model_path("baz-q4")  # cached: not there yet
    before = os.stat(models_dir).st_mtime_ns
    nested = models_dir / "sub" / "baz-q4.gguf"
    _write(nested, 10)
    os.utime(models_dir, ns=(1, before))  # as if "sub" had existed already
    assert models_api._resolve_local_model_path("baz-q4") is None
    monkeypatch.setattr(fs_scan_cache, "TTL_SECONDS", 0.0)
    assert models_api._resolve_local_model_path("baz-q4") == nested

    # A deleted file is never returned, even from a cached listing.
    monkeypatch.setattr(fs_scan_cache, "TTL_SECONDS", 3600.0)
    assert models_api._resolve_local_model_path("bar-q4") == added
    before = os.stat(models_dir).st_mtime_ns
    added.unlink()
    os.utime(models_dir, ns=(1, before))  # keep the cached listing valid
    assert models_api._resolve_local_model_path("bar-q4") is None


async def test_disk_size_lookup_runs_off_the_loop(models_dir, monkeypatch):
    from ocabra.api.internal import models as models_api

    _write(models_dir / "bitnet" / "foo-q4.gguf", 1234)
    on_loop = _record_threads(monkeypatch, models_api, "_resolve_local_model_path")
    state = SimpleNamespace(model_id="llama_cpp/foo-q4", extra_config={})

    assert await models_api._resolve_disk_size_bytes(state, {}) == 1234
    assert on_loop == [False]


async def test_capability_fallbacks_run_off_the_loop(monkeypatch):
    from ocabra.api.internal import models as models_api

    on_loop = _record_threads(monkeypatch, models_api, "_apply_capability_fallbacks")
    state = SimpleNamespace(
        backend_type="vllm",
        backend_model_id="nowhere/model",
        extra_config={"context_length": 8192},
        status=SimpleNamespace(value="configured"),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    payload = await models_api._resolve_capabilities_payload(request, state, {})

    assert payload["context_length"] == 8192
    assert on_loop and not any(on_loop)


async def test_llama_cpp_memory_estimate_runs_off_the_loop(monkeypatch):
    from ocabra.api.internal import models as models_api

    on_loop: list[bool] = []

    def fake_estimate(*, estimate, **_kwargs):
        on_loop.append(_on_event_loop())
        return {**estimate, "source": "fake"}

    monkeypatch.setattr(models_api, "_estimate_llama_cpp_memory", fake_estimate)

    class _Backend:
        async def get_vram_estimate_mb(self, *_a, **_k):
            return 100

    class _Pool:
        async def get_backend(self, _backend_type):
            return _Backend()

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(worker_pool=_Pool(), gpu_manager=None))
    )
    state = SimpleNamespace(
        backend_type="llama_cpp",
        backend_model_id="foo",
        model_id="llama_cpp/foo",
        preferred_gpu=None,
        extra_config={},
    )

    out = await models_api._build_model_memory_estimate(request, state, {}, None, False)

    assert out["source"] == "fake"
    assert on_loop == [False]


# ── /v1/images/* ─────────────────────────────────────────────────


def test_image_url_mode_saves_off_the_loop(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from ocabra.api.openai import images
    from tests.test_openai_api import TestImageGenerations

    on_loop = _record_threads(monkeypatch, images, "_save_image")
    # The helper sets these globally; monkeypatch restores them afterwards.
    monkeypatch.setattr(settings, "image_outputs_dir", settings.image_outputs_dir)
    monkeypatch.setattr(settings, "public_url", settings.public_url)
    app, _ = TestImageGenerations()._diffusers_app(tmp_path)

    resp = TestClient(app).post(
        "/v1/images/generations",
        json={"model": "sdxl", "prompt": "a cat", "response_format": "url"},
    )

    assert resp.status_code == 200
    name = resp.json()["data"][0]["url"].rsplit("/", 1)[-1]
    assert (tmp_path / name).is_file()
    assert on_loop == [False]


# ── fs_scan_cache ────────────────────────────────────────────────


def test_weights_size_cache_reuses_until_dir_or_ttl_changes(tmp_path, monkeypatch):
    from ocabra.core import fs_scan_cache

    model = tmp_path / "m"
    _write(model / "a.safetensors", 10)
    walks = []
    original = fs_scan_cache._walk_weights_size

    def counting(*args):
        walks.append(args)
        return original(*args)

    monkeypatch.setattr(fs_scan_cache, "_walk_weights_size", counting)

    assert fs_scan_cache.weights_size_bytes(model) == 10
    assert fs_scan_cache.weights_size_bytes(model) == 10
    assert len(walks) == 1

    _write(model / "b.bin", 5)  # new file in the dir: mtime changes
    os.utime(model, ns=(1, os.stat(model).st_mtime_ns + 10**9))
    assert fs_scan_cache.weights_size_bytes(model) == 15
    assert len(walks) == 2

    monkeypatch.setattr(fs_scan_cache, "TTL_SECONDS", 0.0)
    fs_scan_cache.weights_size_bytes(model)
    fs_scan_cache.weights_size_bytes(model)
    assert len(walks) == 4

    assert fs_scan_cache.weights_size_bytes(tmp_path / "missing") == 0


async def test_warm_arch_cache_parses_headers_off_the_event_loop(monkeypatch):
    """ASY-01: synchronous callers hit the cache after the startup warm-up."""
    import threading

    from ocabra.core import vram_planner

    loop_thread = threading.get_ident()
    seen: list[tuple[str, int]] = []

    def fake(path):
        seen.append((str(path), threading.get_ident()))
        return object() if str(path).endswith(".gguf") else None

    monkeypatch.setattr(vram_planner, "arch_from_gguf", fake)
    read = await vram_planner.warm_arch_cache(["/m/a.gguf", "/m/a.gguf", "/m/b.bin", None])
    assert read == 1
    assert [p for p, _ in seen] == ["/m/a.gguf", "/m/b.bin"]  # deduplicated
    assert all(t != loop_thread for _, t in seen)
