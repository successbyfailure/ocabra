"""The api's lifespan: wiring before loops, and a teardown that really runs.

BUG-25: shutdown never stopped ModelManager (its watchdogs and PIN autoload
kept running), never disposed the DB engine, and a failure halfway through
startup left every loop already started alive.

BUG-26: ``model_manager.start()`` launched its loops before ~a dozen
collaborators were injected (service manager, PIN worker resolver, duration
estimator, session registry, batch coordinator…). It only worked because the
PIN autoload waits ``pin_autoload_delay_s`` (60 s).

Every component is a ``create_autospec`` of the real class, so a renamed
method breaks these tests instead of silently passing.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from ocabra import main as main_mod
from ocabra.config import settings


class _FakeSession:
    async def __aenter__(self):
        return AsyncMock()

    async def __aexit__(self, *exc):
        return False


class _Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.mp = monkeypatch
        self.events: list[str] = []
        self.instances: dict[str, MagicMock] = {}

    def component(self, module: str, name: str, *, record=("start", "stop")) -> MagicMock:
        mod = importlib.import_module(module)
        inst = create_autospec(getattr(mod, name), instance=True)
        for method in record:
            if hasattr(inst, method):
                getattr(inst, method).side_effect = lambda *a, _m=method, **k: self.events.append(
                    f"{name}.{_m}"
                )
        self.mp.setattr(mod, name, lambda *a, **k: inst)
        self.instances[name] = inst
        return inst

    def func(self, module: str, name: str, return_value=None) -> AsyncMock:
        mod = importlib.import_module(module)
        fn = AsyncMock(
            return_value=return_value, side_effect=lambda *a, **k: self.events.append(name)
        )
        if return_value is not None:
            fn.side_effect = None
        self.mp.setattr(mod, name, fn)
        return fn

    def fake_loop(self, module: str, name: str) -> None:
        async def _loop(*args, **kwargs):
            stop = next(a for a in (*args, *kwargs.values()) if isinstance(a, asyncio.Event))
            self.events.append(f"{name}.started")
            try:
                await stop.wait()
            finally:
                self.events.append(f"{name}.stopped")

        self.mp.setattr(importlib.import_module(module), name, _loop)


@pytest.fixture
def harness(monkeypatch):
    h = _Harness(monkeypatch)
    for flag, value in (
        ("federation_enabled", False),
        ("federation_join_url", ""),
    ):
        monkeypatch.setattr(settings, flag, value)

    async def _init_redis():
        h.events.append("redis.init")

    async def _close_redis():
        h.events.append("redis.close")

    monkeypatch.setattr("ocabra.redis_client.init_redis", _init_redis)
    monkeypatch.setattr("ocabra.redis_client.close_redis", _close_redis)
    engine = MagicMock()
    engine.dispose = AsyncMock(side_effect=lambda: h.events.append("db.dispose"))
    monkeypatch.setattr("ocabra.database.engine", engine)
    monkeypatch.setattr("ocabra.database.AsyncSessionLocal", lambda: _FakeSession())
    monkeypatch.setattr("ocabra.db.server_config.load_overrides", AsyncMock(return_value={}))
    monkeypatch.setattr("ocabra.db.server_config.save_override", AsyncMock())
    monkeypatch.setattr("ocabra.core.privacy.apply_temp_dir", lambda: None)
    monkeypatch.setattr(
        "ocabra.db.gpu_power_settings.reapply_persisted_settings", AsyncMock(return_value={})
    )

    h.component("ocabra.core.gpu_manager", "GPUManager")
    h.component("ocabra.core.scheduler", "GPUScheduler")
    wp = h.component("ocabra.core.worker_pool", "WorkerPool")
    wp.registered_backends.return_value = {}
    h.component("ocabra.core.model_manager", "ModelManager")
    h.component("ocabra.core.backend_installer", "BackendInstaller")
    h.component("ocabra.core.service_manager", "ServiceManager")
    h.component("ocabra.core.profile_registry", "ProfileRegistry")
    h.instances["ProfileRegistry"].ensure_default_profiles.return_value = 0
    h.instances["ProfileRegistry"].ensure_diarized_profiles.return_value = 0
    h.component("ocabra.core.duration_estimator", "DurationEstimator")
    h.component("ocabra.core.session_registry", "SessionRegistry")
    h.component("ocabra.core.batch_priority", "BatchPriorityCoordinator")
    h.component("ocabra.core.backend_process_manager", "BackendProcessManager")
    reg = h.component("ocabra.registry.ollama_registry", "OllamaRegistry")
    reg.list_installed.return_value = []
    reg.list_loaded.return_value = []
    h.component("ocabra.core.trtllm_compile_manager", "TrtllmCompileManager")
    h.component("ocabra.core.context_probe_service", "ContextProbeService")
    updater = h.component("ocabra.core.self_update", "SelfUpdater")
    h.component("ocabra.core.openai_tts_compat", "OpenAITTSCompatRegistry")
    h.component("ocabra.core.openai_image_compat", "OpenAIImageCompatRegistry")
    h.component("ocabra.core.router_resolver", "RouterResolver")
    h.component("ocabra.agents.mcp_registry", "MCPRegistry")
    monkeypatch.setattr("ocabra.agents.mcp_registry.set_registry", lambda r: None)
    h.component("ocabra.core.batch_processor", "BatchProcessor")
    monkeypatch.setattr(
        "ocabra.core.default_presets.ensure_llm_default_presets",
        AsyncMock(return_value={"created": 0, "skipped": 0}),
    )
    monkeypatch.setattr("ocabra.core.auth_manager.seed_first_admin", AsyncMock())
    monkeypatch.setattr("ocabra.core.secrets_box.rotate_stored_secrets", AsyncMock())
    monkeypatch.setattr(main_mod, "_scan_and_register_trtllm_engines", AsyncMock())

    for loop_name in (
        "_idle_eviction_loop",
        "_schedule_maintenance_loop",
        "_service_idle_unload_loop",
        "_service_health_loop",
        "_image_outputs_cleanup_loop",
        "_ollama_inventory_loop",
    ):
        h.fake_loop("ocabra.main", loop_name)
    h.fake_loop("ocabra.core.files_retention", "retention_loop")

    async def _updater_run(stop):
        h.events.append("self_update.started")
        try:
            await stop.wait()
        finally:
            h.events.append("self_update.stopped")

    updater.run.side_effect = _updater_run
    return h


def _app():
    return SimpleNamespace(state=SimpleNamespace())


@pytest.mark.asyncio
async def test_collaborators_are_wired_before_model_manager_loops_start(harness):
    mm = harness.instances["ModelManager"]
    wp = harness.instances["WorkerPool"]
    seen: dict[str, bool] = {}

    def _on_start():
        seen.update(
            service_manager=mm.set_service_manager.called,
            duration_estimator=mm.set_duration_estimator.called,
            session_registry=mm.set_session_registry.called,
            batch_priority=mm.set_batch_priority.called,
            pin_worker_resolver=getattr(mm, "pin_worker_resolver", None) is not None,
            before_prepare=wp.set_before_prepare.called,
            worker_batch_priority=wp.set_batch_priority.called,
            prompt_progress_sink=wp.set_prompt_progress_sink.called,
            no_loop_yet=not any(e.endswith(".started") for e in harness.events),
        )

    mm.start.side_effect = _on_start
    async with main_mod.lifespan(_app()):
        pass
    assert seen and all(seen.values()), seen


@pytest.mark.asyncio
async def test_shutdown_stops_everything_in_reverse_start_order(harness):
    async with main_mod.lifespan(_app()):
        await asyncio.sleep(0)  # let the loops start
        assert "ModelManager.stop" not in harness.events
    ev = harness.events
    for expected in (
        "BatchProcessor.stop",
        "MCPRegistry.stop",
        "TrtllmCompileManager.stop",
        "BackendProcessManager.stop",
        "ModelManager.stop",
        "SessionRegistry.stop",
        "DurationEstimator.stop",
        "GPUManager.stop",
        "db.dispose",
        "redis.close",
        "_idle_eviction_loop.stopped",
        "self_update.stopped",
        "retention_loop.stopped",
    ):
        assert expected in ev, expected
    order = [
        "BatchProcessor.stop",
        "MCPRegistry.stop",
        "BackendProcessManager.stop",
        "ModelManager.stop",
        "GPUManager.stop",
        "db.dispose",
        "redis.close",
    ]
    assert [ev.index(e) for e in order] == sorted(ev.index(e) for e in order)
    # Loops stop before the components they use.
    assert ev.index("_idle_eviction_loop.stopped") < ev.index("ModelManager.stop")
    # ModelManager.stop() only stops its own loops: nothing unloads models.
    assert not harness.instances["ModelManager"].unload.called


@pytest.mark.asyncio
async def test_failed_startup_tears_down_what_already_started(harness):
    harness.instances["BatchProcessor"].start.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        async with main_mod.lifespan(_app()):
            pytest.fail("startup should have failed")
    ev = harness.events
    assert "ModelManager.stop" in ev
    assert "GPUManager.stop" in ev
    assert "_service_health_loop.stopped" in ev
    assert "self_update.stopped" in ev
    assert "redis.close" in ev
    assert "BatchProcessor.stop" not in ev  # never started


@pytest.mark.asyncio
async def test_a_stuck_stop_is_bounded_and_does_not_block_the_rest(harness, monkeypatch):
    monkeypatch.setattr(main_mod, "_SHUTDOWN_STEP_TIMEOUT_S", 0.2)

    async def _hang():
        await asyncio.Event().wait()

    harness.instances["MCPRegistry"].stop.side_effect = _hang
    harness.instances["BatchProcessor"].stop.side_effect = RuntimeError("stop failed")
    loop = asyncio.get_running_loop()
    async with main_mod.lifespan(_app()):
        shutdown_started = loop.time()
    # The hung step costs one step timeout, not forever.
    assert loop.time() - shutdown_started < 2
    assert "ModelManager.stop" in harness.events
    assert "redis.close" in harness.events
