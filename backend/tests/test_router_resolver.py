"""Unit tests for RouterResolver.

Bloque 20, Etapa 6.

Verifies the resolver's decision walk under the enumerated cases:

    1. primary_loaded (loaded always wins, in-flight count irrelevant)
    2. loadable_no_disruption (unloaded, no busy neighbours → load here)
    3. session_veto → next
    4. loaded but busy → still stay (queueing beats cold-start fallback)
    5. unloaded target + neighbour busy → skip (would disrupt)
    6. loop detected → skip
    7. missing / disabled target → skip
    8. nested router → delegates
    9. routing_enabled=False → skipped entirely (plain profile)
   10. no target survives → falls back to router's own base
   11. router honors first-loadable ordering

Updated 2026-09-15: previously the resolver skipped LOADED-but-busy
targets when the estimator predicted a long remaining wait. That heuristic
caused a production ping-pong (Ollama serving + router picks vLLM 26b →
pressure-eviction fails → worker stuck in error). The rule now is:
loaded wins, period.

The resolver has three collaborators (ProfileRegistry, ModelManager,
optional DurationEstimator + SessionRegistry). All are mocked with tiny
dataclasses so the tests exercise the resolver's own control flow, not
the collaborators' side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import MagicMock

import pytest

from ocabra.core.router_resolver import RouterResolver


class _Status(str, Enum):
    CONFIGURED = "configured"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADED = "unloaded"
    ERROR = "error"


@dataclass
class _FakeProfile:
    profile_id: str
    base_model_id: str
    enabled: bool = True
    routing_targets: list[str] | None = None
    load_overrides: dict | None = None


@dataclass
class _FakeState:
    status: _Status = _Status.LOADED
    model_id: str = ""


@dataclass
class _FakeProfileRegistry:
    profiles: dict[str, _FakeProfile] = field(default_factory=dict)

    async def get(self, profile_id: str) -> _FakeProfile | None:
        return self.profiles.get(profile_id)


@dataclass
class _FakeModelManager:
    states: dict[str, _FakeState] = field(default_factory=dict)
    busy: set[str] = field(default_factory=set)

    async def get_state(self, model_id: str):
        # Import inside the function to avoid a hard import at module level.
        from ocabra.core.model_manager import ModelStatus

        state = self.states.get(model_id)
        if state is None:
            return None
        # Adapt _FakeState.status to the real ModelStatus enum the resolver
        # imports at runtime.
        state.status = ModelStatus(state.status.value)
        state.model_id = model_id
        return state

    def is_busy(self, model_id: str) -> bool:
        return model_id in self.busy

    def any_busy_worker_except(self, target_id, *, also_exclude=()) -> bool:
        excluded = {target_id, *also_exclude}
        return any(m not in excluded for m in self.busy)


def _make_resolver(*profiles, **kwargs) -> tuple[RouterResolver, _FakeProfileRegistry, _FakeModelManager]:
    registry = _FakeProfileRegistry(profiles={p.profile_id: p for p in profiles})
    mm = _FakeModelManager(states=kwargs.get("states", {}))
    if "busy" in kwargs:
        mm.busy = set(kwargs["busy"])
    sr = kwargs.get("session_registry")
    est = kwargs.get("estimator")
    resolver = RouterResolver(
        profile_registry=registry,
        model_manager=mm,
        duration_estimator=est,
        session_registry=sr,
    )
    return resolver, registry, mm


class TestBasicResolution:
    @pytest.mark.asyncio
    async def test_primary_loaded_wins(self):
        router = _FakeProfile("g", "vllm/base", routing_targets=["t1", "t2"])
        t1 = _FakeProfile("t1", "vllm/a")
        t2 = _FakeProfile("t2", "vllm/b")
        resolver, _, _ = _make_resolver(
            router, t1, t2,
            states={"vllm/a": _FakeState(_Status.LOADED)},
        )
        result = await resolver.pick(router)
        assert result.target_profile_id == "t1"
        assert result.reason == "primary_loaded"

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_unloaded(self):
        """Primary UNLOADED but loadable — the resolver still picks it as
        the winner. The 'wait vs fallback' decision is Etapa 3+, ordering
        is respected."""
        router = _FakeProfile("g", "vllm/base", routing_targets=["t1", "t2"])
        t1 = _FakeProfile("t1", "vllm/a")
        t2 = _FakeProfile("t2", "vllm/b")
        resolver, _, _ = _make_resolver(
            router, t1, t2,
            states={"vllm/a": _FakeState(_Status.UNLOADED)},
        )
        result = await resolver.pick(router)
        assert result.target_profile_id == "t1"
        assert result.reason == "loadable_no_disruption"

    @pytest.mark.asyncio
    async def test_no_target_survives_returns_no_immediate_winner(self):
        """All targets missing/disabled/loop → we advertise the router's own
        base as the fallback via reason='no_immediate_winner', matching
        pre-Bloque 20 behaviour so callers don't need a special case."""
        router = _FakeProfile("g", "vllm/base", routing_targets=["missing1", "missing2"])
        resolver, _, _ = _make_resolver(router)
        result = await resolver.pick(router)
        assert result.reason == "no_immediate_winner"
        assert result.target_profile_id == "g"
        assert result.base_model_id == "vllm/base"


class TestSessionVeto:
    @pytest.mark.asyncio
    async def test_session_held_target_is_skipped(self):
        router = _FakeProfile("g", "vllm/base", routing_targets=["t1", "t2"])
        t1 = _FakeProfile("t1", "vllm/a")
        t2 = _FakeProfile("t2", "vllm/b")
        registry_mock = MagicMock()
        registry_mock.has_active_session_holding = lambda k: k == "vllm/a"
        resolver, _, _ = _make_resolver(
            router, t1, t2,
            states={
                "vllm/a": _FakeState(_Status.LOADED),
                "vllm/b": _FakeState(_Status.LOADED),
            },
            session_registry=registry_mock,
        )
        result = await resolver.pick(router)
        # vllm/a is loaded but reserved — must go to vllm/b instead.
        assert result.target_profile_id == "t2"
        assert result.reason == "primary_loaded"


class TestBusyHandling:
    @pytest.mark.asyncio
    async def test_loaded_target_wins_even_when_busy(self):
        """Loaded targets always win regardless of in-flight load. Queueing
        against a hot worker is strictly cheaper than a cold start on the
        fallback, and the router's ``fallback`` list is for the case
        ``primary can't load``, not ``primary is loaded but slow``."""
        router = _FakeProfile("g", "vllm/base", routing_targets=["t1", "t2"])
        t1 = _FakeProfile("t1", "vllm/a")
        t2 = _FakeProfile("t2", "vllm/b")
        resolver, _, _ = _make_resolver(
            router, t1, t2,
            states={
                "vllm/a": _FakeState(_Status.LOADED),
                "vllm/b": _FakeState(_Status.LOADED),
            },
            busy={"vllm/a"},  # even 100 concurrent requests would not skip
        )
        result = await resolver.pick(router)
        assert result.target_profile_id == "t1"
        assert result.reason == "primary_loaded"

    @pytest.mark.asyncio
    async def test_unloaded_target_skipped_when_neighbour_busy(self):
        """Primary is UNLOADED, another worker is currently serving. Loading
        the primary here would likely need to evict the busy neighbour — the
        safer choice is to skip to the next candidate."""
        router = _FakeProfile("g", "vllm/base", routing_targets=["t1", "t2"])
        t1 = _FakeProfile("t1", "vllm/a")
        t2 = _FakeProfile("t2", "vllm/b")
        resolver, _, _ = _make_resolver(
            router, t1, t2,
            states={
                "vllm/a": _FakeState(_Status.UNLOADED),
                "vllm/b": _FakeState(_Status.UNLOADED),
            },
            busy={"vllm/other-model"},  # unrelated but currently busy
        )
        result = await resolver.pick(router)
        # Both t1 and t2 got skipped as "load_would_disrupt_busy"; nothing
        # survived, so we fall back to the router's own base.
        assert result.reason == "no_immediate_winner"

    @pytest.mark.asyncio
    async def test_unloaded_target_accepted_when_no_neighbour_busy(self):
        """Primary UNLOADED, GPU quiet elsewhere → load it. This is the
        common cold-start case for the router's first target."""
        router = _FakeProfile("g", "vllm/base", routing_targets=["t1", "t2"])
        t1 = _FakeProfile("t1", "vllm/a")
        t2 = _FakeProfile("t2", "vllm/b")
        resolver, _, _ = _make_resolver(
            router, t1, t2,
            states={"vllm/a": _FakeState(_Status.UNLOADED)},
            busy=set(),
        )
        result = await resolver.pick(router)
        assert result.target_profile_id == "t1"
        assert result.reason == "loadable_no_disruption"

    @pytest.mark.asyncio
    async def test_router_self_busy_does_not_disqualify_first_target(self):
        """Regression 2026-09-15: the stats middleware credits the router's
        own base_model_id with the in-flight counter for the current request.
        Without masking, ``any_busy_worker_except`` sees the router as a
        busy neighbour of every target and skips them all — the resolver
        would always fall through to ``no_immediate_winner`` and then to
        the router's base. Now the router's own base_model_id is masked
        from the busy check."""
        router = _FakeProfile("g", "ollama/gemma4:26b", routing_targets=["t1", "t2"])
        t1 = _FakeProfile("t1", "ollama/gemma4:26b-ctx160k")
        t2 = _FakeProfile("t2", "ollama/gemma4:26b")
        resolver, _, _ = _make_resolver(
            router, t1, t2,
            states={
                "ollama/gemma4:26b-ctx160k": _FakeState(_Status.UNLOADED),
                "ollama/gemma4:26b": _FakeState(_Status.UNLOADED),
            },
            # Middleware credited the current request under router.base_model_id
            busy={"ollama/gemma4:26b"},
        )
        result = await resolver.pick(router)
        assert result.target_profile_id == "t1"
        assert result.reason == "loadable_no_disruption"


class TestLoopAndMissing:
    @pytest.mark.asyncio
    async def test_missing_target_is_skipped(self):
        router = _FakeProfile("g", "vllm/base", routing_targets=["missing", "t1"])
        t1 = _FakeProfile("t1", "vllm/a")
        resolver, _, _ = _make_resolver(
            router, t1,
            states={"vllm/a": _FakeState(_Status.LOADED)},
        )
        result = await resolver.pick(router)
        assert result.target_profile_id == "t1"

    @pytest.mark.asyncio
    async def test_disabled_target_is_skipped(self):
        router = _FakeProfile("g", "vllm/base", routing_targets=["t1", "t2"])
        t1 = _FakeProfile("t1", "vllm/a", enabled=False)
        t2 = _FakeProfile("t2", "vllm/b")
        resolver, _, _ = _make_resolver(
            router, t1, t2,
            states={"vllm/b": _FakeState(_Status.LOADED)},
        )
        result = await resolver.pick(router)
        assert result.target_profile_id == "t2"

    @pytest.mark.asyncio
    async def test_self_reference_loop_is_skipped(self):
        """Router A → A must not recurse indefinitely — the visited set
        catches the immediate cycle and moves on."""
        router = _FakeProfile("g", "vllm/base", routing_targets=["g", "t1"])
        t1 = _FakeProfile("t1", "vllm/a")
        resolver, _, _ = _make_resolver(
            router, t1,
            states={"vllm/a": _FakeState(_Status.LOADED)},
        )
        result = await resolver.pick(router)
        assert result.target_profile_id == "t1"


class TestFeatureFlag:
    @pytest.mark.asyncio
    async def test_flag_off_treats_router_as_plain_profile(self, monkeypatch):
        """When ``routing_enabled=False`` the resolver never activates —
        the whole subsystem behaves as if no router existed. Emergency kill
        switch for prod."""
        from ocabra.config import settings

        monkeypatch.setattr(settings, "routing_enabled", False)
        router = _FakeProfile("g", "vllm/base", routing_targets=["t1"])
        t1 = _FakeProfile("t1", "vllm/a")
        resolver, _, _ = _make_resolver(router, t1)
        assert resolver.is_router(router) is False
