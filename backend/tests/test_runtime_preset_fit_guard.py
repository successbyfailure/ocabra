"""Guard: a preset change must never leave a profile with no preset that fits.

Regression 2026-11-08: gemma4:26b got a single ``fixed`` 262144 preset, the
planner only grants ~177k on the 3090, and every request (even "hola") got
``400 profile has no enabled runtime preset that fits``. Exercised through the
real HTTP handlers (TestClient), not by calling the helper directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.api.internal import profiles as profiles_api


def _preset(key: str, ctx: int, enabled: bool = True):
    return SimpleNamespace(preset_key=key, context_tokens=ctx, enabled=enabled)


class _Registry:
    def __init__(self, presets):
        self.profile = SimpleNamespace(
            profile_id="gemma4:26b",
            profile_kind="execution",
            base_model_id="llamacpp_mks/gemma4:26b",
            runtime_presets=list(presets),
        )
        self.deleted: list[str] = []
        self.updates: list[dict] = []

    async def get(self, profile_id):
        return self.profile if profile_id == self.profile.profile_id else None

    async def create_runtime_preset(self, _session, _profile_id, payload):
        row = _preset(
            payload["preset_key"], payload["context_tokens"], payload.get("enabled", True)
        )
        self.profile.runtime_presets.append(row)
        return row

    async def update_runtime_preset(self, _session, _profile_id, key, patch):
        row = next(p for p in self.profile.runtime_presets if p.preset_key == key)
        for name, value in patch.items():
            setattr(row, name, value)
        self.updates.append(dict(patch))
        return row

    async def delete_runtime_preset(self, _session, _profile_id, key):
        self.deleted.append(key)
        self.profile.runtime_presets = [
            p for p in self.profile.runtime_presets if p.preset_key != key
        ]


@pytest.fixture
def harness(monkeypatch):
    """App with the profiles router; the planner says 'fits' iff ctx <= 176876."""

    async def fake_resolve(profile, _state, _gpus, force=False):
        return [
            {
                "preset_key": p.preset_key,
                "resolved": True,
                "requested_context": p.context_tokens,
                "effective_context": min(p.context_tokens, 176876),
                "fits": p.context_tokens <= 176876,
            }
            for p in profile.runtime_presets
        ]

    monkeypatch.setattr(profiles_api.runtime_capacity_cache, "resolve", fake_resolve)
    monkeypatch.setattr(
        profiles_api,
        "RuntimePresetOut",
        SimpleNamespace(
            model_validate=lambda p: SimpleNamespace(
                model_dump=lambda mode=None: {
                    "preset_key": p.preset_key,
                    "context_tokens": p.context_tokens,
                }
            )
        ),
    )

    def build(presets):
        app = FastAPI()
        app.include_router(profiles_api.router, prefix="/ocabra")
        registry = _Registry(presets)
        app.state.profile_registry = registry
        app.state.model_manager = SimpleNamespace(
            get_state=AsyncMock(return_value=SimpleNamespace()), gpu_manager=None
        )
        app.dependency_overrides[get_current_user] = lambda: UserContext(
            user_id=None,
            username="__test__",
            role="system_admin",
            group_ids=[],
            accessible_model_ids=set(),
            is_anonymous=False,
        )
        return TestClient(app), registry

    return build


def _create_body(key: str, ctx: int) -> dict:
    return {
        "preset_key": key,
        "label": key,
        "context_tokens": ctx,
        "parallelism": 4,
        "kv_cache_dtype": "q8",
        "placement_policy": "forced",
        "preferred_gpus": [1],
        "is_default": True,
    }


def test_creating_the_only_preset_above_capacity_is_refused_and_rolled_back(harness):
    client, registry = harness([])
    resp = client.post(
        "/ocabra/profiles/gemma4:26b/runtime-presets", json=_create_body("normal", 262144)
    )

    assert resp.status_code == 400
    assert "rechazaría todas las peticiones" in resp.json()["detail"]
    assert registry.deleted == ["normal"]
    assert registry.profile.runtime_presets == []


def test_creating_a_preset_that_fits_is_allowed(harness):
    client, registry = harness([])
    resp = client.post(
        "/ocabra/profiles/gemma4:26b/runtime-presets", json=_create_body("normal", 163840)
    )

    assert resp.status_code == 201
    assert registry.deleted == []


def test_patching_the_last_fitting_preset_above_capacity_is_reverted(harness):
    client, registry = harness([_preset("normal", 163840)])
    resp = client.patch(
        "/ocabra/profiles/gemma4:26b/runtime-presets/normal", json={"context_tokens": 262144}
    )

    assert resp.status_code == 400
    assert registry.profile.runtime_presets[0].context_tokens == 163840  # restored
    assert registry.updates[-1] == {"context_tokens": 163840}


def test_a_non_fitting_preset_is_fine_while_another_one_fits(harness):
    client, _registry = harness([_preset("normal", 163840)])
    resp = client.post(
        "/ocabra/profiles/gemma4:26b/runtime-presets",
        json={**_create_body("huge", 262144), "is_default": False},
    )

    assert resp.status_code == 201


def test_deleting_the_only_fitting_preset_is_refused_but_deleting_the_last_one_is_not(harness):
    client, registry = harness([_preset("normal", 163840), _preset("huge", 262144)])
    resp = client.delete("/ocabra/profiles/gemma4:26b/runtime-presets/normal")
    assert resp.status_code == 400
    assert registry.deleted == []

    client2, registry2 = harness([_preset("normal", 262144)])
    # removing every preset returns the profile to the no-preset path: allowed
    assert client2.delete("/ocabra/profiles/gemma4:26b/runtime-presets/normal").status_code == 200
    assert registry2.deleted == ["normal"]
