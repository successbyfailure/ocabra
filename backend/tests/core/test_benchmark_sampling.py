"""Muestreo del fabricante en ejecuciones de benchmark (override en el servidor)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ocabra.api.internal.benchmarks import BenchmarkRunCreate
from ocabra.benchmarks.sampling_override import (
    BenchmarkSamplingMiddleware,
    apply_override,
    current_sampling_override,
)
from ocabra.core.inference.resolution import merge_profile_defaults

OVERRIDE = {
    "profile": "gemma4:12b",
    "params": {"temperature": 1.0, "top_k": 64, "min_p": 0.0, "max_tokens": 1},
    "api_key": "sk-runner",
}


def _profile(profile_id: str):
    return SimpleNamespace(profile_id=profile_id, request_defaults={"top_k": 20}, assets={})


def test_sin_override_manda_el_cliente() -> None:
    merged = merge_profile_defaults(_profile("gemma4:12b"), {"temperature": 0})
    assert merged == {"top_k": 20, "temperature": 0}


def test_el_override_pisa_el_muestreo_solo_del_perfil_medido() -> None:
    marker = current_sampling_override.set(OVERRIDE)
    try:
        merged = merge_profile_defaults(
            _profile("gemma4:12b"), {"temperature": 0, "max_tokens": 4096}
        )
        other = merge_profile_defaults(_profile("gemma4:26b"), {"temperature": 0})
    finally:
        current_sampling_override.reset(marker)
    assert merged["temperature"] == 1.0 and merged["top_k"] == 64 and merged["min_p"] == 0.0
    # El tope de salida sigue siendo el del benchmark: solo se pisa muestreo.
    assert merged["max_tokens"] == 4096
    assert other == {"top_k": 20, "temperature": 0}
    assert apply_override("gemma4:12b", {"temperature": 0}) == {"temperature": 0}


async def _call(manager, token: str | None) -> dict | None:
    seen: dict = {}

    async def app(scope, receive, send):
        seen["override"] = current_sampling_override.get()

    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
    scope = {
        "type": "http",
        "headers": headers,
        "app": SimpleNamespace(state=SimpleNamespace(benchmark_manager=manager)),
    }
    await BenchmarkSamplingMiddleware(app)(scope, None, None)
    return seen["override"]


@pytest.mark.asyncio
async def test_solo_la_clave_del_runner_activa_el_override() -> None:
    manager = SimpleNamespace(sampling_override=OVERRIDE)
    assert await _call(manager, "sk-runner") == OVERRIDE
    assert await _call(manager, "sk-de-un-usuario") is None
    assert await _call(manager, None) is None
    assert await _call(SimpleNamespace(sampling_override=None), "sk-runner") is None
    assert current_sampling_override.get() is None


def test_la_api_valida_el_modo_de_muestreo() -> None:
    assert BenchmarkRunCreate(app="ifeval", suite="quick", models=["m"]).sampling == "benchmark"
    assert (
        BenchmarkRunCreate(
            app="ifeval", suite="quick", models=["m"], sampling="recommended"
        ).sampling
        == "recommended"
    )
    with pytest.raises(ValidationError):
        BenchmarkRunCreate(app="ifeval", suite="quick", models=["m"], sampling="otro")
    with pytest.raises(ValidationError):
        BenchmarkRunCreate(
            app="ifeval", suite="quick", models=["m"], sampling="recommended", node="Mapache"
        )
