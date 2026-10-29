"""RealtimeSession must load/call the preset-derived worker key for the LLM,
not the bare profile model_id — see resolve_realtime_llm_runtime_plan() in
realtime.py for the resolution side of this fix.

Reproduced live 2026-10-29: every realtime session loaded the bare
base_model_id as its own separate worker instance, orphaned from whatever
preset-derived instance REST traffic was already using for the same model —
on a single GPU that doesn't fit both, each new session evicted the other
and cold-started again.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends.base import BackendCapabilities
from ocabra.core.realtime_session import RealtimeSession


def _make_session(*, llm_worker_key: str | None, load_overrides=None, preferred_gpu=None):
    ws = MagicMock()
    ws.send_text = AsyncMock()

    state = SimpleNamespace(
        capabilities=BackendCapabilities(chat=True),
        backend_model_id="gemma4-26b",
        status=SimpleNamespace(value="loaded"),
        display_name="Gemma 4 26B",
    )
    model_manager = MagicMock()
    model_manager.get_state = AsyncMock(return_value=state)
    model_manager.load = AsyncMock(return_value=state)

    worker_pool = MagicMock()
    worker_pool.get_worker = MagicMock(return_value=None)

    session = RealtimeSession(
        ws=ws,
        model_id="llama_cpp/gemma4-26b",
        worker_pool=worker_pool,
        model_manager=model_manager,
        llm_worker_key=llm_worker_key,
        llm_load_overrides=load_overrides,
        llm_preferred_gpu=preferred_gpu,
    )
    session._send_event = AsyncMock()  # noqa: SLF001
    session._send_error = AsyncMock()  # noqa: SLF001
    return session, model_manager, worker_pool


def test_defaults_worker_key_to_model_id_when_not_provided() -> None:
    session, _, _ = _make_session(llm_worker_key=None)
    assert session.llm_worker_key == session.llm_model_id == "llama_cpp/gemma4-26b"


def test_worker_key_can_differ_from_the_display_model_id() -> None:
    session, _, _ = _make_session(llm_worker_key="llama_cpp/gemma4-26b::02e08fbd7161")
    assert session.llm_model_id == "llama_cpp/gemma4-26b"
    assert session.llm_worker_key == "llama_cpp/gemma4-26b::02e08fbd7161"


@pytest.mark.asyncio
async def test_capability_checks_query_the_worker_key_not_the_display_id() -> None:
    session, model_manager, _ = _make_session(llm_worker_key="llama_cpp/gemma4-26b::abc")

    await session._should_use_native_audio()  # noqa: SLF001

    model_manager.get_state.assert_awaited_with("llama_cpp/gemma4-26b::abc")


@pytest.mark.asyncio
async def test_register_workers_pins_the_worker_key_not_the_display_id() -> None:
    session, _, _ = _make_session(llm_worker_key="llama_cpp/gemma4-26b::abc")
    session._session_registry = MagicMock()  # noqa: SLF001
    session._session_registry.register = AsyncMock()  # noqa: SLF001

    await session._register_workers()  # noqa: SLF001

    workers_held = session._session_registry.register.await_args.kwargs.get(  # noqa: SLF001
        "workers_held"
    )
    assert "llama_cpp/gemma4-26b::abc" in workers_held
    assert "llama_cpp/gemma4-26b" not in workers_held


@pytest.mark.asyncio
async def test_background_load_uses_ensure_worker_loaded_with_the_preset_worker_key() -> None:
    session, model_manager, worker_pool = _make_session(
        llm_worker_key="llama_cpp/gemma4-26b::abc",
        load_overrides={"llama_cpp": {"ctx_size": 65536}},
        preferred_gpu=1,
    )
    # Force the "not yet loaded, go load it" branch.
    model_manager.get_state = AsyncMock(return_value=None)
    worker_pool.get_worker = MagicMock(return_value=None)

    fake_ensure = AsyncMock(return_value=SimpleNamespace())
    with patch("ocabra.api.openai._deps.ensure_worker_loaded", new=fake_ensure):
        await session._load_remaining_models(include_stt=False)  # noqa: SLF001

    fake_ensure.assert_awaited_once()
    args, kwargs = fake_ensure.await_args
    assert args[1] == "llama_cpp/gemma4-26b"  # base_model_id (display id)
    assert args[2] == "llama_cpp/gemma4-26b::abc"  # worker_key (preset-derived)
    assert kwargs["preferred_gpu"] == 1
    model_manager.load.assert_not_awaited()  # must not fall back to the bare-id path
