"""Cold llama.cpp models show the tools/reasoning their own GGUF template declares."""

import struct
from pathlib import Path

import pytest

from ocabra.core import model_manager as mm_module
from ocabra.core.model_manager import (
    LoadPolicy,
    ModelManager,
    ModelState,
    ModelStatus,
    resolve_display_capabilities,
)
from ocabra.core.worker_pool import WorkerPool
from ocabra.registry.local_scanner import gguf_template_capabilities, parse_gguf_chat_template

FLINT_LIKE = (
    "{%- if tools %}<|system|># Tools<tools></tools>{% endif -%}"
    "{%- if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif -%}"
    "<tool_call>{{ fn.name }}</tool_call>"
)


def _gguf(path: Path, template: str | None) -> Path:
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kvs = [(b"general.architecture", "flint")]
    if template is not None:
        kvs.append((b"tokenizer.chat_template", template))
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    for key, value in kvs:
        body += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8) + s(value)
    path.write_bytes(body)
    return path


def test_reads_the_template_and_its_markers(tmp_path):
    path = _gguf(tmp_path / "flint.gguf", FLINT_LIKE)
    assert parse_gguf_chat_template(path) == FLINT_LIKE
    assert gguf_template_capabilities(path) == {"tools": True, "reasoning": True}
    assert gguf_template_capabilities(_gguf(tmp_path / "none.gguf", None)) is None


def _cold_flint(path: Path) -> ModelState:
    return ModelState(
        model_id="llamacpp_mks/flint:40b",
        display_name="flint:40b",
        backend_type="llamacpp_mks",
        load_policy=LoadPolicy.ON_DEMAND,
        status=ModelStatus.CONFIGURED,
        extra_config={"llama_cpp": {"model_path": str(path)}},
    )


@pytest.mark.asyncio
async def test_a_cold_model_lists_what_its_template_declares(tmp_path, monkeypatch):
    """2026-11-11: flint:40b showed tools/reasoning off in the list (no hint in
    its name) although its template has <tool_call> and <think>."""
    from ocabra.backends.base import BackendCapabilities

    monkeypatch.setattr(mm_module, "_TEMPLATE_CAPS_BY_FILE", {})
    state = _cold_flint(_gguf(tmp_path / "flint.gguf", FLINT_LIKE))
    # What production showed: capabilities cached from an older load.
    state.capabilities = BackendCapabilities(
        chat=True, tools=False, reasoning=False, context_length=262144
    )
    before, source = resolve_display_capabilities(state)
    assert source == "cached"
    assert before["tools"] is False and before["reasoning"] is False

    mm = ModelManager(WorkerPool())
    mm._states[state.model_id] = state
    await mm.warm_template_capabilities()

    after, source = resolve_display_capabilities(state)
    assert after["tools"] is True and after["reasoning"] is True
    assert after["context_length"] == 262144  # the rest of the cache is kept


@pytest.mark.asyncio
async def test_a_loaded_model_keeps_what_its_worker_reported(tmp_path, monkeypatch):
    from ocabra.backends.base import BackendCapabilities

    monkeypatch.setattr(mm_module, "_TEMPLATE_CAPS_BY_FILE", {})
    state = _cold_flint(_gguf(tmp_path / "flint.gguf", FLINT_LIKE))
    state.status = ModelStatus.LOADED
    state.capabilities = BackendCapabilities(chat=True, tools=False, reasoning=False)
    mm = ModelManager(WorkerPool())
    mm._states[state.model_id] = state
    await mm.warm_template_capabilities()

    caps, source = resolve_display_capabilities(state)
    assert source == "live"
    assert caps["tools"] is False
