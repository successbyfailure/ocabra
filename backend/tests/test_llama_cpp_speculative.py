"""Sprint 17.4 — speculative decoding compatibility tests.

Covers:
    * GGUF tokenizer fingerprint parsing in ``local_scanner``.
    * The ``/ocabra/models/{id}/speculative-candidates`` endpoint.
    * ``ModelManager._resolve_keep_alive_seconds`` per-model override.
"""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus
from ocabra.registry.local_scanner import (
    LocalScanner,
    parse_gguf_tokenizer_fingerprint,
)

# ---------------------------------------------------------------------------
# Synthetic GGUF helpers
# ---------------------------------------------------------------------------


_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _build_synthetic_gguf(*, vocab_size: int, bos_id: int, eos_id: int) -> bytes:
    """Build a minimal valid GGUF v3 buffer with just the tokenizer keys."""
    header = b"GGUF" + struct.pack("<I", 3)  # magic + version
    header += struct.pack("<Q", 0)  # tensor_count
    header += struct.pack("<Q", 3)  # kv_count

    # tokens array (item_type=string, count=vocab_size with empty strings)
    body = _pack_string("tokenizer.ggml.tokens")
    body += struct.pack("<I", _GGUF_TYPE_ARRAY)
    body += struct.pack("<I", _GGUF_TYPE_STRING)
    body += struct.pack("<Q", vocab_size)
    for _ in range(vocab_size):
        body += _pack_string("")

    # bos
    body += _pack_string("tokenizer.ggml.bos_token_id")
    body += struct.pack("<I", _GGUF_TYPE_UINT32)
    body += struct.pack("<I", bos_id)

    # eos
    body += _pack_string("tokenizer.ggml.eos_token_id")
    body += struct.pack("<I", _GGUF_TYPE_UINT32)
    body += struct.pack("<I", eos_id)

    return header + body


def test_parse_gguf_tokenizer_fingerprint_synthetic(tmp_path: Path) -> None:
    target = tmp_path / "demo.gguf"
    target.write_bytes(_build_synthetic_gguf(vocab_size=128256, bos_id=1, eos_id=2))

    vocab_size, bos_id, eos_id = parse_gguf_tokenizer_fingerprint(target)

    assert vocab_size == 128256
    assert bos_id == 1
    assert eos_id == 2


def test_parse_gguf_tokenizer_fingerprint_returns_none_on_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOTGGUF" + b"\x00" * 32)

    vocab_size, bos_id, eos_id = parse_gguf_tokenizer_fingerprint(bad)

    assert vocab_size is None
    assert bos_id is None
    assert eos_id is None


@pytest.mark.asyncio
async def test_local_scanner_indexes_tokenizer_fingerprint(tmp_path: Path) -> None:
    main = tmp_path / "main.gguf"
    main.write_bytes(_build_synthetic_gguf(vocab_size=4096, bos_id=1, eos_id=2))
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(_build_synthetic_gguf(vocab_size=4096, bos_id=1, eos_id=2))
    incompatible = tmp_path / "other.gguf"
    incompatible.write_bytes(_build_synthetic_gguf(vocab_size=8192, bos_id=1, eos_id=2))

    scanner = LocalScanner()
    models = await scanner.scan(tmp_path)

    fingerprints = {m.path: (m.vocab_size, m.bos_id, m.eos_id) for m in models}
    assert fingerprints[str(main)] == (4096, 1, 2)
    assert fingerprints[str(draft)] == (4096, 1, 2)
    assert fingerprints[str(incompatible)] == (8192, 1, 2)


def _state(
    model_id: str,
    *,
    backend_type: str = "llama_cpp",
    extra_config: dict | None = None,
) -> ModelState:
    return ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type=backend_type,
        load_policy=LoadPolicy.ON_DEMAND,
        status=ModelStatus.LOADED,
        extra_config=extra_config or {},
    )


def test_resolve_keep_alive_uses_section_override() -> None:
    state = _state(
        "llama_cpp/foo",
        extra_config={"llama_cpp": {"keep_alive_seconds": 42}},
    )
    assert ModelManager._resolve_keep_alive_seconds(state, 600) == 42


def test_resolve_keep_alive_uses_top_level_override() -> None:
    state = _state(
        "llama_cpp/foo",
        extra_config={"keep_alive_seconds": 99},
    )
    assert ModelManager._resolve_keep_alive_seconds(state, 600) == 99


def test_resolve_keep_alive_falls_back_to_global() -> None:
    state = _state("llama_cpp/foo", extra_config={})
    assert ModelManager._resolve_keep_alive_seconds(state, 600) == 600


# ---------------------------------------------------------------------------
# /speculative-candidates endpoint
# ---------------------------------------------------------------------------


class _FakeModelConfig:
    def __init__(
        self,
        model_id: str,
        backend_type: str,
        vocab_size: int | None,
        bos_id: int | None,
        eos_id: int | None,
        display_name: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.backend_type = backend_type
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.display_name = display_name


class _FakeResult:
    def __init__(self, items):
        self._items = items

    def scalar_one_or_none(self):
        return self._items[0] if self._items else None

    def scalars(self):
        return self

    def all(self):
        return list(self._items)


class _FakeSession:
    """Returns the queued results in order on each ``execute()`` call."""

    def __init__(self, results):
        self._results = list(results)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, _query):
        if not self._results:
            return _FakeResult([])
        return _FakeResult(self._results.pop(0))


@pytest.mark.asyncio
async def test_list_speculative_candidates_filters_compatible() -> None:
    from ocabra.api.internal import models as models_api

    target_state = _state("llama_cpp/main")
    target_row = _FakeModelConfig("llama_cpp/main", "llama_cpp", 4096, 1, 2)
    candidate = _FakeModelConfig("llama_cpp/draft", "llama_cpp", 4096, 1, 2, display_name="Draft")

    mm = MagicMock()

    async def _get_state(_mid):
        return target_state

    mm.get_state = _get_state

    request = MagicMock()
    request.app.state.model_manager = mm

    session = _FakeSession([[target_row], [candidate]])

    def _factory():
        return session

    # ``AsyncSessionLocal`` is imported inside the endpoint, so we patch the
    # module-level symbol on ``ocabra.database`` directly.
    with patch("ocabra.database.AsyncSessionLocal", _factory):
        payload = await models_api.list_speculative_candidates(
            "llama_cpp/main",
            request,
            _user=MagicMock(is_admin=True),
        )

    assert payload == [
        {
            "model_id": "llama_cpp/draft",
            "display_name": "Draft",
            "vocab_size": 4096,
            "bos_id": 1,
            "eos_id": 2,
        }
    ]


@pytest.mark.asyncio
async def test_list_speculative_candidates_returns_empty_when_no_fingerprint() -> None:
    from ocabra.api.internal import models as models_api

    target_state = _state("llama_cpp/main")
    target_row = _FakeModelConfig("llama_cpp/main", "llama_cpp", None, None, None)

    mm = MagicMock()

    async def _get_state(_mid):
        return target_state

    mm.get_state = _get_state

    request = MagicMock()
    request.app.state.model_manager = mm

    session = _FakeSession([[target_row]])

    with patch("ocabra.database.AsyncSessionLocal", lambda: session):
        payload = await models_api.list_speculative_candidates(
            "llama_cpp/main",
            request,
            _user=MagicMock(is_admin=True),
        )

    assert payload == []
