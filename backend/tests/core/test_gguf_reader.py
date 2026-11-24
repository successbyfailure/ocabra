"""Characterization tests for the unified GGUF reader (DUP-03).

Builds a synthetic GGUF with a string-array vocab and a tensor table and checks
that the five local_scanner extractors, now thin wrappers over
``gguf_reader.read_gguf_header``, reproduce their old outputs.
"""

from __future__ import annotations

import struct
from pathlib import Path

from ocabra.core import gguf_reader
from ocabra.registry import local_scanner


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_string(key: str, value: str) -> bytes:
    return _str(key) + struct.pack("<I", 8) + _str(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _str(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_string_array(key: str, items: list[str]) -> bytes:
    out = _str(key) + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(items))
    for it in items:
        out += _str(it)
    return out


def _tensor(name: str, offset: int) -> bytes:
    # name, n_dims=1, dims=[1], ggml_type=0, offset
    return (
        _str(name)
        + struct.pack("<I", 1)
        + struct.pack("<Q", 1)
        + struct.pack("<I", 0)
        + struct.pack("<Q", offset)
    )


def _build_gguf(tmp_path: Path, *, tensors: list[str]) -> Path:
    kv = (
        _kv_string("general.architecture", "qwen35")
        + _kv_u32("general.file_type", 15)
        + _kv_u32("qwen35.context_length", 262144)
        + _kv_string("tokenizer.chat_template", "{{ '<tool_call>' }}{{ '<think>' }}")
        + _kv_string_array("tokenizer.ggml.tokens", ["a", "b", "c", "d", "e"])
        + _kv_u32("tokenizer.ggml.bos_token_id", 1)
        + _kv_u32("tokenizer.ggml.eos_token_id", 2)
    )
    kv_count = 7
    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", kv_count)
        + kv
    )
    tensor_table = b"".join(_tensor(name, i * 64) for i, name in enumerate(tensors))
    data = header + tensor_table + b"\x00" * 4096  # room so offsets are valid
    p = tmp_path / "model.gguf"
    p.write_bytes(data)
    return p


def test_reader_extracts_metadata_arrays_and_tensors(tmp_path: Path) -> None:
    p = _build_gguf(tmp_path, tensors=["output.weight", "blk.0.nextn.embed.weight"])
    h = gguf_reader.read_gguf_header(p)
    assert h is not None
    assert h.metadata["general.architecture"] == "qwen35"
    assert h.metadata["general.file_type"] == 15
    assert h.array_lengths["tokenizer.ggml.tokens"] == 5
    assert "blk.0.nextn.embed.weight" in h.tensor_names


def test_reader_is_cached_by_path_mtime_size(tmp_path: Path) -> None:
    p = _build_gguf(tmp_path, tensors=["output.weight"])
    gguf_reader._read_gguf_header_cached.cache_clear()
    gguf_reader.read_gguf_header(p)
    gguf_reader.read_gguf_header(p)
    info = gguf_reader._read_gguf_header_cached.cache_info()
    assert info.hits >= 1


def test_tokenizer_fingerprint(tmp_path: Path) -> None:
    p = _build_gguf(tmp_path, tensors=["output.weight"])
    assert local_scanner.parse_gguf_tokenizer_fingerprint(p) == (5, 1, 2)


def test_file_type(tmp_path: Path) -> None:
    p = _build_gguf(tmp_path, tensors=["output.weight"])
    assert local_scanner.parse_gguf_file_type(p) == 15


def test_context_length(tmp_path: Path) -> None:
    p = _build_gguf(tmp_path, tensors=["output.weight"])
    assert local_scanner.parse_gguf_context_length(p) == 262144


def test_chat_template_and_capabilities(tmp_path: Path) -> None:
    p = _build_gguf(tmp_path, tensors=["output.weight"])
    assert "<tool_call>" in local_scanner.parse_gguf_chat_template(p)
    caps = local_scanner.gguf_template_capabilities(p)
    assert caps == {"tools": True, "reasoning": True}


def test_speculative_type_mtp(tmp_path: Path) -> None:
    p = _build_gguf(tmp_path, tensors=["output.weight", "blk.0.nextn.embed.weight"])
    assert local_scanner.parse_gguf_speculative_type(p) == "draft-mtp"


def test_speculative_type_none_without_markers(tmp_path: Path) -> None:
    p = _build_gguf(tmp_path, tensors=["output.weight", "blk.0.attn.weight"])
    assert local_scanner.parse_gguf_speculative_type(p) is None


def test_missing_file_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "nope.gguf"
    assert gguf_reader.read_gguf_header(missing) is None
    assert local_scanner.parse_gguf_tokenizer_fingerprint(missing) == (None, None, None)
    assert local_scanner.parse_gguf_file_type(missing) is None
    assert local_scanner.parse_gguf_context_length(missing) is None
    assert local_scanner.parse_gguf_chat_template(missing) is None
    assert local_scanner.parse_gguf_speculative_type(missing) is None


def test_not_a_gguf_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 64)
    assert gguf_reader.read_gguf_header(p) is None
