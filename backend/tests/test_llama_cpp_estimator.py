"""Tests for the deterministic llama.cpp / GGUF VRAM estimator (Sprint 17.2)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from ocabra.core.llama_cpp_estimator import (
    _KV_BYTES_PER_ELEMENT,
    estimate_vram,
    estimate_vram_safe,
)
from ocabra.schemas.backend_load import LlamaCppLoadConfig

# Path to a real Qwen2.5-0.5B GGUF that already lives on the dev machine; tests
# fall back to a synthetic fixture when this file is unavailable in CI.
_REAL_GGUF = Path(
    "/docker/ai-models/ocabra/models/huggingface/Qwen--Qwen2.5-0.5B-Instruct-GGUF/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)


# ---------------------------------------------------------------------------
# Synthetic GGUF builder — minimal header that the estimator can consume.
# ---------------------------------------------------------------------------


def _write_string(buf: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def _write_kv_string(buf: bytearray, key: str, value: str) -> None:
    _write_string(buf, key)
    buf += struct.pack("<I", 8)  # GGUF_TYPE_STRING
    _write_string(buf, value)


def _write_kv_uint32(buf: bytearray, key: str, value: int) -> None:
    _write_string(buf, key)
    buf += struct.pack("<I", 4)  # GGUF_TYPE_UINT32
    buf += struct.pack("<I", value)


def _build_synthetic_gguf(
    path: Path,
    *,
    arch: str = "llama",
    n_layers: int = 32,
    embedding_length: int = 4096,
    head_count: int = 32,
    head_count_kv: int = 8,
    padding_bytes: int = 1024,
) -> Path:
    body = bytearray()
    body += b"GGUF"
    body += struct.pack("<I", 3)  # version
    body += struct.pack("<Q", 0)  # tensor_count
    # We will rewrite kv_count once we know how many we wrote.
    kv_count_offset = len(body)
    body += struct.pack("<Q", 0)
    kv_count = 0

    _write_kv_string(body, "general.architecture", arch)
    kv_count += 1
    _write_kv_uint32(body, f"{arch}.block_count", n_layers)
    kv_count += 1
    _write_kv_uint32(body, f"{arch}.embedding_length", embedding_length)
    kv_count += 1
    _write_kv_uint32(body, f"{arch}.attention.head_count", head_count)
    kv_count += 1
    _write_kv_uint32(body, f"{arch}.attention.head_count_kv", head_count_kv)
    kv_count += 1

    body[kv_count_offset : kv_count_offset + 8] = struct.pack("<Q", kv_count)
    # Fake "weights" so model_bytes is non-trivial.
    body += b"\x00" * padding_bytes

    path.write_bytes(bytes(body))
    return path


# ---------------------------------------------------------------------------
# Synthetic-fixture tests (always run)
# ---------------------------------------------------------------------------


def test_estimate_vram_synthetic_default(tmp_path: Path) -> None:
    gguf = _build_synthetic_gguf(tmp_path / "tiny.gguf")
    config = LlamaCppLoadConfig(ctx_size=4096, batch_size=512)

    result = estimate_vram(str(gguf), config)

    assert result["model_bytes"] == gguf.stat().st_size
    # n_layers=32, n_kv_heads=8, head_dim=128, ctx=4096, f16=2 bytes per element,
    # both K and V at 2 bytes each ⇒ 32 × 8 × 128 × 4096 × (2+2) = 536_870_912
    expected_kv = 32 * 8 * 128 * 4096 * (2 + 2)
    assert result["kv_bytes"] == expected_kv
    # Compute buffer: batch * embedding * 4 = 512 * 4096 * 4
    assert result["compute_buffer_bytes"] == 512 * 4096 * 4
    assert result["total_bytes"] == (
        result["model_bytes"] + result["kv_bytes"] + result["compute_buffer_bytes"]
    )


def test_estimate_vram_kv_quant_reduces_kv_bytes(tmp_path: Path) -> None:
    gguf = _build_synthetic_gguf(tmp_path / "tiny.gguf")
    base = estimate_vram(str(gguf), LlamaCppLoadConfig(ctx_size=4096, batch_size=512))
    quantized = estimate_vram(
        str(gguf),
        LlamaCppLoadConfig(
            ctx_size=4096,
            batch_size=512,
            flash_attn=True,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
        ),
    )
    # q4_0 = 0.5 bytes per element; should be exactly 1/4 of the f16 KV bytes.
    assert quantized["kv_bytes"] * 4 == base["kv_bytes"]
    assert quantized["kv_bytes"] < base["kv_bytes"]
    # Other components unchanged.
    assert quantized["model_bytes"] == base["model_bytes"]
    assert quantized["compute_buffer_bytes"] == base["compute_buffer_bytes"]


@pytest.mark.parametrize(
    "cache_type",
    ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"],
)
def test_estimate_vram_each_kv_cache_type_uses_correct_factor(
    tmp_path: Path, cache_type: str
) -> None:
    gguf = _build_synthetic_gguf(tmp_path / "tiny.gguf")
    config = LlamaCppLoadConfig(
        ctx_size=2048,
        batch_size=256,
        flash_attn=True,
        cache_type_k=cache_type,  # type: ignore[arg-type]
        cache_type_v=cache_type,  # type: ignore[arg-type]
    )
    result = estimate_vram(str(gguf), config)

    bytes_per_elem = _KV_BYTES_PER_ELEMENT[cache_type]
    expected = int(32 * 8 * 128 * 2048 * bytes_per_elem * 2)  # K + V
    assert result["kv_bytes"] == expected


def test_estimate_vram_scales_with_ctx_size(tmp_path: Path) -> None:
    gguf = _build_synthetic_gguf(tmp_path / "tiny.gguf")
    small = estimate_vram(str(gguf), LlamaCppLoadConfig(ctx_size=2048))
    big = estimate_vram(str(gguf), LlamaCppLoadConfig(ctx_size=4096))
    # KV cache is linear in context length.
    assert big["kv_bytes"] == small["kv_bytes"] * 2


def test_estimate_vram_safe_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert estimate_vram_safe(str(tmp_path / "nope.gguf"), LlamaCppLoadConfig()) is None


def test_estimate_vram_raises_for_non_gguf_file(tmp_path: Path) -> None:
    bogus = tmp_path / "not.gguf"
    bogus.write_bytes(b"NOPE" + b"\x00" * 256)
    with pytest.raises(ValueError, match="Not a GGUF file"):
        estimate_vram(str(bogus), LlamaCppLoadConfig())


def test_estimate_vram_uses_defaults_when_metadata_missing(tmp_path: Path) -> None:
    """A GGUF without architecture metadata still returns a sensible estimate."""
    body = bytearray()
    body += b"GGUF"
    body += struct.pack("<I", 3)  # version
    body += struct.pack("<Q", 0)  # tensor_count
    body += struct.pack("<Q", 0)  # kv_count = 0
    body += b"\x00" * 256
    path = tmp_path / "empty.gguf"
    path.write_bytes(bytes(body))

    result = estimate_vram(str(path), LlamaCppLoadConfig(ctx_size=2048))

    assert result["model_bytes"] == path.stat().st_size
    # n_layers=32, n_kv_heads=8, head_dim=128 (defaults), ctx=2048, f16
    assert result["kv_bytes"] == 32 * 8 * 128 * 2048 * 4


# ---------------------------------------------------------------------------
# Real-GGUF smoke test (skipped when the model file isn't present).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _REAL_GGUF.is_file(), reason="Qwen2.5-0.5B GGUF not available")
def test_estimate_vram_real_qwen_gguf() -> None:
    config = LlamaCppLoadConfig(ctx_size=8192, batch_size=512)
    result = estimate_vram(str(_REAL_GGUF), config)

    assert result["model_bytes"] == _REAL_GGUF.stat().st_size
    assert result["kv_bytes"] > 0
    assert result["compute_buffer_bytes"] > 0
    assert result["total_bytes"] == (
        result["model_bytes"] + result["kv_bytes"] + result["compute_buffer_bytes"]
    )

    # Cuantizar K y V debe reducir kv_bytes.
    quantized = estimate_vram(
        str(_REAL_GGUF),
        LlamaCppLoadConfig(
            ctx_size=8192,
            batch_size=512,
            flash_attn=True,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
        ),
    )
    assert quantized["kv_bytes"] < result["kv_bytes"]


# ---------------------------------------------------------------------------
# Schema validator tests.
# ---------------------------------------------------------------------------


def test_load_config_rejects_quantized_v_without_flash_attn() -> None:
    with pytest.raises(ValueError, match="flash_attn"):
        LlamaCppLoadConfig(cache_type_v="q4_0")


def test_load_config_accepts_quantized_v_with_flash_attn() -> None:
    cfg = LlamaCppLoadConfig(flash_attn=True, cache_type_v="q4_0")
    assert cfg.cache_type_v == "q4_0"


def test_load_config_allows_quantized_k_without_flash_attn() -> None:
    cfg = LlamaCppLoadConfig(cache_type_k="q4_0")
    assert cfg.cache_type_k == "q4_0"
    assert cfg.cache_type_v is None
