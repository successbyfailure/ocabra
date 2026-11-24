"""Row-lookup tables (token / per-layer embeddings) do not count as resident RAM.

qwen3.8-flash-next-coder keeps a 27 GB per-layer embedding table in host RAM,
in the second part of a split GGUF; counted as resident, the editor said the
model did not fit the node's RAM (2026-11-24).
"""

from __future__ import annotations

import struct
from pathlib import Path

from ocabra.core import host_memory

MIB = 1024 * 1024


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _write_gguf(path: Path, tensors: list[tuple[str, int]]) -> Path:
    """GGUF v3 with ``tensors`` = [(name, size_bytes)], data laid out in order."""
    kv = _str("general.architecture") + struct.pack("<I", 8) + _str("qwen4exp")
    table = b""
    offset = 0
    for name, size in tensors:
        table += _str(name) + struct.pack("<I", 1) + struct.pack("<Q", 1) + struct.pack("<I", 0)
        table += struct.pack("<Q", offset)
        offset += size
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", 1)
    head += kv + table
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head + b"\0" * pad)
        f.truncate(len(head) + pad + offset)
    return path


def test_lookup_tables_are_summed_across_split_parts(tmp_path):
    first = _write_gguf(
        tmp_path / "m-00001-of-00002.gguf",
        [("token_embd.weight", 2 * MIB), ("blk.0.ffn_up_exps.weight", 8 * MIB)],
    )
    _write_gguf(tmp_path / "m-00002-of-00002.gguf", [("per_layer_token_embd.weight", 30 * MIB)])
    assert host_memory.lookup_tables_mb(first) == 32


def test_a_single_file_counts_its_own_tables(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf", [("token_embd.weight", 3 * MIB), ("output.weight", 3 * MIB)])
    assert host_memory.lookup_tables_mb(path) == 3


def test_lookup_tables_leave_the_resident_need():
    # 28 GB on the CPU side, 27.4 of them a per-layer table: ~600 MB resident.
    assert host_memory.llama_ram_need_mb(28000, model_file_mb=56000, mlock=False, lookup_mb=27400) == 600


def test_mlock_still_pins_the_whole_file():
    assert host_memory.llama_ram_need_mb(28000, model_file_mb=56000, mlock=True, lookup_mb=27400) == 56000


def test_experts_still_count():
    assert host_memory.llama_ram_need_mb(12000, model_file_mb=20000, mlock=False) == 12000


async def test_a_draft_with_experts_in_ram_counts_on_the_host(tmp_path, monkeypatch):
    """--cpu-moe-draft leaves the MTP draft's experts in RAM: llama-fit-params
    only projects the main model, so the backend adds them."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from ocabra.backends.llama_cpp_backend import LlamaCppBackend
    from ocabra.core import llama_fit

    model = _write_gguf(tmp_path / "m.gguf", [("token_embd.weight", MIB)])
    draft = _write_gguf(
        tmp_path / "mtp.gguf",
        [("blk.0.ffn_up_exps.weight", 40 * MIB), ("blk.0.attn_q.weight", 2 * MIB)],
    )
    backend = LlamaCppBackend()
    monkeypatch.setattr(backend, "_resolve_model_file", lambda *a: model)
    monkeypatch.setattr(backend, "_resolve_draft_model_path", lambda *a: draft)
    monkeypatch.setattr(backend, "_get_binary_path", lambda *a: "/bin/llama-server")
    monkeypatch.setattr(
        llama_fit,
        "project",
        AsyncMock(return_value=SimpleNamespace(gpus={0: {"weights": 100}}, host={"model": 500}, host_mb=500)),
    )
    extra = {"llama_cpp": {"speculative": {"spec_type": "draft-mtp", "draft_model_id": str(draft), "draft_cpu_moe": True}}}
    out = await backend.project_memory("m", [0], extra)
    assert out["host_breakdown"]["draft"] == 40
    assert out["host_mb"] == 540
