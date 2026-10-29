import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ocabra.api.internal import models as models_api
from ocabra.backends.base import BackendCapabilities
from ocabra.core.model_manager import ModelState, ModelStatus


class _FakeBackend:
    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        assert model_id == "My--Engine"
        return BackendCapabilities(completion=True, streaming=True, context_length=0)

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        assert model_id in {"My--Engine", "Qwen/Qwen3-32B-AWQ", "demo"}
        return 18000

    async def estimate_memory_profile(self, model_id: str, gpu_index: int, extra_config: dict):
        assert model_id == "Qwen/Qwen3-32B-AWQ"
        assert gpu_index == 1
        assert extra_config["vllm"]["max_model_len"] == 7800
        return {
            "status": "ok",
            "model_loading_memory_mb": 18575,
            "available_kv_cache_mb": 1956,
            "estimated_max_model_len": 7824,
            "maximum_concurrency": 1.0,
            "requested_context_length": 7800,
        }


class _FakeWorkerPool:
    async def get_backend(self, backend_type: str):
        assert backend_type == "tensorrt_llm"
        return _FakeBackend()


def test_apply_capability_fallbacks_uses_tensorrt_extra_config_context():
    state = SimpleNamespace(
        backend_type="tensorrt_llm",
        extra_config={"context_length": 4096},
    )

    payload = models_api._apply_capability_fallbacks(state, {"context_length": 0})

    assert payload["context_length"] == 4096


async def test_serialize_model_state_enriches_capabilities_from_backend():
    state = SimpleNamespace(
        model_id="tensorrt_llm/My--Engine",
        backend_model_id="My--Engine",
        backend_type="tensorrt_llm",
        extra_config={"context_length": 4096},
        to_dict=lambda: {
            "model_id": "tensorrt_llm/My--Engine",
            "backend_model_id": "My--Engine",
            "backend_type": "tensorrt_llm",
            "capabilities": {"context_length": 0},
        },
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker_pool=_FakeWorkerPool())))

    payload = await models_api._serialize_model_state(request, state, {})

    assert payload["capabilities"]["context_length"] == 4096


def test_apply_capability_fallbacks_prefers_tensorrt_extra_config_when_backend_reports_smaller_context():
    state = SimpleNamespace(
        backend_type="tensorrt_llm",
        backend_model_id="Engine",
        extra_config={"context_length": 8192},
    )

    payload = models_api._apply_capability_fallbacks(state, {"context_length": 512})

    assert payload["context_length"] == 8192


def test_apply_capability_fallbacks_reads_tokenizer_model_max_length(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "huggingface" / "Qwen--Qwen3.5-9B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text('{"model_max_length": 262144}')

    monkeypatch.setattr(models_api.settings, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(models_api.settings, "hf_cache_dir", "")

    state = SimpleNamespace(
        backend_type="vllm",
        backend_model_id="Qwen/Qwen3.5-9B",
        extra_config={},
    )

    payload = models_api._apply_capability_fallbacks(state, {"context_length": 0})

    assert payload["context_length"] == 262144


class _FakeManagerForOverlay:
    def __init__(self, canonical, derived):
        self._canonical = canonical
        self._derived = derived

    async def list_states(self, *, include_derived: bool = False):
        if include_derived:
            return self._canonical + self._derived
        return self._canonical


async def test_overlay_loaded_derived_state_surfaces_runtime_fields_on_base_entry():
    """qwen3.8:27b dashboard bug (2026-10-26): the runtime-preset system
    loads a derived worker key (base_model_id::<hash>), not the base
    model_id itself, so the base ModelState never transitions to LOADED.
    GET /ocabra/models (list_states without include_derived) only returns
    canonical entries, so it showed 'configured' forever — the dashboard's
    'Modelos activos' widget read exactly this list and showed "Sin modelos
    cargados" while GPU1 sat at 95% util / 17.7GB used serving real
    traffic."""
    base = ModelState(
        model_id="llama_cpp/qwen3.8-27b",
        display_name="qwen3.8:27b",
        backend_type="llama_cpp",
        status=ModelStatus.CONFIGURED,
    )
    derived = ModelState(
        model_id="llama_cpp/qwen3.8-27b::617f901a9ef7",
        display_name="qwen3.8:27b (override)",
        backend_type="llama_cpp",
        status=ModelStatus.LOADED,
        current_gpu=[0, 1],
        vram_used_mb=24751,
        loaded_at=datetime(2026, 10, 26, 21, 19, 24, tzinfo=timezone.utc),
        capabilities=BackendCapabilities(chat=True, completion=True, tools=True, streaming=True),
    )
    mm = _FakeManagerForOverlay([base], [derived])

    overlaid = await models_api._overlay_loaded_derived_state(mm, [base])

    assert len(overlaid) == 1
    result = overlaid[0]
    assert result.model_id == "llama_cpp/qwen3.8-27b"
    assert result.display_name == "qwen3.8:27b"
    assert result.status == ModelStatus.LOADED
    assert result.current_gpu == [0, 1]
    assert result.vram_used_mb == 24751
    assert result.loaded_at == derived.loaded_at
    # capabilities-fix regression (2026-10-26, tool-calling incident): the
    # base entry's own cached capabilities are stale defaults from before
    # it was ever loaded — the derived worker's live-probed capabilities
    # (e.g. tools=True once the chat-template override that was silently
    # stripping tool-calling support got removed) must win, not the base's.
    assert result.capabilities.tools is True


async def test_overlay_loaded_derived_state_leaves_already_loaded_base_alone():
    base = ModelState(
        model_id="ollama/mistral:7b",
        display_name="mistral:7b",
        backend_type="ollama",
        status=ModelStatus.LOADED,
        current_gpu=[0],
        vram_used_mb=5000,
    )
    mm = _FakeManagerForOverlay([base], [])

    overlaid = await models_api._overlay_loaded_derived_state(mm, [base])

    assert overlaid[0] is base


async def test_overlay_loaded_derived_state_leaves_unloaded_base_without_derived_sibling():
    base = ModelState(
        model_id="ollama/unused:7b",
        display_name="unused:7b",
        backend_type="ollama",
        status=ModelStatus.CONFIGURED,
    )
    mm = _FakeManagerForOverlay([base], [])

    overlaid = await models_api._overlay_loaded_derived_state(mm, [base])

    assert overlaid[0] is base


async def test_serialize_model_state_uses_extra_config_model_path_for_disk_size(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    models_root.mkdir()
    model_file = models_root / "weights.gguf"
    model_file.write_bytes(b"0123456789")
    monkeypatch.setattr(models_api.settings, "models_dir", str(models_root))

    async def _immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(models_api.asyncio, "to_thread", _immediate_to_thread)

    state = SimpleNamespace(
        model_id="bitnet/demo",
        backend_model_id="demo",
        backend_type="bitnet",
        extra_config={"model_path": str(model_file)},
        to_dict=lambda: {
            "model_id": "bitnet/demo",
            "backend_model_id": "demo",
            "backend_type": "bitnet",
            "capabilities": {"context_length": 0},
        },
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker_pool=None)))

    payload = await models_api._serialize_model_state(request, state, {})

    assert payload["disk_size_bytes"] == 10


@pytest.mark.asyncio
async def test_resolve_extra_config_path_rejects_paths_outside_models_dir(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    models_root.mkdir()
    outside_path = tmp_path / "outside"
    outside_path.mkdir()

    monkeypatch.setattr(models_api.settings, "models_dir", str(models_root))

    payload = models_api._resolve_extra_config_path({"model_path": str(outside_path)}, "model_path")

    assert payload is None


@pytest.mark.asyncio
async def test_resolve_extra_config_path_finds_llama_cpp_nested_model_path(tmp_path, monkeypatch):
    """llama.cpp stores model_path nested under extra_config['llama_cpp'],
    not at the top level like other backends — a model_path at the top of
    extra_config (the only place this used to look) never matches a real
    llama_cpp model's config, so every memory-estimate/disk-size lookup for
    it silently fell back to the heuristic. Regression for that gap."""
    models_root = tmp_path / "models"
    models_root.mkdir()
    gguf = models_root / "model.gguf"
    gguf.write_bytes(b"GGUF")

    monkeypatch.setattr(models_api.settings, "models_dir", str(models_root))

    payload = models_api._resolve_extra_config_path(
        {"llama_cpp": {"model_path": str(gguf)}}, "model_path"
    )

    assert payload == gguf


@pytest.mark.asyncio
async def test_resolve_extra_config_path_allows_ollama_shared_models_dir(tmp_path, monkeypatch):
    """GGUF weights llama_cpp reuses from Ollama's own blob store live
    outside settings.models_dir by design (docker-compose mounts
    /data/ollama_models separately) — they must still resolve, not be
    rejected by the models_dir-only safety check."""
    models_root = tmp_path / "models"
    models_root.mkdir()
    ollama_root = tmp_path / "ollama_models"
    blob = ollama_root / "blobs" / "sha256-abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"GGUF")

    monkeypatch.setattr(models_api.settings, "models_dir", str(models_root))
    monkeypatch.setattr(models_api.settings, "ollama_shared_models_dir", str(ollama_root))

    payload = models_api._resolve_extra_config_path(
        {"llama_cpp": {"model_path": str(blob)}}, "model_path"
    )

    assert payload == blob


@pytest.mark.asyncio
async def test_delete_model_files_rejects_symlink_escape(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    huggingface_root = models_root / "huggingface"
    huggingface_root.mkdir(parents=True)
    outside_path = tmp_path / "outside-model"
    outside_path.mkdir()
    symlink_path = huggingface_root / "malicious"
    os.symlink(outside_path, symlink_path)

    monkeypatch.setattr(models_api.settings, "models_dir", str(models_root))

    with pytest.raises(models_api.HTTPException) as exc_info:
        await models_api._delete_model_files("vllm/malicious", "vllm")

    assert exc_info.value.status_code == 400
    assert outside_path.exists()


def test_resolve_local_model_path_finds_matching_gguf(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    models_root.mkdir()
    gguf_path = models_root / "somewhere" / "gguf" / "ggml-model-i2_s.gguf"
    gguf_path.parent.mkdir(parents=True)
    gguf_path.write_bytes(b"gguf")

    monkeypatch.setattr(models_api.settings, "models_dir", str(models_root))
    monkeypatch.setattr(models_api.settings, "hf_cache_dir", "")

    resolved = models_api._resolve_local_model_path("microsoft/BitNet-b1.58-2B-4T-gguf::ggml-model-i2_s")

    assert resolved == gguf_path


def test_resolve_local_model_path_matches_gguf_with_dots_in_registered_name(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    models_root.mkdir()
    gguf_path = models_root / "tiiuae--falcon3-1b-instruct-1.58bit-gguf-i2_s.gguf"
    gguf_path.write_bytes(b"gguf")

    monkeypatch.setattr(models_api.settings, "models_dir", str(models_root))
    monkeypatch.setattr(models_api.settings, "hf_cache_dir", "")

    resolved = models_api._resolve_local_model_path("tiiuae--falcon3-1b-instruct-1.58bit-gguf-i2_s")

    assert resolved == gguf_path


@pytest.mark.asyncio
async def test_get_models_storage_returns_filesystem_usage(monkeypatch, tmp_path):
    models_root = tmp_path / "models"
    models_root.mkdir()

    class _Stat:
        f_frsize = 4096
        f_blocks = 100
        f_bavail = 25

    monkeypatch.setattr(models_api.settings, "models_dir", str(models_root))
    monkeypatch.setattr(models_api.os, "statvfs", lambda _path: _Stat())

    payload = await models_api.get_models_storage()

    assert payload == {
        "path": str(models_root),
        "total_bytes": 409600,
        "used_bytes": 307200,
        "free_bytes": 102400,
    }


@pytest.mark.asyncio
async def test_build_model_memory_estimate_uses_vllm_runtime_probe():
    state = SimpleNamespace(
        model_id="vllm/Qwen/Qwen3-32B-AWQ",
        backend_model_id="Qwen/Qwen3-32B-AWQ",
        backend_type="vllm",
        preferred_gpu=None,
        extra_config={"vllm": {"max_model_len": 7800}},
        capabilities=SimpleNamespace(context_length=32768),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                worker_pool=SimpleNamespace(get_backend=lambda _backend_type: _FakeBackend()),
                gpu_manager=SimpleNamespace(
                    get_state=lambda _gpu: SimpleNamespace(total_vram_mb=24576, free_vram_mb=24115)
                ),
            )
        )
    )

    async def _get_backend(_backend_type: str):
        return _FakeBackend()

    async def _get_gpu(_gpu: int):
        return SimpleNamespace(total_vram_mb=24576, free_vram_mb=24115)

    request.app.state.worker_pool.get_backend = _get_backend
    request.app.state.gpu_manager.get_state = _get_gpu

    payload = await models_api._build_model_memory_estimate(
        request=request,
        state=state,
        extra_config={"vllm": {"max_model_len": 7800}},
        preferred_gpu=1,
        run_probe=True,
    )

    assert payload["source"] == "runtime_probe"
    assert payload["gpu_index"] == 1
    assert payload["estimated_weights_mb"] == 18000
    assert payload["model_loading_memory_mb"] == 18575
    assert payload["estimated_kv_cache_mb"] == 1956
    assert payload["estimated_max_context_length"] == 7824


@pytest.mark.asyncio
async def test_build_model_memory_estimate_marks_missing_tensorrt_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(models_api.settings, "models_dir", str(tmp_path / "models"))
    state = SimpleNamespace(
        model_id="tensorrt_llm/demo",
        backend_model_id="demo",
        backend_type="tensorrt_llm",
        preferred_gpu=None,
        extra_config={},
        capabilities=SimpleNamespace(context_length=0),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                worker_pool=SimpleNamespace(),
                gpu_manager=SimpleNamespace(),
            )
        )
    )

    async def _get_backend(_backend_type: str):
        return _FakeBackend()

    async def _get_gpu(_gpu: int):
        return SimpleNamespace(total_vram_mb=24576, free_vram_mb=24115)

    request.app.state.worker_pool.get_backend = _get_backend
    request.app.state.gpu_manager.get_state = _get_gpu

    payload = await models_api._build_model_memory_estimate(
        request=request,
        state=state,
        extra_config={},
        preferred_gpu=None,
        run_probe=False,
    )

    assert payload["status"] == "error"
    assert payload["engine_present"] is False


# ---------------------------------------------------------------------------
# _estimate_llama_cpp_memory — unified with vram_planner.py (no more separate
# llama_cpp_estimator.py breakdown; see docs/tasks/gpu-unlock-benchmark-plan.md)
# ---------------------------------------------------------------------------

_DENSE_GGUF_KV = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 32,
    "qwen2.attention.head_count_kv": 8,
    "qwen2.attention.key_length": 128,
    "qwen2.attention.value_length": 128,
    "qwen2.embedding_length": 4096,
    "qwen2.context_length": 32768,
}

_HYBRID_GGUF_KV = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 65,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.full_attention_interval": 4,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.inner_size": 6144,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.group_count": 16,
    "qwen35.embedding_length": 5120,
    "qwen35.context_length": 262144,
}


def _gguf_bytes(kv: dict) -> bytes:
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", 0)
    out += struct.pack("<Q", len(kv))
    for key, value in kv.items():
        key_bytes = key.encode("utf-8")
        out += struct.pack("<Q", len(key_bytes)) + key_bytes
        if isinstance(value, str):
            out += struct.pack("<I", 8)
            val_bytes = value.encode("utf-8")
            out += struct.pack("<Q", len(val_bytes)) + val_bytes
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", int(value))
    return bytes(out)


def _write_gguf(models_dir: Path, kv: dict, *, size_mb: int = 100) -> Path:
    """Write a synthetic GGUF with a real header plus a padded-out file
    size. The padding is a sparse hole (truncate, not actual zero bytes) —
    size_mb can be set to a realistic multi-GB value (e.g. to match a real
    model's file size for a weights_mb calculation) without writing that
    many bytes for real. A previous version wrote the padding literally,
    which — at size_mb=16000 in one caller below — created a genuine ~15GB
    file per test run; pytest's tmp_path retention (last 3 sessions) then
    silently accumulated ~45GB in /tmp across a night of full-suite runs
    and twice filled the host's tmpfs (2026-10-27)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "model.gguf"
    body = _gguf_bytes(kv)
    target_size = max(len(body), size_mb * 1024 * 1024)
    with path.open("wb") as f:
        f.write(body)
        f.truncate(target_size)
    return path


def test_estimate_llama_cpp_memory_matches_vram_planner_for_a_dense_model(tmp_path, monkeypatch):
    from ocabra.core import vram_planner as vp

    models_dir = tmp_path / "models"
    monkeypatch.setattr(models_api.settings, "models_dir", str(models_dir))
    gguf_path = _write_gguf(models_dir, _DENSE_GGUF_KV, size_mb=50)

    state = SimpleNamespace(backend_model_id="dense-model")
    extra_config = {
        "model_path": str(gguf_path),
        "llama_cpp": {"gpu_layers": 999, "ctx_size": 8192, "parallel_slots": 1},
    }
    estimate = {"estimated_max_context_length": None, "notes": []}

    result = models_api._estimate_llama_cpp_memory(
        estimate=estimate,
        state=state,
        extra_config=extra_config,
        heuristic_mb=12345,
        total_vram_mb=24576,
    )

    arch = vp.arch_from_gguf(str(gguf_path))
    expected_total = vp.plan_llama_cpp_vram_mb(
        arch, 50.0, 8192, gpu_layers=999, kv_dtype_bytes=2.0, parallel_slots=1,
    )

    assert result["source"] == "gguf_header"
    assert result["model_loading_memory_mb"] == pytest.approx(expected_total, abs=1)
    assert result["estimated_weights_mb"] == 50
    assert result["estimated_max_context_length"] is not None
    assert result["fits_current_gpu"] is True


def test_estimate_llama_cpp_memory_accounts_for_quantized_kv_cache(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    monkeypatch.setattr(models_api.settings, "models_dir", str(models_dir))
    gguf_path = _write_gguf(models_dir, _DENSE_GGUF_KV, size_mb=50)

    state = SimpleNamespace(backend_model_id="dense-model")
    base_config = {
        "model_path": str(gguf_path),
        "llama_cpp": {"gpu_layers": 999, "ctx_size": 32768, "parallel_slots": 1},
    }
    f16 = models_api._estimate_llama_cpp_memory(
        estimate={"estimated_max_context_length": None, "notes": []},
        state=state, extra_config=base_config, heuristic_mb=0, total_vram_mb=24576,
    )

    quantized_config = {
        "model_path": str(gguf_path),
        "llama_cpp": {
            "gpu_layers": 999, "ctx_size": 32768, "parallel_slots": 1,
            "flash_attn": True, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        },
    }
    q8 = models_api._estimate_llama_cpp_memory(
        estimate={"estimated_max_context_length": None, "notes": []},
        state=state, extra_config=quantized_config, heuristic_mb=0, total_vram_mb=24576,
    )

    # q8_0 KV is half the bytes/element of f16 -> noticeably smaller KV cache.
    assert q8["estimated_kv_cache_mb"] < f16["estimated_kv_cache_mb"]


def test_estimate_llama_cpp_memory_accounts_for_self_speculative_mtp_overhead(tmp_path, monkeypatch):
    """The real production qwen3.8:27b config speculative={'spec_type': 'draft-mtp'}
    must widen the hybrid rs_cache and add the draft-context overhead, not just
    size a plain KV cache — this is what the old llama_cpp_estimator.py module
    (dense-only, no hybrid/MTP awareness) always got wrong."""
    models_dir = tmp_path / "models"
    monkeypatch.setattr(models_api.settings, "models_dir", str(models_dir))
    gguf_path = _write_gguf(models_dir, _HYBRID_GGUF_KV, size_mb=16000)

    state = SimpleNamespace(backend_model_id="qwen3.8-27b")
    plain_config = {
        "model_path": str(gguf_path),
        "llama_cpp": {"gpu_layers": 999, "ctx_size": 32768, "parallel_slots": 3},
    }
    plain = models_api._estimate_llama_cpp_memory(
        estimate={"estimated_max_context_length": None, "notes": []},
        state=state, extra_config=plain_config, heuristic_mb=0, total_vram_mb=24576,
    )

    mtp_config = {
        "model_path": str(gguf_path),
        "llama_cpp": {
            "gpu_layers": 999, "ctx_size": 32768, "parallel_slots": 3,
            "speculative": {"spec_type": "draft-mtp"},
        },
    }
    mtp = models_api._estimate_llama_cpp_memory(
        estimate={"estimated_max_context_length": None, "notes": []},
        state=state, extra_config=mtp_config, heuristic_mb=0, total_vram_mb=24576,
    )

    assert mtp["model_loading_memory_mb"] > plain["model_loading_memory_mb"]
    assert any("draft-mtp" in note for note in mtp["notes"])


def test_estimate_llama_cpp_memory_falls_back_to_heuristic_when_gguf_missing(monkeypatch):
    state = SimpleNamespace(backend_model_id="missing-model")
    estimate = {"estimated_max_context_length": None, "notes": []}

    result = models_api._estimate_llama_cpp_memory(
        estimate=estimate,
        state=state,
        extra_config={"llama_cpp": {"gpu_layers": 999}},
        heuristic_mb=4096,
        total_vram_mb=24576,
    )

    assert result["estimated_weights_mb"] == 4096
    assert "heurística" in result["notes"][0]


def test_estimate_llama_cpp_memory_resolves_ollama_blob_without_gguf_extension(tmp_path, monkeypatch):
    """Real production models reuse weights pulled through Ollama, whose
    blob store names files by content hash ("sha256-<hash>", no extension)
    — requiring a literal ".gguf" suffix on the resolved file (the bug that
    made this panel always fall back to the heuristic for every such model,
    including the flagship qwen3.8:27b) must not reject them."""
    models_dir = tmp_path / "models"
    ollama_dir = tmp_path / "ollama_models"
    monkeypatch.setattr(models_api.settings, "models_dir", str(models_dir))
    monkeypatch.setattr(models_api.settings, "ollama_shared_models_dir", str(ollama_dir))
    blob_dir = ollama_dir / "blobs"
    gguf_path = _write_gguf(blob_dir, _DENSE_GGUF_KV, size_mb=50)
    blob_path = blob_dir / "sha256-deadbeef"
    gguf_path.rename(blob_path)

    state = SimpleNamespace(backend_model_id="dense-model")
    result = models_api._estimate_llama_cpp_memory(
        estimate={"estimated_max_context_length": None, "notes": []},
        state=state,
        extra_config={"llama_cpp": {"model_path": str(blob_path), "gpu_layers": 999, "ctx_size": 8192}},
        heuristic_mb=0,
        total_vram_mb=24576,
    )

    assert result["source"] == "gguf_header"
    assert result["estimated_weights_mb"] == 50


def test_estimate_llama_cpp_memory_falls_back_to_heuristic_when_cpu_only(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    monkeypatch.setattr(models_api.settings, "models_dir", str(models_dir))
    gguf_path = _write_gguf(models_dir, _DENSE_GGUF_KV, size_mb=50)

    state = SimpleNamespace(backend_model_id="dense-model")
    estimate = {"estimated_max_context_length": None, "notes": []}

    result = models_api._estimate_llama_cpp_memory(
        estimate=estimate,
        state=state,
        extra_config={"model_path": str(gguf_path), "llama_cpp": {"gpu_layers": 0}},
        heuristic_mb=0,
        total_vram_mb=24576,
    )

    assert result["estimated_weights_mb"] is None
    assert "CPU" in result["notes"][0]


class TestModelsActivityLiveProgress:
    """2026-10-29 — /ocabra/models/activity now includes per-request live
    progress (input/output token estimates, last-activity age) so the
    model/GPU card can distinguish a slow request from a hung one."""

    @pytest.mark.asyncio
    async def test_activity_includes_per_request_progress(self) -> None:
        from ocabra.core.model_manager import LoadPolicy, ModelManager
        from ocabra.core.worker_pool import WorkerPool

        mm = ModelManager(WorkerPool())
        model_id = "ollama/test-model"
        mm._persisted_model_ids.add(model_id)
        mm._states[model_id] = ModelState(
            model_id=model_id,
            display_name=model_id,
            backend_type="ollama",
            load_policy=LoadPolicy.ON_DEMAND,
        )
        request_id = mm.try_begin_request(
            model_id, max_inflight=32, input_tokens_estimate=500
        )
        mm.update_request_progress(request_id, output_tokens_estimate=120)

        fake_request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(model_manager=mm))
        )
        result = await models_api.get_models_activity(fake_request, _user=None)

        entry = result["activity"][model_id]
        assert entry["inFlight"] == 1
        [req_detail] = entry["requests"]
        assert req_detail["requestId"] == request_id
        assert req_detail["inputTokensEstimate"] == 500
        assert req_detail["outputTokensEstimate"] == 120
        assert req_detail["tokensPerSecond"] is not None
        assert req_detail["lastActivitySecondsAgo"] >= 0

    @pytest.mark.asyncio
    async def test_tokens_per_second_is_none_before_any_output_token(self) -> None:
        """Division by elapsed time before any token has streamed would be
        misleading (reads as 0 t/s, indistinguishable from 'stalled')."""
        from ocabra.core.model_manager import LoadPolicy, ModelManager
        from ocabra.core.worker_pool import WorkerPool

        mm = ModelManager(WorkerPool())
        model_id = "ollama/test-model"
        mm._persisted_model_ids.add(model_id)
        mm._states[model_id] = ModelState(
            model_id=model_id,
            display_name=model_id,
            backend_type="ollama",
            load_policy=LoadPolicy.ON_DEMAND,
        )
        mm.try_begin_request(model_id, max_inflight=32, input_tokens_estimate=500)

        fake_request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(model_manager=mm))
        )
        result = await models_api.get_models_activity(fake_request, _user=None)

        [req_detail] = result["activity"][model_id]["requests"]
        assert req_detail["outputTokensEstimate"] is None
        assert req_detail["tokensPerSecond"] is None


class TestModelsActivityLoadPhase:
    """2026-10-29 — /ocabra/models/activity now tags each in-flight request
    with *why* it's still waiting (phase: loading / waiting_for_gpu /
    prompt_eval / generating), not just an undifferentiated 'no output yet'.
    Reproduced live: a request stuck on a huge (~18k token) prompt that
    _count_input_tokens undercounted as '~128 tok' looked identical to a
    genuine hang without this."""

    @pytest.mark.asyncio
    async def test_phase_is_loading_while_the_worker_is_starting(self) -> None:
        from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelStatus
        from ocabra.core.worker_pool import WorkerPool

        mm = ModelManager(WorkerPool())
        model_id = "ollama/test-model"
        mm._persisted_model_ids.add(model_id)
        mm._states[model_id] = ModelState(
            model_id=model_id, display_name=model_id, backend_type="ollama",
            load_policy=LoadPolicy.ON_DEMAND, status=ModelStatus.LOADING,
        )
        mm.try_begin_request(model_id, max_inflight=32, input_tokens_estimate=500)

        fake_request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(model_manager=mm))
        )
        result = await models_api.get_models_activity(fake_request, _user=None)

        [req_detail] = result["activity"][model_id]["requests"]
        assert req_detail["phase"] == "loading"
        assert req_detail["waitingOnModelId"] is None

    @pytest.mark.asyncio
    async def test_phase_is_waiting_for_gpu_during_eviction_drain_wait(self) -> None:
        from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelStatus
        from ocabra.core.worker_pool import WorkerPool

        mm = ModelManager(WorkerPool())
        model_id = "ollama/test-model"
        mm._persisted_model_ids.add(model_id)
        mm._states[model_id] = ModelState(
            model_id=model_id, display_name=model_id, backend_type="ollama",
            load_policy=LoadPolicy.ON_DEMAND, status=ModelStatus.LOADING,
        )
        mm._eviction_wait_for[model_id] = "ollama/other-busy-model"
        mm.try_begin_request(model_id, max_inflight=32, input_tokens_estimate=500)

        fake_request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(model_manager=mm))
        )
        result = await models_api.get_models_activity(fake_request, _user=None)

        [req_detail] = result["activity"][model_id]["requests"]
        assert req_detail["phase"] == "waiting_for_gpu"
        assert req_detail["waitingOnModelId"] == "ollama/other-busy-model"

    @pytest.mark.asyncio
    async def test_phase_is_prompt_eval_when_loaded_but_no_output_yet(self) -> None:
        from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelStatus
        from ocabra.core.worker_pool import WorkerPool

        mm = ModelManager(WorkerPool())
        model_id = "ollama/test-model"
        mm._persisted_model_ids.add(model_id)
        mm._states[model_id] = ModelState(
            model_id=model_id, display_name=model_id, backend_type="ollama",
            load_policy=LoadPolicy.ON_DEMAND, status=ModelStatus.LOADED,
        )
        mm.try_begin_request(model_id, max_inflight=32, input_tokens_estimate=18240)

        fake_request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(model_manager=mm))
        )
        result = await models_api.get_models_activity(fake_request, _user=None)

        [req_detail] = result["activity"][model_id]["requests"]
        assert req_detail["phase"] == "prompt_eval"

    @pytest.mark.asyncio
    async def test_phase_is_generating_once_output_tokens_have_streamed(self) -> None:
        from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelStatus
        from ocabra.core.worker_pool import WorkerPool

        mm = ModelManager(WorkerPool())
        model_id = "ollama/test-model"
        mm._persisted_model_ids.add(model_id)
        mm._states[model_id] = ModelState(
            model_id=model_id, display_name=model_id, backend_type="ollama",
            load_policy=LoadPolicy.ON_DEMAND, status=ModelStatus.LOADED,
        )
        request_id = mm.try_begin_request(model_id, max_inflight=32)
        mm.update_request_progress(request_id, output_tokens_estimate=50)

        fake_request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(model_manager=mm))
        )
        result = await models_api.get_models_activity(fake_request, _user=None)

        [req_detail] = result["activity"][model_id]["requests"]
        assert req_detail["phase"] == "generating"
