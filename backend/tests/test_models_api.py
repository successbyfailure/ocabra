import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ocabra.api.internal import models as models_api
from ocabra.backends.base import BackendCapabilities


class _FakeBackend:
    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        assert model_id == "My--Engine"
        return BackendCapabilities(completion=True, streaming=True, context_length=0)


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
