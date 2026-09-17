"""Tests for path traversal protection in model and engine deletion.

Covers _is_path_within_base(), _delete_model_files(), and TRT-LLM engine
deletion to ensure paths outside the configured base are always rejected.
"""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.api.internal.models import _is_path_within_base, _delete_model_files
from ocabra.api.internal import trtllm


# ── _is_path_within_base unit tests ──────────────────────────────


class TestIsPathWithinBase:
    def test_simple_child(self, tmp_path):
        child = tmp_path / "models" / "my-model"
        child.mkdir(parents=True)
        assert _is_path_within_base(child, tmp_path) is True

    def test_same_path(self, tmp_path):
        assert _is_path_within_base(tmp_path, tmp_path) is True

    def test_parent_escape_dotdot(self, tmp_path):
        evil = tmp_path / "models" / ".." / ".." / "etc"
        assert _is_path_within_base(evil, tmp_path / "models") is False

    def test_absolute_path_outside(self, tmp_path):
        base = tmp_path / "models"
        base.mkdir()
        outside = tmp_path / "secrets"
        outside.mkdir()
        assert _is_path_within_base(outside, base) is False

    def test_symlink_escaping_base(self, tmp_path):
        """Symlink inside base pointing outside should be rejected."""
        base = tmp_path / "models"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")

        link = base / "evil-link"
        link.symlink_to(outside)

        # resolve() follows symlinks, so the resolved path is outside base
        assert _is_path_within_base(link, base) is False

    def test_nonexistent_child_still_valid(self, tmp_path):
        """Non-existent path under base should still be accepted."""
        child = tmp_path / "models" / "future-model"
        assert _is_path_within_base(child, tmp_path) is True

    def test_dotdot_to_root(self, tmp_path):
        base = tmp_path / "models"
        base.mkdir()
        evil = base / ".." / ".." / ".." / "etc" / "passwd"
        assert _is_path_within_base(evil, base) is False


# ── _delete_model_files path traversal ──────────────────────────


class TestDeleteModelFiles:
    @pytest.mark.asyncio
    async def test_rejects_traversal_in_hf_model_id(self, tmp_path, monkeypatch):
        """Model ID with ../../ in name should not delete outside models dir."""
        from ocabra.api.internal import models as models_mod

        models_dir = tmp_path / "models"
        hf_dir = models_dir / "huggingface"
        hf_dir.mkdir(parents=True)

        # Create target outside hf dir
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("important")

        monkeypatch.setattr(models_mod.settings, "models_dir", str(models_dir))

        # The candidate is computed as hf_dir / model_id.replace("/","--")
        # With a normal model ID the traversal is neutralized by replace("/","--")
        # but we test the _is_path_within_base guard directly via a symlink
        evil_name = "evil--model"
        evil_path = hf_dir / evil_name
        evil_path.symlink_to(outside)

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _delete_model_files(f"vllm/{evil_name.replace('--', '/')}", "vllm")

        assert exc_info.value.status_code == 400
        assert (outside / "keep.txt").exists()

    @pytest.mark.asyncio
    async def test_valid_hf_model_deletion(self, tmp_path, monkeypatch):
        """Legitimate model directory within huggingface/ is deleted."""
        from ocabra.api.internal import models as models_mod

        models_dir = tmp_path / "models"
        hf_dir = models_dir / "huggingface"
        model_dir = hf_dir / "org--mymodel"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("weights")

        monkeypatch.setattr(models_mod.settings, "models_dir", str(models_dir))

        result = await _delete_model_files("vllm/org/mymodel", "vllm")

        assert result == str(model_dir)
        assert not model_dir.exists()

    @pytest.mark.asyncio
    async def test_ollama_model_delegates_to_http_api(self, tmp_path, monkeypatch):
        """Ollama backend delegates to the daemon's HTTP DELETE endpoint
        (the ``ollama`` CLI is NOT installed in the api container, D11).
        Returns ``ollama:<name>`` on success so the caller can log which
        remote resource was cleaned up."""
        from ocabra.api.internal import models as models_mod
        import httpx

        monkeypatch.setattr(models_mod.settings, "models_dir", str(tmp_path))

        mock_response = MagicMock(status_code=200, text="")

        class MockClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def request(self, method, url, **kwargs):
                return mock_response

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: MockClient())

        result = await _delete_model_files("ollama/llama3:8b", "ollama")
        assert result == "ollama:llama3:8b"

    @pytest.mark.asyncio
    async def test_nonexistent_model_returns_none(self, tmp_path, monkeypatch):
        """If model dir doesn't exist on disk, returns None without error."""
        from ocabra.api.internal import models as models_mod

        models_dir = tmp_path / "models"
        (models_dir / "huggingface").mkdir(parents=True)
        monkeypatch.setattr(models_mod.settings, "models_dir", str(models_dir))

        result = await _delete_model_files("vllm/nonexistent/model", "vllm")
        assert result is None

    @pytest.mark.asyncio
    async def test_hf_hub_cache_blobs_are_deleted(self, tmp_path, monkeypatch):
        """The heavy safetensors typically live in ``<hf_cache>/hub/models--<repo>/``
        (the standard huggingface_hub cache), not in ``models/huggingface/``.
        Deleting a model must clean up the hub cache too, otherwise the disk
        stays full (regression from 2026-09-17: ~94 GB of vLLM blobs leaked)."""
        from ocabra.api.internal import models as models_mod

        models_dir = tmp_path / "models"
        (models_dir / "huggingface").mkdir(parents=True)
        hf_cache = tmp_path / "hf_cache"
        hub_model = hf_cache / "hub" / "models--org--mymodel"
        hub_model.mkdir(parents=True)
        (hub_model / "blob").write_text("x" * 1024)

        monkeypatch.setattr(models_mod.settings, "models_dir", str(models_dir))
        monkeypatch.setattr(models_mod.settings, "hf_cache_dir", str(hf_cache))

        result = await _delete_model_files("vllm/org/mymodel", "vllm")

        assert result == str(hub_model)
        assert not hub_model.exists()

    @pytest.mark.asyncio
    async def test_legacy_hf_cache_blobs_are_deleted(self, tmp_path, monkeypatch):
        """Older layouts wrote to ``<hf_cache>/models--<repo>/`` directly
        (before HF Hub moved to the ``hub/`` subdir). Some old installs
        still have blobs there — clean those up too."""
        from ocabra.api.internal import models as models_mod

        models_dir = tmp_path / "models"
        (models_dir / "huggingface").mkdir(parents=True)
        hf_cache = tmp_path / "hf_cache"
        legacy_model = hf_cache / "models--org--mymodel"
        legacy_model.mkdir(parents=True)
        (legacy_model / "blob").write_text("x" * 1024)

        monkeypatch.setattr(models_mod.settings, "models_dir", str(models_dir))
        monkeypatch.setattr(models_mod.settings, "hf_cache_dir", str(hf_cache))

        result = await _delete_model_files("vllm/org/mymodel", "vllm")

        assert result == str(legacy_model)
        assert not legacy_model.exists()

    @pytest.mark.asyncio
    async def test_all_three_locations_deleted_when_present(self, tmp_path, monkeypatch):
        """Some models leave blobs in more than one path (e.g. a fresh pull
        via download manager landed in ``models/huggingface`` and a later
        transformers-triggered load repopulated the hub cache). Delete
        must sweep all three and return them comma-separated."""
        from ocabra.api.internal import models as models_mod

        models_dir = tmp_path / "models"
        pipeline_dir = models_dir / "huggingface" / "org--mymodel"
        pipeline_dir.mkdir(parents=True)
        (pipeline_dir / "weights").write_text("a")
        hf_cache = tmp_path / "hf_cache"
        hub_model = hf_cache / "hub" / "models--org--mymodel"
        hub_model.mkdir(parents=True)
        (hub_model / "blob").write_text("b")
        legacy_model = hf_cache / "models--org--mymodel"
        legacy_model.mkdir(parents=True)
        (legacy_model / "blob").write_text("c")

        monkeypatch.setattr(models_mod.settings, "models_dir", str(models_dir))
        monkeypatch.setattr(models_mod.settings, "hf_cache_dir", str(hf_cache))

        result = await _delete_model_files("vllm/org/mymodel", "vllm")

        assert result is not None
        parts = result.split(",")
        assert len(parts) == 3
        assert not pipeline_dir.exists()
        assert not hub_model.exists()
        assert not legacy_model.exists()


# ── TRT-LLM engine deletion path traversal ───────────────────────


class TestTrtllmEnginePathTraversal:
    @pytest.mark.asyncio
    async def test_rejects_dotdot_engine_name(self, tmp_path, monkeypatch):
        """Engine name with ../ should be rejected."""
        engines_root = tmp_path / "engines"
        engines_root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep")

        monkeypatch.setattr(trtllm.settings, "tensorrt_llm_engines_dir", str(engines_root))

        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(model_manager=SimpleNamespace(_states={})))
        )

        with pytest.raises(trtllm.HTTPException) as exc_info:
            await trtllm.delete_engine("../outside", request)

        assert exc_info.value.status_code == 400
        assert (outside / "keep.txt").exists()

    @pytest.mark.asyncio
    async def test_rejects_symlink_engine_escape(self, tmp_path, monkeypatch):
        """Symlink engine dir pointing outside should be rejected."""
        engines_root = tmp_path / "engines"
        engines_root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "data.bin").write_text("data")

        link = engines_root / "evil-engine"
        link.symlink_to(outside)

        monkeypatch.setattr(trtllm.settings, "tensorrt_llm_engines_dir", str(engines_root))

        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(model_manager=SimpleNamespace(_states={})))
        )

        with pytest.raises(trtllm.HTTPException) as exc_info:
            await trtllm.delete_engine("evil-engine", request)

        assert exc_info.value.status_code == 400
        assert (outside / "data.bin").exists()

    def test_is_path_within_base_trtllm_module(self, tmp_path):
        """Verify trtllm module has its own _is_path_within_base that works."""
        base = tmp_path / "engines"
        base.mkdir()
        child = base / "my-engine"
        child.mkdir()
        assert trtllm._is_path_within_base(child, base) is True
        assert trtllm._is_path_within_base(tmp_path / "other", base) is False
