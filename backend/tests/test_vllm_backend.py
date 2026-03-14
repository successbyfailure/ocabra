"""
Tests for VLLMBackend.

All tests mock subprocess creation — no real vLLM or GPU required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_model_dir(tmp_path: Path) -> Path:
    """Create a fake model directory with config files."""
    return tmp_path


def _make_model(base: Path, model_id: str, config: dict, tok_cfg: dict | None = None) -> Path:
    model_path = base / model_id
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(json.dumps(config))
    if tok_cfg is not None:
        (model_path / "tokenizer_config.json").write_text(json.dumps(tok_cfg))
    return model_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_proc(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    return proc


# ---------------------------------------------------------------------------
# get_capabilities
# ---------------------------------------------------------------------------


class TestGetCapabilities:
    @pytest.mark.asyncio
    async def test_llama_capabilities(self, tmp_model_dir: Path):
        _make_model(
            tmp_model_dir,
            "meta-llama/llama-3-8b",
            {
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 8192,
            },
            {"chat_template": "{% for msg in messages %}...{% endfor %}"},
        )

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            backend = VLLMBackend()
            caps = await backend.get_capabilities("meta-llama/llama-3-8b")

        assert caps.chat is True
        assert caps.tools is True
        assert caps.vision is False
        assert caps.streaming is True
        assert caps.context_length == 8192

    @pytest.mark.asyncio
    async def test_vision_model(self, tmp_model_dir: Path):
        _make_model(
            tmp_model_dir,
            "llava-next",
            {"architectures": ["LlavaNextForConditionalGeneration"]},
        )

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            backend = VLLMBackend()
            caps = await backend.get_capabilities("llava-next")

        assert caps.vision is True
        assert caps.chat is True

    @pytest.mark.asyncio
    async def test_embedding_model(self, tmp_model_dir: Path):
        _make_model(
            tmp_model_dir,
            "bert-base",
            {"architectures": ["BertModel"]},
        )

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            backend = VLLMBackend()
            caps = await backend.get_capabilities("bert-base")

        assert caps.embeddings is True
        assert caps.pooling is True
        assert caps.score is True
        assert caps.chat is False

    @pytest.mark.asyncio
    async def test_sequence_classification_model(self, tmp_model_dir: Path):
        _make_model(
            tmp_model_dir,
            "acme/sentiment-classifier",
            {
                "architectures": ["XLMRobertaForSequenceClassification"],
                "num_labels": 3,
                "id2label": {"0": "negative", "1": "neutral", "2": "positive"},
            },
        )

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            backend = VLLMBackend()
            caps = await backend.get_capabilities("acme/sentiment-classifier")

        assert caps.classification is True
        assert caps.score is True
        assert caps.rerank is False

    @pytest.mark.asyncio
    async def test_cross_encoder_rerank_model(self, tmp_model_dir: Path):
        _make_model(
            tmp_model_dir,
            "BAAI/bge-reranker-base",
            {
                "architectures": ["XLMRobertaForSequenceClassification"],
                "num_labels": 1,
            },
        )

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            backend = VLLMBackend()
            caps = await backend.get_capabilities("BAAI/bge-reranker-base")

        assert caps.rerank is True
        assert caps.score is True

    @pytest.mark.asyncio
    async def test_unknown_architecture_fallback(self, tmp_model_dir: Path):
        _make_model(
            tmp_model_dir,
            "unknown-model",
            {"architectures": ["SomeFutureCausalLM"]},
        )

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            mock_settings.hf_cache_dir = str(tmp_model_dir / "_hf_cache")
            backend = VLLMBackend()
            caps = await backend.get_capabilities("unknown-model")

        # Unknown architectures must stay conservative unless there is explicit chat evidence.
        assert caps.chat is False
        assert caps.completion is True

    @pytest.mark.asyncio
    async def test_unknown_architecture_with_chat_template(self, tmp_model_dir: Path):
        _make_model(
            tmp_model_dir,
            "future-chat",
            {"architectures": ["SomeFutureCausalLM"]},
            {"chat_template": "{% for msg in messages %}{{ msg['content'] }}{% endfor %}"},
        )

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            mock_settings.hf_cache_dir = str(tmp_model_dir / "_hf_cache")
            backend = VLLMBackend()
            caps = await backend.get_capabilities("future-chat")

        # chat_template is explicit evidence that /chat/completions is supported.
        assert caps.chat is True

    @pytest.mark.asyncio
    async def test_no_config_json(self, tmp_model_dir: Path):
        model_path = tmp_model_dir / "no-config"
        model_path.mkdir(parents=True)

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            mock_settings.hf_cache_dir = str(tmp_model_dir / "_hf_cache")
            backend = VLLMBackend()
            caps = await backend.get_capabilities("no-config")

        # No config → no caps inferred except streaming/completion defaults
        assert caps.streaming is True
        assert caps.completion is True
        assert caps.chat is False

    @pytest.mark.asyncio
    async def test_resolves_capabilities_from_hf_cache_snapshot(self, tmp_model_dir: Path):
        hf_cache = tmp_model_dir / "_hf_cache"
        snapshot = hf_cache / "hub" / "models--acme--demo-chat" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps({"architectures": ["LlamaForCausalLM"], "max_position_embeddings": 4096})
        )
        (snapshot / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{% for msg in messages %}...{% endfor %}"})
        )
        refs = hf_cache / "hub" / "models--acme--demo-chat" / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text("abc123")

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir / "models")
            mock_settings.hf_cache_dir = str(hf_cache)
            backend = VLLMBackend()
            caps = await backend.get_capabilities("acme/demo-chat")

        assert caps.chat is True
        assert caps.tools is True
        assert caps.context_length == 4096


# ---------------------------------------------------------------------------
# get_vram_estimate_mb
# ---------------------------------------------------------------------------


class TestGetVramEstimate:
    @pytest.mark.asyncio
    async def test_safetensors_files(self, tmp_model_dir: Path):
        model_path = tmp_model_dir / "my-model"
        model_path.mkdir()
        # Create fake safetensors files totalling 1000 bytes
        (model_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 600)
        (model_path / "model-00002-of-00002.safetensors").write_bytes(b"x" * 400)

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            backend = VLLMBackend()
            estimate = await backend.get_vram_estimate_mb("my-model")

        # Very small models still reserve a minimum baseline for vLLM runtime overhead.
        assert estimate >= 2048

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_model_dir: Path):
        model_path = tmp_model_dir / "empty"
        model_path.mkdir()

        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            backend = VLLMBackend()
            estimate = await backend.get_vram_estimate_mb("empty")

        assert estimate >= 2048


# ---------------------------------------------------------------------------
# load / unload
# ---------------------------------------------------------------------------


class TestLoadUnload:
    @pytest.mark.asyncio
    async def test_load_success(self, tmp_model_dir: Path):
        """load() spawns a subprocess and returns WorkerInfo after /health passes."""
        model_path = tmp_model_dir / "test-llm"
        model_path.mkdir()

        proc = _fake_proc(returncode=None)

        from ocabra.backends.vllm_backend import VLLMBackend

        with (
            patch("ocabra.backends.vllm_backend.settings") as mock_settings,
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
            patch.object(VLLMBackend, "_wait_for_startup", new=AsyncMock()),
            patch.object(VLLMBackend, "get_vram_estimate_mb", new=AsyncMock(return_value=8192)),
        ):
            mock_settings.models_dir = str(tmp_model_dir)
            mock_settings.hf_cache_dir = "/tmp/hf_cache"
            mock_settings.hf_token = ""
            backend = VLLMBackend()
            info = await backend.load("test-llm", gpu_indices=[1], port=18001)

        assert info.model_id == "test-llm"
        assert info.port == 18001
        assert info.pid == 12345
        assert info.vram_used_mb == 8192
        assert info.backend_type == "vllm"

    @pytest.mark.asyncio
    async def test_load_passes_vllm_tuning_flags(self, tmp_model_dir: Path):
        (tmp_model_dir / "test-llm").mkdir()
        proc = _fake_proc(returncode=None)
        create_subprocess = AsyncMock(return_value=proc)

        from ocabra.backends.vllm_backend import VLLMBackend

        with (
            patch("ocabra.backends.vllm_backend.settings") as mock_settings,
            patch("asyncio.create_subprocess_exec", new=create_subprocess),
            patch.object(VLLMBackend, "_wait_for_startup", new=AsyncMock()),
            patch.object(VLLMBackend, "get_vram_estimate_mb", new=AsyncMock(return_value=8192)),
        ):
            mock_settings.models_dir = str(tmp_model_dir)
            mock_settings.hf_cache_dir = "/tmp/hf_cache"
            mock_settings.hf_token = ""
            mock_settings.cuda_device_order = "PCI_BUS_ID"
            mock_settings.vllm_gpu_memory_utilization = 0.85
            mock_settings.vllm_enable_prefix_caching = True
            mock_settings.vllm_max_num_seqs = 16
            mock_settings.vllm_max_num_batched_tokens = 8192
            mock_settings.vllm_tensor_parallel_size = None
            mock_settings.vllm_max_model_len = None
            mock_settings.vllm_enable_chunked_prefill = None
            mock_settings.vllm_swap_space = None
            mock_settings.vllm_kv_cache_dtype = "fp8"
            mock_settings.vllm_enforce_eager = False
            mock_settings.vllm_attention_backend = None
            backend = VLLMBackend()
            await backend.load("test-llm", gpu_indices=[1], port=18001)

        cmd = list(create_subprocess.await_args.args)
        assert "--enable-prefix-caching" in cmd
        assert "--max-num-seqs" in cmd
        assert "16" in cmd
        assert "--max-num-batched-tokens" in cmd
        assert "8192" in cmd
        assert "--tensor-parallel-size" in cmd
        assert "--max-model-len" not in cmd
        assert "--swap-space" not in cmd
        assert "--kv-cache-dtype" in cmd
        assert "fp8" in cmd
        assert "--enforce-eager" not in cmd

    @pytest.mark.asyncio
    async def test_load_prefers_model_vllm_overrides(self, tmp_model_dir: Path):
        (tmp_model_dir / "test-llm").mkdir()
        proc = _fake_proc(returncode=None)
        create_subprocess = AsyncMock(return_value=proc)

        from ocabra.backends.vllm_backend import VLLMBackend

        with (
            patch("ocabra.backends.vllm_backend.settings") as mock_settings,
            patch("asyncio.create_subprocess_exec", new=create_subprocess),
            patch.object(VLLMBackend, "_wait_for_startup", new=AsyncMock()),
            patch.object(VLLMBackend, "get_vram_estimate_mb", new=AsyncMock(return_value=8192)),
        ):
            mock_settings.models_dir = str(tmp_model_dir)
            mock_settings.hf_cache_dir = "/tmp/hf_cache"
            mock_settings.hf_token = ""
            mock_settings.cuda_device_order = "PCI_BUS_ID"
            mock_settings.vllm_gpu_memory_utilization = 0.85
            mock_settings.vllm_enable_prefix_caching = True
            mock_settings.vllm_max_num_seqs = 16
            mock_settings.vllm_max_num_batched_tokens = 8192
            mock_settings.vllm_tensor_parallel_size = None
            mock_settings.vllm_max_model_len = None
            mock_settings.vllm_enable_chunked_prefill = None
            mock_settings.vllm_swap_space = None
            mock_settings.vllm_kv_cache_dtype = "fp8"
            mock_settings.vllm_enforce_eager = False
            mock_settings.vllm_attention_backend = None
            backend = VLLMBackend()
            await backend.load(
                "test-llm",
                gpu_indices=[1],
                port=18001,
                extra_config={
                    "vllm": {
                        "enable_prefix_caching": False,
                        "tensor_parallel_size": 2,
                        "max_model_len": 32768,
                        "max_num_seqs": 4,
                        "max_num_batched_tokens": 2048,
                        "enable_chunked_prefill": True,
                        "swap_space": 8,
                        "kv_cache_dtype": None,
                        "enforce_eager": True,
                        "attention_backend": "FLASH_ATTN",
                    }
                },
            )

        cmd = list(create_subprocess.await_args.args)
        assert "--enable-prefix-caching" not in cmd
        assert "--tensor-parallel-size" in cmd
        assert "2" in cmd
        assert "--max-model-len" in cmd
        assert "32768" in cmd
        assert "--max-num-seqs" in cmd
        assert "4" in cmd
        assert "16" not in cmd
        assert "--max-num-batched-tokens" in cmd
        assert "2048" in cmd
        assert "--enable-chunked-prefill" in cmd
        assert "--swap-space" in cmd
        assert "8" in cmd
        assert "--kv-cache-dtype" not in cmd
        assert "--enforce-eager" in cmd
        assert "--attention-backend" in cmd
        assert "FLASH_ATTN" in cmd

    @pytest.mark.asyncio
    async def test_load_passes_extended_compatibility_flags(self, tmp_model_dir: Path):
        (tmp_model_dir / "test-llm").mkdir()
        proc = _fake_proc(returncode=None)
        create_subprocess = AsyncMock(return_value=proc)

        from ocabra.backends.vllm_backend import VLLMBackend

        with (
            patch("ocabra.backends.vllm_backend.settings") as mock_settings,
            patch("asyncio.create_subprocess_exec", new=create_subprocess),
            patch.object(VLLMBackend, "_wait_for_startup", new=AsyncMock()),
            patch.object(VLLMBackend, "get_vram_estimate_mb", new=AsyncMock(return_value=8192)),
        ):
            mock_settings.models_dir = str(tmp_model_dir)
            mock_settings.hf_cache_dir = "/tmp/hf_cache"
            mock_settings.hf_token = ""
            mock_settings.cuda_device_order = "PCI_BUS_ID"
            mock_settings.vllm_gpu_memory_utilization = 0.85
            mock_settings.vllm_enable_prefix_caching = True
            mock_settings.vllm_model_impl = "auto"
            mock_settings.vllm_runner = None
            mock_settings.vllm_hf_overrides = None
            mock_settings.vllm_chat_template = None
            mock_settings.vllm_tool_call_parser = None
            mock_settings.vllm_reasoning_parser = None
            backend = VLLMBackend()
            await backend.load(
                "test-llm",
                gpu_indices=[1],
                port=18001,
                extra_config={
                    "vllm": {
                        "model_impl": "transformers",
                        "runner": "pooling",
                        "hf_overrides": {"architectures": ["BertModel"]},
                        "chat_template": "/tmp/chat-template.jinja",
                        "chat_template_content_format": "openai",
                        "generation_config": "vllm",
                        "override_generation_config": {"temperature": 0.2},
                        "tool_call_parser": "hermes",
                        "tool_parser_plugin": "acme.plugin",
                        "reasoning_parser": "deepseek_r1",
                        "language_model_only": True,
                    }
                },
            )

        cmd = list(create_subprocess.await_args.args)
        assert "--model-impl" in cmd
        assert "transformers" in cmd
        assert "--runner" in cmd
        assert "pooling" in cmd
        assert "--hf-overrides" in cmd
        assert '{"architectures": ["BertModel"]}' in cmd
        assert "--chat-template" in cmd
        assert "/tmp/chat-template.jinja" in cmd
        assert "--chat-template-content-format" in cmd
        assert "openai" in cmd
        assert "--generation-config" in cmd
        assert "--override-generation-config" in cmd
        assert '{"temperature": 0.2}' in cmd
        assert "--tool-call-parser" in cmd
        assert "hermes" in cmd
        assert "--enable-auto-tool-choice" in cmd
        assert "--tool-parser-plugin" in cmd
        assert "acme.plugin" in cmd
        assert "--reasoning-parser" in cmd
        assert "deepseek_r1" in cmd
        assert "--language-model-only" in cmd

    @pytest.mark.asyncio
    async def test_load_raises_without_port(self):
        from ocabra.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        with pytest.raises(ValueError, match="port"):
            await backend.load("some-model", gpu_indices=[0])

    @pytest.mark.asyncio
    async def test_load_timeout_kills_process(self, tmp_model_dir: Path):
        """If startup times out, the process is killed and TimeoutError is raised."""
        (tmp_model_dir / "slow-model").mkdir()

        proc = _fake_proc(returncode=None)

        from ocabra.backends.vllm_backend import VLLMBackend

        async def _timeout(*_args, **_kwargs):
            raise TimeoutError("startup timeout")

        with (
            patch("ocabra.backends.vllm_backend.settings") as mock_settings,
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
            patch.object(
                VLLMBackend,
                "_wait_for_startup",
                new=AsyncMock(side_effect=TimeoutError("startup timeout")),
            ),
            patch.object(VLLMBackend, "_kill_process", new=AsyncMock()),
        ):
            mock_settings.models_dir = str(tmp_model_dir)
            mock_settings.hf_cache_dir = "/tmp/hf_cache"
            mock_settings.hf_token = ""
            backend = VLLMBackend()
            with pytest.raises(TimeoutError):
                await backend.load("slow-model", gpu_indices=[0], port=18002)

    @pytest.mark.asyncio
    async def test_oom_detected_during_startup(self, tmp_model_dir: Path):
        """Process exiting with rc=137 (OOM kill) raises MemoryError."""
        (tmp_model_dir / "big-model").mkdir()

        # Simulate process dying mid-startup
        proc = _fake_proc(returncode=137)

        from ocabra.backends.vllm_backend import VLLMBackend

        async def _wait_until_dead(model_id, port):
            # Real implementation checks returncode inside loop
            entry = backend._processes.get(model_id)
            if entry and entry[0].returncode == 137:
                raise MemoryError("OOM-killed")

        with (
            patch("ocabra.backends.vllm_backend.settings") as mock_settings,
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
            patch.object(VLLMBackend, "_kill_process", new=AsyncMock()),
        ):
            mock_settings.models_dir = str(tmp_model_dir)
            mock_settings.hf_cache_dir = "/tmp/hf_cache"
            mock_settings.hf_token = ""
            backend = VLLMBackend()
            # Patch _wait_for_startup to use our helper bound to backend
            with patch.object(backend, "_wait_for_startup", side_effect=MemoryError("OOM-killed")):
                with pytest.raises((MemoryError, TimeoutError, Exception)):
                    await backend.load("big-model", gpu_indices=[0], port=18003)

    @pytest.mark.asyncio
    async def test_unload_sends_sigterm(self):
        proc = _fake_proc(returncode=None)
        proc.wait = AsyncMock(return_value=0)

        from ocabra.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        backend._processes["my-model"] = (proc, 18001)

        await backend.unload("my-model")

        proc.terminate.assert_called_once()
        assert "my-model" not in backend._processes

    @pytest.mark.asyncio
    async def test_unload_unknown_model_is_noop(self):
        from ocabra.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        # Should not raise
        await backend.unload("nonexistent-model")


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self):
        proc = _fake_proc(returncode=None)

        from ocabra.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        backend._processes["healthy-model"] = (proc, 18010)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            client_instance.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value = client_instance

            result = await backend.health_check("healthy-model")

        assert result is True

    @pytest.mark.asyncio
    async def test_dead_process(self):
        proc = _fake_proc(returncode=1)

        from ocabra.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        backend._processes["dead-model"] = (proc, 18011)

        result = await backend.health_check("dead-model")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_process(self):
        from ocabra.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        result = await backend.health_check("ghost-model")
        assert result is False


class TestModelPathResolution:
    def test_resolve_hf_layout(self, tmp_model_dir: Path):
        from ocabra.backends.vllm_backend import VLLMBackend

        hf_dir = tmp_model_dir / "huggingface" / "facebook--opt-125m"
        hf_dir.mkdir(parents=True)

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            backend = VLLMBackend()
            resolved = backend._resolve_model_target("facebook/opt-125m")

        assert resolved == str(hf_dir)

    def test_resolve_remote_fallback(self, tmp_model_dir: Path):
        from ocabra.backends.vllm_backend import VLLMBackend

        with patch("ocabra.backends.vllm_backend.settings") as mock_settings:
            mock_settings.models_dir = str(tmp_model_dir)
            backend = VLLMBackend()
            resolved = backend._resolve_model_target("Qwen/Qwen2.5-0.5B-Instruct")

        assert resolved == "Qwen/Qwen2.5-0.5B-Instruct"
