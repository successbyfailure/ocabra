"""
vLLM backend — one subprocess per loaded model.

Each model gets an isolated vLLM OpenAI-compatible API server on a dedicated
port in the range configured by settings.worker_port_range_*.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.backends.base import (
    BackendCapabilities,
    BackendInstallSpec,
    BackendInterface,
    ModalityType,
    WorkerInfo,
)
from ocabra.config import settings
from ocabra.core.backend_installer import venv_nvidia_ld_library_path

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Architecture → capability hints
# ---------------------------------------------------------------------------
_ARCH_CAPS: dict[str, dict[str, Any]] = {
    # Vision / multimodal
    "LlavaNextForConditionalGeneration": {"vision": True, "chat": True},
    "LlavaForConditionalGeneration": {"vision": True, "chat": True},
    "Qwen2VLForConditionalGeneration": {"vision": True, "chat": True, "tools": True},
    "InternVLChatModel": {"vision": True, "chat": True},
    "MiniCPMV": {"vision": True, "chat": True},
    "Phi3VForCausalLM": {"vision": True, "chat": True},
    # Reasoning (long CoT)
    "DeepseekV3ForCausalLM": {"reasoning": True, "chat": True, "tools": True},
    "DeepseekR1ForCausalLM": {"reasoning": True, "chat": True},
    "Qwen3ForCausalLM": {"reasoning": True, "chat": True, "tools": True},
    "Qwen3MoeForCausalLM": {"reasoning": True, "chat": True, "tools": True},
    # General text
    "LlamaForCausalLM": {"chat": True, "tools": True},
    "Llama4ForConditionalGeneration": {"chat": True, "tools": True, "vision": True},
    "MistralForCausalLM": {"chat": True, "tools": True},
    "MixtralForCausalLM": {"chat": True, "tools": True},
    "Phi3ForCausalLM": {"chat": True, "tools": True},
    "Phi3SmallForCausalLM": {"chat": True, "tools": True},
    "GemmaForCausalLM": {"chat": True},
    "Gemma2ForCausalLM": {"chat": True, "tools": True},
    "Qwen2ForCausalLM": {"chat": True, "tools": True},
    "Qwen2MoeForCausalLM": {"chat": True, "tools": True},
    "CohereForCausalLM": {"chat": True, "tools": True},
    "GPTNeoXForCausalLM": {"chat": True, "completion": True},
    "GPT2LMHeadModel": {"completion": True},
    # Embeddings
    "BertModel": {"embeddings": True, "pooling": True, "score": True},
    "RobertaModel": {"embeddings": True, "pooling": True, "score": True},
    "XLMRobertaModel": {"embeddings": True, "pooling": True, "score": True},
    "DistilBertModel": {"embeddings": True, "pooling": True, "score": True},
    "E5Model": {"embeddings": True, "pooling": True, "score": True},
    "BertForSequenceClassification": {"classification": True, "score": True},
    "RobertaForSequenceClassification": {"classification": True, "score": True},
    "XLMRobertaForSequenceClassification": {"classification": True, "score": True},
    "DistilBertForSequenceClassification": {"classification": True, "score": True},
    "DebertaV2ForSequenceClassification": {"classification": True, "score": True},
}

_STARTUP_TIMEOUT_S = 120
_SHUTDOWN_TIMEOUT_S = 30
_MIN_VLLM_VRAM_MB = 2048


class VLLMBackend(BackendInterface):
    """
    Manages vLLM subprocess workers.

    One VLLMBackend instance is shared by the WorkerPool.  It keeps a
    registry of running processes keyed by model_id.
    """

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.TEXT_GENERATION, ModalityType.EMBEDDINGS}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Declarative install spec for vllm (Bloque 15 Fase 2)."""

        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-vllm",
            oci_tags={"cuda12": "latest-cuda12"},
            apt_packages=[
                # Triton JIT compiles kernels at runtime and needs a C/C++
                # compiler. The slim base image does not ship gcc/g++; without
                # these, the engine fails with
                # "RuntimeError: Failed to find C compiler" the first time
                # vllm runs (Deuda D11, validated 2026-04-25).
                "gcc",
                "g++",
            ],
            pip_packages=[
                "vllm==0.17.1",
                "torch>=2.5",
            ],
            pip_extra_index_urls=[
                "https://download.pytorch.org/whl/cu124",
            ],
            estimated_size_mb=9500,
            display_name="vLLM",
            description="High-throughput LLM inference engine with PagedAttention",
            tags=["LLM", "GPU", "CUDA"],
        )

    def _resolve_python_bin(self) -> str:
        """Pick the python that runs the vllm OpenAI server.

        Priority: ``<backends_dir>/vllm/metadata.json`` > :data:`sys.executable`.
        """

        try:
            meta_path = Path(settings.backends_dir) / "vllm" / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                bin_path = meta.get("python_bin") if isinstance(meta, dict) else None
                if bin_path and Path(bin_path).is_file():
                    return str(bin_path)
        except (OSError, json.JSONDecodeError):
            pass
        return sys.executable

    def __init__(self) -> None:
        # model_id → (process, port)
        self._processes: dict[str, tuple[asyncio.subprocess.Process, int]] = {}

    # ------------------------------------------------------------------
    # BackendInterface
    # ------------------------------------------------------------------

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        """
        Spawn a vLLM server process for *model_id*.

        Args:
            model_id: Local relative path under models_dir (e.g. ``meta-llama/Meta-Llama-3-8B``).
            gpu_indices: CUDA device indices to use.

        Returns:
            WorkerInfo describing the running process.
        """
        # Port must be pre-assigned by WorkerPool before calling load().
        # Callers (ModelManager) are expected to call worker_pool.assign_port()
        # and pass it via kwargs.
        port: int = kwargs.get("port", 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        extra_config = kwargs.get("extra_config") or {}
        cmd, env, cuda_devices = self._build_launch_spec(
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            extra_config=extra_config,
        )

        logger.info(
            "vllm_starting",
            model_id=model_id,
            port=port,
            gpus=cuda_devices,
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        self._processes[model_id] = (proc, port)

        # Wait until the server is healthy or times out
        try:
            await self._wait_for_startup(model_id, port)
        except TimeoutError:
            await self._kill_process(model_id)
            raise

        vram_mb = await self.get_vram_estimate_mb(model_id)

        logger.info("vllm_ready", model_id=model_id, port=port, pid=proc.pid)
        return WorkerInfo(
            backend_type="vllm",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=proc.pid or 0,
            vram_used_mb=vram_mb,
        )

    async def estimate_memory_profile(
        self,
        model_id: str,
        *,
        gpu_index: int,
        extra_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        port = self._reserve_free_port()
        cmd, env, _cuda_devices = self._build_launch_spec(
            model_id=model_id,
            gpu_indices=[gpu_index],
            port=port,
            extra_config=extra_config or {},
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        readers = [
            asyncio.create_task(self._consume_stream(proc.stdout, stdout_chunks)),
            asyncio.create_task(self._consume_stream(proc.stderr, stderr_chunks)),
        ]
        healthy = False
        error_message: str | None = None
        try:
            await self._wait_for_probe_startup(proc, port)
            healthy = True
        except Exception as exc:
            error_message = str(exc)
        finally:
            await self._terminate_probe_process(proc)
            await asyncio.gather(*readers, return_exceptions=True)

        logs = "\n".join([*stdout_chunks, *stderr_chunks])
        profile = self._parse_memory_profile_logs(logs)
        profile["status"] = "ok" if healthy else "error"
        profile["error"] = error_message
        profile["logs"] = logs[-8000:] if logs else ""
        return profile

    def _get_vllm_option(self, extra_config: dict[str, Any], key: str, default: Any) -> Any:
        camel_key = "".join(
            part.capitalize() if index else part
            for index, part in enumerate(key.split("_"))
        )
        vllm_config = extra_config.get("vllm")
        if isinstance(vllm_config, dict):
            if key in vllm_config:
                return vllm_config[key]
            if camel_key in vllm_config:
                return vllm_config[camel_key]
        if key in extra_config:
            return extra_config[key]
        if camel_key in extra_config:
            return extra_config[camel_key]
        return default

    def _build_launch_spec(
        self,
        *,
        model_id: str,
        gpu_indices: list[int],
        port: int,
        extra_config: dict[str, Any],
    ) -> tuple[list[str], dict[str, str], str]:
        model_target = self._resolve_model_target(model_id)
        cuda_devices = ",".join(str(i) for i in gpu_indices)
        tensor_parallel = self._get_vllm_option(
            extra_config,
            "tensor_parallel_size",
            self._get_setting("vllm_tensor_parallel_size"),
        ) or len(gpu_indices)

        cmd = [
            self._resolve_python_bin(),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_target,
            "--tensor-parallel-size",
            str(tensor_parallel),
            "--gpu-memory-utilization",
            str(
                self._get_vllm_option(
                    extra_config,
                    "gpu_memory_utilization",
                    self._get_setting("vllm_gpu_memory_utilization"),
                )
            ),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--served-model-name",
            model_id,
        ]
        if self._get_vllm_option(
            extra_config,
            "disable_log_requests",
            self._get_setting("vllm_disable_log_requests"),
        ):
            cmd.append("--no-enable-log-requests")
        model_impl = self._get_vllm_option(
            extra_config, "model_impl", self._get_setting("vllm_model_impl")
        )
        if model_impl:
            cmd.extend(["--model-impl", str(model_impl)])
        runner = self._get_vllm_option(extra_config, "runner", self._get_setting("vllm_runner"))
        if runner:
            cmd.extend(["--runner", str(runner)])
        if self._get_vllm_option(
            extra_config, "enable_prefix_caching", self._get_setting("vllm_enable_prefix_caching")
        ):
            cmd.append("--enable-prefix-caching")
        max_num_seqs = self._get_vllm_option(
            extra_config, "max_num_seqs", self._get_setting("vllm_max_num_seqs")
        )
        if max_num_seqs:
            cmd.extend(["--max-num-seqs", str(max_num_seqs)])
        max_num_batched_tokens = self._get_vllm_option(
            extra_config,
            "max_num_batched_tokens",
            self._get_setting("vllm_max_num_batched_tokens"),
        )
        if max_num_batched_tokens:
            cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
        max_model_len = self._get_vllm_option(
            extra_config, "max_model_len", self._get_setting("vllm_max_model_len")
        )
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])
        enable_chunked_prefill = self._get_vllm_option(
            extra_config,
            "enable_chunked_prefill",
            self._get_setting("vllm_enable_chunked_prefill"),
        )
        if enable_chunked_prefill is True:
            cmd.append("--enable-chunked-prefill")
        elif enable_chunked_prefill is False:
            cmd.append("--no-enable-chunked-prefill")
        swap_space = self._get_vllm_option(
            extra_config, "swap_space", self._get_setting("vllm_swap_space")
        )
        if swap_space:
            cmd.extend(["--swap-space", str(swap_space)])
        kv_cache_dtype = self._get_vllm_option(
            extra_config, "kv_cache_dtype", self._get_setting("vllm_kv_cache_dtype")
        )
        if kv_cache_dtype:
            cmd.extend(["--kv-cache-dtype", str(kv_cache_dtype)])
        if self._get_vllm_option(
            extra_config, "enforce_eager", self._get_setting("vllm_enforce_eager")
        ):
            cmd.append("--enforce-eager")
        attention_backend = self._get_vllm_option(
            extra_config, "attention_backend", self._get_setting("vllm_attention_backend")
        )
        if attention_backend:
            cmd.extend(["--attention-backend", str(attention_backend)])
        if self._get_vllm_option(
            extra_config, "trust_remote_code", self._get_setting("vllm_trust_remote_code")
        ):
            cmd.append("--trust-remote-code")
        hf_overrides = self._get_vllm_option(
            extra_config, "hf_overrides", self._get_setting("vllm_hf_overrides")
        )
        if hf_overrides:
            cmd.extend(["--hf-overrides", self._encode_vllm_json_option(hf_overrides)])
        chat_template = self._get_vllm_option(
            extra_config, "chat_template", self._get_setting("vllm_chat_template")
        )
        if chat_template:
            cmd.extend(["--chat-template", str(chat_template)])
        chat_template_content_format = self._get_vllm_option(
            extra_config,
            "chat_template_content_format",
            self._get_setting("vllm_chat_template_content_format"),
        )
        if chat_template_content_format:
            cmd.extend(["--chat-template-content-format", str(chat_template_content_format)])
        generation_config = self._get_vllm_option(
            extra_config, "generation_config", self._get_setting("vllm_generation_config")
        )
        if generation_config:
            cmd.extend(["--generation-config", str(generation_config)])
        override_generation_config = self._get_vllm_option(
            extra_config,
            "override_generation_config",
            self._get_setting("vllm_override_generation_config"),
        )
        if override_generation_config:
            cmd.extend(
                [
                    "--override-generation-config",
                    self._encode_vllm_json_option(override_generation_config),
                ]
            )
        tool_call_parser = self._get_vllm_option(
            extra_config, "tool_call_parser", self._get_setting("vllm_tool_call_parser")
        )
        if tool_call_parser:
            cmd.extend(["--tool-call-parser", str(tool_call_parser), "--enable-auto-tool-choice"])
        tool_parser_plugin = self._get_vllm_option(
            extra_config, "tool_parser_plugin", self._get_setting("vllm_tool_parser_plugin")
        )
        if tool_parser_plugin:
            cmd.extend(["--tool-parser-plugin", str(tool_parser_plugin)])
        reasoning_parser = self._get_vllm_option(
            extra_config, "reasoning_parser", self._get_setting("vllm_reasoning_parser")
        )
        if reasoning_parser:
            cmd.extend(["--reasoning-parser", str(reasoning_parser)])
        if self._get_vllm_option(
            extra_config,
            "language_model_only",
            self._get_setting("vllm_language_model_only"),
        ):
            cmd.append("--language-model-only")

        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": settings.cuda_device_order,
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HOME": settings.hf_cache_dir,
        }
        if settings.hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

        # Slim image has no CUDA toolkit; the loader needs the bundled CUDA
        # libs (libcublas/libcudart/...) on LD_LIBRARY_PATH (Deuda D14).
        nvidia_ld = venv_nvidia_ld_library_path(settings.backends_dir, "vllm")
        if nvidia_ld:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{nvidia_ld}:{existing}" if existing else nvidia_ld

        return cmd, env, cuda_devices

    def _encode_vllm_json_option(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def _get_setting(self, name: str) -> Any:
        value = getattr(settings, name, None)
        module_name = getattr(value.__class__, "__module__", "")
        if module_name.startswith("unittest.mock"):
            return None
        return value

    async def unload(self, model_id: str) -> None:
        """Send SIGTERM, wait up to 30 s, then SIGKILL."""
        await self._kill_process(model_id, graceful=True)
        logger.info("vllm_unloaded", model_id=model_id)

    async def health_check(self, model_id: str) -> bool:
        """Return True if the vLLM /health endpoint replies 200."""
        entry = self._processes.get(model_id)
        if not entry:
            return False
        proc, port = entry
        if proc.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        """
        Infer capabilities from local config.json and tokenizer_config.json.

        Uses conservative defaults for unknown/undetected models to avoid
        over-reporting unsupported capabilities.
        """
        model_path = self._resolve_local_model_dir(model_id)
        caps: dict[str, Any] = {"streaming": True, "completion": True}
        if model_path is None:
            return BackendCapabilities(
                chat=False,
                completion=True,
                tools=False,
                vision=False,
                embeddings=False,
                pooling=False,
                rerank=False,
                classification=False,
                score=False,
                reasoning=False,
                image_generation=False,
                audio_transcription=False,
                tts=False,
                streaming=True,
                context_length=0,
            )

        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with config_path.open() as f:
                    config = json.load(f)
            except Exception:
                config = {}

            # Architecture hints
            architectures: list[str] = config.get("architectures") or []
            for arch in architectures:
                arch_caps = _ARCH_CAPS.get(arch)
                if arch_caps:
                    caps.update(arch_caps)
                    break

            if any(str(arch).endswith("ForSequenceClassification") for arch in architectures):
                caps["classification"] = True

            num_labels = config.get("num_labels")
            id2label = config.get("id2label")
            if (
                isinstance(num_labels, int)
                and num_labels > 1
                or isinstance(id2label, dict)
                and len(id2label) > 1
            ):
                caps["classification"] = True

            # Context length
            for key in ("max_position_embeddings", "max_seq_len", "model_max_length"):
                if key in config:
                    caps["context_length"] = int(config[key])
                    break

        tok_path = model_path / "tokenizer_config.json"
        if tok_path.exists():
            try:
                with tok_path.open() as f:
                    tok_cfg = json.load(f)
                if tok_cfg.get("chat_template"):
                    caps["chat"] = True
            except Exception:
                pass

        gen_path = model_path / "generation_config.json"
        if gen_path.exists():
            try:
                with gen_path.open() as f:
                    gen_cfg = json.load(f)
                if "tool_choice" in json.dumps(gen_cfg):
                    caps["tools"] = True
            except Exception:
                pass

        model_hint = model_id.lower()
        if "cross-encoder" in model_hint or "rerank" in model_hint or "re-rank" in model_hint:
            caps["rerank"] = True
            caps["score"] = True
        if "classifier" in model_hint or "classification" in model_hint:
            caps["classification"] = True

        return BackendCapabilities(
            chat=caps.get("chat", False),
            completion=caps.get("completion", False),
            tools=caps.get("tools", False),
            vision=caps.get("vision", False),
            embeddings=caps.get("embeddings", False),
            pooling=caps.get("pooling", caps.get("embeddings", False)),
            rerank=caps.get("rerank", False),
            classification=caps.get("classification", False),
            score=caps.get("score", caps.get("embeddings", False)),
            reasoning=caps.get("reasoning", False),
            image_generation=False,
            audio_transcription=False,
            tts=False,
            streaming=caps.get("streaming", True),
            context_length=caps.get("context_length", 0),
        )

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        """
        Estimate VRAM from .safetensors file sizes × 1.2 overhead factor.

        Falls back to 0 if the directory does not exist yet.
        """
        model_path = self._resolve_local_model_dir(model_id)
        if model_path is None:
            return 0
        total_bytes = sum(
            p.stat().st_size for p in model_path.rglob("*.safetensors") if p.is_file()
        )
        if total_bytes == 0:
            # Try .bin (older format)
            total_bytes = sum(p.stat().st_size for p in model_path.rglob("*.bin") if p.is_file())
        estimated = int(total_bytes / 1024 / 1024 * 1.2)
        # Reserve a practical baseline for runtime memory (KV cache, graph capture, kernels).
        return max(_MIN_VLLM_VRAM_MB, estimated)

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        """Proxy a request to the vLLM worker."""
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No vLLM worker for model '{model_id}'")
        _, port = entry
        url = f"http://127.0.0.1:{port}{path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return resp.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        """Stream a request to the vLLM worker, yielding raw byte chunks."""
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No vLLM worker for model '{model_id}'")
        _, port = entry
        url = f"http://127.0.0.1:{port}{path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=body) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _wait_for_startup(self, model_id: str, port: int) -> None:
        """Poll /health until success or timeout."""
        deadline = asyncio.get_event_loop().time() + _STARTUP_TIMEOUT_S
        backoff = 1.0
        url = f"http://127.0.0.1:{port}/health"

        while asyncio.get_event_loop().time() < deadline:
            # Check if process already died (OOM etc.)
            entry = self._processes.get(model_id)
            if entry and entry[0].returncode is not None:
                rc = entry[0].returncode
                stderr_tail = await self._read_stderr_tail(entry[0])
                err_suffix = f" stderr={stderr_tail}" if stderr_tail else ""
                if rc == -signal.SIGKILL or rc == 137:
                    raise MemoryError(
                        f"vLLM process for '{model_id}' OOM-killed (rc={rc}).{err_suffix}"
                    )
                raise RuntimeError(
                    f"vLLM process for '{model_id}' exited with rc={rc}.{err_suffix}"
                )

            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return
            except Exception:
                pass

            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 10.0)

        raise TimeoutError(
            f"vLLM worker for '{model_id}' did not become healthy within {_STARTUP_TIMEOUT_S}s"
        )

    async def _wait_for_probe_startup(
        self,
        proc: asyncio.subprocess.Process,
        port: int,
        timeout_s: int = _STARTUP_TIMEOUT_S,
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout_s
        url = f"http://127.0.0.1:{port}/health"

        while asyncio.get_event_loop().time() < deadline:
            if proc.returncode is not None:
                raise RuntimeError(f"Probe process exited with rc={proc.returncode}")
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(0.5)

        raise TimeoutError(f"vLLM probe did not become healthy within {timeout_s}s")

    async def _terminate_probe_process(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        pgid: int | None = None
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, AttributeError, TypeError):
            pgid = None
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                return
            await proc.wait()

    async def _consume_stream(
        self,
        stream: asyncio.StreamReader | None,
        sink: list[str],
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.readline()
            if not chunk:
                return
            sink.append(chunk.decode("utf-8", errors="replace").rstrip())

    def _parse_memory_profile_logs(self, logs: str) -> dict[str, Any]:
        def _extract_float(pattern: str) -> float | None:
            match = re.search(pattern, logs, flags=re.MULTILINE)
            return float(match.group(1)) if match else None

        def _extract_int(pattern: str) -> int | None:
            match = re.search(pattern, logs, flags=re.MULTILINE)
            return int(match.group(1).replace(",", "")) if match else None

        model_loading_gib = _extract_float(r"Model loading took ([0-9.]+) GiB memory")
        available_kv_gib = _extract_float(r"Available KV cache memory: ([0-9.]+) GiB")
        gpu_kv_tokens = _extract_int(r"GPU KV cache size: ([0-9,]+) tokens")
        max_context = _extract_int(r"estimated maximum model length is ([0-9,]+)")
        concurrency_match = re.search(
            r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x",
            logs,
            flags=re.MULTILINE,
        )
        requested_context = (
            int(concurrency_match.group(1).replace(",", "")) if concurrency_match else None
        )
        maximum_concurrency = float(concurrency_match.group(2)) if concurrency_match else None
        return {
            "model_loading_memory_mb": int(model_loading_gib * 1024) if model_loading_gib is not None else None,
            "available_kv_cache_mb": int(available_kv_gib * 1024) if available_kv_gib is not None else None,
            "gpu_kv_cache_tokens": gpu_kv_tokens,
            "estimated_max_model_len": max_context,
            "requested_context_length": requested_context,
            "maximum_concurrency": maximum_concurrency,
        }

    @staticmethod
    def _reserve_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    async def _kill_process(self, model_id: str, graceful: bool = False) -> None:
        entry = self._processes.pop(model_id, None)
        if not entry:
            return
        proc, _ = entry
        if proc.returncode is not None:
            return

        pgid: int | None = None
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, AttributeError, TypeError):
            pgid = None

        if graceful:
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    proc.terminate()  # SIGTERM
                await asyncio.wait_for(proc.wait(), timeout=float(_SHUTDOWN_TIMEOUT_S))
                return
            except (TimeoutError, ProcessLookupError):
                logger.warning("vllm_sigterm_timeout", model_id=model_id)

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()  # SIGKILL
            await proc.wait()
        except ProcessLookupError:
            pass

    async def _read_stderr_tail(self, proc: asyncio.subprocess.Process, limit: int = 4000) -> str:
        if proc.stderr is None:
            return ""
        try:
            stderr_data = await asyncio.wait_for(proc.stderr.read(), timeout=0.3)
        except Exception:
            return ""
        if not stderr_data:
            return ""
        text = stderr_data.decode("utf-8", errors="replace").strip()
        if len(text) <= limit:
            return text
        head = text[: limit // 2]
        tail = text[-(limit // 2) :]
        return f"{head}\n...[truncated]...\n{tail}"

    def _resolve_local_model_dir(self, model_id: str) -> Path | None:
        base = Path(settings.models_dir)
        direct = base / model_id
        if direct.exists() and direct.is_dir():
            return direct

        hf_layout = base / "huggingface" / model_id.replace("/", "--")
        if hf_layout.exists() and hf_layout.is_dir():
            return hf_layout

        cached = self._resolve_hf_cache_snapshot_dir(model_id)
        if cached is not None:
            return cached

        return None

    def _resolve_hf_cache_snapshot_dir(self, model_id: str) -> Path | None:
        """
        Resolve a model snapshot directory from the Hugging Face cache layout.

        Expected structure:
        <hf_cache_dir>/hub/models--org--repo/{refs,snapshots}/...
        """
        hf_cache_dir = getattr(settings, "hf_cache_dir", "")
        if not hf_cache_dir:
            return None

        model_cache_root = Path(hf_cache_dir) / "hub" / f"models--{model_id.replace('/', '--')}"
        snapshots_dir = model_cache_root / "snapshots"
        if not snapshots_dir.exists() or not snapshots_dir.is_dir():
            return None

        ref_main = model_cache_root / "refs" / "main"
        if ref_main.exists():
            try:
                commit = ref_main.read_text(encoding="utf-8").strip()
            except Exception:
                commit = ""
            if commit:
                snap = snapshots_dir / commit
                if snap.exists() and snap.is_dir():
                    return snap

        candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _resolve_model_target(self, model_id: str) -> str:
        """
        Resolve model target for vLLM.

        Priority:
        1. Local legacy layout: <models_dir>/<model_id>
        2. Local HF layout: <models_dir>/huggingface/<repo-with--separators>
        3. Fallback to raw model_id (remote repo resolution by vLLM/HF)
        """
        local_dir = self._resolve_local_model_dir(model_id)
        if local_dir:
            return str(local_dir)
        return model_id
