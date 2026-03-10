"""
vLLM backend — one subprocess per loaded model.

Each model gets an isolated vLLM OpenAI-compatible API server on a dedicated
port in the range configured by settings.worker_port_range_*.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.backends.base import BackendCapabilities, BackendInterface, WorkerInfo
from ocabra.config import settings

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
    "BertModel": {"embeddings": True},
    "RobertaModel": {"embeddings": True},
    "XLMRobertaModel": {"embeddings": True},
    "DistilBertModel": {"embeddings": True},
    "E5Model": {"embeddings": True},
}

_STARTUP_TIMEOUT_S = 120
_SHUTDOWN_TIMEOUT_S = 30


class VLLMBackend(BackendInterface):
    """
    Manages vLLM subprocess workers.

    One VLLMBackend instance is shared by the WorkerPool.  It keeps a
    registry of running processes keyed by model_id.
    """

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
        from ocabra.core.worker_pool import WorkerPool  # avoid circular at module level

        # Port must be pre-assigned by WorkerPool before calling load().
        # Callers (ModelManager) are expected to call worker_pool.assign_port()
        # and pass it via kwargs.
        port: int = kwargs.get("port", 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        model_path = str(Path(settings.models_dir) / model_id)

        cuda_devices = ",".join(str(i) for i in gpu_indices)
        tensor_parallel = len(gpu_indices)

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--tensor-parallel-size", str(tensor_parallel),
            "--gpu-memory-utilization", "0.90",
            "--port", str(port),
            "--host", "127.0.0.1",
            "--served-model-name", model_id,
            "--disable-log-requests",
        ]

        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HOME": settings.hf_cache_dir,
        }
        if settings.hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

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

        Falls back to ``{"chat": True, "completion": True, "streaming": True}``
        for unknown architectures.
        """
        model_path = Path(settings.models_dir) / model_id
        caps: dict[str, Any] = {"streaming": True, "completion": True}

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
            else:
                # Unknown architecture — assume basic chat
                caps["chat"] = True

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

        return BackendCapabilities(
            chat=caps.get("chat", False),
            completion=caps.get("completion", False),
            tools=caps.get("tools", False),
            vision=caps.get("vision", False),
            embeddings=caps.get("embeddings", False),
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
        model_path = Path(settings.models_dir) / model_id
        total_bytes = sum(
            p.stat().st_size
            for p in model_path.rglob("*.safetensors")
            if p.is_file()
        )
        if total_bytes == 0:
            # Try .bin (older format)
            total_bytes = sum(
                p.stat().st_size
                for p in model_path.rglob("*.bin")
                if p.is_file()
            )
        return int(total_bytes / 1024 / 1024 * 1.2)

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

    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        """Stream a request to the vLLM worker, yielding raw byte chunks."""
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No vLLM worker for model '{model_id}'")
        _, port = entry
        url = f"http://127.0.0.1:{port}{path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=body) as resp:
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
                if rc == -signal.SIGKILL or rc == 137:
                    raise MemoryError(f"vLLM process for '{model_id}' OOM-killed (rc={rc})")
                raise RuntimeError(f"vLLM process for '{model_id}' exited with rc={rc}")

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

    async def _kill_process(self, model_id: str, graceful: bool = False) -> None:
        entry = self._processes.pop(model_id, None)
        if not entry:
            return
        proc, _ = entry
        if proc.returncode is not None:
            return

        if graceful:
            try:
                proc.terminate()  # SIGTERM
                await asyncio.wait_for(proc.wait(), timeout=float(_SHUTDOWN_TIMEOUT_S))
                return
            except (asyncio.TimeoutError, ProcessLookupError):
                logger.warning("vllm_sigterm_timeout", model_id=model_id)

        try:
            proc.kill()  # SIGKILL
            await proc.wait()
        except ProcessLookupError:
            pass
