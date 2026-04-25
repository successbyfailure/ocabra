from __future__ import annotations

import asyncio
import json
import os
import signal
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
from ocabra.core.backend_installer import (
    venv_cuda_home,
    venv_nvidia_ld_library_path,
)

logger = structlog.get_logger(__name__)

_ARCH_CAPS: dict[str, dict[str, Any]] = {
    "LlavaNextForConditionalGeneration": {"vision": True, "chat": True},
    "LlavaForConditionalGeneration": {"vision": True, "chat": True},
    "Qwen2VLForConditionalGeneration": {"vision": True, "chat": True, "tools": True},
    "InternVLChatModel": {"vision": True, "chat": True},
    "DeepseekV3ForCausalLM": {"reasoning": True, "chat": True, "tools": True},
    "DeepseekR1ForCausalLM": {"reasoning": True, "chat": True},
    "Qwen3ForCausalLM": {"reasoning": True, "chat": True, "tools": True},
    "Qwen3MoeForCausalLM": {"reasoning": True, "chat": True, "tools": True},
    "LlamaForCausalLM": {"chat": True, "tools": True},
    "MistralForCausalLM": {"chat": True, "tools": True},
    "MixtralForCausalLM": {"chat": True, "tools": True},
    "BertModel": {"embeddings": True, "pooling": True, "score": True},
    "RobertaModel": {"embeddings": True, "pooling": True, "score": True},
}

_MIN_SGLANG_VRAM_MB = 2048
_SHUTDOWN_TIMEOUT_S = 30
_WORKER_PATH = Path(__file__).resolve().parents[1] / "workers" / "sglang_worker.py"


class SGLangBackend(BackendInterface):

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.TEXT_GENERATION, ModalityType.EMBEDDINGS}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Declarative install spec for the sglang backend (Bloque 15 Fase 2)."""

        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-sglang",
            oci_tags={"cuda12": "latest-cuda12"},
            apt_packages=[
                # SGLang's tvm_ffi JIT-compiles inline C++/CUDA at runtime.
                # gcc/g++ are mandatory; ninja-build speeds up the JIT.
                "gcc",
                "g++",
                "ninja-build",
            ],
            pip_packages=[
                "sglang==0.5.9",
                # CUDA toolkit wheels — provide nvcc + headers + libs that
                # tvm_ffi looks for under CUDA_HOME (Deuda D12, validated
                # 2026-04-25). venv_cuda_home() in core.backend_installer
                # builds a fake CUDA_HOME that points at these wheels.
                "nvidia-cuda-nvcc-cu12",
                "nvidia-cuda-runtime-cu12",
                "nvidia-cuda-cccl-cu12",
            ],
            pip_extra_index_urls=[
                "https://download.pytorch.org/whl/cu124",
            ],
            estimated_size_mb=9000,
            display_name="SGLang",
            description="High-throughput LLM inference engine focused on structured generation",
            tags=["LLM", "GPU", "CUDA"],
        )

    def __init__(self) -> None:
        self._processes: dict[str, tuple[asyncio.subprocess.Process, int]] = {}

    def _resolve_python_bin(self) -> str:
        """Pick the python that runs the sglang server inside the worker.

        Priority: ``<backends_dir>/sglang/metadata.json`` > legacy
        ``settings.sglang_python_bin`` > :data:`sys.executable`.
        """

        try:
            meta_path = Path(settings.backends_dir) / "sglang" / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                bin_path = meta.get("python_bin") if isinstance(meta, dict) else None
                if bin_path and Path(bin_path).is_file():
                    return str(bin_path)
        except (OSError, json.JSONDecodeError):
            pass

        legacy = (settings.sglang_python_bin or "").strip()
        if legacy and Path(legacy).is_file():
            return legacy
        return sys.executable

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")
        if not _WORKER_PATH.exists():
            raise FileNotFoundError(f"sglang_worker.py not found at '{_WORKER_PATH}'")

        extra_config = kwargs.get("extra_config") or {}
        model_target = self._resolve_model_target(model_id)
        tensor_parallel = self._get_option(
            extra_config,
            "tensor_parallel_size",
            settings.sglang_tensor_parallel_size,
        ) or max(1, len(gpu_indices))

        sglang_python = self._resolve_python_bin()
        cmd = [
            sys.executable,
            str(_WORKER_PATH),
            "--python-bin",
            sglang_python,
            "--server-module",
            settings.sglang_server_module,
            "--model-id",
            model_id,
            "--model-path",
            model_target,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tp",
            str(tensor_parallel),
            "--mem-fraction-static",
            str(
                self._get_option(
                    extra_config,
                    "mem_fraction_static",
                    settings.sglang_mem_fraction_static,
                )
            ),
            "--served-model-name",
            model_id,
        ]
        attention_backend = self._get_option(extra_config, "attention_backend", None)
        if attention_backend:
            cmd.extend(["--attention-backend", str(attention_backend)])
        prefill_attention_backend = self._get_option(extra_config, "prefill_attention_backend", None)
        if prefill_attention_backend:
            cmd.extend(["--prefill-attention-backend", str(prefill_attention_backend)])
        decode_attention_backend = self._get_option(extra_config, "decode_attention_backend", None)
        if decode_attention_backend:
            cmd.extend(["--decode-attention-backend", str(decode_attention_backend)])
        sampling_backend = self._get_option(extra_config, "sampling_backend", None)
        if sampling_backend:
            cmd.extend(["--sampling-backend", str(sampling_backend)])
        if bool(self._get_option(extra_config, "disable_cuda_graph", False)):
            cmd.append("--disable-cuda-graph")
        context_length = self._get_option(
            extra_config,
            "context_length",
            settings.sglang_context_length,
        )
        if context_length:
            cmd.extend(["--context-length", str(context_length)])
        if self._get_option(
            extra_config,
            "trust_remote_code",
            settings.sglang_trust_remote_code,
        ):
            cmd.append("--trust-remote-code")
        if self._get_option(
            extra_config,
            "disable_radix_cache",
            settings.sglang_disable_radix_cache,
        ):
            cmd.append("--disable-radix-cache")

        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": settings.cuda_device_order,
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_indices),
            "HF_HOME": settings.hf_cache_dir,
        }
        if settings.hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

        # SGLang JIT-compiles CUDA kernels on first run; tvm_ffi requires
        # CUDA_HOME and the loader needs the bundled CUDA runtime libs on
        # LD_LIBRARY_PATH. Both are derived from the venv wheels (Deuda D12).
        cuda_home = venv_cuda_home(settings.backends_dir, "sglang")
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env["CUDA_PATH"] = cuda_home
            env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
        nvidia_ld = venv_nvidia_ld_library_path(settings.backends_dir, "sglang")
        if nvidia_ld:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{nvidia_ld}:{existing}" if existing else nvidia_ld

        logger.info("sglang_starting", model_id=model_id, port=port, tp=tensor_parallel)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._processes[model_id] = (process, port)

        try:
            await self._wait_for_startup(
                model_id,
                port,
                timeout_s=int(
                    self._get_option(
                        extra_config,
                        "startup_timeout_s",
                        settings.sglang_startup_timeout_s,
                    )
                ),
            )
        except Exception:
            await self._kill_process(model_id)
            raise

        return WorkerInfo(
            backend_type="sglang",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid or 0,
            vram_used_mb=await self.get_vram_estimate_mb(model_id),
        )

    async def unload(self, model_id: str) -> None:
        await self._kill_process(model_id, graceful=True)

    async def health_check(self, model_id: str) -> bool:
        entry = self._processes.get(model_id)
        if not entry:
            return False
        process, port = entry
        if process.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"http://127.0.0.1:{port}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        model_path = self._resolve_local_model_dir(model_id)
        caps: dict[str, Any] = {"streaming": True, "completion": True, "chat": True, "tools": True}
        if model_path is not None:
            config_path = model_path / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:
                    config = {}
                for arch in config.get("architectures") or []:
                    arch_caps = _ARCH_CAPS.get(arch)
                    if arch_caps:
                        caps.update(arch_caps)
                        break
                for key in ("max_position_embeddings", "max_seq_len", "model_max_length"):
                    if key in config:
                        caps["context_length"] = int(config[key])
                        break
            tok_path = model_path / "tokenizer_config.json"
            if tok_path.exists():
                try:
                    tokenizer = json.loads(tok_path.read_text(encoding="utf-8"))
                except Exception:
                    tokenizer = {}
                if tokenizer.get("chat_template"):
                    caps["chat"] = True

        normalized = model_id.lower()
        if any(token in normalized for token in ("embed", "embedding", "bge", "e5")):
            caps["chat"] = False
            caps["tools"] = False
            caps["embeddings"] = True
            caps["pooling"] = True
            caps["score"] = True
        if any(token in normalized for token in ("llava", "vision", "minicpmv", "vl")):
            caps["vision"] = True
        if any(token in normalized for token in ("deepseek-r1", "qwen3", "reason")):
            caps["reasoning"] = True

        return BackendCapabilities(
            chat=caps.get("chat", False),
            completion=caps.get("completion", False),
            tools=caps.get("tools", False),
            vision=caps.get("vision", False),
            embeddings=caps.get("embeddings", False),
            pooling=caps.get("pooling", caps.get("embeddings", False)),
            reasoning=caps.get("reasoning", False),
            streaming=True,
            context_length=int(caps.get("context_length", 0)),
            score=caps.get("score", caps.get("embeddings", False)),
        )

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        model_path = self._resolve_local_model_dir(model_id)
        if model_path is None:
            return _MIN_SGLANG_VRAM_MB
        total_bytes = sum(path.stat().st_size for path in model_path.rglob("*.safetensors") if path.is_file())
        if total_bytes == 0:
            total_bytes = sum(path.stat().st_size for path in model_path.rglob("*.bin") if path.is_file())
        estimated = int(total_bytes / 1024 / 1024 * 1.2)
        return max(_MIN_SGLANG_VRAM_MB, estimated)

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No SGLang worker for model '{model_id}'")
        _, port = entry
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"http://127.0.0.1:{port}{path}", json=body)
            response.raise_for_status()
            return response.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No SGLang worker for model '{model_id}'")
        _, port = entry
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", f"http://127.0.0.1:{port}{path}", json=body) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    def _get_option(self, extra_config: dict[str, Any], key: str, default: Any) -> Any:
        nested = extra_config.get("sglang")
        if isinstance(nested, dict) and key in nested:
            return nested[key]
        return extra_config.get(key, default)

    async def _wait_for_startup(self, model_id: str, port: int, timeout_s: int) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        url = f"http://127.0.0.1:{port}/health"
        while asyncio.get_running_loop().time() < deadline:
            entry = self._processes.get(model_id)
            if entry and entry[0].returncode is not None:
                rc = entry[0].returncode
                stderr_tail = await self._read_stderr_tail(entry[0])
                suffix = f" stderr={stderr_tail}" if stderr_tail else ""
                if rc in {-signal.SIGKILL, 137}:
                    raise MemoryError(f"SGLang worker for '{model_id}' was OOM-killed (rc={rc}).{suffix}")
                raise RuntimeError(f"SGLang worker for '{model_id}' exited with rc={rc}.{suffix}")
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1)
        raise TimeoutError(f"SGLang worker for '{model_id}' did not become healthy within {timeout_s}s")

    async def _kill_process(self, model_id: str, graceful: bool = False) -> None:
        entry = self._processes.pop(model_id, None)
        if not entry:
            return
        process, _ = entry
        if process.returncode is not None:
            return
        if graceful:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=float(_SHUTDOWN_TIMEOUT_S))
                return
            except (TimeoutError, ProcessLookupError):
                logger.warning("sglang_sigterm_timeout", model_id=model_id)
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass

    async def _read_stderr_tail(self, process: asyncio.subprocess.Process, limit: int = 4000) -> str:
        if process.stderr is None:
            return ""
        try:
            data = await asyncio.wait_for(process.stderr.read(), timeout=0.3)
        except Exception:
            return ""
        if not data:
            return ""
        text = data.decode("utf-8", errors="replace").strip()
        if len(text) <= limit:
            return text
        half = limit // 2
        return f"{text[:half]}\n...[truncated]...\n{text[-half:]}"

    def _resolve_model_target(self, model_id: str) -> str:
        local = self._resolve_local_model_dir(model_id)
        if local is not None:
            return str(local)
        return model_id

    def _resolve_local_model_dir(self, model_id: str) -> Path | None:
        base = Path(settings.models_dir)
        direct = base / model_id
        if direct.exists() and direct.is_dir():
            return direct

        hf_layout = base / "huggingface" / model_id.replace("/", "--")
        if hf_layout.exists() and hf_layout.is_dir():
            return hf_layout

        hf_cache_dir = getattr(settings, "hf_cache_dir", "")
        if hf_cache_dir:
            model_cache_root = Path(hf_cache_dir) / "hub" / f"models--{model_id.replace('/', '--')}"
            snapshots_dir = model_cache_root / "snapshots"
            if snapshots_dir.exists() and snapshots_dir.is_dir():
                candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
                if candidates:
                    return max(candidates, key=lambda path: path.stat().st_mtime)
        return None
