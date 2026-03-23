from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.backends.base import BackendCapabilities, BackendInterface, WorkerInfo
from ocabra.config import settings

logger = structlog.get_logger(__name__)

_MIN_TRT_VRAM_MB = 4096
_SHUTDOWN_TIMEOUT_S = 30
_WORKER_PATH = Path(__file__).resolve().parents[1] / "workers" / "tensorrt_llm_worker.py"


class TensorRTLLMBackend(BackendInterface):
    def __init__(self) -> None:
        self._processes: dict[str, tuple[asyncio.subprocess.Process, int]] = {}
        self.disabled_reason = self._detect_disabled_reason()

    def is_enabled(self) -> bool:
        return self.disabled_reason is None

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        if not self.is_enabled():
            raise RuntimeError(self.disabled_reason or "TensorRT-LLM backend is disabled")
        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")
        if not _WORKER_PATH.exists():
            raise FileNotFoundError(f"tensorrt_llm_worker.py not found at '{_WORKER_PATH}'")

        extra_config = kwargs.get("extra_config") or {}
        engine_dir = self._resolve_engine_dir(model_id, extra_config)
        tokenizer_path = self._resolve_tokenizer_path(model_id, extra_config, engine_dir)
        cmd = [
            sys.executable,
            str(_WORKER_PATH),
            "--serve-bin",
            settings.tensorrt_llm_serve_bin,
            "--model-id",
            model_id,
            "--engine-dir",
            str(engine_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend",
            str(self._get_option(extra_config, "backend", settings.tensorrt_llm_backend)),
        ]
        if tokenizer_path is not None:
            cmd.extend(["--tokenizer-path", str(tokenizer_path)])
        max_batch_size = self._get_option(
            extra_config,
            "max_batch_size",
            settings.tensorrt_llm_max_batch_size,
        )
        if max_batch_size:
            cmd.extend(["--max-batch-size", str(max_batch_size)])
        context_length = self._get_option(
            extra_config,
            "context_length",
            settings.tensorrt_llm_context_length,
        )
        if context_length:
            cmd.extend(["--max-num-tokens", str(context_length)])
        if self._get_option(
            extra_config,
            "trust_remote_code",
            settings.tensorrt_llm_trust_remote_code,
        ):
            cmd.append("--trust-remote-code")

        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": settings.cuda_device_order,
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_indices),
        }

        logger.info("tensorrt_llm_starting", model_id=model_id, port=port, engine_dir=str(engine_dir))
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
                        settings.tensorrt_llm_startup_timeout_s,
                    )
                ),
            )
        except Exception:
            await self._kill_process(model_id)
            raise

        return WorkerInfo(
            backend_type="tensorrt_llm",
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
        normalized = model_id.lower()
        return BackendCapabilities(
            chat="embed" not in normalized,
            completion=True,
            tools="embed" not in normalized,
            vision=any(token in normalized for token in ("vision", "vl", "llava")),
            embeddings=any(token in normalized for token in ("embed", "embedding", "bge", "e5")),
            reasoning=any(token in normalized for token in ("deepseek-r1", "qwen3", "reason")),
            streaming=True,
            context_length=int(settings.tensorrt_llm_context_length or 0),
        )

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        if not self.is_enabled():
            return 0
        try:
            engine_dir = self._resolve_engine_dir(model_id, {})
        except Exception:
            return _MIN_TRT_VRAM_MB
        total_bytes = sum(path.stat().st_size for path in engine_dir.rglob("*") if path.is_file())
        estimated = int(total_bytes / 1024 / 1024 * 1.1)
        return max(_MIN_TRT_VRAM_MB, estimated)

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No TensorRT-LLM worker for model '{model_id}'")
        _, port = entry
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"http://127.0.0.1:{port}{path}", json=body)
            response.raise_for_status()
            return response.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No TensorRT-LLM worker for model '{model_id}'")
        _, port = entry
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", f"http://127.0.0.1:{port}{path}", json=body) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    def _detect_disabled_reason(self) -> str | None:
        if not settings.tensorrt_llm_enabled:
            return "feature flag TENSORRT_LLM_ENABLED=false"
        if not shutil.which(settings.tensorrt_llm_serve_bin) and not Path(
            settings.tensorrt_llm_serve_bin
        ).exists():
            return f"serve binary not found: {settings.tensorrt_llm_serve_bin}"
        return None

    def _get_option(self, extra_config: dict[str, Any], key: str, default: Any) -> Any:
        nested = extra_config.get("tensorrt_llm")
        if isinstance(nested, dict) and key in nested:
            return nested[key]
        return extra_config.get(key, default)

    def _resolve_engine_dir(self, model_id: str, extra_config: dict[str, Any]) -> Path:
        explicit = self._get_option(extra_config, "engine_dir", None)
        if explicit:
            path = Path(str(explicit))
            if path.exists() and path.is_dir():
                return path

        direct = Path(model_id)
        if direct.exists() and direct.is_dir():
            return direct

        if settings.tensorrt_llm_engines_dir:
            root = Path(settings.tensorrt_llm_engines_dir)
            candidates = [root / model_id, root / model_id.replace("/", "--")]
            for candidate in candidates:
                if candidate.exists() and candidate.is_dir():
                    return candidate

        raise FileNotFoundError(f"TensorRT-LLM engine directory not found for '{model_id}'")

    def _resolve_tokenizer_path(
        self,
        model_id: str,
        extra_config: dict[str, Any],
        engine_dir: Path,
    ) -> Path | None:
        explicit = self._get_option(extra_config, "tokenizer_path", None) or settings.tensorrt_llm_tokenizer_path
        if explicit:
            path = Path(str(explicit))
            if path.exists():
                return path

        for candidate in (
            Path(settings.models_dir) / model_id,
            Path(settings.models_dir) / "huggingface" / model_id.replace("/", "--"),
            engine_dir,
            engine_dir.parent,
        ):
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

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
                    raise MemoryError(
                        f"TensorRT-LLM worker for '{model_id}' was OOM-killed (rc={rc}).{suffix}"
                    )
                raise RuntimeError(
                    f"TensorRT-LLM worker for '{model_id}' exited with rc={rc}.{suffix}"
                )
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1)
        raise TimeoutError(
            f"TensorRT-LLM worker for '{model_id}' did not become healthy within {timeout_s}s"
        )

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
                logger.warning("tensorrt_llm_sigterm_timeout", model_id=model_id)
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
