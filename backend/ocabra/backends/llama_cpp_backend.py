from __future__ import annotations

import asyncio
import os
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

_DEFAULT_TOTAL_LAYERS = 32
_DEFAULT_STARTUP_TIMEOUT_S = 30
_SHUTDOWN_TIMEOUT_S = 20
_WORKER_PATH = Path(__file__).resolve().parents[1] / "workers" / "llama_cpp_worker.py"


class LlamaCppBackend(BackendInterface):
    def __init__(self) -> None:
        self._processes: dict[str, tuple[asyncio.subprocess.Process, int]] = {}
        self._model_configs: dict[str, dict[str, Any]] = {}

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")
        if not _WORKER_PATH.exists():
            raise FileNotFoundError(f"llama_cpp_worker.py not found at '{_WORKER_PATH}'")

        extra_config = kwargs.get("extra_config") or {}
        model_file = self._resolve_model_file(model_id, extra_config)
        options = self._build_options(extra_config)
        options["model_file"] = str(model_file)
        self._model_configs[model_id] = options

        cmd = [
            sys.executable,
            str(_WORKER_PATH),
            "--server-bin",
            settings.llama_cpp_server_bin,
            "--model-id",
            model_id,
            "--model-path",
            str(model_file),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(options["ctx_size"]),
            "--batch-size",
            str(options["batch_size"]),
            "--ubatch-size",
            str(options["ubatch_size"]),
            "--gpu-layers",
            str(options["gpu_layers"]),
            "--alias",
            model_id,
        ]
        if options["threads"] is not None:
            cmd.extend(["--threads", str(options["threads"])])
        if options["flash_attn"]:
            cmd.append("--flash-attn")
        if options["mlock"]:
            cmd.append("--mlock")
        if options["embedding"]:
            cmd.append("--embedding")

        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": settings.cuda_device_order,
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_indices)
            if options["gpu_layers"] > 0
            else "",
        }

        logger.info(
            "llama_cpp_starting",
            model_id=model_id,
            port=port,
            gpu_layers=options["gpu_layers"],
        )
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
                timeout_s=int(options.get("startup_timeout_s") or _DEFAULT_STARTUP_TIMEOUT_S),
            )
        except Exception:
            await self._kill_process(model_id)
            raise

        vram_used_mb = await self.get_vram_estimate_mb(model_id)
        return WorkerInfo(
            backend_type="llama_cpp",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid or 0,
            vram_used_mb=vram_used_mb,
        )

    async def unload(self, model_id: str) -> None:
        await self._kill_process(model_id, graceful=True)
        self._model_configs.pop(model_id, None)

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
        options = self._model_configs.get(model_id, {})
        normalized = model_id.lower()
        embeddings = bool(options.get("embedding")) or any(
            token in normalized for token in ("embed", "embedding", "bge", "e5")
        )
        vision = any(token in normalized for token in ("llava", "vision", "minicpmv", "vl"))
        reasoning = any(token in normalized for token in ("deepseek-r1", "qwen3", "reason"))
        tools = not embeddings and any(
            token in normalized for token in ("llama", "mistral", "mixtral", "qwen", "tool")
        )
        return BackendCapabilities(
            chat=not embeddings,
            completion=True,
            tools=tools,
            vision=vision,
            embeddings=embeddings,
            reasoning=reasoning,
            streaming=True,
            context_length=int(options.get("ctx_size", settings.llama_cpp_ctx_size)),
        )

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        options = self._model_configs.get(model_id, {})
        gpu_layers = int(options.get("gpu_layers", settings.llama_cpp_gpu_layers))
        if gpu_layers <= 0:
            return 0

        total_layers = max(1, int(options.get("total_layers", _DEFAULT_TOTAL_LAYERS)))
        model_file = options.get("model_file")
        if model_file:
            size_bytes = await asyncio.to_thread(lambda: Path(model_file).stat().st_size)
            total_mb = max(1, int(size_bytes / (1024 * 1024)))
        else:
            total_mb = max(1, int(options.get("model_vram_mb", 4096)))
        return max(1, int(total_mb * min(gpu_layers, total_layers) / total_layers))

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No llama.cpp worker for model '{model_id}'")
        _, port = entry
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"http://127.0.0.1:{port}{path}", json=body)
            response.raise_for_status()
            return response.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No llama.cpp worker for model '{model_id}'")
        _, port = entry
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", f"http://127.0.0.1:{port}{path}", json=body) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    def _build_options(self, extra_config: dict[str, Any]) -> dict[str, Any]:
        options = {
            "gpu_layers": int(
                self._get_option(extra_config, "gpu_layers", settings.llama_cpp_gpu_layers)
            ),
            "ctx_size": int(self._get_option(extra_config, "ctx_size", settings.llama_cpp_ctx_size)),
            "threads": self._to_int_or_none(
                self._get_option(extra_config, "threads", settings.llama_cpp_threads)
            ),
            "batch_size": int(
                self._get_option(extra_config, "batch_size", settings.llama_cpp_batch_size)
            ),
            "ubatch_size": int(
                self._get_option(extra_config, "ubatch_size", settings.llama_cpp_ubatch_size)
            ),
            "flash_attn": bool(
                self._get_option(extra_config, "flash_attn", settings.llama_cpp_flash_attn)
            ),
            "mlock": bool(self._get_option(extra_config, "mlock", settings.llama_cpp_mlock)),
            "embedding": bool(
                self._get_option(extra_config, "embedding", settings.llama_cpp_embeddings)
            ),
            "startup_timeout_s": int(
                self._get_option(
                    extra_config,
                    "startup_timeout_s",
                    settings.llama_cpp_startup_timeout_s,
                )
            ),
            "total_layers": int(
                self._get_option(extra_config, "total_layers", _DEFAULT_TOTAL_LAYERS)
            ),
        }
        return options

    def _get_option(self, extra_config: dict[str, Any], key: str, default: Any) -> Any:
        nested = extra_config.get("llama_cpp")
        if isinstance(nested, dict) and key in nested:
            return nested[key]
        return extra_config.get(key, default)

    def _resolve_model_file(self, model_id: str, extra_config: dict[str, Any]) -> Path:
        explicit = self._get_option(extra_config, "model_path", None)
        if explicit:
            path = Path(str(explicit))
            if path.is_file():
                return path

        direct = Path(model_id)
        if direct.is_file():
            return direct

        root = Path(settings.models_dir)
        candidates = [
            root / model_id,
            root / f"{model_id}.gguf",
            root / "huggingface" / model_id.replace("/", "--"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        stem = Path(model_id).stem
        matches = [path for path in root.rglob("*.gguf") if path.stem == stem]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)

        raise FileNotFoundError(f"llama.cpp GGUF model not found for '{model_id}' under {root}")

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
                        f"llama.cpp worker for '{model_id}' was OOM-killed (rc={rc}).{suffix}"
                    )
                raise RuntimeError(
                    f"llama.cpp worker for '{model_id}' exited with rc={rc}.{suffix}"
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
            f"llama.cpp worker for '{model_id}' did not become healthy within {timeout_s}s"
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
                logger.warning("llama_cpp_sigterm_timeout", model_id=model_id)

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

    def _to_int_or_none(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)
