from __future__ import annotations

import asyncio
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

_SHUTDOWN_TIMEOUT_S = 20
_DEFAULT_MODEL_MB = 400
_DEFAULT_TOTAL_LAYERS = 32


class BitnetBackend(BackendInterface):
    def __init__(self) -> None:
        self._processes: dict[str, tuple[asyncio.subprocess.Process, int]] = {}
        self._model_configs: dict[str, dict[str, Any]] = {}

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        extra_config = kwargs.get("extra_config") or {}
        model_file = self._resolve_model_file(model_id, extra_config)
        options = self._build_options(extra_config)
        self._model_configs[model_id] = options

        cmd = [
            settings.bitnet_server_bin,
            "--model",
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
            "--parallel",
            str(options["parallel"]),
            "--n-gpu-layers",
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

        if options["gpu_layers"] > 0:
            visible = ",".join(str(i) for i in gpu_indices)
        else:
            visible = ""
        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": settings.cuda_device_order,
            "CUDA_VISIBLE_DEVICES": visible,
        }
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        preferred_paths = ["/usr/local/lib/bitnet", "/usr/local/lib/llama_cpp"]
        ld_parts = preferred_paths + ([current_ld_path] if current_ld_path else [])
        env["LD_LIBRARY_PATH"] = ":".join(part for part in ld_parts if part)

        logger.info("bitnet_starting", model_id=model_id, port=port, gpu_layers=options["gpu_layers"])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._processes[model_id] = (proc, port)

        try:
            await self._wait_for_startup(model_id, port, timeout_s=settings.bitnet_startup_timeout_s)
        except Exception:
            await self._kill_process(model_id)
            raise

        vram_estimate = await self.get_vram_estimate_mb(model_id)
        return WorkerInfo(
            backend_type="bitnet",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=proc.pid or 0,
            vram_used_mb=vram_estimate,
        )

    async def unload(self, model_id: str) -> None:
        await self._kill_process(model_id, graceful=True)
        self._model_configs.pop(model_id, None)

    async def health_check(self, model_id: str) -> bool:
        entry = self._processes.get(model_id)
        if not entry:
            return False
        proc, port = entry
        if proc.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"http://127.0.0.1:{port}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        options = self._model_configs.get(model_id, {})
        context_length = int(options.get("ctx_size", settings.bitnet_ctx_size))
        return BackendCapabilities(
            chat=True,
            completion=True,
            tools=False,
            vision=False,
            embeddings=False,
            streaming=True,
            context_length=context_length,
        )

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        options = self._model_configs.get(model_id, {})
        gpu_layers = int(options.get("gpu_layers", settings.bitnet_gpu_layers))
        if gpu_layers <= 0:
            return 0
        total_layers = max(1, int(options.get("total_layers", _DEFAULT_TOTAL_LAYERS)))
        model_mb = max(1, int(options.get("model_vram_mb", _DEFAULT_MODEL_MB)))
        return int(model_mb * min(gpu_layers, total_layers) / total_layers)

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No BitNet worker for model '{model_id}'")
        _, port = entry
        url = f"http://127.0.0.1:{port}{path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=body)
            response.raise_for_status()
            return response.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No BitNet worker for model '{model_id}'")
        _, port = entry
        url = f"http://127.0.0.1:{port}{path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=body) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    def _build_options(self, extra_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "gpu_layers": int(
                self._get_bitnet_option(extra_config, "gpu_layers", settings.bitnet_gpu_layers)
            ),
            "ctx_size": int(
                self._get_bitnet_option(extra_config, "ctx_size", settings.bitnet_ctx_size)
            ),
            "threads": self._to_int_or_none(
                self._get_bitnet_option(extra_config, "threads", settings.bitnet_threads)
            ),
            "batch_size": int(
                self._get_bitnet_option(extra_config, "batch_size", settings.bitnet_batch_size)
            ),
            "ubatch_size": int(
                self._get_bitnet_option(extra_config, "ubatch_size", settings.bitnet_ubatch_size)
            ),
            "parallel": int(
                self._get_bitnet_option(extra_config, "parallel", settings.bitnet_parallel)
            ),
            "flash_attn": bool(
                self._get_bitnet_option(extra_config, "flash_attn", settings.bitnet_flash_attn)
            ),
            "mlock": bool(self._get_bitnet_option(extra_config, "mlock", settings.bitnet_mlock)),
            "total_layers": int(
                self._get_bitnet_option(extra_config, "total_layers", _DEFAULT_TOTAL_LAYERS)
            ),
            "model_vram_mb": int(
                self._get_bitnet_option(extra_config, "model_vram_mb", _DEFAULT_MODEL_MB)
            ),
        }

    def _get_bitnet_option(self, extra_config: dict[str, Any], key: str, default: Any) -> Any:
        bitnet_config = extra_config.get("bitnet")
        if isinstance(bitnet_config, dict) and key in bitnet_config:
            return bitnet_config[key]
        return extra_config.get(key, default)

    def _resolve_model_file(self, model_id: str, extra_config: dict[str, Any] | None = None) -> Path:
        model_path = (extra_config or {}).get("model_path") if isinstance(extra_config, dict) else None
        if model_path:
            explicit = Path(str(model_path))
            if explicit.is_file():
                return explicit

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

        # Fallback: find by stem for scanners that register `model_ref=path.stem`.
        stem = Path(model_id).stem
        matches = [p for p in root.rglob("*.gguf") if p.stem == stem]
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)

        raise FileNotFoundError(f"BitNet GGUF model not found for '{model_id}' under {root}")

    async def _wait_for_startup(self, model_id: str, port: int, timeout_s: int) -> None:
        deadline = asyncio.get_event_loop().time() + timeout_s
        url = f"http://127.0.0.1:{port}/health"

        while asyncio.get_event_loop().time() < deadline:
            entry = self._processes.get(model_id)
            if entry and entry[0].returncode is not None:
                rc = entry[0].returncode
                stderr_tail = await self._read_stderr_tail(entry[0])
                suffix = f" stderr={stderr_tail}" if stderr_tail else ""
                if rc == -signal.SIGKILL or rc == 137:
                    raise MemoryError(f"BitNet process for '{model_id}' OOM-killed (rc={rc}).{suffix}")
                raise RuntimeError(f"BitNet process for '{model_id}' exited with rc={rc}.{suffix}")

            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)

        raise TimeoutError(f"BitNet worker for '{model_id}' did not become healthy within {timeout_s}s")

    async def _kill_process(self, model_id: str, graceful: bool = False) -> None:
        entry = self._processes.pop(model_id, None)
        if not entry:
            return
        proc, _ = entry
        if proc.returncode is not None:
            return

        if graceful:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=float(_SHUTDOWN_TIMEOUT_S))
                return
            except (TimeoutError, ProcessLookupError):
                logger.warning("bitnet_sigterm_timeout", model_id=model_id)

        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass

    async def _read_stderr_tail(self, proc: asyncio.subprocess.Process, limit: int = 3000) -> str:
        if proc.stderr is None:
            return ""
        try:
            data = await asyncio.wait_for(proc.stderr.read(), timeout=0.3)
        except Exception:
            return ""
        if not data:
            return ""
        text = data.decode("utf-8", errors="replace").strip()
        if len(text) <= limit:
            return text
        return text[-limit:]

    def _to_int_or_none(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return int(value)
