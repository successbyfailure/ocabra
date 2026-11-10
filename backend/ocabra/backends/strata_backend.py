"""Lifecycle adapter for Strata Terminal's OpenAI-compatible server."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.backends.base import BackendCapabilities, BackendInterface, ModalityType, WorkerInfo

logger = structlog.get_logger(__name__)
_STARTUP_TIMEOUT_S = 300
_REQUEST_TIMEOUT_S = 900
_SHUTDOWN_TIMEOUT_S = 30


@dataclass
class _Worker:
    process: asyncio.subprocess.Process
    port: int
    log_file: Any
    log_path: Path
    temp_config_path: Path
    api_key: str
    served_model: str
    config: dict[str, Any]


def _arg_int(args: list[Any], flag: str) -> int:
    values = [str(value) for value in args]
    try:
        return int(values[values.index(flag) + 1])
    except (ValueError, IndexError):
        return 0


class StrataBackend(BackendInterface):
    """Run a prepared Strata config privately under oCabra's lifecycle."""

    def __init__(self) -> None:
        self._processes: dict[str, _Worker] = {}

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.TEXT_GENERATION}

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs: Any) -> WorkerInfo:
        port = int(kwargs.get("port") or 0)
        if not port:
            raise ValueError("load() requires a port")
        if model_id in self._processes:
            raise RuntimeError(f"Strata worker for {model_id!r} is already loaded")

        extra = kwargs.get("extra_config") or {}
        options = self._options(extra)
        config_path = Path(str(options.get("config_path") or ""))
        if not config_path.is_absolute() or not config_path.is_file():  # noqa: ASYNC240
            raise FileNotFoundError(
                "Strata requires extra_config.strata.config_path pointing to an absolute run config"
            )
        try:
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))  # noqa: ASYNC240
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Strata run config {config_path}: {exc}") from exc
        if not isinstance(config, dict) or not isinstance(config.get("args"), list):
            raise ValueError(f"Invalid Strata run config {config_path}: missing args list")

        configured_gpus = config.get("gpu")
        if isinstance(configured_gpus, int):
            configured_gpus = [configured_gpus]
        if configured_gpus and [int(value) for value in configured_gpus] != gpu_indices:
            raise ValueError(
                f"Strata config targets GPUs {configured_gpus}, but oCabra admitted {gpu_indices}"
            )
        self._check_available_ram(int(options.get("min_available_ram_mb") or 0))

        runtime_dir = Path(str(options.get("runtime_dir") or config.get("cwd") or ""))
        python_bin = Path(
            str(options.get("python_bin") or runtime_dir / "python" / "bin" / "python3.12")
        )
        server_path = Path(str(options.get("server_path") or runtime_dir / "serve" / "server.py"))
        for label, path in (
            ("runtime_dir", runtime_dir),
            ("python_bin", python_bin),
            ("server_path", server_path),
        ):
            if not path.exists():  # noqa: ASYNC240
                raise FileNotFoundError(f"Strata {label} does not exist: {path}")

        key = secrets.token_urlsafe(32)
        safe_name = model_id.replace("/", "__").replace(":", "_")
        log_path = Path(tempfile.gettempdir()) / f"strata-worker-{safe_name}.log"
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="ocabra-strata-", suffix=".json", delete=False
        )
        temp_path = Path(handle.name)
        private_config = {
            **config,
            "host": "127.0.0.1",
            "port": port,
            "api_key": key,
            "open_browser": False,
            "log": str(log_path),
            "ocabra": options,
        }
        try:
            json.dump(private_config, handle, indent=2)
        finally:
            handle.close()

        env = dict(os.environ)
        runtime_root = runtime_dir.parent.parent
        env["XDG_DATA_HOME"] = str(options.get("xdg_data_home") or runtime_root / "xdg-data")
        env["XDG_STATE_HOME"] = str(options.get("xdg_state_home") or runtime_root / "xdg-state")
        cmd = [
            str(python_bin),
            str(server_path),
            "--engine",
            "strata",
            "--config",
            str(temp_path),
            "--port",
            str(port),
        ]
        log_file = open(log_path, "ab")  # noqa: ASYNC230, SIM115 - worker lifetime
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(runtime_dir),
                env=env,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        except BaseException:
            log_file.close()
            temp_path.unlink(missing_ok=True)  # noqa: ASYNC240
            raise

        worker = _Worker(
            process=process,
            port=port,
            log_file=log_file,
            log_path=log_path,
            temp_config_path=temp_path,
            api_key=key,
            served_model=str(config.get("model_name") or model_id),
            config=private_config,
        )
        self._processes[model_id] = worker
        logger.info("strata_starting", model_id=model_id, port=port, gpu_indices=gpu_indices)
        try:
            await self._wait_for_startup(
                model_id, int(options.get("startup_timeout_s") or _STARTUP_TIMEOUT_S)
            )
        except BaseException:
            await self._kill_process(model_id)
            raise

        return WorkerInfo(
            backend_type="strata",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid or 0,
            vram_used_mb=await self.get_vram_estimate_mb(model_id, extra),
        )

    async def unload(self, model_id: str) -> None:
        await self._kill_process(model_id, graceful=True)

    async def health_check(self, model_id: str) -> bool:
        worker = self._processes.get(model_id)
        if not worker or worker.process.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"http://127.0.0.1:{worker.port}/health")
            return response.status_code == 200 and bool(response.json().get("loaded"))
        except Exception:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        worker = self._processes.get(model_id)
        config = worker.config if worker else {}
        args = config.get("args") if isinstance(config.get("args"), list) else []
        options = config.get("ocabra") if isinstance(config.get("ocabra"), dict) else {}
        context = _arg_int(args, "--max-context")
        parallel = max(1, int(config.get("parallel") or 1))
        vision = str(config.get("vision") or "").lower() not in {"", "no", "none", "false"}
        return BackendCapabilities(
            chat=True,
            tools=True,
            vision=vision,
            reasoning=True,
            streaming=True,
            context_length=context,
            native_context_length=int(options.get("native_context_length") or 262_144),
            parallel_slots=parallel,
            kv_cache_shared=False,
            context_per_request=context,
        )

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        extra = extra_config or {}
        runtime = extra.get("_runtime") if isinstance(extra.get("_runtime"), dict) else {}
        per_gpu = runtime.get("capacity_vram_by_gpu_mb")
        if isinstance(per_gpu, dict) and per_gpu:
            return sum(max(0, int(value)) for value in per_gpu.values())
        return max(
            2048,
            int(runtime.get("capacity_vram_mb") or self._options(extra).get("vram_total_mb") or 0),
        )

    def request_headers(self, model_id: str) -> dict[str, str]:
        worker = self._worker(model_id)
        return {"Authorization": f"Bearer {worker.api_key}"}

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._worker(model_id)
        payload = {**body, "model": worker.served_model}
        timeout = float(worker.config["ocabra"].get("request_timeout_s") or _REQUEST_TIMEOUT_S)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"http://127.0.0.1:{worker.port}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {worker.api_key}"},
            )
            response.raise_for_status()
            return response.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        worker = self._worker(model_id)
        payload = {**body, "model": worker.served_model}
        timeout = float(worker.config["ocabra"].get("request_timeout_s") or _REQUEST_TIMEOUT_S)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{worker.port}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {worker.api_key}"},
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    @staticmethod
    def _options(extra: dict[str, Any]) -> dict[str, Any]:
        value = extra.get("strata")
        return dict(value) if isinstance(value, dict) else {}

    def _worker(self, model_id: str) -> _Worker:
        worker = self._processes.get(model_id)
        if not worker:
            raise KeyError(f"No Strata worker for model {model_id!r}")
        return worker

    @staticmethod
    def _check_available_ram(minimum_mb: int) -> None:
        if minimum_mb <= 0:
            return
        try:
            line = next(
                line
                for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
                if line.startswith("MemAvailable:")
            )
            available_mb = int(line.split()[1]) // 1024
        except (OSError, StopIteration, ValueError):
            return
        if available_mb < minimum_mb:
            raise MemoryError(
                f"Strata needs {minimum_mb} MiB available RAM; only {available_mb} MiB is available"
            )

    async def _wait_for_startup(self, model_id: str, timeout_s: int) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            worker = self._processes.get(model_id)
            if worker and worker.process.returncode is not None:
                raise RuntimeError(
                    f"Strata worker exited rc={worker.process.returncode}; "
                    f"log_tail={self._log_tail(worker)}"
                )
            if await self.health_check(model_id):
                return
            await asyncio.sleep(1)
        worker = self._processes.get(model_id)
        raise TimeoutError(
            f"Strata worker did not load within {timeout_s}s; "
            f"log_tail={self._log_tail(worker) if worker else ''}"
        )

    async def _kill_process(self, model_id: str, graceful: bool = False) -> None:
        worker = self._processes.pop(model_id, None)
        if not worker:
            return
        process = worker.process
        try:
            if process.returncode is None and graceful:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                    await asyncio.wait_for(process.wait(), timeout=_SHUTDOWN_TIMEOUT_S)
                except (TimeoutError, ProcessLookupError):
                    if process.returncode is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5)
                        except TimeoutError:
                            os.killpg(process.pid, signal.SIGKILL)
                            await process.wait()
            elif process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        finally:
            worker.log_file.close()
            worker.temp_config_path.unlink(missing_ok=True)  # noqa: ASYNC240

    @staticmethod
    def _log_tail(worker: _Worker, limit: int = 4000) -> str:
        try:
            with worker.log_path.open("rb") as handle:
                handle.seek(max(0, worker.log_path.stat().st_size - limit))
                return handle.read().decode("utf-8", errors="replace").strip()[-limit:]
        except OSError:
            return ""
