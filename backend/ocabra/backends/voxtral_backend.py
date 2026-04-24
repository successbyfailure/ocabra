"""Voxtral TTS backend — launches voxtral_worker.py which wraps vllm-omni."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
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

logger = structlog.get_logger(__name__)

WORKER_PATH = Path(__file__).resolve().parents[2] / "workers" / "voxtral_worker.py"

# Voxtral 4B BF16 ≈ 8 GB VRAM
KNOWN_VRAM_MB: dict[str, int] = {
    "voxtral-4b": 8192,
    "voxtral": 8192,
}


@dataclass
class _VoxtralWorker:
    process: asyncio.subprocess.Process
    info: WorkerInfo
    log_file: Any | None = None


class VoxtralBackend(BackendInterface):

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.AUDIO_SPEECH}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Declarative install spec for voxtral (Bloque 15 Fase 2).

        Voxtral wraps the vllm-omni stack, so the venv pulls vllm + vllm-omni.
        """

        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-voxtral",
            oci_tags={"cuda12": "latest-cuda12"},
            pip_packages=[
                "vllm>=0.18.0",
                "vllm-omni>=0.18.0",
            ],
            pip_extra_index_urls=[
                "https://download.pytorch.org/whl/cu124",
            ],
            estimated_size_mb=10000,
            display_name="Voxtral (vllm-omni)",
            description=(
                "Multimodal speech backend wrapping vllm-omni for Voxtral 4B "
                "and similar audio LLMs"
            ),
            tags=["TTS", "GPU", "CUDA"],
        )

    def __init__(self) -> None:
        self._workers: dict[str, _VoxtralWorker] = {}

    def _resolve_python_bin(self) -> str:
        """Pick the python that vllm-omni runs inside.

        Priority: ``<backends_dir>/voxtral/metadata.json`` > legacy
        ``settings.voxtral_python_bin`` > :data:`sys.executable`.
        """

        try:
            meta_path = Path(settings.backends_dir) / "voxtral" / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                bin_path = meta.get("python_bin") if isinstance(meta, dict) else None
                if bin_path and Path(bin_path).is_file():
                    return str(bin_path)
        except (OSError, json.JSONDecodeError):
            pass

        legacy = settings.voxtral_python_bin
        if legacy and Path(legacy).is_file():
            return str(legacy)
        return sys.executable

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs: Any) -> WorkerInfo:
        existing = self._workers.get(model_id)
        if existing and existing.process.returncode is None:
            return existing.info

        if not WORKER_PATH.exists():
            raise FileNotFoundError(f"voxtral_worker.py not found at '{WORKER_PATH}'")

        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        python_bin = self._resolve_python_bin()
        if python_bin == sys.executable and not (
            settings.voxtral_python_bin and Path(settings.voxtral_python_bin).is_file()
        ):
            logger.warning(
                "voxtral_python_bin_missing",
                configured=settings.voxtral_python_bin,
                fallback=python_bin,
            )

        env = os.environ.copy()
        env.update(kwargs.get("env", {}))
        env["PYTHONUNBUFFERED"] = "1"
        env["VOXTRAL_STARTUP_TIMEOUT_S"] = str(settings.voxtral_startup_timeout_s)
        if python_bin != sys.executable:
            env["VOXTRAL_PYTHON_BIN"] = python_bin
        if gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
        if settings.hf_token:
            env.setdefault("HF_TOKEN", settings.hf_token)

        log_path = _worker_log_path(model_id)
        log_file = open(log_path, "ab")  # noqa: SIM115

        process = await asyncio.create_subprocess_exec(
            sys.executable,  # outer worker always uses the system python
            str(WORKER_PATH),
            "--model-id", model_id,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--gpu-indices", ",".join(str(i) for i in gpu_indices),
            env=env,
            stdout=log_file,
            stderr=log_file,
        )

        info = WorkerInfo(
            backend_type="voxtral",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid,
            vram_used_mb=await self.get_vram_estimate_mb(model_id),
        )

        self._workers[model_id] = _VoxtralWorker(process=process, info=info, log_file=log_file)

        timeout_s = max(1, settings.voxtral_startup_timeout_s)
        if not await self._wait_until_healthy(model_id, timeout_s=timeout_s):
            await self.unload(model_id)
            raise RuntimeError(
                f"Voxtral worker failed to start for model '{model_id}' within {timeout_s}s. "
                f"Check logs at {log_path}"
            )

        logger.info(
            "voxtral_worker_started",
            model_id=model_id,
            port=port,
            pid=process.pid,
        )
        return info

    async def unload(self, model_id: str) -> None:
        worker = self._workers.pop(model_id, None)
        if not worker:
            return
        process = worker.process
        try:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=15)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        finally:
            if worker.log_file:
                try:
                    worker.log_file.close()
                except Exception:
                    pass

    async def health_check(self, model_id: str) -> bool:
        worker = self._workers.get(model_id)
        if not worker or worker.process.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"http://127.0.0.1:{worker.info.port}/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(tts=True)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        normalized = model_id.lower()
        for key, mb in KNOWN_VRAM_MB.items():
            if key in normalized:
                return mb
        return 8192

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"Voxtral worker for '{model_id}' is not loaded")
        endpoint = path if path.startswith("/") else f"/{path}"
        if endpoint == "/voices":
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"http://127.0.0.1:{worker.info.port}/voices")
            r.raise_for_status()
            return r.json()

        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                f"http://127.0.0.1:{worker.info.port}{endpoint}", json=body,
            )
        r.raise_for_status()
        return {
            "content": r.content,
            "content_type": r.headers.get("content-type", "audio/mpeg"),
        }

    async def forward_stream(
        self, model_id: str, path: str, body: dict,
    ) -> AsyncIterator[bytes]:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"Voxtral worker for '{model_id}' is not loaded")
        endpoint = path if path.startswith("/") else f"/{path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{worker.info.port}{endpoint}",
                json=body,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk

    async def _wait_until_healthy(self, model_id: str, timeout_s: int = 300) -> bool:
        attempts = max(1, timeout_s * 2)
        for _ in range(attempts):
            if await self.health_check(model_id):
                return True
            # Also check if the process died
            worker = self._workers.get(model_id)
            if worker and worker.process.returncode is not None:
                return False
            await asyncio.sleep(0.5)
        return False


def _worker_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/voxtral-worker-{safe}.log"
