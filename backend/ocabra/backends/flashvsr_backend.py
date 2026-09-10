"""Backend de escalado de vídeo con FlashVSR v1.1 (difusión de un paso).

Nivel de máxima calidad. Medido en una RTX 3090 con salida 1080p: 3,23 fps y
19,1 GB de VRAM, es decir ~9x el tiempo real. Ocupa la GPU casi entera, así
que va detrás de una cola y conviene dejarlo en ``on_demand``.

Para vídeos completos el nivel rápido (``upscaler_backend``, Real-ESRGAN
Compact) es ~28x más veloz con 51 MiB; este solo se justifica cuando la
calidad manda y el material es corto.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
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

BACKEND_TYPE = "flashvsr"

# Medido sobre una RTX 3090 con salida 1080p: 19,1 GB con un segmento de 2 s y
# 19,2 GB con uno de 10 s. La VRAM la fija la resolución de salida, no la
# duración, así que un valor fijo es correcto aquí. Se reserva algo de margen
# para el pico del decodificador de la VAE.
_VRAM_ESTIMATE_MB = 20480


class FlashVSRBackend(BackendInterface):
    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.VIDEO_UPSCALING}

    @property
    def install_spec(self) -> BackendInstallSpec:
        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-flashvsr",
            oci_tags={"cuda12": "latest-cuda12"},
            # torch 2.6 + cu124 es la combinación con la que se validó la
            # compilación de Block-Sparse-Attention en sm_86.
            pip_packages=[
                "torch==2.6.0",
                "torchvision==0.21.0",
                # diffsynth lo exige con la etiqueta +cu124 exacta.
                "torchaudio==2.6.0",
                "numpy<2",
            ],
            pip_extra_index_urls=[
                "https://download.pytorch.org/whl/cu124",
            ],
            # nvcc hace falta de verdad: BSA no publica ruedas precompiladas.
            apt_packages=["ffmpeg", "git", "build-essential", "ninja-build"],
            git_repo="https://github.com/OpenImagingLab/FlashVSR",
            git_ref="main",
            post_install_script="backend/scripts/install_flashvsr.sh",
            estimated_size_mb=22000,
            display_name="FlashVSR (difusión, máxima calidad)",
            description=(
                "Escalado de vídeo 4x por difusión de un paso. Máxima calidad "
                "visual, ~9x el tiempo real y 19 GB de VRAM: solo para clips "
                "cortos y siempre en cola."
            ),
            tags=["Video", "GPU", "CUDA", "Slow"],
        )

    def __init__(self) -> None:
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._workers: dict[str, WorkerInfo] = {}

    def _resolve_python_bin(self) -> str:
        try:
            meta_path = Path(settings.backends_dir) / BACKEND_TYPE / "metadata.json"
            if not meta_path.exists():
                return sys.executable
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return sys.executable
        bin_path = meta.get("python_bin") if isinstance(meta, dict) else None
        return str(bin_path) if bin_path else sys.executable

    def _worker_script_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "workers" / "flashvsr_worker.py"
        if not script.exists():
            raise FileNotFoundError(f"Worker script not found: {script}")
        return script

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        if not gpu_indices:
            raise ValueError("At least one GPU index is required")

        model_path = Path(settings.models_dir) / model_id
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")
        gpu_index = gpu_indices[0]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        nvidia_ld = venv_nvidia_ld_library_path(settings.backends_dir, BACKEND_TYPE)
        if nvidia_ld:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{nvidia_ld}:{existing}" if existing else nvidia_ld

        cmd = [
            self._resolve_python_bin(),
            str(self._worker_script_path()),
            "--model-id", model_id,
            "--model-path", str(model_path),
            "--src-dir", str(Path(settings.backends_dir) / BACKEND_TYPE / "src"),
            "--port", str(port),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=open(f"/tmp/flashvsr_worker_{model_id.replace('/', '_')}.log", "wb"),
            env=env,
        )

        try:
            await self._wait_until_healthy(port=port, process=process, timeout_s=600)
        except Exception:
            await self._terminate_process(process)
            raise

        info = WorkerInfo(
            backend_type=BACKEND_TYPE,
            model_id=model_id,
            gpu_indices=[gpu_index],
            port=port,
            pid=process.pid,
            vram_used_mb=_VRAM_ESTIMATE_MB,
        )
        self._processes[model_id] = process
        self._workers[model_id] = info
        logger.info(
            "flashvsr_worker_loaded",
            model_id=model_id, gpu_index=gpu_index, port=port, pid=process.pid,
        )
        return info

    async def unload(self, model_id: str) -> None:
        process = self._processes.pop(model_id, None)
        self._workers.pop(model_id, None)
        if process is None:
            return
        await self._terminate_process(process)
        logger.info("flashvsr_worker_unloaded", model_id=model_id)

    async def health_check(self, model_id: str) -> bool:
        worker = self._workers.get(model_id)
        if worker is None:
            return False
        return await self._health_check_port(worker.port)

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(video_upscaling=True, streaming=False)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        return _VRAM_ESTIMATE_MB

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if worker is None:
            raise KeyError(f"No worker found for model '{model_id}'")
        url = f"http://127.0.0.1:{worker.port}{path}"
        async with httpx.AsyncClient(timeout=1800.0) as client:
            response = await client.post(url, json=body)
            response.raise_for_status()
            return response.json()

    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        raise RuntimeError("FlashVSR backend does not support streaming")

    async def upscale_video(self, model_id: str, video: bytes, **kwargs: Any) -> bytes:
        worker = self._workers.get(model_id)
        if worker is None:
            raise KeyError(f"No worker found for model '{model_id}'")
        params = {k: str(v) for k, v in kwargs.items() if v is not None}
        url = f"http://127.0.0.1:{worker.port}/upscale"
        # Sin límite corto de tiempo: un segmento largo puede tardar minutos y
        # abortarlo a mitad tira todo el trabajo de GPU ya hecho.
        async with httpx.AsyncClient(timeout=3600.0) as client:
            response = await client.post(
                url, params=params, files={"file": ("segment.mp4", video, "video/mp4")}
            )
            response.raise_for_status()
            return response.content

    async def _wait_until_healthy(
        self, *, port: int, process: asyncio.subprocess.Process, timeout_s: int
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"FlashVSR worker exited with code {process.returncode}")
            if await self._health_check_port(port):
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"FlashVSR worker health check timed out after {timeout_s}s")

    async def _health_check_port(self, port: int) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://127.0.0.1:{port}/health")
                if response.status_code != 200:
                    return False
                return bool(response.json().get("ok", False))
        except Exception:
            return False

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=30)
        except TimeoutError:
            process.kill()
            await process.wait()
