import asyncio
import json
import math
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

logger = structlog.get_logger(__name__)


_SIZE_MAP: dict[str, tuple[int, int]] = {
    "256x256": (256, 256),
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    "1792x1024": (1792, 1024),
    "1024x1792": (1024, 1792),
}


class DiffusersBackend(BackendInterface):

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.IMAGE_GENERATION}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Declarative install spec for the diffusers backend (Bloque 15 Fase 2)."""

        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-diffusers",
            oci_tags={"cuda12": "latest-cuda12"},
            pip_packages=[
                "torch>=2.5",
                "diffusers>=0.31",
                "accelerate>=1.2",
                "transformers>=4.47",
                "Pillow>=11.0",
                "safetensors>=0.4",
                "numpy",
            ],
            pip_extra_index_urls=[
                "https://download.pytorch.org/whl/cu124",
            ],
            estimated_size_mb=6000,
            display_name="Diffusers (Stable Diffusion / SDXL / FLUX)",
            description=(
                "Image generation backend for Stable Diffusion, SDXL and "
                "FLUX models via Hugging Face diffusers"
            ),
            tags=["Image", "GPU", "CUDA"],
        )

    def __init__(self) -> None:
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._workers: dict[str, WorkerInfo] = {}

    def _resolve_python_bin(self) -> str:
        """Return the python interpreter that should launch the diffusers worker."""

        try:
            meta_path = Path(settings.backends_dir) / "diffusers" / "metadata.json"
            if not meta_path.exists():
                return sys.executable
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return sys.executable
        bin_path = meta.get("python_bin") if isinstance(meta, dict) else None
        return str(bin_path) if bin_path else sys.executable

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        if not gpu_indices:
            raise ValueError("At least one GPU index is required")

        model_path = Path(settings.models_dir) / model_id
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        port = int(kwargs.get("port") or 0)
        if port == 0:
            port = await self._assign_port()
        gpu_index = gpu_indices[0]

        worker_script = self._worker_script_path()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["DIFFUSERS_TORCH_DTYPE"] = settings.diffusers_torch_dtype
        env["DIFFUSERS_ENABLE_TORCH_COMPILE"] = str(settings.diffusers_enable_torch_compile).lower()
        env["DIFFUSERS_ENABLE_XFORMERS"] = str(settings.diffusers_enable_xformers).lower()
        env["DIFFUSERS_OFFLOAD_MODE"] = settings.diffusers_offload_mode
        env["DIFFUSERS_ALLOW_TF32"] = str(settings.diffusers_allow_tf32).lower()

        cmd = [
            self._resolve_python_bin(),
            str(worker_script),
            "--model-id",
            model_id,
            "--model-path",
            str(model_path),
            "--port",
            str(port),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )

        try:
            await self._wait_until_healthy(port=port, process=process, timeout_s=180)
        except Exception:
            await self._terminate_process(process)
            raise

        vram_estimate = await self.get_vram_estimate_mb(model_id)
        info = WorkerInfo(
            backend_type="diffusers",
            model_id=model_id,
            gpu_indices=[gpu_index],
            port=port,
            pid=process.pid,
            vram_used_mb=vram_estimate,
        )

        self._processes[model_id] = process
        self._workers[model_id] = info

        logger.info(
            "diffusers_worker_loaded",
            model_id=model_id,
            gpu_index=gpu_index,
            port=port,
            pid=process.pid,
        )
        return info

    async def unload(self, model_id: str) -> None:
        process = self._processes.pop(model_id, None)
        self._workers.pop(model_id, None)

        if process is None:
            return

        await self._terminate_process(process)
        logger.info("diffusers_worker_unloaded", model_id=model_id)

    async def health_check(self, model_id: str) -> bool:
        worker = self._workers.get(model_id)
        if worker is None:
            return False
        return await self._health_check_port(worker.port)

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(image_generation=True, streaming=False)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        model_path = Path(settings.models_dir) / model_id
        if not model_path.exists():
            return 0

        total_bytes = 0
        for tensor_file in model_path.rglob("*.safetensors"):
            if tensor_file.is_file():
                total_bytes += tensor_file.stat().st_size

        total_mb = total_bytes / (1024 * 1024)
        return int(math.ceil(total_mb * 1.3))

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if worker is None:
            raise KeyError(f"No worker found for model '{model_id}'")

        if path.endswith("/images/generations"):
            payload = self._translate_openai_image_request(body)
            request_path = "/generate"
        else:
            payload = body
            request_path = path

        url = f"http://127.0.0.1:{worker.port}{request_path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            worker_payload = response.json()

        if request_path == "/generate":
            return {
                "created": int(time.time()),
                "data": [
                    {"b64_json": image["b64_json"]}
                    for image in worker_payload.get("images", [])
                    if "b64_json" in image
                ],
            }

        return worker_payload

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        raise RuntimeError("Diffusers backend does not support streaming")

    async def _wait_until_healthy(
        self,
        *,
        port: int,
        process: asyncio.subprocess.Process,
        timeout_s: int,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"Diffusers worker exited with code {process.returncode}")
            if await self._health_check_port(port):
                return
            await asyncio.sleep(1)

        raise TimeoutError(f"Diffusers worker health check timed out after {timeout_s}s")

    async def _health_check_port(self, port: int) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://127.0.0.1:{port}/health")
                if response.status_code != 200:
                    return False
                data = response.json()
                return bool(data.get("ok", False))
        except Exception:
            return False

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=30)
            return
        except TimeoutError:
            process.kill()
            await process.wait()

    async def _assign_port(self) -> int:
        raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

    def _worker_script_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        worker_script = repo_root / "workers" / "diffusers_worker.py"
        if not worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")
        return worker_script

    def _translate_openai_image_request(self, body: dict) -> dict:
        size = body.get("size", "1024x1024")
        if size not in _SIZE_MAP:
            allowed = ", ".join(sorted(_SIZE_MAP.keys()))
            raise ValueError(f"Unsupported image size '{size}'. Allowed values: {allowed}")

        width, height = _SIZE_MAP[size]
        payload: dict[str, Any] = {
            "prompt": body["prompt"],
            "width": width,
            "height": height,
            "num_images": int(body.get("n", 1)),
        }

        optional_fields = [
            "negative_prompt",
            "num_inference_steps",
            "guidance_scale",
            "seed",
        ]
        for field in optional_fields:
            if field in body and body[field] is not None:
                payload[field] = body[field]

        return payload
