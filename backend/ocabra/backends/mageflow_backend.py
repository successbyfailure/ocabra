"""Backend for Microsoft Mage-Flow image generation / editing.

Mage-Flow uses a custom ``MageFlowPipeline`` (not a native diffusers pipeline)
and a distinct dependency stack (torch 2.13 / transformers 5.5 / diffusers
0.38), so it cannot share the diffusers backend's venv or worker. This backend
spawns a dedicated ``mageflow_worker.py`` subprocess running in
``/data/backends/mage/venv`` and speaks the same worker HTTP contract as the
diffusers worker (``/generate`` / ``/edit`` / ``/health``), so the OpenAI image
endpoints route to it unchanged.

The worker runs with SDPA attention (no flash-attn compile) — see
``mageflow_worker.py`` for details.
"""

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
from ocabra.core.backend_installer import venv_nvidia_ld_library_path

logger = structlog.get_logger(__name__)


class MageFlowBackend(BackendInterface):

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.IMAGE_GENERATION}

    @property
    def install_spec(self) -> BackendInstallSpec:
        # NOTE: Mage-Flow's venv is provisioned out-of-band (git clone of
        # microsoft/Mage + ``pip install -e mage_flow`` + an ``attn_type``
        # default patch to "sdpa"), which the pip-only installer cannot
        # express. These packages document the base stack; the editable
        # ``mage_flow`` package and the sdpa patch are applied by the provision
        # script. torch is pinned to the cu126 wheel so flash-attn (if ever
        # added) builds against a CUDA major matching the host toolkit.
        return BackendInstallSpec(
            oci_image="",
            oci_tags={},
            pip_packages=[
                "torch==2.13.0",
                "torchvision==0.28.0",
                "diffusers==0.38.0",
                "transformers==5.5.0",
                "accelerate==1.13.0",
                "safetensors==0.8.0",
                "einops==0.8.2",
                "pillow==12.3.0",
                "loguru==0.7.3",
                "numpy",
            ],
            pip_extra_index_urls=[
                "https://download.pytorch.org/whl/cu126",
            ],
            estimated_size_mb=20000,
            display_name="Mage-Flow (Microsoft)",
            description=(
                "Native-resolution 4B image generation & editing via Microsoft "
                "Mage-Flow (custom MageFlowPipeline, SDPA attention)"
            ),
            tags=["Image", "GPU", "CUDA"],
        )

    def __init__(self) -> None:
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._workers: dict[str, WorkerInfo] = {}

    def _resolve_python_bin(self) -> str:
        """Return the python interpreter that launches the Mage-Flow worker."""
        try:
            meta_path = Path(settings.backends_dir) / "mage" / "metadata.json"
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
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")
        gpu_index = gpu_indices[0]

        worker_script = self._worker_script_path()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # Turbo checkpoints denoise in 4 steps / cfg 1.0; the worker uses these
        # when the caller doesn't override (see mageflow_worker._resolve_*).
        env.setdefault("MAGE_DEFAULT_STEPS", "4")
        env.setdefault("MAGE_DEFAULT_CFG", "1.0")

        nvidia_ld = venv_nvidia_ld_library_path(settings.backends_dir, "mage")
        if nvidia_ld:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{nvidia_ld}:{existing}" if existing else nvidia_ld

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
            stderr=open(f"/tmp/mageflow_worker_{model_id.replace('/', '_')}.log", "wb"),
            env=env,
        )

        try:
            # Mage-Flow load is heavy (~70s: 4B DiT + Qwen3-VL text encoder + VAE).
            await self._wait_until_healthy(port=port, process=process, timeout_s=300)
        except Exception:
            await self._terminate_process(process)
            raise

        vram_estimate = await self.get_vram_estimate_mb(model_id)
        info = WorkerInfo(
            backend_type="mage",
            model_id=model_id,
            gpu_indices=[gpu_index],
            port=port,
            pid=process.pid,
            vram_used_mb=vram_estimate,
        )
        self._processes[model_id] = process
        self._workers[model_id] = info
        logger.info(
            "mageflow_worker_loaded",
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
        logger.info("mageflow_worker_unloaded", model_id=model_id)

    async def health_check(self, model_id: str) -> bool:
        worker = self._workers.get(model_id)
        if worker is None:
            return False
        return await self._health_check_port(worker.port)

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(image_generation=True, streaming=False)

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        model_path = Path(settings.models_dir) / model_id
        if not model_path.exists():
            return 0
        # Mage-Flow keeps the DiT, the Qwen3-VL text encoder AND the VAE
        # resident on-GPU (no CPU offload in the custom pipeline), so — unlike
        # the diffusers estimate — we sum every weight file. Measured resident
        # ~17.6 GB / peak ~18.3 GB for the 4B Turbo checkpoint.
        candidates = [p for p in model_path.rglob("*.safetensors") if p.is_file()]
        total_bytes = sum(p.stat().st_size for p in candidates)
        total_mb = total_bytes / (1024 * 1024)
        return int(math.ceil(total_mb * 1.1))

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if worker is None:
            raise KeyError(f"No worker found for model '{model_id}'")
        request_path = "/edit" if path.rstrip("/").endswith("edit") else "/generate"
        url = f"http://127.0.0.1:{worker.port}{request_path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=body)
            response.raise_for_status()
            return response.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        raise RuntimeError("Mage-Flow backend does not support streaming")

    async def _wait_until_healthy(
        self, *, port: int, process: asyncio.subprocess.Process, timeout_s: int
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"Mage-Flow worker exited with code {process.returncode}")
            if await self._health_check_port(port):
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"Mage-Flow worker health check timed out after {timeout_s}s")

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

    def _worker_script_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        worker_script = repo_root / "workers" / "mageflow_worker.py"
        if not worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")
        return worker_script
