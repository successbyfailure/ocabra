"""Backend de generación de vídeo sobre un ComfyUI sin interfaz (LTX-2.5).

Cada modelo es un workflow de ComfyUI en formato API más la carpeta de pesos
``<MODELS_DIR>/video`` (estructura del repo de LTX-2.5 en Hugging Face). El
worker (``workers/comfyui_video_worker.py``) arranca su propio ComfyUI, que
oCabra carga, descarga y expulsa como cualquier otro modelo; no tiene nada que
ver con el servicio interactivo ``comfyui``.

Se usa ComfyUI y no ``ltx-pipelines``/Diffusers porque los pesos int8
«convrot» (37 GB entre DiT y codificador) solo los entiende ComfyUI; el camino
de PyTorch necesita los bf16 (68 GB). Medido en una RTX 3090: 1280x704 y 5 s
con audio en 46 s en caliente.

``extra_config.comfyui_video``:
  model_dir   carpeta relativa a MODELS_DIR (por defecto ``video``)
  workflow    nombre del JSON en ``workers/comfyui_video_workflows`` (``ltx25_t2v``)
  comfy_args  opciones extra para ComfyUI (p. ej. ``["--lowvram"]``)
  max_seconds tope de duración por petición (por defecto 10)
  vram_mb     VRAM reservada (por defecto 23500: la 3090 entera)
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

from ocabra.backends._health_http import health_client
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

BACKEND_TYPE = "comfyui_video"
CONFIG_SECTION = "comfyui_video"

# Pico medido con 1280x704 y 5 s: 23,7 GB. ComfyUI reparte la VRAM que
# encuentra (carga dinámica), así que se reserva la tarjeta entera.
DEFAULT_VRAM_MB = 23500
DEFAULT_WORKFLOW = "ltx25_t2v"
DEFAULT_MODEL_DIR = "video"
DEFAULT_MAX_SECONDS = 10
# ComfyUI pin: mismo commit que la imagen del servicio interactivo, el único
# validado con LTX-2.5 (nodos LTXV y kernels int8).
COMFYUI_REF = "995456d0188ffb51032c28f41527cfa82d87d8a9"


def workflows_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "workers" / "comfyui_video_workflows"


def section(extra_config: dict | None) -> dict[str, Any]:
    value = (extra_config or {}).get(CONFIG_SECTION)
    return value if isinstance(value, dict) else {}


class ComfyUIVideoBackend(BackendInterface):
    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.VIDEO_GENERATION}

    @property
    def install_spec(self) -> BackendInstallSpec:
        return BackendInstallSpec(
            # ComfyUI pide torch reciente con cu130 en las series 20+; el
            # driver del hub (615) da CUDA 13.
            pip_packages=["torch", "torchvision", "torchaudio"],
            pip_extra_index_urls=["https://download.pytorch.org/whl/cu130"],
            # gcc: Triton compila kernels al codificar el texto de LTX-2.5.
            apt_packages=["ffmpeg", "gcc", "libc6-dev", "git"],
            git_repo="https://github.com/Comfy-Org/ComfyUI.git",
            git_ref=COMFYUI_REF,
            post_install_script="backend/scripts/install_comfyui_video.sh",
            estimated_size_mb=9000,
            display_name="ComfyUI (vídeo)",
            description=(
                "Generación de vídeo con audio (LTX-2.5) sobre un ComfyUI sin "
                "interfaz gestionado por oCabra. Ocupa una GPU de 24 GB entera."
            ),
            tags=["Video", "GPU", "CUDA"],
        )

    def __init__(self) -> None:
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._workers: dict[str, WorkerInfo] = {}
        self._comfy_ports: dict[str, int] = {}
        self._configs: dict[str, dict[str, Any]] = {}

    # ── Rutas ───────────────────────────────────────────────────────────
    def _resolve_python_bin(self) -> str:
        meta_path = Path(settings.backends_dir) / BACKEND_TYPE / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return sys.executable
        bin_path = meta.get("python_bin") if isinstance(meta, dict) else None
        return str(bin_path) if bin_path else sys.executable

    def _worker_script_path(self) -> Path:
        script = Path(__file__).resolve().parents[2] / "workers" / "comfyui_video_worker.py"
        if not script.exists():
            raise FileNotFoundError(f"Worker script not found: {script}")
        return script

    @staticmethod
    def workflow_path(name: str) -> Path:
        # Solo nombres de fichero del directorio de workflows, nunca rutas.
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"Invalid workflow name: {name!r}")
        path = workflows_dir() / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Workflow not found: {path.name}")
        return path

    # ── Ciclo de vida ───────────────────────────────────────────────────
    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        if not gpu_indices:
            raise ValueError("At least one GPU index is required")
        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")
        cfg = section(kwargs.get("extra_config"))
        model_dir = Path(settings.models_dir) / str(cfg.get("model_dir") or DEFAULT_MODEL_DIR)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Video model directory not found: {model_dir}")
        workflow = self.workflow_path(str(cfg.get("workflow") or DEFAULT_WORKFLOW))
        src_dir = Path(settings.backends_dir) / BACKEND_TYPE / "src"
        if not (src_dir / "main.py").exists():
            raise FileNotFoundError(
                f"ComfyUI not installed at {src_dir}; install the backend first"
            )

        gpu_index = gpu_indices[0]
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        nvidia_ld = venv_nvidia_ld_library_path(settings.backends_dir, BACKEND_TYPE)
        if nvidia_ld:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{nvidia_ld}:{existing}" if existing else nvidia_ld

        comfy_port = await self._acquire_aux_port()
        cmd = [
            self._resolve_python_bin(),
            str(self._worker_script_path()),
            "--model-id", model_id,
            "--model-path", str(model_dir),
            "--src-dir", str(src_dir),
            "--workflow", str(workflow),
            "--port", str(port),
            "--comfy-port", str(comfy_port),
        ]  # fmt: skip
        for arg in cfg.get("comfy_args") or []:
            cmd.append(f"--comfy-arg={arg}")

        log_path = f"/tmp/comfyui_video_worker_{model_id.replace('/', '_')}.log"
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=open(log_path, "ab"),  # noqa: ASYNC230,SIM115 — fd de vida larga para el hijo
            env=env,
        )
        self._processes[model_id] = process
        self._comfy_ports[model_id] = comfy_port
        try:
            await self._wait_until_healthy(port=port, process=process, timeout_s=300)
        except BaseException:
            if self._processes.get(model_id) is process:
                self._processes.pop(model_id, None)
            await self._terminate_process(process)
            self._release_aux_port(self._comfy_ports.pop(model_id, 0))
            raise

        vram_mb = int(cfg.get("vram_mb") or DEFAULT_VRAM_MB)
        info = WorkerInfo(
            backend_type=BACKEND_TYPE,
            model_id=model_id,
            gpu_indices=[gpu_index],
            port=port,
            pid=process.pid,
            vram_used_mb=vram_mb,
        )
        self._workers[model_id] = info
        self._configs[model_id] = cfg
        logger.info(
            "comfyui_video_worker_loaded",
            model_id=model_id,
            gpu_index=gpu_index,
            port=port,
            comfy_port=comfy_port,
            workflow=workflow.name,
        )
        return info

    async def unload(self, model_id: str) -> None:
        process = self._processes.pop(model_id, None)
        self._workers.pop(model_id, None)
        self._configs.pop(model_id, None)
        port = self._comfy_ports.pop(model_id, 0)
        if process is not None:
            await self._terminate_process(process)
        self._release_aux_port(port)
        logger.info("comfyui_video_worker_unloaded", model_id=model_id)

    async def health_check(self, model_id: str) -> bool:
        worker = self._workers.get(model_id)
        if worker is None:
            return False
        return await self._health_check_port(worker.port)

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(video_generation=True, streaming=False)

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        return int(section(extra_config).get("vram_mb") or DEFAULT_VRAM_MB)

    def max_seconds(self, model_id: str) -> int:
        return int(self._configs.get(model_id, {}).get("max_seconds") or DEFAULT_MAX_SECONDS)

    # ── Peticiones ──────────────────────────────────────────────────────
    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if worker is None:
            raise KeyError(f"No worker found for model '{model_id}'")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"http://127.0.0.1:{worker.port}{path}", json=body)
            response.raise_for_status()
            return response.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        raise RuntimeError("comfyui_video backend does not support streaming")

    async def generate_video(
        self, model_id: str, params: dict[str, Any]
    ) -> tuple[bytes, dict[str, str]]:
        """Genera un vídeo. Devuelve (MP4, cabeceras ``x-ocabra-*`` del worker)."""
        worker = self._workers.get(model_id)
        if worker is None:
            raise KeyError(f"No worker found for model '{model_id}'")
        # Sin tope corto: abortar a mitad tira todo el trabajo de GPU hecho.
        async with httpx.AsyncClient(timeout=3600.0) as client:
            response = await client.post(f"http://127.0.0.1:{worker.port}/generate", json=params)
            response.raise_for_status()
            stats = {k: v for k, v in response.headers.items() if k.lower().startswith("x-ocabra-")}
            return response.content, stats

    async def get_progress(self, model_id: str) -> dict[str, Any] | None:
        worker = self._workers.get(model_id)
        if worker is None:
            return None
        try:
            client = health_client()
            r = await client.get(f"http://127.0.0.1:{worker.port}/progress", timeout=2.0)
            return r.json() if r.status_code == 200 else None
        except Exception:  # noqa: BLE001 — informativo
            return None

    # ── Internos ────────────────────────────────────────────────────────
    async def _wait_until_healthy(
        self, *, port: int, process: asyncio.subprocess.Process, timeout_s: int
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"ComfyUI video worker exited with code {process.returncode}")
            if await self._health_check_port(port):
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"ComfyUI video worker health check timed out after {timeout_s}s")

    async def _health_check_port(self, port: int) -> bool:
        try:
            client = health_client()
            response = await client.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            return response.status_code == 200 and bool(response.json().get("ok", False))
        except Exception:  # noqa: BLE001
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
