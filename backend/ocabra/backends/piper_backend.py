"""Piper TTS backend — launches piper_worker.py as a subprocess.

Piper (rhasspy/piper1-gpl, VITS + ONNX Runtime) is CPU-only and needs no GPU
at all — it exists to serve the "fast lane" of the voice assistant pipeline
(sub-250ms TTFB measured on this host) without touching the VRAM that the
production LLM and the cloning-tier TTS backends (Chatterbox, Qwen3-TTS)
need on the 3060/3090.
"""

from __future__ import annotations

import asyncio
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
from ocabra.core.backend_installer import read_backend_metadata

logger = structlog.get_logger(__name__)

WORKER_PATH = Path(__file__).resolve().parents[2] / "workers" / "piper_worker.py"

FORMAT_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


@dataclass
class _PiperWorker:
    process: asyncio.subprocess.Process
    info: WorkerInfo
    log_file: Any | None = None


class PiperBackend(BackendInterface):
    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.AUDIO_SPEECH}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Declarative install spec for Piper.

        Pure CPU: ONNX Runtime + the ``piper-tts`` wheel (bundles its own
        espeak-ng-data, no apt packages needed). No torch, no CUDA extra
        index — this is what keeps it out of the VRAM budget entirely.
        """
        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-piper",
            oci_tags={"cpu": "latest-cpu"},
            pip_packages=[
                "piper-tts>=1.8.0",
                "huggingface_hub>=0.24",
                "soundfile>=0.12",
                "numpy",
            ],
            estimated_size_mb=200,
            display_name="Piper (VITS, CPU)",
            description=(
                "Fast, CPU-only TTS (ONNX VITS) — no voice cloning, but zero "
                "VRAM and low latency. Each voice is a separate ~60MB model."
            ),
            tags=["TTS", "CPU"],
        )

    def __init__(self) -> None:
        self._workers: dict[str, _PiperWorker] = {}

    def _resolve_python_bin(self) -> str:
        meta = read_backend_metadata(settings.backends_dir, "piper") or {}
        bin_path = meta.get("python_bin") if isinstance(meta, dict) else None
        if bin_path and Path(bin_path).is_file():
            return str(bin_path)
        configured = settings.piper_python_bin
        if configured and Path(configured).is_file():
            return str(configured)
        return sys.executable

    @staticmethod
    def _voice_name(model_id: str) -> str:
        """Strip the ``piper/`` prefix to get the bare voice name."""
        name = model_id
        if name.startswith("piper/"):
            name = name[len("piper/") :]
        return name

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs: Any) -> WorkerInfo:
        existing = self._workers.get(model_id)
        if existing and existing.process.returncode is None:
            return existing.info

        if not WORKER_PATH.exists():
            raise FileNotFoundError(f"piper_worker.py not found at '{WORKER_PATH}'")

        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        extra_config = kwargs.get("extra_config") or {}
        piper_cfg = extra_config.get("piper") if isinstance(extra_config, dict) else None
        piper_cfg = piper_cfg if isinstance(piper_cfg, dict) else {}
        onnx_override = piper_cfg.get("onnx_path")
        config_override = piper_cfg.get("config_path")

        voice_name = str(piper_cfg.get("voice") or self._voice_name(model_id))
        voices_dir = Path(settings.models_dir) / "piper"

        python_bin = self._resolve_python_bin()

        env = os.environ.copy()
        env.update(kwargs.get("env", {}))
        env["PYTHONUNBUFFERED"] = "1"
        if settings.hf_token:
            env.setdefault("HF_TOKEN", settings.hf_token)
        # CPU-only — no CUDA_VISIBLE_DEVICES set regardless of gpu_indices.

        log_path = _worker_log_path(model_id)
        log_file = open(log_path, "ab")  # noqa: SIM115

        args = [
            python_bin,
            str(WORKER_PATH),
            "--model-id",
            model_id,
            "--voice",
            voice_name,
            "--voices-dir",
            str(voices_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if onnx_override:
            args += ["--onnx-path", str(onnx_override)]
        if config_override:
            args += ["--config-path", str(config_override)]

        process = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )

        info = WorkerInfo(
            backend_type="piper",
            model_id=model_id,
            gpu_indices=[],
            port=port,
            pid=process.pid,
            vram_used_mb=0,
        )

        self._workers[model_id] = _PiperWorker(process=process, info=info, log_file=log_file)

        # First load may need to download the voice (~60MB) from HF —
        # give it more headroom than the in-memory-only warm path needs.
        if not await self._wait_until_healthy(model_id, timeout_s=120):
            await self.unload(model_id)
            raise RuntimeError(
                f"Piper worker failed to start for model '{model_id}'. "
                f"Check logs at {log_path}"
            )

        logger.info("piper_worker_started", model_id=model_id, voice=voice_name, port=port, pid=process.pid)
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
                    await asyncio.wait_for(process.wait(), timeout=10)
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
        return BackendCapabilities(tts=True, streaming=True)

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        # CPU-only backend; no VRAM reservation, ever.
        return 0

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"Piper worker for '{model_id}' is not loaded")
        endpoint = path if path.startswith("/") else f"/{path}"

        if endpoint == "/voices":
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"http://127.0.0.1:{worker.info.port}/voices")
            r.raise_for_status()
            return r.json()

        response_format = str(body.get("response_format") or "mp3").lower()
        payload = {
            "text": str(body.get("input") or ""),
            "voice": str(body.get("voice") or "default"),
            "speed": float(body.get("speed") or 1.0),
            "response_format": response_format,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"http://127.0.0.1:{worker.info.port}{endpoint}",
                json=payload,
            )
        r.raise_for_status()
        return {
            "content": r.content,
            "content_type": r.headers.get(
                "content-type", _content_type_for_format(response_format)
            ),
        }

    async def forward_stream(
        self,
        model_id: str,
        path: str,
        body: dict,
    ) -> AsyncIterator[bytes]:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"Piper worker for '{model_id}' is not loaded")
        endpoint = path if path.startswith("/") else f"/{path}"

        response_format = str(body.get("response_format") or "mp3").lower()
        payload = {
            "text": str(body.get("input") or ""),
            "voice": str(body.get("voice") or "default"),
            "speed": float(body.get("speed") or 1.0),
            "response_format": response_format,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{worker.info.port}{endpoint}",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk

    async def _wait_until_healthy(self, model_id: str, timeout_s: int = 120) -> bool:
        attempts = max(1, timeout_s * 2)
        for _ in range(attempts):
            if await self.health_check(model_id):
                return True
            worker = self._workers.get(model_id)
            if worker and worker.process.returncode is not None:
                return False
            await asyncio.sleep(0.5)
        return False


def _content_type_for_format(response_format: str) -> str:
    return FORMAT_CONTENT_TYPES.get(response_format.lower(), "audio/mpeg")


def _worker_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/piper-worker-{safe}.log"
