"""Chatterbox TTS backend — launches chatterbox_worker.py as a subprocess."""

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

WORKER_PATH = Path(__file__).resolve().parents[2] / "workers" / "chatterbox_worker.py"

# Chatterbox has no named speakers — voice cloning is done via voice_ref.
# Map all OpenAI voices to "default".
CHATTERBOX_VOICE_MAPPINGS: dict[str, str] = {
    "alloy": "default",
    "echo": "default",
    "fable": "default",
    "nova": "default",
    "onyx": "default",
    "shimmer": "default",
}

CHATTERBOX_LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "pl",
    "ro",
    "hu",
    "cs",
    "el",
    "sv",
    "da",
    "fi",
    "bg",
    "hr",
    "sk",
    "sl",
    "lt",
    "lv",
    "et",
    "mt",
]

KNOWN_VRAM_MB: dict[str, int] = {
    "chatterbox-turbo": 4096,
}

FORMAT_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


@dataclass
class _ChatterboxWorker:
    process: asyncio.subprocess.Process
    info: WorkerInfo
    log_file: Any | None = None


class ChatterboxBackend(BackendInterface):

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.AUDIO_SPEECH}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Declarative install spec for chatterbox (Bloque 15 Fase 2)."""

        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-chatterbox",
            oci_tags={"cuda12": "latest-cuda12"},
            pip_packages=[
                "torch>=2.5",
                "torchaudio>=2.5",
                "chatterbox-tts>=0.1.7",
                "soundfile>=0.12",
                "numpy",
            ],
            pip_extra_index_urls=[
                "https://download.pytorch.org/whl/cu124",
            ],
            estimated_size_mb=4500,
            display_name="Chatterbox (Resemble AI)",
            description=(
                "Multilingual TTS with zero-shot voice cloning across 28 "
                "languages (Resemble AI Chatterbox)"
            ),
            tags=["TTS", "GPU", "CUDA"],
        )

    def __init__(self) -> None:
        self._workers: dict[str, _ChatterboxWorker] = {}

    def _resolve_python_bin(self) -> str:
        """Pick the python that should launch the chatterbox worker.

        Priority:
        1. ``<backends_dir>/chatterbox/metadata.json`` -> ``python_bin``
           (modular install via :class:`BackendInstaller`).
        2. ``settings.chatterbox_python_bin`` (legacy fat-image venv at
           ``/opt/chatterbox-venv``).
        3. :data:`sys.executable` as a final fallback.
        """

        try:
            meta_path = Path(settings.backends_dir) / "chatterbox" / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                bin_path = meta.get("python_bin") if isinstance(meta, dict) else None
                if bin_path and Path(bin_path).is_file():
                    return str(bin_path)
        except (OSError, json.JSONDecodeError):
            pass

        legacy = settings.chatterbox_python_bin
        if legacy and Path(legacy).is_file():
            return str(legacy)
        return sys.executable

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs: Any) -> WorkerInfo:
        existing = self._workers.get(model_id)
        if existing and existing.process.returncode is None:
            return existing.info

        if not WORKER_PATH.exists():
            raise FileNotFoundError(f"chatterbox_worker.py not found at '{WORKER_PATH}'")

        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        python_bin = self._resolve_python_bin()
        if python_bin == sys.executable and not Path(settings.chatterbox_python_bin).is_file():
            logger.warning(
                "chatterbox_python_bin_missing",
                configured=settings.chatterbox_python_bin,
                fallback=python_bin,
            )

        env = os.environ.copy()
        env.update(kwargs.get("env", {}))
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
        if settings.hf_token:
            env.setdefault("HF_TOKEN", settings.hf_token)

        log_path = _worker_log_path(model_id)
        log_file = open(log_path, "ab")  # noqa: SIM115

        process = await asyncio.create_subprocess_exec(
            python_bin,
            str(WORKER_PATH),
            "--model-id",
            model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpu-indices",
            ",".join(str(i) for i in gpu_indices),
            env=env,
            stdout=log_file,
            stderr=log_file,
        )

        info = WorkerInfo(
            backend_type="chatterbox",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid,
            vram_used_mb=await self.get_vram_estimate_mb(model_id),
        )

        self._workers[model_id] = _ChatterboxWorker(process=process, info=info, log_file=log_file)

        if not await self._wait_until_healthy(model_id, timeout_s=300):
            await self.unload(model_id)
            raise RuntimeError(
                f"Chatterbox worker failed to start for model '{model_id}'. "
                f"Check logs at {log_path}"
            )

        logger.info(
            "chatterbox_worker_started",
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
        return BackendCapabilities(tts=True, streaming=True)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        normalized = model_id.lower()
        for key, mb in KNOWN_VRAM_MB.items():
            if key in normalized:
                return mb
        # Full Chatterbox model
        if "chatterbox" in normalized:
            return 8192
        return 4096

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"Chatterbox worker for '{model_id}' is not loaded")
        endpoint = path if path.startswith("/") else f"/{path}"

        if endpoint == "/voices":
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"http://127.0.0.1:{worker.info.port}/voices")
            r.raise_for_status()
            return r.json()

        # Map OpenAI voice names to Chatterbox
        openai_voice = str(body.get("voice") or "alloy").lower()
        response_format = str(body.get("response_format") or "mp3").lower()
        payload = {
            "text": str(body.get("input") or ""),
            "voice": _map_openai_voice(openai_voice),
            "language": str(body.get("language") or "en"),
            "speed": float(body.get("speed") or 1.0),
            "response_format": response_format,
        }
        voice_ref = body.get("voice_ref") or body.get("reference_audio")
        if voice_ref:
            payload["voice_ref"] = voice_ref

        async with httpx.AsyncClient(timeout=300.0) as client:
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
            raise KeyError(f"Chatterbox worker for '{model_id}' is not loaded")
        endpoint = path if path.startswith("/") else f"/{path}"

        openai_voice = str(body.get("voice") or "alloy").lower()
        response_format = str(body.get("response_format") or "mp3").lower()
        payload = {
            "text": str(body.get("input") or ""),
            "voice": _map_openai_voice(openai_voice),
            "language": str(body.get("language") or "en"),
            "speed": float(body.get("speed") or 1.0),
            "response_format": response_format,
        }
        voice_ref = body.get("voice_ref") or body.get("reference_audio")
        if voice_ref:
            payload["voice_ref"] = voice_ref

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{worker.info.port}{endpoint}",
                json=payload,
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
            worker = self._workers.get(model_id)
            if worker and worker.process.returncode is not None:
                return False
            await asyncio.sleep(0.5)
        return False


def _map_openai_voice(openai_voice: str) -> str:
    return CHATTERBOX_VOICE_MAPPINGS.get(openai_voice.lower(), "default")


def _content_type_for_format(response_format: str) -> str:
    return FORMAT_CONTENT_TYPES.get(response_format.lower(), "audio/mpeg")


def _worker_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/chatterbox-worker-{safe}.log"
