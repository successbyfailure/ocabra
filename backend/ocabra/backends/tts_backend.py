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

OPENAI_VOICES = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")

VOICE_MAPPINGS: dict[str, dict[str, str]] = {
    "qwen3-tts": {
        "alloy": "zh-CN-XiaoxiaoNeural",
        "echo": "zh-CN-YunxiNeural",
        "fable": "en-GB-SoniaNeural",
        "onyx": "en-US-GuyNeural",
        "nova": "en-US-AriaNeural",
        "shimmer": "en-US-JennyNeural",
    },
    "kokoro": {
        "alloy": "af",
        "echo": "am",
        "fable": "bf",
        "onyx": "bm",
        "nova": "cf",
        "shimmer": "cm",
    },
    "bark": {
        "alloy": "v2/en_speaker_0",
        "echo": "v2/en_speaker_1",
        "fable": "v2/en_speaker_2",
        "onyx": "v2/en_speaker_3",
        "nova": "v2/en_speaker_4",
        "shimmer": "v2/en_speaker_5",
    },
}

KNOWN_VRAM_MB = {
    "qwen3-tts": 8192,
    "kokoro-82m": 1024,
    "kokoro": 1024,
    "bark": 6144,
}

FORMAT_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
WORKER_PATH = Path(__file__).resolve().parents[2] / "workers" / "tts_worker.py"


@dataclass
class _TTSWorker:
    process: asyncio.subprocess.Process
    info: WorkerInfo


class TTSBackend(BackendInterface):

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.AUDIO_SPEECH}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Declarative install spec for the tts backend (Bloque 15 Fase 2).

        ``pip_packages`` mirrors the worker imports + ``Dockerfile.tts``.  Pulls
        torch CUDA wheels via ``pip_extra_index_urls`` and seeds the venv with
        the FastAPI core runtime so ``tts_worker.py`` can boot.
        """

        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-tts",
            oci_tags={"cuda12": "latest-cuda12"},
            pip_packages=[
                "torch>=2.5",
                "torchaudio>=2.5",
                # qwen-tts pins transformers exactly (e.g. 4.57.3); declaring
                # a >=5.0 floor here breaks resolution.  We let qwen-tts and
                # kokoro pull in the transformers version they need.
                "qwen-tts>=0.1.1",
                "kokoro>=0.9",
                "soundfile>=0.12",
                "numpy",
            ],
            pip_extra_index_urls=[
                "https://download.pytorch.org/whl/cu124",
            ],
            estimated_size_mb=5000,
            display_name="TTS (Qwen3-TTS / Kokoro / Bark)",
            description=(
                "Text-to-speech backend serving Qwen3-TTS, Kokoro and Bark "
                "voices through a unified worker"
            ),
            tags=["TTS", "GPU", "CUDA"],
        )

    def __init__(self) -> None:
        self._workers: dict[str, _TTSWorker] = {}

    def _resolve_python_bin(self) -> str:
        """Return the python interpreter that should launch the tts worker.

        Reads ``<backends_dir>/tts/metadata.json`` (written by the
        :class:`~ocabra.core.backend_installer.BackendInstaller` on a source
        install) and returns its ``python_bin`` entry when present.  Falls
        back to :data:`sys.executable` for the fat image where TTS deps live
        in the main interpreter.
        """

        try:
            meta_path = Path(settings.backends_dir) / "tts" / "metadata.json"
            if not meta_path.exists():
                return sys.executable
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return sys.executable
        bin_path = meta.get("python_bin") if isinstance(meta, dict) else None
        return str(bin_path) if bin_path else sys.executable

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        existing = self._workers.get(model_id)
        if existing and existing.process.returncode is None:
            return existing.info

        if not WORKER_PATH.exists():
            raise FileNotFoundError(f"tts_worker.py not found at '{WORKER_PATH}'")

        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")
        env = os.environ.copy()
        env.update(kwargs.get("env", {}))
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)

        process = await asyncio.create_subprocess_exec(
            self._resolve_python_bin(),
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
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        info = WorkerInfo(
            backend_type="tts",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid,
            vram_used_mb=await self.get_vram_estimate_mb(model_id),
        )

        self._workers[model_id] = _TTSWorker(process=process, info=info)

        if not await self._wait_until_healthy(model_id):
            await self.unload(model_id)
            raise RuntimeError(f"TTS worker failed to start for model '{model_id}'")

        logger.info("tts_worker_started", model_id=model_id, port=port, pid=process.pid)
        return info

    async def unload(self, model_id: str) -> None:
        worker = self._workers.pop(model_id, None)
        if not worker:
            return

        process = worker.process
        if process.returncode is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=8)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def health_check(self, model_id: str) -> bool:
        worker = self._workers.get(model_id)
        if not worker or worker.process.returncode is not None:
            return False

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://127.0.0.1:{worker.info.port}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(tts=True)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        normalized = model_id.lower()
        for key, size_mb in KNOWN_VRAM_MB.items():
            if key in normalized:
                return size_mb
        return 4096

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"TTS worker for '{model_id}' is not loaded")

        endpoint = path if path.startswith("/") else f"/{path}"

        if endpoint == "/voices":
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"http://127.0.0.1:{worker.info.port}/voices")
            response.raise_for_status()
            return response.json()

        response_format = str(body.get("response_format") or "mp3").lower()
        openai_voice = str(body.get("voice") or "alloy").lower()
        payload = {
            "input": str(body.get("input") or ""),
            "voice": self._map_openai_voice(model_id, openai_voice),
            "response_format": response_format,
            "speed": float(body.get("speed") or 1.0),
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{worker.info.port}{endpoint}", json=payload
            )
        response.raise_for_status()

        return {
            "content": response.content,
            "content_type": response.headers.get(
                "content-type", _content_type_for_format(response_format)
            ),
        }

    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"TTS worker for '{model_id}' is not loaded")

        endpoint = path if path.startswith("/") else f"/{path}"
        response_format = str(body.get("response_format") or "mp3").lower()
        openai_voice = str(body.get("voice") or "alloy").lower()
        payload = {
            "input": str(body.get("input") or ""),
            "voice": self._map_openai_voice(model_id, openai_voice),
            "response_format": response_format,
            "speed": float(body.get("speed") or 1.0),
        }

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

    async def _wait_until_healthy(self, model_id: str, retries: int = 60) -> bool:
        for _ in range(retries):
            if await self.health_check(model_id):
                return True
            await asyncio.sleep(0.5)
        return False

    def _map_openai_voice(self, model_id: str, openai_voice: str) -> str:
        family = _infer_tts_family(model_id)
        mapping = VOICE_MAPPINGS[family]
        return mapping.get(openai_voice, mapping["alloy"])


def _infer_tts_family(model_id: str) -> str:
    normalized = model_id.lower()
    if "qwen3-tts" in normalized or "qwen" in normalized:
        return "qwen3-tts"
    if "kokoro" in normalized:
        return "kokoro"
    if "bark" in normalized:
        return "bark"
    return "kokoro"


def _content_type_for_format(response_format: str) -> str:
    return FORMAT_CONTENT_TYPES.get(response_format.lower(), "audio/mpeg")
