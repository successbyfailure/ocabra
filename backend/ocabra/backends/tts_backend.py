import asyncio
import os
import socket
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.backends.base import BackendCapabilities, BackendInterface, WorkerInfo

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
WORKER_PATH = Path(__file__).resolve().parents[3] / "workers" / "tts_worker.py"


@dataclass
class _TTSWorker:
    process: asyncio.subprocess.Process
    info: WorkerInfo


class TTSBackend(BackendInterface):
    def __init__(self) -> None:
        self._workers: dict[str, _TTSWorker] = {}

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        existing = self._workers.get(model_id)
        if existing and existing.process.returncode is None:
            return existing.info

        if not WORKER_PATH.exists():
            raise FileNotFoundError(f"tts_worker.py not found at '{WORKER_PATH}'")

        port = int(kwargs.get("port") or _find_free_port())
        env = os.environ.copy()
        env.update(kwargs.get("env", {}))
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)

        process = await asyncio.create_subprocess_exec(
            sys.executable,
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
