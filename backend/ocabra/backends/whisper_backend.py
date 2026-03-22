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

from ocabra.backends.base import BackendCapabilities, BackendInterface, WorkerInfo
from ocabra.config import settings

logger = structlog.get_logger(__name__)

KNOWN_VRAM_MB = {
    "tiny": 300,
    "base": 500,
    "small": 1200,
    "medium": 3000,
    "large-v3": 6000,
    "large-v3-turbo": 3500,
    "whisper-tiny": 300,
    "whisper-base": 500,
    "whisper-small": 1200,
    "whisper-medium": 3000,
    "whisper-large-v3": 6000,
    "whisper-large-v3-turbo": 3500,
    "faster-whisper-large-v3-turbo-latam-int8-ct2": 3600,
}
OPENAI_TO_FASTER_MODEL_ID = {
    "openai/whisper-tiny": "tiny",
    "openai/whisper-base": "base",
    "openai/whisper-small": "small",
    "openai/whisper-medium": "medium",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
}
DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
DIARIZATION_OVERHEAD_MB = 1200
WORKER_PATH = Path(__file__).resolve().parents[2] / "workers" / "whisper_worker.py"


@dataclass
class _WhisperWorker:
    process: asyncio.subprocess.Process
    info: WorkerInfo
    log_file: Any | None = None


class WhisperBackend(BackendInterface):
    def __init__(self) -> None:
        self._workers: dict[str, _WhisperWorker] = {}

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        existing = self._workers.get(model_id)
        if existing and existing.process.returncode is None:
            return existing.info

        if not WORKER_PATH.exists():
            raise FileNotFoundError(f"whisper_worker.py not found at '{WORKER_PATH}'")

        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        extra_config = kwargs.get("extra_config") or {}
        worker_model_id = _resolve_worker_model_id(model_id, extra_config)
        diarization_enabled = _should_enable_diarization(model_id, extra_config)
        diarization_model_id = _resolve_diarization_model_id(extra_config)

        env = os.environ.copy()
        env.update(kwargs.get("env", {}))
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
        if diarization_enabled and settings.hf_token and not env.get("HF_TOKEN"):
            env["HF_TOKEN"] = settings.hf_token

        args = [
            sys.executable,
            str(WORKER_PATH),
            "--model-id",
            worker_model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpu-indices",
            ",".join(str(i) for i in gpu_indices),
        ]
        if diarization_enabled:
            args.extend(["--diarize", "--diarization-model-id", diarization_model_id])

        log_path = _worker_log_path(model_id)
        log_file = open(log_path, "ab")  # noqa: PTH123

        process = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )

        info = WorkerInfo(
            backend_type="whisper",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid,
            vram_used_mb=await self.get_vram_estimate_mb(model_id),
        )

        self._workers[model_id] = _WhisperWorker(process=process, info=info, log_file=log_file)

        if not await self._wait_until_healthy(model_id):
            await self.unload(model_id)
            raise RuntimeError(f"Whisper worker failed to start for model '{model_id}'")

        logger.info(
            "whisper_worker_started",
            model_id=model_id,
            worker_model_id=worker_model_id,
            diarization_enabled=diarization_enabled,
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
                    await asyncio.wait_for(process.wait(), timeout=8)
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
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://127.0.0.1:{worker.info.port}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(audio_transcription=True)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        normalized = model_id.lower()
        base_model_id = _resolve_worker_model_id(model_id, {})
        base_normalized = base_model_id.lower()

        for key, size_mb in KNOWN_VRAM_MB.items():
            if key in normalized or key in base_normalized:
                return size_mb + (
                    DIARIZATION_OVERHEAD_MB if _should_enable_diarization(model_id, {}) else 0
                )

        for model_path in (
            Path(settings.models_dir) / model_id,
            Path(settings.models_dir) / base_model_id,
        ):
            if model_path.exists() and model_path.is_dir():
                total_bytes = sum(
                    file_path.stat().st_size
                    for file_path in model_path.rglob("*")
                    if file_path.is_file()
                )
                if total_bytes > 0:
                    base = max(512, int(total_bytes / (1024 * 1024) * 1.4))
                    if _should_enable_diarization(model_id, {}):
                        base += DIARIZATION_OVERHEAD_MB
                    return base

        base = 2048
        if _should_enable_diarization(model_id, {}):
            base += DIARIZATION_OVERHEAD_MB
        return base

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"Whisper worker for '{model_id}' is not loaded")

        endpoint = path if path.startswith("/") else f"/{path}"
        files, form_data = _build_transcription_multipart(body)

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{worker.info.port}{endpoint}",
                files=files,
                data=form_data,
            )
        response.raise_for_status()

        response_format = str(body.get("response_format", "json")).lower()
        if response_format in {"json", "verbose_json"}:
            return response.json()

        return response.text

    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        payload = await self.forward_request(model_id, path, body)
        if isinstance(payload, (dict, list)):
            yield json.dumps(payload).encode("utf-8")
            return
        if isinstance(payload, str):
            yield payload.encode("utf-8")
            return
        if isinstance(payload, bytes):
            yield payload

    async def _wait_until_healthy(self, model_id: str, retries: int = 60) -> bool:
        for _ in range(retries):
            if await self.health_check(model_id):
                return True
            await asyncio.sleep(0.5)
        return False



def _worker_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/whisper-worker-{safe}.log"

def _build_transcription_multipart(
    body: dict,
) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, str | list[str]]]:
    file_name, file_bytes, content_type = _extract_audio_file(body)

    files = {
        "file": (file_name, file_bytes, content_type),
    }

    form_data: dict[str, str | list[str]] = {}
    if body.get("language"):
        form_data["language"] = str(body["language"])
    if body.get("response_format"):
        form_data["response_format"] = str(body["response_format"])

    granularities = body.get("timestamp_granularities") or body.get(
        "timestamp_granularities[]"
    )
    if granularities is not None:
        if isinstance(granularities, list):
            form_data["timestamp_granularities"] = [str(value) for value in granularities]
        else:
            form_data["timestamp_granularities"] = [str(granularities)]

    if body.get("prompt"):
        form_data["prompt"] = str(body["prompt"])

    if body.get("temperature") is not None:
        form_data["temperature"] = str(body["temperature"])

    if body.get("diarize") is not None:
        form_data["diarize"] = str(body["diarize"]).lower()

    return files, form_data


def _extract_audio_file(body: dict) -> tuple[str, bytes, str]:
    payload = body.get("file")

    if payload is None:
        raise ValueError("Missing 'file' field in transcription request")

    if isinstance(payload, bytes):
        return "audio.wav", payload, "application/octet-stream"

    if isinstance(payload, tuple):
        if len(payload) == 2:
            filename, data = payload
            return str(filename), bytes(data), "application/octet-stream"
        if len(payload) == 3:
            filename, data, content_type = payload
            return str(filename), bytes(data), str(content_type)

    if isinstance(payload, dict):
        filename = str(payload.get("filename") or "audio.wav")
        data = payload.get("content")
        content_type = str(payload.get("content_type") or "application/octet-stream")
        if data is None:
            raise ValueError("Audio file payload requires 'content'")
        return filename, bytes(data), content_type

    read = getattr(payload, "read", None)
    if callable(read):
        content = read()
        if asyncio.iscoroutine(content):
            raise ValueError("Async file readers are not supported in WhisperBackend payload")
        filename = getattr(payload, "filename", "audio.wav")
        content_type = getattr(payload, "content_type", "application/octet-stream")
        return str(filename), bytes(content), str(content_type)

    raise ValueError("Unsupported 'file' payload format for Whisper transcription")


def _resolve_worker_model_id(model_id: str, extra_config: dict) -> str:
    configured = extra_config.get("base_model_id")
    if isinstance(configured, str) and configured.strip():
        candidate = configured.strip()
    elif "::" in model_id:
        candidate = model_id.split("::", 1)[0]
    else:
        candidate = model_id

    return OPENAI_TO_FASTER_MODEL_ID.get(candidate.lower(), candidate)


def _should_enable_diarization(model_id: str, extra_config: dict) -> bool:
    configured = extra_config.get("diarization_enabled")
    if isinstance(configured, bool):
        return configured
    return "diariz" in model_id.lower()


def _resolve_diarization_model_id(extra_config: dict) -> str:
    value = extra_config.get("diarization_model_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return DEFAULT_DIARIZATION_MODEL_ID
