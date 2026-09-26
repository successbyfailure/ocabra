"""VibeASR backend — speech-to-text via microsoft/VibeVoice-ASR-BitNet.

Launches ``vibeasr_worker.py``, an HTTP shim that owns the persistent native
``asr_stream_server`` (VAE encoder + ternary LM decoder loaded once). CPU-first
and near real-time (RTF ~1 on ~8 threads); registers under the
``AUDIO_TRANSCRIPTION`` modality and serves the ``/transcribe`` contract the
OpenAI ``/v1/audio/transcriptions`` route posts to.
"""

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
from ocabra.core.backend_installer import read_backend_metadata, venv_nvidia_ld_library_path

logger = structlog.get_logger(__name__)

WORKER_PATH = Path(__file__).resolve().parents[2] / "workers" / "vibeasr_worker.py"


@dataclass
class _VibeAsrWorker:
    process: asyncio.subprocess.Process
    info: WorkerInfo
    log_file: Any | None = None


class VibeAsrBackend(BackendInterface):
    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.AUDIO_TRANSCRIPTION}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Native build of microsoft/VibeASR.cpp (asr_stream_server) + a small
        FastAPI worker (core runtime) that wraps its stdin/stdout protocol."""
        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-vibeasr",
            oci_tags={"cpu": "latest-cpu", "cuda12": "latest-cuda12"},
            apt_packages=["build-essential", "cmake", "git", "ca-certificates", "ffmpeg"],
            git_repo="https://github.com/microsoft/VibeASR.cpp.git",
            git_ref="main",
            git_recursive=True,
            post_install_script="backend/scripts/install_vibeasr.sh",
            extra_bins={"stream_server": "bin/asr_stream_server", "infer": "bin/asr_infer"},
            include_core_runtime=True,
            estimated_size_mb=400,
            display_name="VibeASR (1.58-bit STT)",
            description=(
                "Speech-to-text via microsoft/VibeVoice-ASR-BitNet — ternary LM "
                "decoder + VAE encoder, CPU-first near real-time transcription."
            ),
            tags=["STT", "GGUF", "CPU"],
        )

    def __init__(self) -> None:
        self._workers: dict[str, _VibeAsrWorker] = {}

    def _metadata(self) -> dict[str, Any]:
        meta = read_backend_metadata(settings.backends_dir, "vibeasr")
        return meta if isinstance(meta, dict) else {}

    def _resolve_python_bin(self) -> str:
        python_bin = self._metadata().get("python_bin")
        if isinstance(python_bin, str) and python_bin and Path(python_bin).is_file():
            return python_bin
        configured = settings.vibeasr_python_bin
        if configured and Path(configured).is_file():
            return str(configured)
        return sys.executable

    def _resolve_stream_bin(self) -> str:
        extra = self._metadata().get("extra_bins")
        if isinstance(extra, dict):
            bin_path = extra.get("stream_server")
            if bin_path and Path(bin_path).is_file():
                return str(bin_path)
        return settings.vibeasr_stream_server_bin

    def _resolve_models(self, model_id: str, cfg: dict[str, Any]) -> tuple[str, str]:
        """Resolve (vae_model, lm_model) GGUF paths for the model dir.

        The VAE encoder is the ``i8_s`` GGUF, the LM decoder the ``i2_s`` one.
        Explicit ``extra_config`` paths win.
        """
        vae = cfg.get("vae_model")
        lm = cfg.get("lm_model")
        if vae and lm:
            return str(vae), str(lm)

        root = Path(settings.models_dir)
        flat_id = model_id.replace("/", "--")
        candidates = [
            Path(str(vae)).parent if vae else None,
            Path(str(lm)).parent if lm else None,
            root / model_id,
            root / flat_id,
            root / "huggingface" / flat_id,
            Path(model_id),
        ]
        model_dir = next((c for c in candidates if c is not None and c.is_dir()), None)
        if model_dir is None:
            # Locate a directory containing the complete pair. Never combine a
            # VAE from one downloaded model with an LM from another.
            parents = sorted({path.parent for path in root.rglob("*.gguf")})
            model_dir = next(
                (
                    parent
                    for parent in parents
                    if any(
                        "i8_s" in path.name.lower() or "vae" in path.name.lower()
                        for path in parent.glob("*.gguf")
                    )
                    and any(
                        "i2_s" in path.name.lower() or "lm" in path.name.lower()
                        for path in parent.glob("*.gguf")
                    )
                ),
                None,
            )
            if model_dir is None:
                raise FileNotFoundError(
                    "VibeASR needs a co-located VAE (i8_s) and LM (i2_s) "
                    f"GGUF pair under '{root}'."
                )

        ggufs = list(model_dir.rglob("*.gguf"))
        if vae is None:
            vae_match = next(
                (p for p in ggufs if "i8_s" in p.name.lower() or "vae" in p.name.lower()), None
            )
            vae = str(vae_match) if vae_match else ""
        if lm is None:
            lm_match = next(
                (p for p in ggufs if "i2_s" in p.name.lower() or "lm" in p.name.lower()), None
            )
            lm = str(lm_match) if lm_match else ""
        if not vae or not lm:
            raise FileNotFoundError(
                f"VibeASR needs a VAE (i8_s) and LM (i2_s) GGUF under '{model_dir}'; "
                f"found vae='{vae}', lm='{lm}'."
            )
        return str(vae), str(lm)

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs: Any) -> WorkerInfo:
        existing = self._workers.get(model_id)
        if existing and existing.process.returncode is None:
            return existing.info
        if not WORKER_PATH.exists():
            raise FileNotFoundError(f"vibeasr_worker.py not found at '{WORKER_PATH}'")

        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        extra_config = kwargs.get("extra_config") or {}
        raw_cfg = extra_config.get("vibeasr")
        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
        stream_bin = self._resolve_stream_bin()
        vae_model, lm_model = self._resolve_models(model_id, cfg)
        threads = int(cfg.get("threads") or settings.vibeasr_threads)

        env = os.environ.copy()
        env.update(kwargs.get("env", {}))
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
        # The native asr_stream_server ships its libggml/libllama next to it.
        ld_parts = [str(Path(stream_bin).parent)]
        nvidia_ld = venv_nvidia_ld_library_path(settings.backends_dir, "vibeasr")
        if nvidia_ld:
            ld_parts.append(nvidia_ld)
        prev = env.get("LD_LIBRARY_PATH", "")
        if prev:
            ld_parts.append(prev)
        env["LD_LIBRARY_PATH"] = ":".join(p for p in ld_parts if p)

        python_bin = self._resolve_python_bin()
        args = [
            python_bin,
            str(WORKER_PATH),
            "--model-id",
            model_id,
            "--stream-server-bin",
            stream_bin,
            "--vae-model",
            vae_model,
            "--lm-model",
            lm_model,
            "--threads",
            str(threads),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpu-indices",
            ",".join(str(i) for i in gpu_indices),
        ]

        log_path = _worker_log_path(model_id)
        log_file = open(log_path, "ab")  # noqa: SIM115, ASYNC230

        logger.info("vibeasr_starting", model_id=model_id, port=port, threads=threads)
        process = await asyncio.create_subprocess_exec(
            *args, env=env, stdout=log_file, stderr=log_file
        )

        info = WorkerInfo(
            backend_type="vibeasr",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid or 0,
            vram_used_mb=await self.get_vram_estimate_mb(model_id),
        )
        self._workers[model_id] = _VibeAsrWorker(process=process, info=info, log_file=log_file)

        timeout_s = max(1, int(settings.vibeasr_startup_timeout_s))
        if not await self._wait_until_healthy(model_id, timeout_s=timeout_s):
            await self.unload(model_id)
            raise RuntimeError(
                f"VibeASR worker failed to start for '{model_id}' within {timeout_s}s. "
                f"Check logs at {log_path}"
            )
        logger.info("vibeasr_worker_started", model_id=model_id, port=port, pid=process.pid)
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
                except Exception:  # noqa: BLE001
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

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        # CPU-first backend; no VRAM reservation.
        return 0

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"VibeASR worker for '{model_id}' is not loaded")
        endpoint = path if path.startswith("/") else f"/{path}"

        name, data, content_type = _extract_audio_file(body)
        files = {"file": (name, data, content_type)}
        form_data: dict[str, str] = {}
        if body.get("language"):
            form_data["language"] = str(body["language"])
        if body.get("response_format"):
            form_data["response_format"] = str(body["response_format"])
        if body.get("temperature") is not None:
            form_data["temperature"] = str(body["temperature"])

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{worker.info.port}{endpoint}", files=files, data=form_data
            )
        response.raise_for_status()
        response_format = str(body.get("response_format", "json")).lower()
        if response_format in {"json", "verbose_json"}:
            return response.json()
        return response.text

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        payload = await self.forward_request(model_id, path, body)
        if isinstance(payload, (dict, list)):
            yield json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            yield payload.encode("utf-8")
        elif isinstance(payload, bytes):
            yield payload

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


def _worker_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/vibeasr-worker-{safe}.log"


def _extract_audio_file(body: dict) -> tuple[str, bytes, str]:
    payload = body.get("file")
    if payload is None:
        raise ValueError("Missing 'file' field in transcription request")
    if isinstance(payload, bytes):
        return "audio.wav", payload, "application/octet-stream"
    if isinstance(payload, tuple):
        if len(payload) == 2:
            name, data = payload
            return str(name), bytes(data), "application/octet-stream"
        if len(payload) == 3:
            name, data, content_type = payload
            return str(name), bytes(data), str(content_type)
    if isinstance(payload, dict):
        name = str(payload.get("filename") or "audio.wav")
        data = payload.get("content")
        if data is None:
            raise ValueError("Audio file payload requires 'content'")
        return name, bytes(data), str(payload.get("content_type") or "application/octet-stream")
    read = getattr(payload, "read", None)
    if callable(read):
        return (
            str(getattr(payload, "filename", "audio.wav")),
            bytes(read()),
            str(getattr(payload, "content_type", "application/octet-stream")),
        )
    raise ValueError("Unsupported 'file' payload format for VibeASR transcription")
