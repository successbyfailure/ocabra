"""Backend for ACE-Step 1.5 music generation.

Launches the ACE-Step API server (https://github.com/ace-step/ACE-Step-1.5) as a
subprocess on an assigned port, then wraps its async task queue (submit → poll →
download) into the synchronous BackendInterface contract.

Model IDs handled by this backend:
    acestep/turbo   — acestep-v15-turbo   (8 inference steps, recommended)
    acestep/sft     — acestep-v15-sft     (50 steps, higher quality)
    acestep/base    — acestep-v15-base    (base model, no SFT)

Setup requirement:
    The ACE-Step project must be installed at settings.acestep_project_dir
    (default /docker/ACE-Step-1.5) and its virtual environment created via
    ``uv sync`` inside that directory, producing ``.venv/bin/python``.
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

from ocabra.backends.base import BackendCapabilities, BackendInterface, ModalityType, WorkerInfo
from ocabra.config import settings

logger = structlog.get_logger(__name__)

# VRAM estimates in MB (DiT + LM combined, conservative)
KNOWN_VRAM_MB: dict[str, int] = {
    "turbo": 10_000,
    "sft": 10_000,
    "base": 8_000,
    "acestep/turbo": 10_000,
    "acestep/sft": 10_000,
    "acestep/base": 8_000,
}
DEFAULT_VRAM_MB = 10_000

# Map ocabra model ID suffix → ACE-Step config path
_MODEL_CONFIG: dict[str, str] = {
    "turbo": "acestep-v15-turbo",
    "sft": "acestep-v15-sft",
    "base": "acestep-v15-base",
}
DEFAULT_CONFIG = "acestep-v15-turbo"


@dataclass
class _AceStepWorker:
    process: asyncio.subprocess.Process
    info: WorkerInfo
    log_file: Any | None = None


@dataclass
class _ExternalWorker:
    """Represents an ACE-Step instance managed externally (e.g. a Docker service)."""
    info: WorkerInfo


_AnyWorker = _AceStepWorker | _ExternalWorker

# Sentinel pid for externally-managed instances
_EXTERNAL_PID = -1


class AceStepBackend(BackendInterface):

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.IMAGE_GENERATION}
    """Manages ACE-Step API server instances.

    Two operating modes:
    - **Subprocess mode** (default): spawns `python -m acestep.api_server` as a child
      process from the ACE-Step project venv. Requires settings.acestep_project_dir
      to contain a valid `uv sync`-created `.venv`.
    - **External mode**: when settings.acestep_external_api_url is set, the backend
      connects to an already-running ACE-Step REST API (e.g. a Docker service running
      with ACESTEP_MODE=api). No subprocess is spawned; the external URL is used
      for all requests.
    """

    def __init__(self) -> None:
        self._workers: dict[str, _AnyWorker] = {}

    @property
    def _external_url(self) -> str:
        return (settings.acestep_external_api_url or "").rstrip("/")

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        # ── External mode ─────────────────────────────────────────────────
        if self._external_url:
            existing = self._workers.get(model_id)
            if existing:
                return existing.info

            # Verify the external service is reachable before "loading"
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self._external_url}/health")
                    resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(
                    f"ACE-Step external service at '{self._external_url}' is not reachable: {exc}"
                ) from exc

            # Parse port from URL for WorkerInfo (best-effort)
            from urllib.parse import urlparse
            parsed = urlparse(self._external_url)
            port = parsed.port or 8001

            info = WorkerInfo(
                backend_type="acestep",
                model_id=model_id,
                gpu_indices=gpu_indices,
                port=port,
                pid=_EXTERNAL_PID,
                vram_used_mb=await self.get_vram_estimate_mb(model_id),
            )

            self._workers[model_id] = _ExternalWorker(info=info)
            logger.info("acestep_external_worker_registered", model_id=model_id, url=self._external_url)
            return info

        # ── Subprocess mode ────────────────────────────────────────────────
        existing = self._workers.get(model_id)
        if existing and isinstance(existing, _AceStepWorker) and existing.process.returncode is None:
            return existing.info

        project_dir = Path(settings.acestep_project_dir)
        python_bin = _resolve_python(project_dir)

        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        env = os.environ.copy()
        env.update(kwargs.get("env", {}))
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)

        # Point ACE-Step at the shared HF cache so models are not re-downloaded
        env.setdefault("HF_HOME", settings.hf_cache_dir)
        env.setdefault("TRANSFORMERS_CACHE", settings.hf_cache_dir)

        args = [
            python_bin,
            "-m", "acestep.api_server",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--no-init",  # lazy-load model on first request to avoid double-init
        ]

        log_path = _worker_log_path(model_id)
        log_file = open(log_path, "ab")  # noqa: PTH123

        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(project_dir),
            env=env,
            stdout=log_file,
            stderr=log_file,
        )

        info = WorkerInfo(
            backend_type="acestep",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid,
            vram_used_mb=await self.get_vram_estimate_mb(model_id),
        )
        self._workers[model_id] = _AceStepWorker(process=process, info=info, log_file=log_file)

        timeout_s = max(1, int(settings.acestep_startup_timeout_s))
        if not await self._wait_until_healthy(model_id, timeout_s=timeout_s):
            await self.unload(model_id)
            raise RuntimeError(
                f"ACE-Step worker failed to start for model '{model_id}' within {timeout_s}s. "
                f"Check log: {log_path}"
            )

        logger.info("acestep_worker_started", model_id=model_id, port=port, pid=process.pid)
        return info

    async def unload(self, model_id: str) -> None:
        worker = self._workers.pop(model_id, None)
        if not worker:
            return
        # External workers have no process to terminate
        if not isinstance(worker, _AceStepWorker):
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
        base_url = self._worker_base_url(model_id)
        if not base_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _worker_base_url(self, model_id: str) -> str | None:
        """Return the base URL for a loaded worker (subprocess or external)."""
        if self._external_url:
            return self._external_url
        worker = self._workers.get(model_id)
        if not worker or not isinstance(worker, _AceStepWorker):
            return None
        if worker.process.returncode is not None:
            return None
        return f"http://127.0.0.1:{worker.info.port}"

    # ── capabilities / estimation ──────────────────────────────────────────

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(music_generation=True)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        key = model_id.lower()
        for name, vram in KNOWN_VRAM_MB.items():
            if name in key:
                return vram
        return DEFAULT_VRAM_MB

    # ── request forwarding ─────────────────────────────────────────────────

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        """Submit a generation task, poll until complete, return (audio_bytes, content_type, meta).

        The caller (API endpoint) receives a tuple:
            (bytes, str, dict)  →  (audio data, MIME type, ACE-Step metadata)
        """
        base_url = self._worker_base_url(model_id)
        if not base_url:
            raise KeyError(f"ACE-Step worker for '{model_id}' is not loaded")
        timeout_s = max(10, int(settings.acestep_generation_timeout_s))
        poll_interval = max(0.5, float(settings.acestep_poll_interval_s))

        # ── 1. submit task ────────────────────────────────────────────────
        task_payload = _build_release_task_payload(model_id, body)
        async with httpx.AsyncClient(timeout=30.0) as client:
            submit_resp = await client.post(f"{base_url}/release_task", json=task_payload)
        submit_resp.raise_for_status()
        submit_data = submit_resp.json()

        task_id: str | None = None
        resp_data = submit_data.get("data") or {}
        if isinstance(resp_data, dict):
            task_id = resp_data.get("task_id")
        if not task_id:
            raise RuntimeError(f"ACE-Step returned no task_id: {submit_data}")

        logger.debug("acestep_task_submitted", task_id=task_id, model_id=model_id)

        # ── 2. poll for result ────────────────────────────────────────────
        elapsed = 0.0
        result_item: dict | None = None
        async with httpx.AsyncClient(timeout=10.0) as client:
            while elapsed < timeout_s:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                poll_resp = await client.post(
                    f"{base_url}/query_result",
                    json={"task_id_list": [task_id]},
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()

                items = poll_data.get("data") or []
                if not items:
                    continue

                item = items[0]
                status = item.get("status")

                if status == 2:
                    raise RuntimeError(f"ACE-Step generation failed for task {task_id}")
                if status == 1:
                    result_item = item
                    break

        if result_item is None:
            raise TimeoutError(
                f"ACE-Step generation timed out after {timeout_s}s (task_id={task_id})"
            )

        # ── 3. parse result and read audio file ───────────────────────────
        raw_result = result_item.get("result", "[]")
        try:
            audio_list: list[dict] = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        except json.JSONDecodeError:
            audio_list = []

        if not audio_list or not isinstance(audio_list, list):
            raise RuntimeError(f"ACE-Step returned empty result for task {task_id}")

        first = audio_list[0]
        file_path = first.get("file", "")
        if not file_path:
            raise RuntimeError(f"ACE-Step result has no file path (task_id={task_id})")

        ext = Path(file_path).suffix.lower()

        # Prefer reading from the local filesystem (subprocess mode).
        # In external/Docker mode the file lives inside the remote container,
        # so we fall back to downloading via the ACE-Step /v1/audio endpoint.
        if Path(file_path).is_file():
            audio_bytes = Path(file_path).read_bytes()
        else:
            from urllib.parse import urlencode
            audio_url = f"{base_url}/v1/audio?{urlencode({'path': file_path})}"
            async with httpx.AsyncClient(timeout=60.0) as dl_client:
                dl_resp = await dl_client.get(audio_url)
                dl_resp.raise_for_status()
            audio_bytes = dl_resp.content
            if not audio_bytes:
                raise RuntimeError(
                    f"ACE-Step /v1/audio returned empty content for '{file_path}' (task_id={task_id})"
                )
        content_type = _EXT_TO_MIME.get(ext, "audio/mpeg")
        meta = {
            "task_id": task_id,
            "prompt": first.get("prompt", ""),
            "lyrics": first.get("lyrics", ""),
            "metas": first.get("metas", {}),
            "file": file_path,
        }

        logger.info(
            "acestep_generation_done",
            task_id=task_id,
            model_id=model_id,
            file=file_path,
            elapsed_s=round(elapsed, 1),
        )
        return (audio_bytes, content_type, meta)

    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        result = await self.forward_request(model_id, path, body)
        if isinstance(result, tuple):
            audio_bytes, _ct, _meta = result
            yield audio_bytes
        elif isinstance(result, bytes):
            yield result

    # ── helpers ────────────────────────────────────────────────────────────

    async def _wait_until_healthy(self, model_id: str, timeout_s: int = 120) -> bool:
        attempts = max(1, timeout_s * 2)
        for _ in range(attempts):
            if await self.health_check(model_id):
                return True
            await asyncio.sleep(0.5)
        return False


# ── module-level helpers ───────────────────────────────────────────────────────

_EXT_TO_MIME: dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def _resolve_python(project_dir: Path) -> str:
    """Return the Python binary to use for launching the ACE-Step server.

    Preference order:
    1. <project_dir>/.venv/bin/python   (uv-managed venv)
    2. sys.executable (current process) – only safe if acestep is installed in it
    """
    venv_python = project_dir / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)

    raise RuntimeError(
        f"ACE-Step virtual environment not found at '{venv_python}'. "
        f"Run the following to set it up:\n"
        f"  cd {project_dir} && uv sync\n"
        f"Then restart oCabra."
    )


def _resolve_config(model_id: str) -> str:
    """Return the ACE-Step config/model path string for a given canonical model_id."""
    suffix = model_id.split("/", 1)[-1].lower()  # e.g. "turbo" from "acestep/turbo"
    return _MODEL_CONFIG.get(suffix, DEFAULT_CONFIG)


def _build_release_task_payload(model_id: str, body: dict) -> dict:
    """Build the JSON payload for ACE-Step's POST /release_task endpoint."""
    payload: dict[str, Any] = {}

    # Copy supported generation parameters from the incoming request body
    for key in (
        "prompt", "lyrics", "bpm", "key_scale", "time_signature",
        "audio_duration", "inference_steps", "guidance_scale",
        "batch_size", "seed", "thinking", "vocal_language",
        "lm_temperature", "sample_query",
    ):
        if body.get(key) is not None:
            payload[key] = body[key]

    # Default to batch_size=1 if not specified
    payload.setdefault("batch_size", 1)

    # Map response_format to audio_format (mp3/wav/flac)
    fmt = str(body.get("response_format") or body.get("audio_format") or "mp3").lower()
    if fmt not in {"mp3", "wav", "flac"}:
        fmt = "mp3"
    payload["audio_format"] = fmt

    # Pass the model config path so ACE-Step knows which DiT to use
    payload["model"] = _resolve_config(model_id)

    return payload


def _worker_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/acestep-worker-{safe}.log"
