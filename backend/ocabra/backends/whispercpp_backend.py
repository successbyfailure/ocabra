"""whisper.cpp backend — native CPU speech-to-text via ``whisper-server``.

``whisper-server`` is launched directly as the worker (no Python wrapper):
it already serves ``/health`` and, via ``--inference-path /transcribe``, the
same path the OpenAI ``/v1/audio/transcriptions`` route posts to. Intended as
a lower-priority fallback target inside the ``recomendado-stt`` router — see
[[project_multi_gpu_vram_race_incidents]] context: RouterResolver's existing
warm/cold-start logic gives occupancy-based CPU/GPU balancing for free, no
new scheduling code needed.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
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

_INFERENCE_PATH = "/transcribe"


def _worker_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/whispercpp-worker-{safe}.log"


@dataclass
class _WhisperCppWorker:
    process: asyncio.subprocess.Process
    port: int
    log_file: Any


class WhisperCppBackend(BackendInterface):
    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.AUDIO_TRANSCRIPTION}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Native build of ggml-org/whisper.cpp — ``whisper-server`` is the
        worker itself, so no ``include_core_runtime`` venv is needed."""
        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-whispercpp",
            oci_tags={"cpu": "latest-cpu"},
            apt_packages=[
                "build-essential",
                "cmake",
                "git",
                "ninja-build",
                "ca-certificates",
            ],
            git_repo="https://github.com/ggerganov/whisper.cpp",
            git_ref="master",
            post_install_script="backend/scripts/install_whispercpp.sh",
            extra_bins={
                "server": "bin/whisper-server",
                "cli": "bin/whisper-cli",
                "quantize": "bin/whisper-quantize",
            },
            include_core_runtime=False,
            estimated_size_mb=900,
            display_name="whisper.cpp (CPU)",
            description=(
                "Native ggml/whisper.cpp STT server. CPU-only fallback for "
                "transcription, intended to absorb load when the GPU "
                "whisper variants are busy."
            ),
            tags=["STT", "CPU", "GGML"],
        )

    def __init__(self) -> None:
        self._processes: dict[str, _WhisperCppWorker] = {}

    def _resolve_server_bin(self) -> str:
        """Pick the ``whisper-server`` binary path.

        Priority:
        1. ``settings.whispercpp_custom_server_bin`` (explicit custom fork).
        2. ``<backends_dir>/whisper_cpp/metadata.json`` -> ``extra_bins.server``
           (modular install via :class:`BackendInstaller`).
        3. ``settings.whispercpp_server_bin`` (legacy fat-image path).
        """
        custom_bin = settings.whispercpp_custom_server_bin
        if isinstance(custom_bin, str) and custom_bin.strip():
            custom_path = Path(custom_bin)
            if not custom_path.is_file():
                raise FileNotFoundError(f"Custom whisper-server not found at '{custom_path}'")
            return str(custom_path)
        meta = read_backend_metadata(settings.backends_dir, "whisper_cpp")
        if meta is not None:
            extra = meta.get("extra_bins") if isinstance(meta, dict) else None
            if isinstance(extra, dict):
                bin_path = extra.get("server")
                if bin_path and Path(bin_path).is_file():
                    return str(bin_path)
        return settings.whispercpp_server_bin

    def _resolve_model_file(self, model_id: str, extra_config: dict[str, Any]) -> Path:
        cfg = extra_config.get("whisper_cpp") if isinstance(extra_config, dict) else None
        explicit = (cfg or {}).get("model_path") if isinstance(cfg, dict) else None
        if not explicit and isinstance(extra_config, dict):
            explicit = extra_config.get("model_path")
        if explicit:
            path = Path(str(explicit))
            if path.is_file():
                return path

        direct = Path(model_id)
        if direct.is_file():
            return direct

        models_dir = Path(settings.backends_dir) / "whisper_cpp" / "models"
        stem = Path(model_id).name
        candidates = [
            models_dir / model_id,
            models_dir / f"{stem}.bin",
            models_dir / f"ggml-{stem}.bin",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        matches = [path for path in models_dir.glob("*.bin") if stem.lower() in path.name.lower()]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)

        raise FileNotFoundError(
            f"whisper.cpp GGML model not found for '{model_id}' under {models_dir}"
        )

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs: Any) -> WorkerInfo:
        existing = self._processes.get(model_id)
        if existing and existing.process.returncode is None:
            return WorkerInfo(
                backend_type="whisper_cpp",
                model_id=model_id,
                gpu_indices=[],
                port=existing.port,
                pid=existing.process.pid or 0,
                vram_used_mb=0,
            )

        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")

        extra_config = kwargs.get("extra_config") or {}
        cfg = extra_config.get("whisper_cpp") if isinstance(extra_config, dict) else None
        cfg = cfg if isinstance(cfg, dict) else {}
        model_file = self._resolve_model_file(model_id, extra_config)
        threads = int(cfg.get("threads") or settings.whispercpp_threads)
        server_bin = self._resolve_server_bin()

        env = os.environ.copy()
        env.update(kwargs.get("env", {}))
        # CPU-only backend: never advertise a GPU to the worker, regardless
        # of what the scheduler assigned (it shouldn't assign any).
        env["CUDA_VISIBLE_DEVICES"] = ""
        # The binary's own RUNPATH points at its original build tree (e.g.
        # /tmp/whisper.cpp/build/bin), not wherever BackendInstaller ends up
        # placing it (/data/backends/whisper_cpp/bin) — it still needs to
        # find its libggml*/libwhisper*.so siblings there.
        lib_dir = str(Path(server_bin).parent)
        prev_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{prev_ld}" if prev_ld else lib_dir

        args = [
            server_bin,
            "--model",
            str(model_file),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--threads",
            str(threads),
            "--no-gpu",
            "--convert",
            "--inference-path",
            _INFERENCE_PATH,
        ]

        log_path = _worker_log_path(model_id)
        log_file = open(log_path, "ab")  # noqa: ASYNC230, SIM115 — worker-lifetime handle

        logger.info("whispercpp_starting", model_id=model_id, port=port, threads=threads)
        process = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )
        self._processes[model_id] = _WhisperCppWorker(process=process, port=port, log_file=log_file)

        try:
            await self._wait_for_startup(
                model_id,
                port,
                timeout_s=int(cfg.get("startup_timeout_s") or settings.whispercpp_startup_timeout_s),
            )
        except BaseException:
            await self._kill_process(model_id)
            raise

        logger.info("whispercpp_worker_started", model_id=model_id, port=port, pid=process.pid)
        return WorkerInfo(
            backend_type="whisper_cpp",
            model_id=model_id,
            gpu_indices=[],
            port=port,
            pid=process.pid or 0,
            vram_used_mb=0,
        )

    async def unload(self, model_id: str) -> None:
        await self._kill_process(model_id, graceful=True)

    async def health_check(self, model_id: str) -> bool:
        entry = self._processes.get(model_id)
        if not entry or entry.process.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"http://127.0.0.1:{entry.port}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(audio_transcription=True)

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        return 0

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No whisper.cpp worker for model '{model_id}'")

        endpoint = path if path.startswith("/") else f"/{path}"
        files, form_data = _build_transcription_multipart(body)

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{entry.port}{endpoint}",
                files=files,
                data=form_data,
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
            return
        if isinstance(payload, str):
            yield payload.encode("utf-8")
            return
        if isinstance(payload, bytes):
            yield payload

    async def _wait_for_startup(self, model_id: str, port: int, timeout_s: int) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        url = f"http://127.0.0.1:{port}/health"
        while asyncio.get_running_loop().time() < deadline:
            entry = self._processes.get(model_id)
            if entry and entry.process.returncode is not None:
                rc = entry.process.returncode
                log_tail = self._read_log_tail(model_id)
                suffix = f" log_tail={log_tail}" if log_tail else ""
                if rc in {-signal.SIGKILL, 137}:
                    raise MemoryError(
                        f"whisper.cpp worker for '{model_id}' was killed (rc={rc}).{suffix}"
                    )
                raise RuntimeError(
                    f"whisper.cpp worker for '{model_id}' exited with rc={rc}.{suffix}"
                )
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(
            f"whisper.cpp worker for '{model_id}' did not become healthy within {timeout_s}s"
        )

    async def _kill_process(self, model_id: str, graceful: bool = False) -> None:
        entry = self._processes.pop(model_id, None)
        if not entry:
            return
        process = entry.process
        if process.returncode is not None:
            entry.log_file.close()
            return

        if graceful:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=20.0)
                entry.log_file.close()
                return
            except (TimeoutError, ProcessLookupError):
                logger.warning(
                    "whispercpp_sigterm_timeout",
                    model_id=model_id,
                    log_tail=self._read_log_tail(model_id),
                )

        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass
        finally:
            entry.log_file.close()

    def _read_log_tail(self, model_id: str, limit: int = 4000) -> str:
        try:
            path = _worker_log_path(model_id)
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                if size > limit:
                    f.seek(size - limit)
                data = f.read()
        except OSError:
            return ""
        if not data:
            return ""
        text = data.decode("utf-8", errors="replace").strip()
        return text if len(text) <= limit else text[-limit:]


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
            raise ValueError("Async file readers are not supported in WhisperCppBackend payload")
        filename = getattr(payload, "filename", "audio.wav")
        content_type = getattr(payload, "content_type", "application/octet-stream")
        return str(filename), bytes(content), str(content_type)

    raise ValueError("Unsupported 'file' payload format for whisper.cpp transcription")
