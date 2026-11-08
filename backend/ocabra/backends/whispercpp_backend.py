"""whisper.cpp backend — native CPU speech-to-text via ``whisper-server``.

``whisper-server`` is launched directly as the worker (no Python wrapper):
it already serves ``/health`` and, via ``--inference-path /transcribe``, the
same path the OpenAI ``/v1/audio/transcriptions`` route posts to. Intended as
a lower-priority fallback target inside the ``recomendado-stt`` router — see
[[project_multi_gpu_vram_race_incidents]] context: RouterResolver's existing
warm/cold-start logic gives occupancy-based CPU/GPU balancing for free, no
new scheduling code needed.

Speaker diarization (``diarize=true``) has no equivalent inside whisper.cpp
itself — ``whisper-server`` is a native binary with no Python to lazy-load
pyannote into, unlike ``workers/whisper_worker.py`` (faster-whisper). Instead,
diarization requests are forwarded to ``workers/diarization_worker.py``, a
small pyannote-only sidecar process launched lazily on first use and reused
for the life of this model's worker — see ``_ensure_diarization_worker``.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
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
_DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
_DIARIZATION_WORKER_PATH = Path(__file__).resolve().parents[2] / "workers" / "diarization_worker.py"


def _worker_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/whispercpp-worker-{safe}.log"


def _diarization_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/whispercpp-diarize-{safe}.log"


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _is_diarize_requested(raw: Any) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() not in {"", "0", "false", "no"}


@dataclass
class _WhisperCppWorker:
    process: asyncio.subprocess.Process
    port: int
    log_file: Any


@dataclass
class _DiarizationSidecar:
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
        self._diarization_workers: dict[str, _DiarizationSidecar] = {}
        self._diarization_locks: dict[str, asyncio.Lock] = {}
        self._extra_configs: dict[str, dict[str, Any]] = {}

    def _resolve_server_bin(self) -> str:
        """Pick the ``whisper-server`` binary path.

        Priority:
        1. ``<backends_dir>/whisper_cpp/metadata.json`` -> ``extra_bins.server``
           (modular install via :class:`BackendInstaller`).
        2. ``settings.whispercpp_server_bin`` (legacy fat-image path).

        There was a higher-priority ``whispercpp_custom_server_bin`` setting,
        mirroring the llama.cpp one. Removed unused (2026-11-06): it was never
        set anywhere — not in ``.env``, not in ``.env.example``, and the live
        process read ``None`` — while whisper.cpp, unlike llama.cpp, has no
        Makespace fork that would have justified it. A custom build belongs in
        its own backend with its own install_spec, the way ``llamacpp_mks``
        does it, rather than in a second silent way to swap the binary.
        """
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

    def _resolve_diarization_python_bin(self) -> str:
        """pyannote.audio lives in the ``whisper`` backend's own venv (it's
        already a declared dependency there, see whisper_backend.py's
        install_spec) — reuse it instead of installing a second copy just
        for whisper.cpp's diarization sidecar."""
        meta = read_backend_metadata(settings.backends_dir, "whisper")
        python_bin = meta.get("python_bin") if isinstance(meta, dict) else None
        if isinstance(python_bin, str) and python_bin and Path(python_bin).is_file():
            return python_bin
        raise FileNotFoundError(
            "whisper_cpp diarization requires the 'whisper' backend to be installed "
            "(pyannote.audio lives in its venv) — install it from the /backends page."
        )

    async def _ensure_diarization_worker(
        self, model_id: str, extra_config: dict[str, Any]
    ) -> _DiarizationSidecar:
        existing = self._diarization_workers.get(model_id)
        if existing and existing.process.returncode is None:
            return existing

        lock = self._diarization_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            # Re-check after acquiring the lock — a concurrent caller may have
            # already finished launching it while we were waiting.
            existing = self._diarization_workers.get(model_id)
            if existing and existing.process.returncode is None:
                return existing

            cfg = extra_config.get("whisper_cpp") if isinstance(extra_config, dict) else None
            cfg = cfg if isinstance(cfg, dict) else {}
            diarization_model_id = str(
                cfg.get("diarization_model_id") or _DEFAULT_DIARIZATION_MODEL_ID
            )
            python_bin = self._resolve_diarization_python_bin()
            port = _reserve_free_port()

            env = os.environ.copy()
            if settings.hf_token and not env.get("HF_TOKEN"):
                env["HF_TOKEN"] = settings.hf_token
            env["CUDA_VISIBLE_DEVICES"] = ""

            args = [
                python_bin,
                str(_DIARIZATION_WORKER_PATH),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--diarization-model-id",
                diarization_model_id,
                "--device",
                "cpu",
            ]

            log_path = _diarization_log_path(model_id)
            log_file = open(log_path, "ab")  # noqa: ASYNC230, SIM115 — sidecar-lifetime handle

            logger.info("whispercpp_diarization_starting", model_id=model_id, port=port)
            process = await asyncio.create_subprocess_exec(
                *args,
                env=env,
                stdout=log_file,
                stderr=log_file,
            )
            sidecar = _DiarizationSidecar(process=process, port=port, log_file=log_file)
            self._diarization_workers[model_id] = sidecar

            timeout_s = int(settings.whispercpp_diarization_startup_timeout_s)
            deadline = asyncio.get_running_loop().time() + timeout_s
            url = f"http://127.0.0.1:{port}/health"
            healthy = False
            while asyncio.get_running_loop().time() < deadline:
                if process.returncode is not None:
                    break
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        response = await client.get(url)
                    if response.status_code == 200:
                        healthy = True
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.5)

            if not healthy:
                self._diarization_workers.pop(model_id, None)
                log_tail = self._read_log_tail(_diarization_log_path(model_id))
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
                finally:
                    log_file.close()
                raise RuntimeError(
                    f"whisper.cpp diarization sidecar for '{model_id}' did not become "
                    f"healthy within {timeout_s}s. log_tail={log_tail}"
                )

            logger.info("whispercpp_diarization_started", model_id=model_id, port=port)
            return sidecar

    async def _kill_diarization_worker(self, model_id: str) -> None:
        sidecar = self._diarization_workers.pop(model_id, None)
        if not sidecar:
            return
        process = sidecar.process
        try:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        finally:
            sidecar.log_file.close()

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
        self._extra_configs[model_id] = extra_config
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
        # Decode every input with ffmpeg instead of whisper.cpp's built-in
        # miniaudio. Upstream tries miniaudio first and only falls back to
        # ffmpeg when miniaudio *fails to open* the file — never when it opens
        # it and reads it wrong. That leaves this backend decoding the same
        # bytes with a different library than the GPU one (faster-whisper,
        # which decodes through ffmpeg), so the two can disagree about how
        # much audio a file contains with no error anywhere; the response's
        # ``duration`` is literally decoded samples / 16 kHz, so a short read
        # looks like a short clip. Making ffmpeg the only decoder removes that
        # class of divergence. ffmpeg ships in the image (/usr/bin/ffmpeg) and
        # the build has WHISPER_COMMON_FFMPEG, which is what ``--convert``
        # relies on.
        env["WHISPER_COMMON_MINIAUDIO_SKIP"] = "1"

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
            # ``whisper-server`` defaults this to "en", and with an "en" target
            # Whisper *translates* what it hears instead of transcribing it: a
            # Spanish clip comes back in English, every word counted as an
            # error. faster-whisper (the GPU backend) auto-detects, so the same
            # request gave one language or another depending on which worker
            # served it, with no error anywhere — and ``recomendado-stt`` falls
            # back to this worker when the cards are busy, so it reached real
            # users. Reproduced 2026-11-08 with a 3,8 s clip: "El informe sobre
            # el presupuesto europeo se aprobó ayer por la tarde" came back as
            # "The report on European budget was approved yesterday".
            #
            # It goes on the command line, not in the per-request form, because
            # plain transcriptions never pass through this backend's
            # ``forward_request`` — the audio endpoint posts them to the worker
            # directly and only routes diarized ones through here. A caller's
            # own ``language`` still overrides this per request.
            "--language",
            "auto",
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
                timeout_s=int(
                    cfg.get("startup_timeout_s") or settings.whispercpp_startup_timeout_s
                ),
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
        await self._kill_diarization_worker(model_id)
        self._extra_configs.pop(model_id, None)
        self._diarization_locks.pop(model_id, None)

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
        requested_format = str(body.get("response_format", "json")).lower()
        diarize_requested = _is_diarize_requested(body.get("diarize"))

        # Diarizing needs whisper.cpp's own segments (with word timestamps) to
        # merge speaker labels into, regardless of what the caller actually
        # asked for — the real requested_format is applied after merging.
        transcribe_body = {**body, "response_format": "verbose_json"} if diarize_requested else body
        files, form_data = _build_transcription_multipart(transcribe_body)

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{entry.port}{endpoint}",
                files=files,
                data=form_data,
            )
        response.raise_for_status()

        if not diarize_requested:
            if requested_format in {"json", "verbose_json"}:
                return response.json()
            return response.text

        result = response.json()
        sidecar = await self._ensure_diarization_worker(
            model_id, self._extra_configs.get(model_id) or {}
        )
        file_name, file_bytes, content_type = _extract_audio_file(body)
        diar_files = {"file": (file_name, file_bytes, content_type)}
        diar_data = {
            "segments_json": json.dumps(result.get("segments", [])),
            "response_format": requested_format,
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            diar_response = await client.post(
                f"http://127.0.0.1:{sidecar.port}/diarize",
                files=diar_files,
                data=diar_data,
            )
        diar_response.raise_for_status()
        if requested_format in {"json", "verbose_json"}:
            return diar_response.json()
        return diar_response.text

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
                log_tail = self._read_log_tail(_worker_log_path(model_id))
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
                    log_tail=self._read_log_tail(_worker_log_path(model_id)),
                )

        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass
        finally:
            entry.log_file.close()

    def _read_log_tail(self, path: str, limit: int = 4000) -> str:
        try:
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
    # No default here on purpose: the worker is launched with
    # ``--language auto`` (see ``load``), which covers every caller — this
    # function only runs for diarized requests, the plain ones are posted to
    # the worker straight from the audio endpoint.
    if body.get("language"):
        form_data["language"] = str(body["language"])
    if body.get("response_format"):
        form_data["response_format"] = str(body["response_format"])

    granularities = body.get("timestamp_granularities") or body.get("timestamp_granularities[]")
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
