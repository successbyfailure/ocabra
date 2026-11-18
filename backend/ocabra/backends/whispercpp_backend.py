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
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.backends._process_group import terminate_process_group
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


def _parse_optional_bool(raw: Any) -> bool | None:
    """OpenAI/faster-whisper style boolean form value; None if absent/unknown."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


# Logged once per process: VAD needs a Silero GGML model that is not installed.
_vad_unavailable_logged = False


def whisper_models_dirs() -> list[Path]:
    """Where whisper.cpp GGML models (and the Silero VAD one) live, in order.

    ``<models_dir>/whisper_cpp`` first. The old place, ``<backends_dir>/
    whisper_cpp/models``, is inside the backend's own folder, which a reinstall
    or an OCI update replaces wholesale: the models were lost that way on
    2026-11-17. It is still searched so nodes not migrated keep working.
    """
    return [
        Path(settings.models_dir) / "whisper_cpp",
        Path(settings.backends_dir) / "whisper_cpp" / "models",
    ]


def _log_vad_unavailable_once(model_id: str, models_dir: Path) -> None:
    global _vad_unavailable_logged
    if _vad_unavailable_logged:
        return
    _vad_unavailable_logged = True
    logger.warning(
        "whispercpp_vad_unavailable",
        model_id=model_id,
        detail=(
            "no Silero VAD GGML model found (expected ggml-silero-*.bin under "
            f"{models_dir} or whisper_cpp.vad_model in extra_config); "
            "vad_filter is accepted but has no effect on whisper.cpp"
        ),
    )


def transcription_option_fields(body: dict, *, vad_available: bool) -> dict[str, str]:
    """Translate faster-whisper style options into ``whisper-server`` fields.

    - ``condition_on_previous_text=false`` → ``max_context=0``: whisper.cpp
      feeds the text already decoded as prompt for each next 30 s window,
      capped at ``max_context`` tokens; 0 turns that off. ``true`` is the
      server's default, nothing to send. The initial ``prompt`` shares that
      budget (0 would drop it too), so with a ``prompt`` the prompt wins and
      nothing is sent. ``whisper-server``'s ``no_context`` is not a request
      field and is always on — it only clears context *between* requests.
    - ``vad_filter`` → ``vad``: only when the worker was launched with a VAD
      model (``--vad-model``); asking whisper.cpp for VAD without one fails
      the whole transcription, so with no model ``true`` is dropped.

    Absent or unparseable values send nothing, leaving the launch defaults
    (VAD on when a VAD model is installed, like faster-whisper's default).
    """
    fields: dict[str, str] = {}
    condition = _parse_optional_bool(body.get("condition_on_previous_text"))
    if condition is False and not body.get("prompt"):
        fields["max_context"] = "0"
    vad = _parse_optional_bool(body.get("vad_filter"))
    if vad is False:
        fields["vad"] = "false"
    elif vad is True and vad_available:
        fields["vad"] = "true"
    return fields


def _is_diarize_requested(raw: Any) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() not in {"", "0", "false", "no"}


@dataclass
class _WhisperCppWorker:
    process: asyncio.subprocess.Process
    port: int
    log_file: Any
    vad_available: bool = False


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
            # A path recorded before the models moved out of the backend
            # folder: the same file in the current models folder.
            for models_dir in whisper_models_dirs():
                moved = models_dir / path.name
                if moved.is_file():
                    return moved

        direct = Path(model_id)
        if direct.is_file():
            return direct

        stem = Path(model_id).name
        dirs = whisper_models_dirs()
        for models_dir in dirs:
            for candidate in (
                models_dir / model_id,
                models_dir / f"{stem}.bin",
                models_dir / f"ggml-{stem}.bin",
            ):
                if candidate.is_file():
                    return candidate

        for models_dir in dirs:
            try:
                matches = [
                    path
                    for path in models_dir.glob("*.bin")
                    if stem.lower() in path.name.lower() and "silero" not in path.name.lower()
                ]
            except OSError:
                continue
            if matches:
                return max(matches, key=lambda path: path.stat().st_mtime)

        raise FileNotFoundError(
            f"whisper.cpp GGML model not found for '{model_id}' under "
            + " or ".join(str(d) for d in dirs)
        )

    def _resolve_vad_model(self, extra_config: dict[str, Any]) -> Path | None:
        """Silero VAD GGML model for ``--vad-model``, or None if not installed.

        ``whisper_cpp.vad_model`` in extra_config wins; otherwise the newest
        ``ggml-silero*.bin`` next to the whisper models. Nothing is downloaded.
        """
        cfg = extra_config.get("whisper_cpp") if isinstance(extra_config, dict) else None
        explicit = cfg.get("vad_model") if isinstance(cfg, dict) else None
        if explicit:
            path = Path(str(explicit))
            return path if path.is_file() else None
        matches: list[Path] = []
        for models_dir in whisper_models_dirs():
            try:
                matches += [p for p in models_dir.glob("ggml-silero*.bin") if p.is_file()]
            except OSError:
                continue
        if not matches:
            return None
        return max(matches, key=lambda path: path.stat().st_mtime)

    def transcription_option_fields(self, model_id: str, body: dict) -> dict[str, str]:
        """``whisper-server`` form fields for ``vad_filter`` and
        ``condition_on_previous_text`` in ``body`` (see the module function).

        For callers that post to the worker directly instead of through
        :meth:`forward_request` (the plain, non-diarized transcriptions).
        """
        entry = self._processes.get(model_id)
        return transcription_option_fields(
            body, vad_available=bool(getattr(entry, "vad_available", False))
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
                # Own process group, stopped as a whole (see _process_group).
                start_new_session=True,
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
                    await terminate_process_group(process, grace_s=0, graceful=False)
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
        try:
            await terminate_process_group(sidecar.process, grace_s=10.0)
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
            # Where ``--convert`` writes ffmpeg's WAV. The default is ".", the
            # worker's cwd (/app): not writable since the api runs as uid 1000
            # (v0.106.2), so every transcription failed with "FFmpeg
            # conversion failed" (found 2026-11-18 testing VAD).
            "--tmp-dir",
            tempfile.gettempdir(),
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
            # No ``--max-context 0`` here even though faster-whisper defaults
            # to condition_on_previous_text=false: in whisper.cpp the client's
            # ``prompt`` travels through the same context budget, so a launch
            # default of 0 would silently drop every prompt. Per request only,
            # see ``transcription_option_fields``.
        ]
        # Same default as faster-whisper (vad_filter=true), but whisper.cpp
        # needs a Silero GGML model for it — without one, VAD stays off and
        # ``vad_filter`` is accepted and ignored.
        # ``whisper_cpp.vad_filter: false`` in extra_config keeps the model
        # loaded (requests can still ask for VAD) but off by default.
        vad_model = self._resolve_vad_model(extra_config)
        if vad_model is not None:
            args += ["--vad-model", str(vad_model)]
            if cfg.get("vad_filter", True) is not False:
                args.append("--vad")
        else:
            _log_vad_unavailable_once(model_id, whisper_models_dirs()[0])

        log_path = _worker_log_path(model_id)
        log_file = open(log_path, "ab")  # noqa: ASYNC230, SIM115 — worker-lifetime handle

        logger.info("whispercpp_starting", model_id=model_id, port=port, threads=threads)
        process = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=log_file,
            stderr=log_file,
            # Own process group: whisper-server's ``--convert`` runs ffmpeg as
            # a child, and stopping only the server could leave it behind.
            start_new_session=True,
        )
        self._processes[model_id] = _WhisperCppWorker(
            process=process,
            port=port,
            log_file=log_file,
            vad_available=vad_model is not None,
        )

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
        files, form_data = _build_transcription_multipart(
            transcribe_body, vad_available=bool(getattr(entry, "vad_available", False))
        )

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
        try:
            await terminate_process_group(entry.process, grace_s=20.0, graceful=graceful)
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
    *,
    vad_available: bool = False,
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

    form_data.update(transcription_option_fields(body, vad_available=vad_available))

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
