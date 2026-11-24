import asyncio
import json
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.backends._health_http import health_client
from ocabra.backends._startup import worker_exit_error
from ocabra.backends.base import (
    BackendCapabilities,
    BackendInstallSpec,
    BackendInterface,
    ModalityType,
    WorkerInfo,
)
from ocabra.config import settings
from ocabra.core.backend_installer import venv_nvidia_ld_library_path

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


def _worker_log_path(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"/tmp/tts-worker-{safe}.log"


@dataclass
class _TTSWorker:
    process: asyncio.subprocess.Process
    info: WorkerInfo
    log_file: Any | None = None


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

        # Slim image needs the venv's CUDA libs on LD path (Deuda D14).
        nvidia_ld = venv_nvidia_ld_library_path(settings.backends_dir, "tts")
        if nvidia_ld:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{nvidia_ld}:{existing}" if existing else nvidia_ld

        log_path = _worker_log_path(model_id)
        # noqa SIM115/ASYNC230: fd real y de vida larga que consume
        # create_subprocess_exec; se cierra en unload().
        log_file = open(log_path, "ab")  # noqa: SIM115, ASYNC230

        try:
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
                # stdout/stderr a un fichero real, no a PIPE: nadie leia esas
                # tuberias durante el servicio normal, asi que un worker
                # suficientemente hablador llena el buffer del kernel, se bloquea
                # en su propio write() y se queda colgado mientras /health sigue
                # respondiendo (mismo fallo ya corregido en llama_cpp y sglang el
                # 2026-10-30; este backend se quedo sin arreglar).
                stdout=log_file,
                stderr=log_file,
            )
        except BaseException:
            log_file.close()
            raise

        info = WorkerInfo(
            backend_type="tts",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=process.pid,
            vram_used_mb=await self.get_vram_estimate_mb(model_id),
        )

        self._workers[model_id] = _TTSWorker(process=process, info=info, log_file=log_file)

        if not await self._wait_until_healthy(model_id):
            # Read before unload(): unload terminates a live worker too.
            exit_code = process.returncode
            await self.unload(model_id)
            if exit_code is not None:
                raise await asyncio.to_thread(
                    worker_exit_error, "TTS", model_id, exit_code, log_path
                )
            raise RuntimeError(f"TTS worker failed to start for model '{model_id}'")

        logger.info("tts_worker_started", model_id=model_id, port=port, pid=process.pid)
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
            # Close the log even when the worker had already died (BUG-46: the
            # early return for an exited process leaked the descriptor).
            if worker.log_file:
                with suppress(Exception):
                    worker.log_file.close()

    async def health_check(self, model_id: str) -> bool:
        worker = self._workers.get(model_id)
        if not worker or worker.process.returncode is not None:
            return False

        try:
            client = health_client()
            response = await client.get(f"http://127.0.0.1:{worker.info.port}/health", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(tts=True)

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        normalized = model_id.lower()
        for key, size_mb in KNOWN_VRAM_MB.items():
            if key in normalized:
                return size_mb
        return 4096

    async def _wait_until_healthy(self, model_id: str, retries: int = 60) -> bool:
        """Poll /health; return False early if the worker process has exited."""
        for _ in range(retries):
            worker = self._workers.get(model_id)
            if worker is None or worker.process.returncode is not None:
                return False
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
