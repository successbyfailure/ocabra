import asyncio
from pathlib import Path

from ocabra.schemas.registry import LocalModel


class LocalScanner:
    async def scan(self, models_dir: Path) -> list[LocalModel]:
        return await asyncio.to_thread(self._scan_sync, models_dir)

    def _scan_sync(self, models_dir: Path) -> list[LocalModel]:
        models: list[LocalModel] = []

        if not models_dir.exists():
            return models

        for path in models_dir.rglob("*"):
            if path.is_dir() and (path / "config.json").exists():
                size = self._dir_size(path)
                backend_type = self._detect_hf_backend(path)
                models.append(
                    LocalModel(
                        model_ref=path.name,
                        path=str(path),
                        source="huggingface",
                        backend_type=backend_type,
                        size_gb=size,
                    )
                )
                continue

            if path.is_dir() and (path / "Modelfile").exists():
                size = self._dir_size(path)
                models.append(
                    LocalModel(
                        model_ref=path.name,
                        path=str(path),
                        source="ollama",
                        backend_type="ollama",
                        size_gb=size,
                    )
                )
                continue

            if path.is_file() and path.suffix.lower() == ".gguf":
                backend_type = "bitnet" if self._is_bitnet_gguf(path) else "vllm"
                models.append(
                    LocalModel(
                        model_ref=path.stem,
                        path=str(path),
                        source="gguf",
                        backend_type=backend_type,
                        size_gb=path.stat().st_size / (1024**3),
                    )
                )
                continue

        unique: dict[str, LocalModel] = {model.path: model for model in models}
        return sorted(unique.values(), key=lambda m: m.model_ref)

    def _dir_size(self, root: Path) -> float | None:
        total = 0
        for child in root.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total / (1024**3) if total > 0 else None

    def _detect_hf_backend(self, path: Path) -> str:
        """Detect the appropriate backend for a HuggingFace model directory.

        Checks config.json and directory name for known model families
        (Chatterbox, Whisper, TTS, Diffusers) and returns the matching
        backend type.  Defaults to ``vllm``.
        """
        import json

        name_lower = path.name.lower()

        # Chatterbox TTS models
        if "chatterbox" in name_lower:
            return "chatterbox"

        # Whisper / faster-whisper models
        if "whisper" in name_lower:
            return "whisper"

        try:
            config = json.loads((path / "config.json").read_text())
        except Exception:
            config = {}

        model_type = str(config.get("model_type", "")).lower()

        if model_type == "whisper":
            return "whisper"

        # Diffusion models (e.g., Stable Diffusion, SDXL, Flux)
        if model_type in {"stable-diffusion", "sdxl"} or (path / "model_index.json").exists():
            return "diffusers"

        # TTS models (Kokoro, Bark, etc.)
        if "kokoro" in name_lower or "bark" in name_lower:
            return "tts"

        return "vllm"

    def _is_bitnet_gguf(self, path: Path) -> bool:
        name = path.name.lower()
        if "bitnet" in name or "i2_s" in name:
            return True

        # Best-effort header probe: avoid full file scan.
        try:
            with path.open("rb") as file:
                head = file.read(32768).lower()
        except OSError:
            return False
        return b"bitnet" in head or b"i2_s" in head
