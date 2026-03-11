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
                models.append(
                    LocalModel(
                        model_ref=path.name,
                        path=str(path),
                        source="huggingface",
                        backend_type="vllm",
                        size_gb=size,
                    )
                )
                continue

            if path.is_file() and path.suffix.lower() == ".gguf":
                models.append(
                    LocalModel(
                        model_ref=path.stem,
                        path=str(path),
                        source="gguf",
                        backend_type="vllm",
                        size_gb=path.stat().st_size / (1024**3),
                    )
                )
                continue

            if path.is_file() and path.name == "Modelfile":
                size = self._dir_size(path.parent)
                models.append(
                    LocalModel(
                        model_ref=path.parent.name,
                        path=str(path.parent),
                        source="ollama",
                        backend_type="ollama",
                        size_gb=size,
                    )
                )

        unique: dict[str, LocalModel] = {model.path: model for model in models}
        return sorted(unique.values(), key=lambda m: m.model_ref)

    def _dir_size(self, root: Path) -> float | None:
        total = 0
        for child in root.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total / (1024**3) if total > 0 else None
