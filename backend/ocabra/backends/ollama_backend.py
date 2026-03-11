from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

from ocabra.backends.base import BackendCapabilities, BackendInterface, WorkerInfo
from ocabra.config import settings
from ocabra.registry.ollama_registry import OllamaRegistry


class OllamaBackend(BackendInterface):
    def __init__(self) -> None:
        self._registry = OllamaRegistry()
        self._loaded: set[str] = set()

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        _ = kwargs
        await self._registry.load(model_id)
        self._loaded.add(model_id)

        parsed = urlparse(settings.ollama_base_url)
        port = int(parsed.port or 11434)
        return WorkerInfo(
            backend_type="ollama",
            model_id=model_id,
            gpu_indices=[],
            port=port,
            pid=0,
            vram_used_mb=0,
        )

    async def unload(self, model_id: str) -> None:
        await self._registry.unload(model_id)
        self._loaded.discard(model_id)

    async def health_check(self, model_id: str) -> bool:
        loaded = await self._registry.list_loaded()
        return model_id in loaded

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        model = model_id.lower()
        embeds = "embed" in model or "nomic-embed" in model or "mxbai-embed" in model
        vision = "llava" in model or "vision" in model or "vl" in model
        return BackendCapabilities(
            chat=not embeds,
            completion=not embeds,
            embeddings=embeds,
            vision=vision,
            streaming=True,
            context_length=8192,
        )

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        _ = model_id
        return 0

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        raise RuntimeError(f"Ollama backend does not support direct forwarding on path '{path}'")

    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        _ = model_id, path, body
        if False:
            yield b""
