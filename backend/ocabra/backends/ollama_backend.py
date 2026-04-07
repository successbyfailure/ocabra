from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from ocabra.backends.base import BackendCapabilities, BackendInterface, ModalityType, WorkerInfo
from ocabra.config import settings
from ocabra.registry.ollama_registry import OllamaRegistry

logger = structlog.get_logger(__name__)


class OllamaBackend(BackendInterface):

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.TEXT_GENERATION, ModalityType.EMBEDDINGS}
    def __init__(self) -> None:
        self._registry = OllamaRegistry()
        self._loaded: set[str] = set()
        self._caps_cache: dict[str, BackendCapabilities] = {}

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
        cached = self._caps_cache.get(model_id)
        if cached is not None:
            return cached

        model = model_id.lower()
        embeds = "embed" in model or "nomic-embed" in model or "mxbai-embed" in model
        vision = "llava" in model or "vision" in model or "vl" in model
        tools = False
        context_length = 8192

        # Query Ollama /api/show for accurate capabilities
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{settings.ollama_base_url}/api/show",
                    json={"name": model_id},
                )
                if r.status_code == 200:
                    data = r.json()
                    template = data.get("template", "")
                    # Tool support: check template markers OR known model families
                    tools = "{{.Tools}}" in template or ".ToolCalls" in template
                    if not tools:
                        family = data.get("details", {}).get("family", "")
                        # Modern model families support tools natively in Ollama
                        # even without template markers
                        _tool_families = {
                            "gemma4", "gemma3", "gemma2",
                            "qwen3", "qwen2.5", "qwen2",
                            "llama4", "llama3.3", "llama3.2", "llama3.1", "llama3",
                            "mistral", "mixtral", "ministral",
                            "phi4", "phi3.5", "phi3",
                            "command-r",
                            "deepseek", "deepseek2",
                            "nemotron",
                            "devstral",
                            "glm-4",
                        }
                        tools = family.lower() in _tool_families
                    # Detect vision from template or family
                    family = data.get("details", {}).get("family", "")
                    if "vl" in family or "vision" in family or "llava" in family:
                        vision = True
                    # Extract context length from model_info or parameters
                    model_info = data.get("model_info", {})
                    for key, val in model_info.items():
                        if "context_length" in key and isinstance(val, (int, float)):
                            context_length = int(val)
                            break
        except Exception as exc:
            logger.debug("ollama_show_failed", model_id=model_id, error=str(exc))

        caps = BackendCapabilities(
            chat=not embeds,
            completion=not embeds,
            embeddings=embeds,
            vision=vision,
            tools=tools,
            streaming=True,
            context_length=context_length,
        )
        self._caps_cache[model_id] = caps
        return caps

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
