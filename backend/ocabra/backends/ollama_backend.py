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
        _ = gpu_indices
        # Honour the oCabra load_policy when telling Ollama how long to keep
        # the weights in memory. For warm/pin models we pass keep_alive=-1
        # ("never evict") so Ollama doesn't silently drop the weights after
        # its global ``OLLAMA_KEEP_ALIVE`` window — that would turn what
        # oCabra still considers a LOADED state into a cold start on the
        # next request. on_demand falls back to the default keep_alive.
        load_policy = kwargs.get("load_policy")
        keep_alive: int | str | None = None
        if load_policy in ("warm", "pin"):
            keep_alive = -1
        await self._registry.load(model_id, keep_alive=keep_alive)
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
        # Heuristic fallbacks (used only when /api/show doesn't expose
        # ``capabilities``). Ollama 0.5+ overrides every flag below.
        embeds = "embed" in model or "nomic-embed" in model or "mxbai-embed" in model
        vision = "llava" in model or "vision" in model or "vl" in model
        tools = False
        reasoning = False
        audio_input = False
        video_input = False
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
                    # Ollama 0.5+ exposes a top-level ``capabilities`` array
                    # (e.g. ["completion", "vision", "audio", "tools",
                    # "thinking"]). When present it's the source of truth and
                    # supersedes every heuristic below.
                    advertised = data.get("capabilities") or []
                    if isinstance(advertised, list) and advertised:
                        adv_set = {str(c).lower() for c in advertised}
                        # Ollama vocabulary -> oCabra flags. Aliases ("embed"
                        # vs "embedding", "thinking" vs "reasoning") are
                        # accepted defensively.
                        if "completion" in adv_set:
                            embeds = embeds and "embedding" in adv_set
                        if "embedding" in adv_set or "embed" in adv_set:
                            embeds = True
                        vision = "vision" in adv_set or vision
                        tools = "tools" in adv_set
                        reasoning = "thinking" in adv_set or "reasoning" in adv_set
                        audio_input = "audio" in adv_set
                        video_input = "video" in adv_set
                    else:
                        template = data.get("template", "")
                        tools = "{{.Tools}}" in template or ".ToolCalls" in template
                        family = data.get("details", {}).get("family", "")
                        if not tools:
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
                        if "vl" in family or "vision" in family or "llava" in family:
                            vision = True
                    # Reasoning fallback: some models emit <think> blocks without
                    # advertising a "thinking" capability. The chat template is
                    # the authoritative signal (matches the llama_cpp path).
                    if not reasoning:
                        tmpl_l = str(data.get("template", "")).lower()
                        if any(m in tmpl_l for m in ("<think>", "reasoning_content")):
                            reasoning = True

                    # Report the EFFECTIVE served context, not the native one.
                    # Native comes from model_info; the actual window is the
                    # Modelfile ``num_ctx`` PARAMETER when baked (e.g. our
                    # ``-ctx*`` variants), else Ollama's default
                    # (OLLAMA_CONTEXT_LENGTH), capped at native. Reporting native
                    # here made clients send more context than the model serves
                    # → silent truncation (the whole -ctx-variant saga).
                    model_info = data.get("model_info", {})
                    native_ctx = 0
                    for key, val in model_info.items():
                        if "context_length" in key and isinstance(val, (int, float)):
                            native_ctx = int(val)
                            break
                    baked_ctx = None
                    params = data.get("parameters", "")
                    if isinstance(params, str):
                        for line in params.splitlines():
                            parts = line.split()
                            if len(parts) >= 2 and parts[0] == "num_ctx":
                                try:
                                    baked_ctx = int(parts[1])
                                except ValueError:
                                    pass
                    effective_ctx = baked_ctx or settings.ollama_default_num_ctx
                    if native_ctx > 0:
                        effective_ctx = min(effective_ctx, native_ctx)
                    context_length = effective_ctx
        except Exception as exc:
            logger.debug("ollama_show_failed", model_id=model_id, error=str(exc))

        # Embedding models never chat/tool-call/reason — the family heuristic
        # (e.g. qwen3-embedding matching the qwen3 tool family) would otherwise
        # advertise tools=True on a pure embedder.
        if embeds:
            tools = False
            reasoning = False
            vision = False

        caps = BackendCapabilities(
            chat=not embeds,
            completion=not embeds,
            embeddings=embeds,
            vision=vision,
            tools=tools,
            reasoning=reasoning,
            audio_input=audio_input,
            video_input=video_input,
            streaming=True,
            context_length=context_length,
        )
        self._caps_cache[model_id] = caps
        return caps

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
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
