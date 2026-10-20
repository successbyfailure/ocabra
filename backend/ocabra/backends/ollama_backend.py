import json
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
        self._registries: dict[str, OllamaRegistry] = {
            settings.ollama_base_url.rstrip("/"): self._registry
        }
        self._model_endpoints: dict[str, str] = {}
        self._loaded: set[str] = set()
        self._caps_cache: dict[str, BackendCapabilities] = {}
        # Real size_vram observed via /api/ps after a successful load, keyed by
        # model_id. Reused by get_vram_estimate_mb so the scheduler sees Ollama
        # models' actual footprint (context-size KV cache included) on repeat
        # loads, instead of always reporting 0 — see get_vram_estimate_mb.
        self._observed_vram_mb: dict[str, int] = {}

    @staticmethod
    def _configured_pools() -> list[dict[str, Any]]:
        raw = settings.ollama_pools_json.strip()
        if not raw:
            return []
        try:
            pools = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("OLLAMA_POOLS_JSON is not valid JSON") from exc
        if not isinstance(pools, list):
            raise RuntimeError("OLLAMA_POOLS_JSON must be a JSON array")
        return [pool for pool in pools if isinstance(pool, dict)]

    def _select_pool(
        self, gpu_indices: list[int], extra_config: dict[str, Any] | None
    ) -> tuple[str, list[int]]:
        pools = self._configured_pools()
        if not pools:
            return settings.ollama_base_url.rstrip("/"), []
        runtime = extra_config.get("_runtime", {}) if isinstance(extra_config, dict) else {}
        required_parallelism = max(1, int(runtime.get("parallelism") or 1))
        eligible: list[tuple[str, list[int], bool]] = []
        for pool in pools:
            url = str(pool.get("url") or "").rstrip("/")
            gpus = [int(index) for index in pool.get("gpus", [])]
            parallelism = max(1, int(pool.get("num_parallel") or 1))
            multi_gpu = bool(pool.get("multi_gpu"))
            if url and parallelism >= required_parallelism:
                eligible.append((url, gpus, multi_gpu))
        # Multi-GPU preset: necesita un pool que exponga TODAS las GPUs pedidas
        # a la vez (p. ej. gemma4:26b max → preferred_gpus=[0,1] + pool shared).
        # Preferimos el pool más pequeño que cumpla para no monopolizar el
        # shared cuando un single-GPU bastaría. La hint canónica es
        # ``_runtime.gpu_indices`` del coordinator; ``gpu_indices`` entrante
        # suele venir con 1 GPU del admit incluso para presets multi-GPU.
        runtime_gpu_indices = runtime.get("gpu_indices") or []
        effective_multi = (
            [int(idx) for idx in runtime_gpu_indices]
            if len(runtime_gpu_indices) > 1
            else gpu_indices if len(gpu_indices) > 1 else []
        )
        if effective_multi:
            required = set(effective_multi)
            matches = [
                (url, gpus) for url, gpus, multi in eligible
                if multi and required.issubset(set(gpus))
            ]
            if not matches:
                raise RuntimeError(
                    f"no Ollama pool exposes GPUs {sorted(required)} with multi_gpu=true "
                    f"and parallelism {required_parallelism}"
                )
            matches.sort(key=lambda item: len(item[1]))
            return matches[0]
        desired = gpu_indices[0] if len(gpu_indices) == 1 else None
        # Pool single-GPU preferido: evita elegir el pool shared cuando una
        # sola GPU basta (reserva al shared para presets multi-GPU reales).
        single_gpu_candidates = [
            (url, gpus) for url, gpus, multi in eligible if not multi
        ]
        if desired is not None:
            match = next(
                (item for item in single_gpu_candidates if desired in item[1]),
                None,
            )
            if match is None:
                # Fallback: cualquier pool (incluido shared) que incluya la
                # GPU pedida, por si no hay single-GPU pool configurado.
                match = next(
                    (
                        (url, gpus)
                        for url, gpus, _multi in eligible
                        if desired in gpus
                    ),
                    None,
                )
            if match is None:
                raise RuntimeError(
                    f"no Ollama pool for GPU {desired} with parallelism "
                    f"{required_parallelism}"
                )
            return match
        if single_gpu_candidates:
            return single_gpu_candidates[0]
        if not eligible:
            raise RuntimeError(
                f"no Ollama pool supports parallelism {required_parallelism}"
            )
        return (eligible[0][0], eligible[0][1])

    def _registry_for(self, endpoint: str) -> OllamaRegistry:
        return self._registries.setdefault(endpoint, OllamaRegistry(endpoint))

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
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
        extra_config = kwargs.get("extra_config")
        endpoint, actual_gpus = self._select_pool(gpu_indices, extra_config)
        registry = self._registry_for(endpoint)
        ollama_config = (
            extra_config.get("ollama", {}) if isinstance(extra_config, dict) else {}
        )
        options: dict[str, object] = {}
        num_ctx = ollama_config.get("num_ctx") if isinstance(ollama_config, dict) else None
        if isinstance(num_ctx, int) and num_ctx > 0:
            options["num_ctx"] = num_ctx
        await registry.load(
            model_id,
            keep_alive=keep_alive,
            options=options or None,
        )
        self._loaded.add(model_id)

        # Real footprint (weights + KV cache for this model's configured
        # context) straight from Ollama, so the scheduler can actually reason
        # about this model's VRAM usage instead of treating every Ollama load
        # as free. Best-effort: a slow/failed /api/ps here must never fail an
        # otherwise-successful load.
        vram_used_mb = 0
        try:
            for detail in await registry.list_loaded_details():
                if detail["name"] == model_id or detail["model"] == model_id:
                    vram_used_mb = int(detail["size_vram"] / (1024 * 1024))
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning("ollama_vram_probe_failed", model_id=model_id, error=str(exc))
        if vram_used_mb > 0:
            self._observed_vram_mb[model_id] = vram_used_mb

        self._model_endpoints[model_id] = endpoint
        parsed = urlparse(endpoint)
        port = int(parsed.port or 11434)
        return WorkerInfo(
            backend_type="ollama",
            model_id=model_id,
            gpu_indices=actual_gpus,
            port=port,
            pid=0,
            vram_used_mb=vram_used_mb,
            endpoint_url=endpoint,
        )

    async def unload(self, model_id: str, *, endpoint_url: str | None = None) -> None:
        endpoint = endpoint_url or self._model_endpoints.get(
            model_id, settings.ollama_base_url.rstrip("/")
        )
        await self._registry_for(endpoint).unload(model_id)
        if self._model_endpoints.get(model_id) == endpoint:
            self._model_endpoints.pop(model_id, None)
        self._loaded.discard(model_id)

    async def health_check(self, model_id: str) -> bool:
        endpoint = self._model_endpoints.get(
            model_id, settings.ollama_base_url.rstrip("/")
        )
        loaded = await self._registry_for(endpoint).list_loaded()
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
        native_context_length = 0

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
                    native_context_length = native_ctx
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
            native_context_length=native_context_length,
        )
        self._caps_cache[model_id] = caps
        return caps

    # Disk size undercounts real VRAM for large-context models — the KV cache
    # for a big num_ctx can add several extra GB on top of the raw weights.
    # Applied only as a fallback, before we've ever seen this model actually
    # loaded (see _observed_vram_mb).
    _DISK_SIZE_VRAM_MULTIPLIER = 1.3

    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        """Best-effort VRAM estimate so the scheduler can reason about Ollama
        loads the same way it does vLLM/diffusers ones (pressure-eviction,
        wait-for-service-gpu, ...). Previously always returned 0, which made
        every Ollama load look "free" regardless of what else was resident —
        the scheduler approved it instantly and the actual contention only
        surfaced later as a raw httpx ReadTimeout hitting Ollama's own load
        endpoint (reproduced 2026-09-29: cascading model-swap timeouts while
        a different Ollama model was mid-generation).

        Prefers the real size_vram observed the last time this exact model
        was loaded (accounts for this model's configured context size); falls
        back to on-disk blob size with a safety margin when never observed.
        """
        _ = extra_config
        observed = self._observed_vram_mb.get(model_id)
        if observed:
            return observed
        try:
            for detail in await self._registry.list_installed_details():
                if detail["name"] == model_id or detail["model"] == model_id:
                    size_mb = detail["size"] / (1024 * 1024)
                    return int(size_mb * self._DISK_SIZE_VRAM_MULTIPLIER)
        except Exception as exc:  # noqa: BLE001 — never block scheduling on this
            logger.warning("ollama_vram_estimate_failed", model_id=model_id, error=str(exc))
        return 0

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        raise RuntimeError(f"Ollama backend does not support direct forwarding on path '{path}'")

    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        _ = model_id, path, body
        if False:
            yield b""
