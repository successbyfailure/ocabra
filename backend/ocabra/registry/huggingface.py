import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_models, model_info, snapshot_download

from ocabra.backends.vllm_recipes import get_vllm_recipe
from ocabra.config import settings
from ocabra.registry.vllm_runtime_probe import VLLMRuntimeProbeService
from ocabra.schemas.registry import (
    HFModelCard,
    HFModelDetail,
    HFModelVariant,
    HFVLLMRuntimeProbe,
    HFVLLMSupport,
)

_BACKEND_BY_TASK = {
    "text-generation": "vllm",
    "text2text-generation": "vllm",
    "image-text-to-text": "vllm",
    "feature-extraction": "vllm",
    "sentence-similarity": "vllm",
    "image-to-image": "diffusers",
    "text-to-image": "diffusers",
    "image-to-video": "diffusers",
    "text-to-video": "diffusers",
    "automatic-speech-recognition": "whisper",
    "text-to-speech": "tts",
}

_BACKEND_BY_LIBRARY = {
    "diffusers": "diffusers",
    "transformers": "vllm",
    "sentence-transformers": "vllm",
    "adapter-transformers": "vllm",
    "qwen-tts": "tts",
    "parler-tts": "tts",
    "bark": "tts",
    "coqui": "tts",
    "kokoro": "tts",
    "chatterbox": "chatterbox",
}

_UNSUPPORTED_TOKENIZER_CLASSES = {"tokenizersbackend", "tokenizersbackendfast"}
_POOLING_TASKS = {"feature-extraction", "sentence-similarity"}
_GENERATE_TASKS = {"text-generation", "text2text-generation", "image-text-to-text"}
_MULTIMODAL_TASKS = {"image-text-to-text"}
_NATIVE_VLLM_ARCHITECTURES = {
    "CohereForCausalLM",
    "DeepseekR1ForCausalLM",
    "DeepseekV3ForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "GPT2LMHeadModel",
    "GPTNeoXForCausalLM",
    "InternVLChatModel",
    "Llama4ForConditionalGeneration",
    "LlamaForCausalLM",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "MiniCPMV",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Phi3ForCausalLM",
    "Phi3SmallForCausalLM",
    "Phi3VForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen2VLForConditionalGeneration",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
}
_POOLING_ARCHITECTURES = {
    "BertModel",
    "DistilBertModel",
    "E5Model",
    "RobertaModel",
    "XLMRobertaModel",
}
_TRANSFORMERS_BACKEND_ARCHITECTURES = {
    "GraniteForCausalLM",
    "GlmForCausalLM",
    "JambaForCausalLM",
    "MiniMaxText01ForCausalLM",
    "OlmoForCausalLM",
}
_CHATY_REPO_HINTS = ("-instruct", " instruct", "-chat", " chat", "assistant", "tool", "function")
_TOOL_PARSER_HINTS = {
    "hermes": "hermes",
    "qwen3": "qwen3_json",
    "functiongemma": "gemma",
    "granite": "granite",
}
_REASONING_PARSER_HINTS = {
    "deepseek-r1": "deepseek_r1",
    "deepseek r1": "deepseek_r1",
    "qwen3": "qwen3",
}
_NEMO_STT_ARTIFACTS = {
    "nvidia/parakeet-tdt-0.6b-v3": "parakeet-tdt-0.6b-v3.nemo",
    "nvidia/canary-1b-v2": "canary-1b-v2.nemo",
}


@dataclass
class _CompatibilityAssessment:
    compatibility: str = "unknown"
    reason: str | None = None
    vllm_support: HFVLLMSupport | None = None

    @property
    def installable(self) -> bool:
        return self.compatibility != "unsupported"


class HuggingFaceRegistry:
    def __init__(self) -> None:
        self._runtime_probe = VLLMRuntimeProbeService()

    async def infer_backend_for_repo(self, repo_id: str, artifact: str | None = None) -> str:
        if artifact and artifact.lower().endswith(".gguf"):
            raise ValueError(
                "HF GGUF artifacts are not supported; use the Ollama registry flow instead"
            )
        info = await asyncio.to_thread(
            lambda: model_info(
                repo_id=repo_id, files_metadata=False, token=settings.hf_token or None
            )
        )
        siblings = [
            {"rfilename": s.rfilename, "size": getattr(s, "size", None)}
            for s in (info.siblings or [])
        ]
        return self._infer_backend(
            task=getattr(info, "pipeline_tag", None),
            siblings=siblings,
            tags=list(getattr(info, "tags", []) or []),
            library_name=getattr(info, "library_name", None),
            repo_id=repo_id,
        )

    async def search(self, query: str, task: str | None, limit: int) -> list[HFModelCard]:
        def _run() -> list[Any]:
            return list(
                list_models(
                    search=query or None,
                    pipeline_tag=task or None,
                    limit=limit,
                    token=settings.hf_token or None,
                )
            )

        models = await asyncio.to_thread(_run)
        cards: list[HFModelCard] = []
        for model in models:
            siblings = list(getattr(model, "siblings", []) or [])
            size_bytes = sum(
                int(getattr(s, "size", 0) or 0)
                for s in siblings
                if str(getattr(s, "rfilename", "")).endswith(".safetensors")
                or str(getattr(s, "rfilename", "")).endswith(".bin")
            )
            vllm_support = self._infer_vllm_support(
                task=getattr(model, "pipeline_tag", None),
                tags=list(getattr(model, "tags", []) or []),
                library_name=getattr(model, "library_name", None),
                repo_id=model.id,
                sibling_names=[str(getattr(s, "rfilename", "")) for s in siblings],
            )
            cards.append(
                HFModelCard(
                    repo_id=model.id,
                    model_name=model.id.split("/")[-1],
                    task=getattr(model, "pipeline_tag", None),
                    downloads=int(getattr(model, "downloads", 0) or 0),
                    likes=int(getattr(model, "likes", 0) or 0),
                    size_gb=(size_bytes / (1024**3)) if size_bytes > 0 else None,
                    tags=list(getattr(model, "tags", []) or []),
                    gated=bool(getattr(model, "gated", False)),
                    suggested_backend=self._infer_backend(
                        task=getattr(model, "pipeline_tag", None),
                        siblings=[
                            {
                                "rfilename": getattr(s, "rfilename", ""),
                                "size": getattr(s, "size", None),
                            }
                            for s in siblings
                        ],
                        tags=list(getattr(model, "tags", []) or []),
                        library_name=getattr(model, "library_name", None),
                        repo_id=model.id,
                    ),
                    compatibility=self._compatibility_from_vllm_support(vllm_support),
                    compatibility_reason=self._compatibility_reason_from_support(vllm_support),
                    vllm_support=vllm_support,
                    created_at=getattr(model, "created_at", None),
                    last_modified=getattr(model, "last_modified", None),
                )
            )
        return cards

    async def get_model_detail(self, repo_id: str) -> HFModelDetail:
        info = await asyncio.to_thread(
            lambda: model_info(
                repo_id=repo_id, files_metadata=True, token=settings.hf_token or None
            )
        )

        siblings = [
            {
                "rfilename": s.rfilename,
                "size": getattr(s, "size", None),
            }
            for s in (info.siblings or [])
        ]

        safetensors_bytes = sum(
            int(s.get("size") or 0)
            for s in siblings
            if str(s.get("rfilename", "")).endswith(".safetensors")
        )

        task = getattr(info, "pipeline_tag", None)
        suggested_backend = self._infer_backend(
            task=task,
            siblings=siblings,
            tags=list(getattr(info, "tags", []) or []),
            library_name=getattr(info, "library_name", None),
            repo_id=repo_id,
        )
        compatibility = await self._assess_repo_compatibility(
            repo_id=repo_id,
            backend_hint=suggested_backend,
            sibling_names=[str(s.get("rfilename", "")) for s in siblings],
            task=task,
            tags=list(getattr(info, "tags", []) or []),
            library_name=getattr(info, "library_name", None),
        )

        return HFModelDetail(
            repo_id=info.id,
            model_name=info.id.split("/")[-1],
            task=task,
            downloads=int(getattr(info, "downloads", 0) or 0),
            likes=int(getattr(info, "likes", 0) or 0),
            size_gb=(safetensors_bytes / (1024**3)) if safetensors_bytes > 0 else None,
            tags=list(getattr(info, "tags", []) or []),
            gated=bool(getattr(info, "gated", False)),
            siblings=siblings,
            readme_excerpt=None,
            suggested_backend=suggested_backend,
            estimated_vram_gb=(safetensors_bytes / (1024**3) * 1.3)
            if safetensors_bytes > 0
            else None,
            compatibility=compatibility.compatibility,
            compatibility_reason=compatibility.reason,
            vllm_support=compatibility.vllm_support,
        )

    async def get_variants(self, repo_id: str) -> list[HFModelVariant]:
        info = await asyncio.to_thread(
            lambda: model_info(
                repo_id=repo_id, files_metadata=True, token=settings.hf_token or None
            )
        )
        tags = list(getattr(info, "tags", []) or [])
        task = getattr(info, "pipeline_tag", None)
        library_name = getattr(info, "library_name", None)
        siblings = list(info.siblings or [])
        items = [{"name": s.rfilename, "size": int(getattr(s, "size", 0) or 0)} for s in siblings]
        names = [i["name"] for i in items]
        backend_hint = self._infer_backend(
            task=task,
            siblings=[{"rfilename": name} for name in names],
            tags=tags,
            library_name=library_name,
            repo_id=repo_id,
        )
        variants: list[HFModelVariant] = []
        compatibility = await self._assess_repo_compatibility(
            repo_id=repo_id,
            backend_hint=backend_hint,
            sibling_names=names,
            task=task,
            tags=tags,
            library_name=library_name,
        )

        nemo_variants = self._nemo_variants(repo_id=repo_id, items=items)
        if nemo_variants:
            return nemo_variants

        # Standard variant (download config + safetensors/bin tokenizer files).
        has_standard = any(
            n.endswith(".safetensors") or n.endswith(".bin")
            for n in names
        )
        if has_standard:
            total = sum(
                i["size"]
                for i in items
                if i["name"].endswith(".safetensors") or i["name"].endswith(".bin")
            )
            variants.append(
                HFModelVariant(
                    variant_id="standard",
                    label="standard (safetensors/bin)",
                    artifact=None,
                    size_gb=(total / (1024**3)) if total > 0 else None,
                    format="safetensors/bin",
                    quantization=None,
                    backend_type=backend_hint,
                    is_default=True,
                    installable=compatibility.installable,
                    compatibility=compatibility.compatibility,
                    compatibility_reason=compatibility.reason,
                    vllm_support=compatibility.vllm_support,
                )
            )

        if not variants:
            variants.append(
                HFModelVariant(
                    variant_id="default",
                    label="default",
                    artifact=None,
                    size_gb=None,
                    format="unknown",
                    quantization=None,
                    backend_type=backend_hint,
                    is_default=True,
                    installable=compatibility.installable,
                    compatibility=compatibility.compatibility,
                    compatibility_reason=compatibility.reason,
                    vllm_support=compatibility.vllm_support,
                )
            )

        return variants

    async def download(
        self,
        repo_id: str,
        target_dir: Path,
        progress_callback: Callable[[float, float | None], None],
        artifact: str | None = None,
    ) -> Path:
        if artifact and artifact.lower().endswith(".gguf"):
            raise ValueError(
                "HF GGUF artifacts are not supported; use the Ollama registry flow instead"
            )

        await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
        info = await asyncio.to_thread(
            lambda: model_info(
                repo_id=repo_id, files_metadata=False, token=settings.hf_token or None
            )
        )
        sibling_names = [s.rfilename for s in (info.siblings or [])]
        backend_hint = self._infer_backend(
            task=getattr(info, "pipeline_tag", None),
            siblings=[{"rfilename": name} for name in sibling_names],
            tags=list(getattr(info, "tags", []) or []),
            library_name=getattr(info, "library_name", None),
            repo_id=repo_id,
        )
        resolved_artifact = self._resolve_default_artifact(
            repo_id=repo_id,
            sibling_names=sibling_names,
            backend_hint=backend_hint,
            artifact=artifact,
        )
        if not self._has_supported_hf_payload(
            sibling_names,
            backend_hint,
            artifact=resolved_artifact,
        ):
            supported = self._supported_payload_hint(backend_hint=backend_hint)
            available_exts = sorted(
                {
                    Path(name).suffix.lower()
                    for name in sibling_names
                    if Path(name).suffix
                }
            )
            raise ValueError(
                "This Hugging Face repo does not expose supported native weights for oCabra "
                f"(backend_hint={backend_hint}, supported={supported}, "
                f"available_extensions={available_exts or ['<none>']})"
            )
        compatibility = await self._assess_repo_compatibility(
            repo_id=repo_id,
            backend_hint=backend_hint,
            sibling_names=sibling_names,
            task=getattr(info, "pipeline_tag", None),
            tags=list(getattr(info, "tags", []) or []),
            library_name=getattr(info, "library_name", None),
        )
        if not compatibility.installable:
            raise ValueError(
                compatibility.reason
                or "This Hugging Face repo is not compatible with the current vLLM stack"
            )
        allow_patterns = self._download_allow_patterns(
            sibling_names, backend_hint, artifact=resolved_artifact
        )

        from tqdm.auto import tqdm as _BaseTqdm

        _start = time.monotonic()

        class _ProgressTqdm(_BaseTqdm):
            def update(self, n=1):
                super().update(n)
                elapsed = max(time.monotonic() - _start, 1e-6)
                speed_mbs = (self.n / (1024 * 1024)) / elapsed if self.n else None
                pct = 0.0
                if self.total and self.total > 0:
                    pct = min(100.0, (self.n / self.total) * 100.0)
                progress_callback(pct, speed_mbs)

        def _run() -> str:
            return snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                token=settings.hf_token or None,
                cache_dir=settings.hf_cache_dir or None,
                allow_patterns=allow_patterns,
                tqdm_class=_ProgressTqdm,
            )

        downloaded = await asyncio.to_thread(_run)
        progress_callback(100.0, None)
        return Path(downloaded)

    def _infer_backend(
        self,
        task: str | None,
        siblings: list[dict],
        tags: list[str] | None = None,
        library_name: str | None = None,
        repo_id: str | None = None,
    ) -> str:
        normalized_task = (task or "").strip().lower()
        normalized_library = (library_name or "").strip().lower()
        normalized_tags = {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}

        if normalized_task in _BACKEND_BY_TASK:
            return _BACKEND_BY_TASK[normalized_task]

        if normalized_library in _BACKEND_BY_LIBRARY:
            return _BACKEND_BY_LIBRARY[normalized_library]

        if "diffusers" in normalized_tags:
            return "diffusers"
        if normalized_tags & {"text-to-speech", "tts"}:
            # Check for Chatterbox specifically before generic TTS
            repo_name_early = (repo_id or "").lower()
            if "chatterbox" in repo_name_early or normalized_library == "chatterbox":
                return "chatterbox"
            return "tts"
        if normalized_tags & {"automatic-speech-recognition", "asr", "whisper"}:
            return "whisper"

        if task in _BACKEND_BY_TASK:
            return _BACKEND_BY_TASK[task]

        names = [str(s.get("rfilename", "")).lower() for s in siblings]
        repo_name = (repo_id or "").lower()

        if self._looks_like_diffusers_repo(
            names, normalized_task, normalized_library, normalized_tags, repo_name
        ):
            return "diffusers"
        if "chatterbox" in repo_name or normalized_library == "chatterbox":
            return "chatterbox"
        if self._looks_like_tts_repo(repo_name, normalized_tags, normalized_library, names):
            return "tts"
        return "vllm"

    def _has_supported_hf_payload(
        self,
        names: list[str],
        backend_hint: str,
        artifact: str | None = None,
    ) -> bool:
        lowered = [name.lower() for name in names]
        if artifact:
            return artifact.lower() in lowered
        if backend_hint == "diffusers":
            return "model_index.json" in lowered
        if backend_hint == "tts":
            return any(name.endswith((".safetensors", ".bin")) for name in lowered)
        if backend_hint == "chatterbox":
            return any(name.endswith((".safetensors", ".bin")) for name in lowered)
        if backend_hint == "whisper":
            return any(name.endswith((".safetensors", ".bin", ".pt", ".nemo")) for name in lowered)
        return any(name.endswith((".safetensors", ".bin")) for name in lowered)

    def _looks_like_diffusers_repo(
        self,
        names: list[str],
        task: str,
        library_name: str,
        tags: set[str],
        repo_name: str,
    ) -> bool:
        if library_name == "diffusers" or "diffusers" in tags:
            return True
        if task in {"text-to-image", "image-to-image", "text-to-video", "image-to-video"}:
            return True
        if "model_index.json" not in names:
            return False

        diffusers_dirs = (
            "unet/",
            "vae/",
            "scheduler/",
            "text_encoder/",
            "text_encoder_2/",
            "transformer/",
        )
        diffusers_files = {"scheduler_config.json", "feature_extractor/preprocessor_config.json"}
        if any(name.startswith(diffusers_dirs) for name in names):
            return True
        if any(name in diffusers_files for name in names):
            return True
        if any(
            token in repo_name for token in ("stable-diffusion", "sdxl", "flux", "wan", "kolors")
        ):
            return True
        return False

    def _looks_like_tts_repo(
        self,
        repo_name: str,
        tags: set[str],
        library_name: str,
        names: list[str],
    ) -> bool:
        if library_name in {"qwen-tts", "parler-tts", "bark", "coqui", "kokoro"}:
            return True
        if tags & {"text-to-speech", "tts"}:
            return True
        if any(token in repo_name for token in ("tts", "parler", "kokoro", "bark")):
            return True
        return "model_index.json" in names and any("voice" in name for name in names)

    def _download_allow_patterns(
        self,
        siblings: list[str],
        backend_hint: str,
        artifact: str | None = None,
    ) -> list[str] | None:
        names = [name.lower() for name in siblings]
        if artifact:
            return [
                artifact,
                "*.json",
                "*.model",
                "*.txt",
                "tokenizer*",
                "vocab*",
                "merges*",
                "README*",
                "LICENSE*",
            ]

        if backend_hint == "diffusers" or any("model_index.json" in name for name in names):
            return None

        if backend_hint == "whisper":
            return [
                "*.nemo",
                "*.safetensors",
                "*.bin",
                "*.pt",
                "*.json",
                "*.model",
                "*.txt",
                "tokenizer*",
                "vocab*",
                "merges*",
                "README*",
                "LICENSE*",
            ]

        return [
            "*.safetensors",
            "*.bin",
            "*.json",
            "*.model",
            "*.tiktoken",
            "*.txt",
            "tokenizer*",
            "vocab*",
            "merges*",
            "README*",
            "LICENSE*",
        ]

    def _resolve_default_artifact(
        self,
        repo_id: str,
        sibling_names: list[str],
        backend_hint: str,
        artifact: str | None,
    ) -> str | None:
        if artifact:
            return artifact
        if backend_hint != "whisper":
            return None

        canonical = _NEMO_STT_ARTIFACTS.get(repo_id.lower())
        lowered_to_original = {name.lower(): name for name in sibling_names}
        if canonical and canonical.lower() in lowered_to_original:
            return lowered_to_original[canonical.lower()]

        nemo_candidates = sorted(
            name for name in sibling_names if name.lower().endswith(".nemo")
        )
        if nemo_candidates:
            return nemo_candidates[0]
        return None

    def _supported_payload_hint(self, backend_hint: str) -> str:
        if backend_hint == "diffusers":
            return "model_index.json"
        if backend_hint == "tts":
            return ".safetensors|.bin"
        if backend_hint == "chatterbox":
            return ".safetensors|.bin"
        if backend_hint == "whisper":
            return ".nemo|.safetensors|.bin|.pt"
        return ".safetensors|.bin"

    def _nemo_variants(self, repo_id: str, items: list[dict[str, Any]]) -> list[HFModelVariant]:
        nemo_items = [item for item in items if item["name"].lower().endswith(".nemo")]
        if not nemo_items:
            return []

        preferred = _NEMO_STT_ARTIFACTS.get(repo_id.lower())
        variants: list[HFModelVariant] = []
        for idx, item in enumerate(sorted(nemo_items, key=lambda entry: entry["name"].lower())):
            filename = item["name"]
            variant_id = Path(filename).stem.replace("/", "_")
            is_default = False
            if preferred and filename.lower() == preferred.lower():
                is_default = True
            elif not preferred and idx == 0:
                is_default = True
            variants.append(
                HFModelVariant(
                    variant_id=variant_id,
                    label=f"nemo ({Path(filename).name})",
                    artifact=filename,
                    size_gb=(item["size"] / (1024**3)) if item["size"] > 0 else None,
                    format="nemo",
                    quantization=None,
                    backend_type="whisper",
                    is_default=is_default,
                    installable=True,
                    compatibility="compatible",
                    compatibility_reason="NVIDIA NeMo speech artifact detected.",
                    vllm_support=None,
                )
            )
        return variants

    def _extract_quant_from_filename(self, filename: str) -> str | None:
        low = filename.lower()
        m = re.search(r"(q\d+_k_[msl]|q\d+_k|q\d_[01]|q8_0|iq\d(?:_[a-z0-9]+)?)", low)
        if m:
            return m.group(1).upper()
        if "bf16" in low:
            return "BF16"
        if "fp16" in low or "f16" in low:
            return "FP16"
        return None

    async def _assess_repo_compatibility(
        self,
        repo_id: str,
        backend_hint: str,
        sibling_names: list[str],
        task: str | None,
        tags: list[str] | None,
        library_name: str | None,
    ) -> _CompatibilityAssessment:
        if backend_hint != "vllm":
            return _CompatibilityAssessment(compatibility="compatible")

        lowered = {name.lower() for name in sibling_names}
        tokenizer_config = await self._load_repo_json(repo_id, "tokenizer_config.json", lowered)
        config = await self._load_repo_json(repo_id, "config.json", lowered)
        quantize_config = await self._load_repo_json(repo_id, "quantize_config.json", lowered)
        vllm_support = self._infer_vllm_support(
            task=task,
            tags=tags,
            library_name=library_name,
            repo_id=repo_id,
            sibling_names=sibling_names,
            config=config,
            tokenizer_config=tokenizer_config,
        )
        runtime_probe = await self._probe_vllm_runtime_support(
            repo_id=repo_id,
            config=config,
            tokenizer_config=tokenizer_config,
            support=vllm_support,
        )
        vllm_support.runtime_probe = runtime_probe
        if runtime_probe.recommended_model_impl is not None:
            vllm_support.model_impl = runtime_probe.recommended_model_impl
        if runtime_probe.recommended_runner is not None:
            vllm_support.runner = runtime_probe.recommended_runner
        if (
            runtime_probe.status == "needs_remote_code"
            and "trust_remote_code" not in vllm_support.required_overrides
        ):
            vllm_support.required_overrides = sorted(
                {*vllm_support.required_overrides, "trust_remote_code"}
            )
        override_by_probe_status = {
            "missing_chat_template": "chat_template",
            "missing_tool_parser": "tool_call_parser",
            "missing_reasoning_parser": "reasoning_parser",
            "needs_hf_overrides": "hf_overrides",
        }
        implied_override = override_by_probe_status.get(runtime_probe.status)
        if implied_override and implied_override not in vllm_support.required_overrides:
            vllm_support.required_overrides = sorted(
                {*vllm_support.required_overrides, implied_override}
            )

        tokenizer_class = str((tokenizer_config or {}).get("tokenizer_class") or "").strip()
        if tokenizer_class.lower() in _UNSUPPORTED_TOKENIZER_CLASSES:
            return _CompatibilityAssessment(
                compatibility="unsupported",
                reason=(
                    "Este repo declara tokenizer_class="
                    f"{tokenizer_class}, que la combinacion actual de transformers/vLLM no puede importar."
                ),
                vllm_support=vllm_support,
            )

        quant_method = self._extract_quant_method(config=config, quantize_config=quantize_config)
        if quant_method == "gptq":
            return _CompatibilityAssessment(
                compatibility="warning",
                reason="Repo GPTQ detectado. Puede funcionar, pero algunos GPTQ custom fallan en vLLM.",
                vllm_support=vllm_support,
            )

        if runtime_probe.status in {"unsupported_tokenizer", "unsupported_architecture"}:
            return _CompatibilityAssessment(
                compatibility="unsupported",
                reason=runtime_probe.reason,
                vllm_support=vllm_support,
            )
        if runtime_probe.status in {
            "needs_remote_code",
            "missing_chat_template",
            "missing_tool_parser",
            "missing_reasoning_parser",
            "needs_hf_overrides",
        }:
            return _CompatibilityAssessment(
                compatibility="warning",
                reason=runtime_probe.reason,
                vllm_support=vllm_support,
            )
        if vllm_support.classification == "transformers_backend":
            return _CompatibilityAssessment(
                compatibility="warning",
                reason="Compatibilidad prevista via Transformers backend; mas cobertura, menos rendimiento.",
                vllm_support=vllm_support,
            )
        if vllm_support.classification == "pooling":
            return _CompatibilityAssessment(
                compatibility="compatible",
                reason="Modelo de pooling/embeddings detectado. No se sirve como chat generativo.",
                vllm_support=vllm_support,
            )

        return _CompatibilityAssessment(
            compatibility=self._compatibility_from_vllm_support(vllm_support),
            reason=self._compatibility_reason_from_support(vllm_support) or runtime_probe.reason,
            vllm_support=vllm_support,
        )

    def _compatibility_from_vllm_support(self, support: HFVLLMSupport | None) -> str:
        if support is None:
            return "unknown"
        if support.classification == "unsupported":
            return "unsupported"
        if support.classification == "transformers_backend":
            return "warning"
        if support.classification in {"native_vllm", "pooling"}:
            return "compatible"
        return "unknown"

    def _compatibility_reason_from_support(self, support: HFVLLMSupport | None) -> str | None:
        if support is None:
            return None
        if support.classification == "transformers_backend":
            return "vLLM probablemente requiere Transformers backend para este repo."
        if support.classification == "pooling":
            return "vLLM debe arrancarse en modo pooling para este repo."
        if support.required_overrides:
            return "Requiere configuracion extra: " + ", ".join(support.required_overrides)
        return None

    def _infer_vllm_support(
        self,
        task: str | None,
        tags: list[str] | None,
        library_name: str | None,
        repo_id: str,
        sibling_names: list[str],
        config: dict[str, Any] | None = None,
        tokenizer_config: dict[str, Any] | None = None,
    ) -> HFVLLMSupport | None:
        backend_hint = self._infer_backend(
            task=task,
            siblings=[{"rfilename": name} for name in sibling_names],
            tags=tags,
            library_name=library_name,
            repo_id=repo_id,
        )
        if backend_hint != "vllm":
            return None

        normalized_task = (task or "").strip().lower()
        repo_hint = repo_id.strip().lower().replace("_", " ")
        archs = [
            str(arch) for arch in ((config or {}).get("architectures") or []) if str(arch).strip()
        ]
        task_mode = self._task_mode(normalized_task, archs)
        classification = "unknown"
        model_impl: str | None = None
        runner: str | None = None

        if normalized_task in _POOLING_TASKS or any(
            arch in _POOLING_ARCHITECTURES for arch in archs
        ):
            classification = "pooling"
            runner = "pooling"
            model_impl = "transformers"
        elif any(arch in _NATIVE_VLLM_ARCHITECTURES for arch in archs):
            classification = "native_vllm"
            runner = "generate"
            model_impl = "vllm"
        elif normalized_task in _GENERATE_TASKS:
            classification = "transformers_backend"
            runner = "generate"
            model_impl = "transformers"
        elif any(arch in _TRANSFORMERS_BACKEND_ARCHITECTURES for arch in archs):
            classification = "transformers_backend"
            runner = "generate"
            model_impl = "transformers"

        required_overrides: list[str] = []
        tokenizer_class = str((tokenizer_config or {}).get("tokenizer_class") or "").strip().lower()
        if tokenizer_class in _UNSUPPORTED_TOKENIZER_CLASSES:
            classification = "unsupported"
        if (
            not (tokenizer_config or {}).get("chat_template")
            and runner == "generate"
            and any(hint in repo_hint for hint in _CHATY_REPO_HINTS)
        ):
            required_overrides.append("chat_template")
        if any(key in repo_hint for key in _TOOL_PARSER_HINTS):
            required_overrides.append("tool_call_parser")
        if any(key in repo_hint for key in _REASONING_PARSER_HINTS):
            required_overrides.append("reasoning_parser")
        if str((config or {}).get("auto_map") or "").strip():
            required_overrides.append("trust_remote_code")
        if isinstance((config or {}).get("rope_scaling"), dict):
            required_overrides.append("hf_overrides")

        recipe = get_vllm_recipe(
            repo_id=repo_id,
            architectures=archs,
            tokenizer_config=tokenizer_config,
        )
        recipe_notes: list[str] = []
        suggested_config: dict[str, Any] = {}
        suggested_tuning: dict[str, Any] = {}
        recipe_id: str | None = None
        recipe_model_impl: str | None = None
        recipe_runner: str | None = None
        if recipe is not None:
            recipe_id = recipe.recipe_id
            recipe_notes = recipe.notes
            recipe_model_impl = recipe.model_impl
            recipe_runner = recipe.runner
            suggested_config = recipe.suggested_config
            suggested_tuning = recipe.suggested_tuning
            if recipe.model_impl is not None:
                model_impl = recipe.model_impl
            if recipe.runner is not None:
                runner = recipe.runner
            required_overrides.extend(recipe.required_overrides)

        label = {
            "native_vllm": "native",
            "transformers_backend": "transformers backend",
            "pooling": "pooling",
            "unsupported": "unsupported",
            "unknown": "unknown",
        }[classification]

        return HFVLLMSupport(
            classification=classification,
            label=label,
            model_impl=model_impl,
            runner=runner,
            task_mode=task_mode,
            required_overrides=sorted(set(required_overrides)),
            recipe_id=recipe_id,
            recipe_notes=recipe_notes,
            recipe_model_impl=recipe_model_impl,  # type: ignore[arg-type]
            recipe_runner=recipe_runner,  # type: ignore[arg-type]
            suggested_config=suggested_config,
            suggested_tuning=suggested_tuning,
        )

    def _task_mode(self, task: str, archs: list[str]) -> str | None:
        if task in _POOLING_TASKS or any(arch in _POOLING_ARCHITECTURES for arch in archs):
            if task in _MULTIMODAL_TASKS:
                return "multimodal_pooling"
            return "pooling"
        if task in _GENERATE_TASKS or archs:
            if task in _MULTIMODAL_TASKS or any("ConditionalGeneration" in arch for arch in archs):
                return "multimodal_generate"
            return "generate"
        return None

    async def _probe_vllm_runtime_support(
        self,
        repo_id: str,
        config: dict[str, Any] | None,
        tokenizer_config: dict[str, Any] | None,
        support: HFVLLMSupport | None,
    ) -> HFVLLMRuntimeProbe:
        cached = self._runtime_probe.get_cached(repo_id)
        if cached is not None:
            return cached

        if support is None:
            probe = HFVLLMRuntimeProbe(status="unknown")
            return self._runtime_probe.set_cached(repo_id, probe)

        tokenizer_class = str((tokenizer_config or {}).get("tokenizer_class") or "").strip().lower()
        if tokenizer_class in _UNSUPPORTED_TOKENIZER_CLASSES:
            probe = HFVLLMRuntimeProbe(
                status="unsupported_tokenizer",
                reason=f"Tokenizer no soportado por el stack actual: {tokenizer_class}.",
                recommended_model_impl=support.model_impl,  # type: ignore[arg-type]
                recommended_runner=support.runner,  # type: ignore[arg-type]
                tokenizer_load=False,
                config_load=True,
            )
            return self._runtime_probe.set_cached(repo_id, probe)

        runtime_probe = await self._runtime_probe.probe_runtime(repo_id, support)
        if runtime_probe.status != "unavailable":
            return runtime_probe

        return await self._probe_vllm_lightweight_support(
            repo_id=repo_id,
            config=config,
            tokenizer_config=tokenizer_config,
            support=support,
        )

    async def _probe_vllm_lightweight_support(
        self,
        repo_id: str,
        config: dict[str, Any] | None,
        tokenizer_config: dict[str, Any] | None,
        support: HFVLLMSupport,
    ) -> HFVLLMRuntimeProbe:
        try:
            from transformers import AutoConfig, AutoTokenizer
        except Exception:
            probe = HFVLLMRuntimeProbe(
                status="unavailable",
                reason="Transformers no esta disponible en este entorno para ejecutar el probe ligero.",
                recommended_model_impl=support.model_impl,  # type: ignore[arg-type]
                recommended_runner=support.runner,  # type: ignore[arg-type]
            )
            return self._runtime_probe.set_cached(repo_id, probe)

        target = repo_id
        config_ok = False
        tokenizer_ok = False
        try:
            await asyncio.to_thread(
                AutoConfig.from_pretrained,
                target,
                trust_remote_code=False,
                token=settings.hf_token or None,
                cache_dir=settings.hf_cache_dir or None,
            )
            config_ok = True
        except Exception as exc:
            message = str(exc).lower()
            if "trust_remote_code=true" in message or "remote code" in message:
                probe = HFVLLMRuntimeProbe(
                    status="needs_remote_code",
                    reason="El probe de config requiere trust_remote_code.",
                    recommended_model_impl=support.model_impl,  # type: ignore[arg-type]
                    recommended_runner=support.runner,  # type: ignore[arg-type]
                    config_load=False,
                )
                return self._runtime_probe.set_cached(repo_id, probe)
            if support.classification == "unknown":
                probe = HFVLLMRuntimeProbe(
                    status="unsupported_architecture",
                    reason=f"El config no se pudo cargar en el probe ligero: {exc}",
                    recommended_model_impl=support.model_impl,  # type: ignore[arg-type]
                    recommended_runner=support.runner,  # type: ignore[arg-type]
                    config_load=False,
                )
                return self._runtime_probe.set_cached(repo_id, probe)

        try:
            await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                target,
                trust_remote_code=False,
                token=settings.hf_token or None,
                cache_dir=settings.hf_cache_dir or None,
            )
            tokenizer_ok = True
        except Exception as exc:
            message = str(exc).lower()
            if "trust_remote_code=true" in message or "remote code" in message:
                probe = HFVLLMRuntimeProbe(
                    status="needs_remote_code",
                    reason="El probe de tokenizer requiere trust_remote_code.",
                    recommended_model_impl=support.model_impl,  # type: ignore[arg-type]
                    recommended_runner=support.runner,  # type: ignore[arg-type]
                    tokenizer_load=False,
                    config_load=config_ok,
                )
                return self._runtime_probe.set_cached(repo_id, probe)

        status = {
            "native_vllm": "supported_native",
            "transformers_backend": "supported_transformers_backend",
            "pooling": "supported_pooling",
        }.get(support.classification, "unknown")
        probe = HFVLLMRuntimeProbe(
            status=status,  # type: ignore[arg-type]
            reason=None,
            recommended_model_impl=support.model_impl,  # type: ignore[arg-type]
            recommended_runner=support.runner,  # type: ignore[arg-type]
            tokenizer_load=tokenizer_ok if tokenizer_ok or tokenizer_config is not None else None,
            config_load=config_ok if config_ok or config is not None else None,
        )
        return self._runtime_probe.set_cached(repo_id, probe)

    async def _load_repo_json(
        self,
        repo_id: str,
        filename: str,
        sibling_names: set[str],
    ) -> dict[str, Any] | None:
        if filename.lower() not in sibling_names:
            return None
        return await asyncio.to_thread(self._load_repo_json_sync, repo_id, filename)

    def _load_repo_json_sync(self, repo_id: str, filename: str) -> dict[str, Any] | None:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=settings.hf_token or None,
                cache_dir=settings.hf_cache_dir or None,
            )
        except Exception:
            return None
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return None

    def _extract_quant_method(
        self,
        config: dict[str, Any] | None,
        quantize_config: dict[str, Any] | None,
    ) -> str | None:
        for payload in (config or {}, quantize_config or {}):
            quant_cfg = payload.get("quantization_config") if isinstance(payload, dict) else None
            if isinstance(quant_cfg, dict):
                method = (
                    str(quant_cfg.get("quant_method") or quant_cfg.get("format") or "")
                    .strip()
                    .lower()
                )
                if method:
                    return method
            if isinstance(payload, dict):
                method = (
                    str(payload.get("quant_method") or payload.get("format") or "").strip().lower()
                )
                if method:
                    return method
        return None
