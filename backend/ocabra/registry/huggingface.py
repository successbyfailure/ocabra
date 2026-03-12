import asyncio
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import list_models, model_info, snapshot_download

from ocabra.config import settings
from ocabra.schemas.registry import HFModelCard, HFModelDetail, HFModelVariant

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
}


class HuggingFaceRegistry:
    async def infer_backend_for_repo(self, repo_id: str, artifact: str | None = None) -> str:
        if artifact and artifact.lower().endswith(".gguf"):
            raise ValueError("HF GGUF artifacts are not supported; use the Ollama registry flow instead")
        info = await asyncio.to_thread(
            lambda: model_info(repo_id=repo_id, files_metadata=False, token=settings.hf_token or None)
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
            cards.append(
                HFModelCard(
                    repo_id=model.id,
                    model_name=model.id.split("/")[-1],
                    task=getattr(model, "pipeline_tag", None),
                    downloads=int(getattr(model, "downloads", 0) or 0),
                    likes=int(getattr(model, "likes", 0) or 0),
                    size_gb=None,
                    tags=list(getattr(model, "tags", []) or []),
                    gated=bool(getattr(model, "gated", False)),
                    suggested_backend=self._infer_backend(
                        task=getattr(model, "pipeline_tag", None),
                        siblings=[],
                        tags=list(getattr(model, "tags", []) or []),
                        library_name=getattr(model, "library_name", None),
                        repo_id=model.id,
                    ),
                )
            )
        return cards

    async def get_model_detail(self, repo_id: str) -> HFModelDetail:
        info = await asyncio.to_thread(
            lambda: model_info(repo_id=repo_id, files_metadata=True, token=settings.hf_token or None)
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
            estimated_vram_gb=(safetensors_bytes / (1024**3) * 1.3) if safetensors_bytes > 0 else None,
        )

    async def get_variants(self, repo_id: str) -> list[HFModelVariant]:
        info = await asyncio.to_thread(
            lambda: model_info(repo_id=repo_id, files_metadata=True, token=settings.hf_token or None)
        )
        siblings = list(info.siblings or [])
        items = [
            {"name": s.rfilename, "size": int(getattr(s, "size", 0) or 0)}
            for s in siblings
        ]
        names = [i["name"] for i in items]
        variants: list[HFModelVariant] = []

        # Standard vLLM variant (download config + safetensors/bin tokenizer files).
        has_standard = any(n.endswith(".safetensors") or n.endswith(".bin") for n in names)
        if has_standard:
            total = sum(
                i["size"] for i in items
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
                    backend_type="vllm",
                    is_default=True,
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
                    backend_type=await self.infer_backend_for_repo(repo_id),
                    is_default=True,
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
            raise ValueError("HF GGUF artifacts are not supported; use the Ollama registry flow instead")

        await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
        info = await asyncio.to_thread(
            lambda: model_info(repo_id=repo_id, files_metadata=False, token=settings.hf_token or None)
        )
        sibling_names = [s.rfilename for s in (info.siblings or [])]
        backend_hint = self._infer_backend(
            task=getattr(info, "pipeline_tag", None),
            siblings=[{"rfilename": name} for name in sibling_names],
            tags=list(getattr(info, "tags", []) or []),
            library_name=getattr(info, "library_name", None),
            repo_id=repo_id,
        )
        if not self._has_supported_hf_payload(sibling_names, backend_hint):
            raise ValueError("This Hugging Face repo does not expose supported native weights for oCabra")
        allow_patterns = self._download_allow_patterns(sibling_names, backend_hint, artifact=artifact)

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
            return "tts"
        if normalized_tags & {"automatic-speech-recognition", "asr", "whisper"}:
            return "whisper"

        if task in _BACKEND_BY_TASK:
            return _BACKEND_BY_TASK[task]

        names = [str(s.get("rfilename", "")).lower() for s in siblings]
        repo_name = (repo_id or "").lower()

        if self._looks_like_diffusers_repo(names, normalized_task, normalized_library, normalized_tags, repo_name):
            return "diffusers"
        if self._looks_like_tts_repo(repo_name, normalized_tags, normalized_library, names):
            return "tts"
        return "vllm"

    def _has_supported_hf_payload(self, names: list[str], backend_hint: str) -> bool:
        lowered = [name.lower() for name in names]
        if backend_hint == "diffusers":
            return "model_index.json" in lowered
        if backend_hint == "tts":
            return any(name.endswith((".safetensors", ".bin")) for name in lowered)
        if backend_hint == "whisper":
            return any(name.endswith((".safetensors", ".bin", ".pt")) for name in lowered)
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

        diffusers_dirs = ("unet/", "vae/", "scheduler/", "text_encoder/", "text_encoder_2/", "transformer/")
        diffusers_files = {"scheduler_config.json", "feature_extractor/preprocessor_config.json"}
        if any(name.startswith(diffusers_dirs) for name in names):
            return True
        if any(name in diffusers_files for name in names):
            return True
        if any(token in repo_name for token in ("stable-diffusion", "sdxl", "flux", "wan", "kolors")):
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
