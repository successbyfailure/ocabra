import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import list_models, model_info, snapshot_download

from ocabra.config import settings
from ocabra.schemas.registry import HFModelCard, HFModelDetail

_BACKEND_BY_TASK = {
    "text-generation": "vllm",
    "image-to-image": "diffusers",
    "text-to-image": "diffusers",
    "automatic-speech-recognition": "whisper",
    "text-to-speech": "tts",
}


class HuggingFaceRegistry:
    async def search(self, query: str, task: str | None, limit: int) -> list[HFModelCard]:
        def _run() -> list[Any]:
            return list(
                list_models(
                    search=query or None,
                    task=task or None,
                    limit=limit,
                    full=False,
                    cardData=False,
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
        suggested_backend = self._infer_backend(task=task, siblings=siblings)

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

    async def download(
        self,
        repo_id: str,
        target_dir: Path,
        progress_callback: Callable[[float, float | None], None],
    ) -> Path:
        await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)

        class _ProgressTqdm:
            def __init__(self, *args, **kwargs):
                self.total = float(kwargs.get("total") or 0)
                self.n = 0.0
                self.start = time.monotonic()

            def update(self, n: int = 1) -> None:
                self.n += n
                elapsed = max(time.monotonic() - self.start, 1e-6)
                speed = (self.n / (1024 * 1024)) / elapsed
                pct = 0.0
                if self.total > 0:
                    pct = min(100.0, (self.n / self.total) * 100.0)
                progress_callback(pct, speed)

            def close(self) -> None:
                return None

        def _run() -> str:
            return snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                token=settings.hf_token or None,
                tqdm_class=_ProgressTqdm,
            )

        downloaded = await asyncio.to_thread(_run)
        progress_callback(100.0, None)
        return Path(downloaded)

    def _infer_backend(self, task: str | None, siblings: list[dict]) -> str:
        if task in _BACKEND_BY_TASK:
            return _BACKEND_BY_TASK[task]

        names = [str(s.get("rfilename", "")).lower() for s in siblings]
        if any(name.endswith(".gguf") for name in names):
            return "vllm"
        if any("model_index.json" in name for name in names):
            return "diffusers"
        return "vllm"
