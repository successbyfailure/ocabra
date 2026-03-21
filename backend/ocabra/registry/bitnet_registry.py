from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from huggingface_hub import list_models, model_info

from ocabra.config import settings
from ocabra.schemas.registry import HFModelCard, HFModelVariant


class BitnetRegistry:
    def _is_bitnet_repo(self, repo_id: str, tags: list[str] | None = None) -> bool:
        low_repo = repo_id.lower()
        if "bitnet" in low_repo or "1.58bit" in low_repo:
            return True
        return any("bitnet" in str(tag).lower() or "1.58bit" in str(tag).lower() for tag in (tags or []))

    def _is_bitnet_file(self, filename: str) -> bool:
        low = filename.lower()
        if not low.endswith(".gguf"):
            return False
        return "bitnet" in low or "i2_s" in low or "1.58bit" in low

    def _extract_quant(self, filename: str) -> str | None:
        low = filename.lower()
        m = re.search(r"(i2_s|q\d+_k_[msl]|q\d+_k|q\d_[01]|q8_0|iq\d(?:_[a-z0-9]+)?)", low)
        if not m:
            return None
        return m.group(1).upper()

    async def search(self, query: str, limit: int = 20) -> list[HFModelCard]:
        q = query.strip()

        def _run() -> list[Any]:
            # Broad query, then strict bitnet filtering.
            return list(list_models(search=q or "bitnet", limit=max(limit * 4, 40), token=settings.hf_token or None))

        models = await asyncio.to_thread(_run)
        cards: list[HFModelCard] = []
        for model in models:
            repo_id = model.id
            tags = list(getattr(model, "tags", []) or [])
            siblings = list(getattr(model, "siblings", []) or [])
            names = [str(getattr(s, "rfilename", "")) for s in siblings]
            bitnet_gguf = [n for n in names if self._is_bitnet_file(n)]

            if not bitnet_gguf and not self._is_bitnet_repo(repo_id, tags):
                continue

            size_bytes = sum(
                int(getattr(s, "size", 0) or 0)
                for s in siblings
                if str(getattr(s, "rfilename", "")).lower() in {name.lower() for name in bitnet_gguf}
            )

            cards.append(
                HFModelCard(
                    repo_id=repo_id,
                    model_name=repo_id.split("/")[-1],
                    task=getattr(model, "pipeline_tag", None),
                    downloads=int(getattr(model, "downloads", 0) or 0),
                    likes=int(getattr(model, "likes", 0) or 0),
                    size_gb=(size_bytes / (1024**3)) if size_bytes > 0 else None,
                    tags=tags,
                    gated=bool(getattr(model, "gated", False)),
                    suggested_backend="bitnet",
                    compatibility="compatible",
                    compatibility_reason=None,
                    vllm_support=None,
                )
            )
            if len(cards) >= limit:
                break

        return cards

    async def get_variants(self, repo_id: str) -> list[HFModelVariant]:
        info = await asyncio.to_thread(
            lambda: model_info(repo_id=repo_id, files_metadata=True, token=settings.hf_token or None)
        )

        siblings = list(info.siblings or [])
        variants: list[HFModelVariant] = []

        for sibling in siblings:
            name = str(getattr(sibling, "rfilename", ""))
            if not self._is_bitnet_file(name):
                continue
            size = int(getattr(sibling, "size", 0) or 0)
            variants.append(
                HFModelVariant(
                    variant_id=f"gguf::{name}",
                    label=f"gguf · {Path(name).name}",
                    artifact=name,
                    size_gb=(size / (1024**3)) if size > 0 else None,
                    format="gguf",
                    quantization=self._extract_quant(name),
                    backend_type="bitnet",
                    is_default=("i2_s" in name.lower()),
                    installable=True,
                    compatibility="compatible",
                    compatibility_reason=None,
                    vllm_support=None,
                )
            )

        if variants and not any(v.is_default for v in variants):
            variants[0].is_default = True

        return variants
