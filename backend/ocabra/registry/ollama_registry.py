import asyncio
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ocabra.config import settings
from ocabra.schemas.registry import OllamaModelCard


class OllamaRegistry:
    async def search(self, query: str) -> list[OllamaModelCard]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get("https://ollama.com/api/models", params={"q": query})
            response.raise_for_status()
            payload = response.json()

        models = payload.get("models", payload if isinstance(payload, list) else [])
        cards: list[OllamaModelCard] = []
        for item in models:
            name = str(item.get("name") or item.get("model") or "")
            if query and query.lower() not in name.lower():
                continue
            cards.append(
                OllamaModelCard(
                    name=name,
                    description=str(item.get("description") or ""),
                    tags=list(item.get("tags") or []),
                    size_gb=(float(item["size"]) / (1024**3)) if item.get("size") else None,
                    pulls=int(item.get("pulls") or 0),
                )
            )
        return cards

    async def pull(
        self,
        model_ref: str,
        progress_callback: Callable[[float, float | None], None],
    ) -> Path:
        if shutil.which("ollama") is None:
            raise RuntimeError("'ollama' binary is not available in PATH")

        proc = await asyncio.create_subprocess_exec(
            "ollama",
            "pull",
            model_ref,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        started = time.monotonic()
        last_pct = 0.0

        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text_line = line.decode("utf-8", errors="ignore").strip()
            if not text_line:
                continue

            try:
                data = json.loads(text_line)
            except json.JSONDecodeError:
                continue

            completed = float(data.get("completed") or 0.0)
            total = float(data.get("total") or 0.0)
            pct = last_pct
            if total > 0:
                pct = min(100.0, completed / total * 100.0)

            elapsed = max(time.monotonic() - started, 1e-6)
            speed = (completed / (1024 * 1024)) / elapsed if completed else None
            progress_callback(pct, speed)
            last_pct = pct

        return_code = await proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ollama pull failed for {model_ref} (exit={return_code})")

        progress_callback(100.0, None)
        return Path(settings.models_dir) / "ollama" / model_ref.replace(":", "_")
