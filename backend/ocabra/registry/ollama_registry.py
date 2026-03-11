import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ocabra.config import settings
from ocabra.schemas.registry import OllamaModelCard, OllamaModelVariant

logger = logging.getLogger(__name__)

# Known popular models as fallback when the registry is unreachable
_FALLBACK_MODELS = [
    "llama3.3", "llama3.2", "llama3.1", "gemma3", "qwen2.5",
    "deepseek-r1", "mistral", "phi4", "phi4-mini", "llava",
    "nomic-embed-text", "mxbai-embed-large", "codellama",
]


class OllamaRegistry:
    _LIBRARY_RE = re.compile(r"/library/([a-zA-Z0-9._:-]+)")
    _MODEL_HREF_RE = re.compile(
        r"/library/(?P<name>[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+)",
        re.IGNORECASE,
    )
    _TEXT_RE = re.compile(r"<[^>]+>")
    _SIZE_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>KB|MB|GB|TB)", re.IGNORECASE)
    _CTX_RE = re.compile(r"(\d+[kKmM])\s+context window")

    def _url(self, path: str) -> str:
        base = settings.ollama_base_url.rstrip("/")
        return f"{base}{path}"

    async def search(self, query: str) -> list[OllamaModelCard]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://ollama.com/search",
                    params={"q": query},
                )
                response.raise_for_status()
                html = response.text
        except Exception as e:
            logger.warning("ollama_search_unavailable: %s — returning fallback list", e)
            names = [n for n in _FALLBACK_MODELS if not query or query.lower() in n]
            return [OllamaModelCard(name=n, description="", tags=[], size_gb=None, pulls=0) for n in names]

        names = list(dict.fromkeys(self._LIBRARY_RE.findall(html)))
        if query:
            q = query.lower()
            names = [name for name in names if q in name.lower()]
        if not names:
            names = [n for n in _FALLBACK_MODELS if not query or query.lower() in n]

        cards: list[OllamaModelCard] = []
        for name in names[:80]:
            cards.append(
                OllamaModelCard(
                    name=name,
                    description="",
                    tags=[],
                    size_gb=None,
                    pulls=0,
                )
            )
        return cards

    async def get_variants(self, model_name: str) -> list[OllamaModelVariant]:
        model_name = model_name.strip().lower()
        if ":" in model_name:
            model_name = model_name.split(":", 1)[0]

        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(f"https://ollama.com/library/{model_name}/tags")
            response.raise_for_status()
            html = response.text

        variants: dict[str, OllamaModelVariant] = {}
        for match in self._MODEL_HREF_RE.finditer(html):
            full_name = str(match.group("name")).strip()
            if not full_name.startswith(f"{model_name}:"):
                continue

            idx = match.start()
            after_window = html[idx: min(len(html), idx + 1600)]
            around_window = html[max(0, idx - 1200): min(len(html), idx + 1800)]
            plain_after = self._to_plain_text(after_window)
            plain_around = self._to_plain_text(around_window)
            plain = plain_after or plain_around
            if not plain:
                continue

            size_gb = self._extract_size_gb(plain_after) or self._extract_size_gb(plain_around)
            context_window = self._extract_context_window(plain_after) or self._extract_context_window(plain_around)
            modality = self._extract_modality(plain_after) or self._extract_modality(plain_around)
            updated_hint = self._extract_updated_hint(plain_after) or self._extract_updated_hint(plain_around)

            tag = full_name.split(":", 1)[1]
            parameter_size = self._extract_parameter_size(tag)
            quantization = self._extract_quantization(tag)

            variants[full_name] = OllamaModelVariant(
                name=full_name,
                tag=tag,
                size_gb=size_gb,
                parameter_size=parameter_size,
                quantization=quantization,
                context_window=context_window,
                modality=modality,
                updated_hint=updated_hint,
            )

        if not variants:
            return [
                OllamaModelVariant(
                    name=f"{model_name}:latest",
                    tag="latest",
                    size_gb=None,
                    parameter_size=None,
                    quantization=None,
                    context_window=None,
                    modality=None,
                    updated_hint=None,
                )
            ]

        def _sort_key(v: OllamaModelVariant) -> tuple[int, float, str]:
            # Prefer generic tags first, then smaller binaries for faster install.
            generic = 0 if v.tag in {"latest", "8b", "7b", "4b", "3b", "2b", "1b"} else 1
            size = v.size_gb if v.size_gb is not None else 9999.0
            return (generic, size, v.name)

        return sorted(variants.values(), key=_sort_key)

    def _extract_size_gb(self, text: str) -> float | None:
        m = self._SIZE_RE.search(text)
        if not m:
            return None
        num = float(m.group("num"))
        unit = m.group("unit").upper()
        if unit == "KB":
            return num / (1024**2)
        if unit == "MB":
            return num / 1024
        if unit == "GB":
            return num
        if unit == "TB":
            return num * 1024
        return None

    def _extract_context_window(self, text: str) -> str | None:
        m = self._CTX_RE.search(text)
        if not m:
            m2 = re.search(r"\b(\d+(?:\.\d+)?[kK])\b", text)
            return m2.group(1).upper() if m2 else None
        return m.group(1).upper()

    def _extract_modality(self, text: str) -> str | None:
        if "Text, Image" in text:
            return "text+image"
        if "Image" in text and "Text" not in text:
            return "image"
        if "Text" in text:
            return "text"
        return None

    def _extract_updated_hint(self, text: str) -> str | None:
        m = re.search(r"(\d+\s+(?:minute|hour|day|week|month|year)s?\s+ago)", text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        return None

    def _extract_parameter_size(self, tag: str) -> str | None:
        first = tag.split("-", 1)[0].lower()
        if re.fullmatch(r"\d+(?:\.\d+)?[bm]", first):
            return first
        m = re.search(r"(\d+(?:\.\d+)?[bm])", tag.lower())
        return m.group(1) if m else None

    def _extract_quantization(self, tag: str) -> str | None:
        low = tag.lower()
        for token in ("q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "iq2", "iq3", "iq4", "bf16", "fp16", "qat"):
            if token in low:
                return token.upper()
        return None

    def _to_plain_text(self, html: str) -> str:
        plain = self._TEXT_RE.sub(" ", html)
        return " ".join(plain.split())

    async def list_installed_details(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(self._url("/api/tags"))
            response.raise_for_status()
            payload = response.json()

        models = payload.get("models", []) if isinstance(payload, dict) else []
        details: list[dict] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if not name:
                continue
            details.append(
                {
                    "name": name,
                    "model": str(item.get("model") or name),
                    "size": int(item.get("size") or 0),
                    "modified_at": str(item.get("modified_at") or ""),
                }
            )
        return details

    async def list_installed(self) -> list[str]:
        details = await self.list_installed_details()
        return [d["name"] for d in details]

    async def list_loaded(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(self._url("/api/ps"))
            response.raise_for_status()
            payload = response.json()

        models = payload.get("models", []) if isinstance(payload, dict) else []
        names: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                names.append(name)
        return names

    async def load(self, model_ref: str, keep_alive: str | int | None = None) -> None:
        payload: dict[str, object] = {
            "model": model_ref,
            "prompt": "",
            "stream": False,
            "keep_alive": settings.ollama_keep_alive if keep_alive is None else keep_alive,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(self._url("/api/generate"), json=payload)
            response.raise_for_status()

    async def unload(self, model_ref: str) -> None:
        await self.load(model_ref, keep_alive=0)

    async def delete(self, model_ref: str) -> None:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.delete(self._url("/api/delete"), json={"model": model_ref})
            response.raise_for_status()

    async def pull(
        self,
        model_ref: str,
        progress_callback: Callable[[float, float | None], None],
    ) -> Path:
        started = time.monotonic()
        last_pct = 0.0
        saw_success = False

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                self._url("/api/pull"),
                json={"model": model_ref, "stream": True},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    text_line = line.strip()
                    if not text_line:
                        continue

                    try:
                        data = json.loads(text_line)
                    except json.JSONDecodeError:
                        continue

                    status = str(data.get("status") or "").lower()
                    if status == "success":
                        saw_success = True
                        continue
                    if status == "error":
                        error = str(data.get("error") or f"ollama pull failed for {model_ref}")
                        raise RuntimeError(error)

                    completed = float(data.get("completed") or 0.0)
                    total = float(data.get("total") or 0.0)
                    pct = last_pct
                    if total > 0:
                        pct = min(100.0, completed / total * 100.0)

                    elapsed = max(time.monotonic() - started, 1e-6)
                    speed = (completed / (1024 * 1024)) / elapsed if completed else None
                    progress_callback(pct, speed)
                    last_pct = pct

        if not saw_success:
            logger.warning("ollama_pull_missing_success_marker", model_ref=model_ref)

        progress_callback(100.0, None)
        return Path(settings.models_dir) / "ollama" / model_ref.replace(":", "_")
