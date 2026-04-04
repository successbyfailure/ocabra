from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime

from fastapi import Depends

from ocabra.api._deps_auth import UserContext, get_current_user

async def get_ollama_user(
    user: UserContext = Depends(get_current_user),
) -> UserContext:
    """Resolve auth for Ollama-compatible endpoints.

    Delegates to ``get_current_user`` which already handles:
    - Bearer API key resolution.
    - Cookie JWT resolution.
    - Anonymous access when ``require_api_key_ollama=False``.
    - HTTP 401 when ``require_api_key_ollama=True`` and no credentials provided.

    Returns:
        Resolved :class:`UserContext` for the caller.

    Raises:
        HTTPException 401: When authentication is required but missing or invalid.
    """
    return user


OPTION_MAP: dict[str, str] = {
    "num_predict": "max_tokens",
    "num_ctx": "max_model_len",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop",
    "seed": "seed",
    "repeat_penalty": "repetition_penalty",
}

NATIVE_OPTION_ALIAS_MAP: dict[str, str] = {
    "max_tokens": "num_predict",
    "num_predict": "num_predict",
    "num_ctx": "num_ctx",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop",
    "seed": "seed",
    "repeat_penalty": "repeat_penalty",
}


def now_iso_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_native_passthrough_body(
    payload: Mapping[str, object],
    *,
    model: str,
    stream: bool,
    content_keys: tuple[str, ...],
    passthrough_keys: tuple[str, ...] = (),
) -> dict:
    body: dict = {
        "model": model,
        "stream": stream,
    }
    for key in (*content_keys, *passthrough_keys):
        if key in payload:
            body[key] = payload[key]

    options = payload.get("options")
    merged_options = dict(options) if isinstance(options, Mapping) else {}
    for source_key, target_key in NATIVE_OPTION_ALIAS_MAP.items():
        if source_key in payload and target_key not in merged_options:
            merged_options[target_key] = payload[source_key]
    body["options"] = merged_options
    return body


def apply_option_map(body: dict, options: object) -> None:
    if not isinstance(options, Mapping):
        return

    for key, value in options.items():
        mapped = OPTION_MAP.get(str(key))
        if mapped:
            body[mapped] = value


async def iter_sse_payloads(source: AsyncIterator[bytes]) -> AsyncIterator[dict | str]:
    """Parse `data: ...\\n\\n` SSE frames and yield decoded payloads."""
    buffer = ""
    async for chunk in source:
        if not chunk:
            continue
        buffer += chunk.decode("utf-8", errors="ignore")

        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in raw_event.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if not data_lines:
                continue

            data = "\n".join(data_lines).strip()
            if not data:
                continue
            if data == "[DONE]":
                yield "[DONE]"
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue
