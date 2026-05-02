"""Pure translation helpers between OpenAI tool-calling and MCP semantics.

These helpers are deliberately framework-free so they can be exercised in
isolation by the unit tests.

Plan: docs/tasks/agents-mcp-plan.md — section "Contratos" / Formato OpenAI ↔ MCP.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ocabra.agents.mcp_client import MCPTool, MCPToolResult

# Sensitive fields that we never persist or echo back in stats.
DEFAULT_REDACT_FIELDS: tuple[str, ...] = (
    "authorization",
    "password",
    "token",
    "api_key",
    "secret",
)

# OpenAI's function-name regex: ``^[a-zA-Z0-9_-]{1,64}$``.
_NAME_OK = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_NAME_SCRUB = re.compile(r"[^a-zA-Z0-9_-]+")


def sanitize_openai_function_name(raw: str) -> str:
    """Return a name that matches the OpenAI ``functions`` regex.

    Non-conforming chars are collapsed to ``_``; the result is truncated to
    64 chars.  The function never returns an empty string — when ``raw`` is
    empty after scrubbing it falls back to ``"tool"``.
    """
    if not raw:
        return "tool"
    if _NAME_OK.match(raw):
        return raw[:64]
    cleaned = _NAME_SCRUB.sub("_", raw).strip("_")
    if not cleaned:
        cleaned = "tool"
    return cleaned[:64]


def mcp_tool_to_openai(alias: str, tool: MCPTool) -> dict[str, Any]:
    """Convert an :class:`MCPTool` to the OpenAI ``tools[]`` schema.

    The exposed name is namespaced as ``{alias}_{tool.name}`` so callers can
    distinguish tools coming from different MCP servers without collisions.
    The original raw name is sanitised against the OpenAI regex.
    """
    namespaced = (
        f"{sanitize_openai_function_name(alias)}_{sanitize_openai_function_name(tool.name)}"
    )
    namespaced = sanitize_openai_function_name(namespaced)
    parameters = tool.input_schema or {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": namespaced,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }


def parse_openai_tool_call(tool_call: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Parse an OpenAI ``tool_call`` payload into ``(alias, tool_name, args)``.

    Raises:
        ValueError: When the payload shape is invalid or the function name has
            no namespace prefix.
    """
    if not isinstance(tool_call, dict):
        raise ValueError("tool_call must be a dict")
    fn = tool_call.get("function") or {}
    if not isinstance(fn, dict):
        raise ValueError("tool_call.function must be a dict")
    raw_name = fn.get("name")
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError("tool_call.function.name is required")
    if "_" not in raw_name:
        raise ValueError(f"tool name '{raw_name}' has no namespace prefix")
    alias, tool_name = raw_name.split("_", 1)
    if not alias or not tool_name:
        raise ValueError(f"tool name '{raw_name}' has empty namespace or tool")

    raw_args = fn.get("arguments")
    if raw_args is None or raw_args == "":
        args: dict[str, Any] = {}
    elif isinstance(raw_args, dict):
        args = dict(raw_args)
    elif isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool arguments are not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must decode to a JSON object")
        args = parsed
    else:
        raise ValueError(f"tool arguments have unsupported type {type(raw_args).__name__}")
    return alias, tool_name, args


def _stringify_block(block: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Return ``(text_part, image_url_part_or_none)`` for a single MCP content block."""
    btype = (block or {}).get("type")
    if btype == "text":
        return str(block.get("text") or ""), None
    if btype == "image":
        # MCP image blocks: {"type": "image", "data": <b64>, "mimeType": "image/png"}
        data = block.get("data") or block.get("image") or ""
        mime = block.get("mimeType") or block.get("mime_type") or "image/png"
        if data:
            return "", {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }
        return "[image omitted]", None
    if btype == "resource":
        # Resource blocks: surface uri + name as text for the LLM.
        resource = block.get("resource") or {}
        uri = resource.get("uri") or block.get("uri") or ""
        name = resource.get("name") or block.get("name") or ""
        return f"[resource {name or ''} {uri}]".strip(), None
    # Fallback: serialise as JSON to keep some signal for the LLM.
    return json.dumps(block, ensure_ascii=False), None


def mcp_result_to_openai_message(
    tool_call_id: str,
    result: MCPToolResult,
    *,
    vision_capable: bool = False,
) -> dict[str, Any]:
    """Convert an :class:`MCPToolResult` into an OpenAI ``role=tool`` message.

    The OpenAI spec accepts either a plain string or a list of content parts
    in ``content``.  We use the list form when the result has at least one
    image and *vision_capable* is true; otherwise we concatenate text blocks
    into a single string and substitute ``[image omitted]`` for any visual
    part the model cannot read.
    """
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []
    has_image_block = False
    for block in result.content or []:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        if block.get("type") == "image":
            has_image_block = True
        text_part, image_part = _stringify_block(block)
        if image_part is not None:
            image_parts.append(image_part)
        if text_part:
            text_parts.append(text_part)

    if vision_capable and image_parts:
        content_parts: list[dict[str, Any]] = []
        joined_text = "\n".join(p for p in text_parts if p)
        if joined_text:
            content_parts.append({"type": "text", "text": joined_text})
        content_parts.extend(image_parts)
        content: Any = content_parts
    else:
        if has_image_block and not vision_capable:
            text_parts.append("[image omitted]")
        content = "\n".join(p for p in text_parts if p)

    msg: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
    if result.is_error:
        # Surface upstream errors but keep the message valid for the LLM loop.
        if isinstance(content, str):
            msg["content"] = content or "tool_error"
    return msg


def redact_args(
    args: dict[str, Any],
    redact_fields: list[str] | tuple[str, ...] = DEFAULT_REDACT_FIELDS,
) -> dict[str, Any]:
    """Return a deep copy of *args* with sensitive fields replaced by ``"[redacted]"``.

    Field matching is case-insensitive and recursive into nested dicts/lists.
    """
    sensitive = {f.lower() for f in redact_fields}

    def _walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: ("[redacted]" if str(k).lower() in sensitive else _walk(v))
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [_walk(item) for item in value]
        return value

    walked = _walk(args)
    if not isinstance(walked, dict):
        return {}
    return walked


def truncate_summary(text: str, max_bytes: int) -> str:
    """Return *text* truncated so its UTF-8 byte length does not exceed *max_bytes*."""
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[: max_bytes - 1].decode("utf-8", errors="ignore")
    return truncated + "…"


def summarise_result(result: MCPToolResult, *, max_bytes: int) -> str:
    """Produce a short text summary of an MCP result for stats persistence."""
    chunks: list[str] = []
    for block in result.content or []:
        if not isinstance(block, dict):
            chunks.append(str(block))
            continue
        if block.get("type") == "text":
            chunks.append(str(block.get("text") or ""))
        elif block.get("type") == "image":
            chunks.append("[image]")
        elif block.get("type") == "resource":
            resource = block.get("resource") or {}
            chunks.append(f"[resource {resource.get('uri') or ''}]")
        else:
            chunks.append(json.dumps(block, ensure_ascii=False))
    summary = "\n".join(c for c in chunks if c)
    return truncate_summary(summary, max_bytes)
