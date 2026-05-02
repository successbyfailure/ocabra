"""Unit tests for the OpenAI ↔ MCP translation helpers (Stream B / Fase 2)."""

from __future__ import annotations

import pytest

from ocabra.agents.mcp_client import MCPTool, MCPToolResult
from ocabra.agents.translation import (
    DEFAULT_REDACT_FIELDS,
    mcp_result_to_openai_message,
    mcp_tool_to_openai,
    parse_openai_tool_call,
    redact_args,
    sanitize_openai_function_name,
    summarise_result,
    truncate_summary,
)


def test_mcp_tool_to_openai_namespaces_and_sanitises():
    tool = MCPTool(
        name="create-issue",
        description="Open a new issue",
        input_schema={"type": "object", "properties": {"title": {"type": "string"}}},
    )
    out = mcp_tool_to_openai("git#hub", tool)
    assert out["type"] == "function"
    assert out["function"]["name"].startswith("git_hub")
    # name must satisfy the OpenAI regex
    import re

    assert re.match(r"^[a-zA-Z0-9_-]{1,64}$", out["function"]["name"])
    assert out["function"]["parameters"] == tool.input_schema
    assert out["function"]["description"] == "Open a new issue"


def test_mcp_tool_to_openai_empty_schema_normalised():
    tool = MCPTool(name="ping", description="", input_schema=None)  # type: ignore[arg-type]
    out = mcp_tool_to_openai("svc", tool)
    assert out["function"]["parameters"] == {"type": "object", "properties": {}}


def test_parse_openai_tool_call_happy_path():
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "github_create_issue", "arguments": '{"title": "x"}'},
    }
    alias, name, args = parse_openai_tool_call(tc)
    assert alias == "github"
    assert name == "create_issue"
    assert args == {"title": "x"}


def test_parse_openai_tool_call_dict_arguments():
    tc = {"function": {"name": "fs_read", "arguments": {"path": "/tmp/x"}}}
    alias, name, args = parse_openai_tool_call(tc)
    assert alias == "fs"
    assert name == "read"
    assert args == {"path": "/tmp/x"}


def test_parse_openai_tool_call_invalid_json_raises():
    tc = {"function": {"name": "fs_read", "arguments": "not-json"}}
    with pytest.raises(ValueError):
        parse_openai_tool_call(tc)


def test_parse_openai_tool_call_no_namespace_raises():
    tc = {"function": {"name": "noseparator", "arguments": "{}"}}
    with pytest.raises(ValueError):
        parse_openai_tool_call(tc)


def test_parse_openai_tool_call_empty_arguments_ok():
    tc = {"function": {"name": "fs_list", "arguments": ""}}
    alias, name, args = parse_openai_tool_call(tc)
    assert (alias, name, args) == ("fs", "list", {})


def test_redact_args_lowercases_and_recurses():
    args = {
        "Authorization": "Bearer xyz",
        "data": {"password": "p", "ok": "v"},
        "list": [{"api_key": "k"}, {"keep": "this"}],
    }
    redacted = redact_args(args)
    assert redacted["Authorization"] == "[redacted]"
    assert redacted["data"]["password"] == "[redacted]"
    assert redacted["data"]["ok"] == "v"
    assert redacted["list"][0]["api_key"] == "[redacted]"
    assert redacted["list"][1]["keep"] == "this"


def test_redact_args_default_fields_match_plan():
    expected = {"authorization", "password", "token", "api_key", "secret"}
    assert set(DEFAULT_REDACT_FIELDS) == expected


def test_mcp_result_to_openai_message_text_only():
    result = MCPToolResult(
        content=[
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
        ],
        is_error=False,
    )
    msg = mcp_result_to_openai_message("call_1", result, vision_capable=False)
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "Hello\nworld"


def test_mcp_result_to_openai_message_image_no_vision_omitted():
    result = MCPToolResult(
        content=[
            {"type": "text", "text": "see below"},
            {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        ],
    )
    msg = mcp_result_to_openai_message("call_1", result, vision_capable=False)
    assert isinstance(msg["content"], str)
    assert "[image omitted]" in msg["content"]


def test_mcp_result_to_openai_message_image_with_vision_keeps_image():
    result = MCPToolResult(
        content=[
            {"type": "text", "text": "see below"},
            {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        ],
    )
    msg = mcp_result_to_openai_message("call_1", result, vision_capable=True)
    assert isinstance(msg["content"], list)
    has_image = any(isinstance(p, dict) and p.get("type") == "image_url" for p in msg["content"])
    assert has_image


def test_truncate_summary_handles_multibyte():
    s = "ñ" * 100
    truncated = truncate_summary(s, max_bytes=10)
    assert truncated.endswith("…")
    assert len(truncated.encode("utf-8")) <= 11  # 10 bytes + the ellipsis


def test_summarise_result_concats_text_blocks():
    res = MCPToolResult(
        content=[{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]
    )
    out = summarise_result(res, max_bytes=64)
    assert out == "line1\nline2"


def test_sanitize_openai_function_name_passthrough_when_compliant():
    assert sanitize_openai_function_name("a_b-c") == "a_b-c"


def test_sanitize_openai_function_name_scrubs_specials():
    out = sanitize_openai_function_name("git#hub_!issue")
    assert out.replace("__", "_") == "git_hub_issue"
    # Conformant to the OpenAI regex
    import re

    assert re.match(r"^[a-zA-Z0-9_-]{1,64}$", out)


def test_sanitize_openai_function_name_truncates_to_64():
    long_name = "a" * 200
    out = sanitize_openai_function_name(long_name)
    assert len(out) == 64
