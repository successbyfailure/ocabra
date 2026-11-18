"""SEG-11: media URLs in chat requests may not point at private addresses."""

from __future__ import annotations

import socket

import pytest
from fastapi import HTTPException

from ocabra.api.openai import _media_urls as media
from ocabra.config import settings


def _resolves_to(monkeypatch, *addresses: str):
    async def fake(self, host, port, **kwargs):
        family = socket.AF_INET6 if ":" in addresses[0] else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]

    import asyncio

    monkeypatch.setattr(asyncio.get_running_loop().__class__, "getaddrinfo", fake)


def _chat(url: str, kind: str = "image_url") -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}, {"type": kind, kind: {"url": url}}],
            }
        ]
    }


def test_only_non_data_media_urls_are_collected():
    body = {
        "messages": [
            {"role": "system", "content": "plain text"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "image_url", "image_url": "http://a.example/x.png"},
                    {"type": "video_url", "video_url": {"url": "https://b.example/v.mp4"}},
                    {"type": "audio_url", "audio_url": {"url": "http://c.example/a.wav"}},
                ],
            },
        ]
    }
    assert media.media_urls(body) == [
        "http://a.example/x.png",
        "https://b.example/v.mp4",
        "http://c.example/a.wav",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.8.5", "172.18.0.4", "192.168.1.1", "169.254.169.254", "::1"]
)
async def test_private_addresses_are_refused(monkeypatch, address):
    _resolves_to(monkeypatch, address)
    with pytest.raises(HTTPException) as exc:
        await media.check_media_urls(_chat("http://ollama:11434/api/tags"))
    assert exc.value.status_code == 400
    assert "non-public" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_one_private_answer_among_public_ones_is_enough_to_refuse(monkeypatch):
    _resolves_to(monkeypatch, "93.184.216.34", "10.0.0.1")
    assert await media.url_problem("https://mixed.example/x.png")


@pytest.mark.asyncio
async def test_public_addresses_and_data_urls_pass(monkeypatch):
    _resolves_to(monkeypatch, "93.184.216.34")
    await media.check_media_urls(_chat("https://example.com/cat.png"))
    await media.check_media_urls(_chat("data:image/png;base64,AAAA"))


@pytest.mark.asyncio
async def test_other_schemes_are_refused():
    assert await media.url_problem("file:///etc/passwd")
    assert await media.url_problem("gopher://x/")


@pytest.mark.asyncio
async def test_allowlist_by_name_and_by_network(monkeypatch):
    _resolves_to(monkeypatch, "10.0.8.20")
    monkeypatch.setattr(settings, "media_url_allowed_hosts", "images.lan, 10.0.8.0/24")
    assert await media.url_problem("http://images.lan/x.png") is None
    assert await media.url_problem("http://other.lan/x.png") is None  # 10.0.8.20 in the CIDR
    monkeypatch.setattr(settings, "media_url_allowed_hosts", "images.lan")
    assert await media.url_problem("http://other.lan/x.png")


@pytest.mark.asyncio
async def test_allow_private_switch(monkeypatch):
    _resolves_to(monkeypatch, "127.0.0.1")
    monkeypatch.setattr(settings, "media_url_allow_private", True)
    assert await media.url_problem("http://localhost/x.png") is None


@pytest.mark.asyncio
async def test_unresolvable_host_is_refused(monkeypatch):
    async def fail(self, *a, **k):
        raise socket.gaierror("nope")

    import asyncio

    monkeypatch.setattr(asyncio.get_running_loop().__class__, "getaddrinfo", fail)
    assert "cannot resolve" in await media.url_problem("http://nowhere.invalid/x.png")


def test_chat_completions_checks_before_forwarding():
    import inspect

    from ocabra.api.openai import chat

    source = inspect.getsource(chat.chat_completions)
    assert source.index("check_media_urls(body)") < source.index("check_capability")
