"""Media URLs in chat requests may not point at private addresses (SEG-11).

An ``image_url`` / ``video_url`` / ``audio_url`` part with an ``http(s)`` URL is
fetched by the backend (vLLM, llama.cpp…) from inside oCabra's network. Without
this check, anyone allowed to use a vision model could make it request
``http://ollama:11434``, the host's other services, or any LAN address, and
read part of the answer back through the model.

``data:`` URLs are always fine. For ``http(s)``, every address the host
resolves to must be public, unless the host is in ``MEDIA_URL_ALLOWED_HOSTS``
(hostnames or CIDRs, comma-separated) or ``MEDIA_URL_ALLOW_PRIVATE`` is on.

Not covered: the backend resolves the name again when it downloads, so a DNS
answer that changes in between (rebinding) still gets through. Fetching the
media here and inlining it as ``data:`` would close that too.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import urlsplit

from ocabra.config import settings

_MEDIA_PARTS = ("image_url", "video_url", "audio_url")


def media_urls(body: dict[str, Any]) -> list[str]:
    """Every non-``data:`` media URL in the request's messages."""
    urls: list[str] = []
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            for kind in _MEDIA_PARTS:
                value = part.get(kind)
                url = value.get("url") if isinstance(value, dict) else value
                if isinstance(url, str) and url and not url.startswith("data:"):
                    urls.append(url)
    return urls


def _allowed_hosts() -> tuple[set[str], list[ipaddress._BaseNetwork]]:
    names: set[str] = set()
    networks: list[ipaddress._BaseNetwork] = []
    for item in (settings.media_url_allowed_hosts or "").split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            names.add(item)
    return names, networks


async def url_problem(url: str) -> str | None:
    """Why ``url`` may not be fetched, or ``None`` if it may."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "only data:, http: and https: URLs are accepted"
    if settings.media_url_allow_private:
        return None
    host = parts.hostname.lower()
    names, networks = _allowed_hosts()
    if host in names:
        return None
    try:
        infos = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                host, parts.port or (443 if parts.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            ),
            timeout=3.0,
        )  # fmt: skip
    except (OSError, TimeoutError):
        return f"cannot resolve {host}"
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if any(address in network for network in networks):
            continue
        if not address.is_global:
            return f"{host} resolves to a non-public address ({address})"
    return None


async def check_media_urls(body: dict[str, Any]) -> None:
    """Raise a 400 (OpenAI error shape) for the first media URL not allowed."""
    for url in media_urls(body):
        problem = await url_problem(url)
        if problem:
            from ocabra.api.openai._deps import _openai_error

            raise _openai_error(
                f"Media URL not allowed: {problem}. Send the image as a data: URL, or ask "
                "the administrator to allow the host (MEDIA_URL_ALLOWED_HOSTS).",
                "invalid_request_error",
                param="messages",
                code="media_url_not_allowed",
                status_code=400,
            )
