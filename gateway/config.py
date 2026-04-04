"""Gateway configuration: builds the Host → service routing table from env vars."""
from __future__ import annotations

import os
from urllib.parse import urlparse


def _host(url: str) -> str:
    """Return the netloc (hostname[:port]) of a URL, lowercased. Empty string if blank."""
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc.lower()


# Where to reach the ocabra API from inside docker
OCABRA_API_URL: str = os.getenv("OCABRA_API_URL", "http://api:8000").rstrip("/")

# Shared secret for gateway → ocabra internal calls (empty = disabled)
GATEWAY_SERVICE_TOKEN: str = os.getenv("GATEWAY_SERVICE_TOKEN", "")

# Port this gateway process listens on (container-internal; host port set via docker-compose)
GATEWAY_PORT: int = int(os.getenv("GATEWAY_PORT", "9000"))

# Host for the directory page (served by the gateway itself, not proxied)
DIRECTORY_HOST: str = _host(os.getenv("DIRECTORY_UI_URL", ""))

# Seconds between touch calls for the same service (throttle to avoid noise)
TOUCH_INTERVAL_S: int = int(os.getenv("GATEWAY_TOUCH_INTERVAL_S", "30"))

# How many seconds to wait after triggering start before giving up on loading page
STARTUP_TIMEOUT_S: int = int(os.getenv("GATEWAY_STARTUP_TIMEOUT_S", "180"))

# Service definitions: (service_id, UI_URL env var, base_url env var, display name)
_SERVICE_DEFS: list[tuple[str, str, str, str]] = [
    ("hunyuan", "HUNYUAN_UI_URL", "HUNYUAN_BASE_URL", "Hunyuan3D"),
    ("comfyui", "COMFYUI_UI_URL", "COMFYUI_BASE_URL", "ComfyUI"),
    ("a1111",   "A1111_UI_URL",   "A1111_BASE_URL",   "Automatic1111"),
    ("acestep", "ACESTEP_UI_URL", "ACESTEP_BASE_URL",  "ACE-Step"),
]

# Map: hostname → {service_id, upstream, display_name, ui_url}
SERVICE_BY_HOST: dict[str, dict] = {}

for _sid, _ui_var, _base_var, _name in _SERVICE_DEFS:
    _ui_url  = os.getenv(_ui_var, "")
    _base_url = os.getenv(_base_var, "")
    _h = _host(_ui_url)
    if _h and _base_url:
        SERVICE_BY_HOST[_h] = {
            "service_id":   _sid,
            "upstream":     _base_url.rstrip("/"),
            "display_name": _name,
            "ui_url":       _ui_url,
        }

# Reverse map: service_id → host (for cross-linking from directory page)
SERVICE_ID_TO_HOST: dict[str, str] = {v["service_id"]: k for k, v in SERVICE_BY_HOST.items()}
