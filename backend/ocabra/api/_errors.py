"""One error format per API family (REF-10).

The handlers raise ``HTTPException`` with whatever ``detail`` they like (an
OpenAI-style ``{"error": {...}}`` dict most of the time, sometimes a plain
string or an Ollama-style ``{"error": "text"}``), and FastAPI answers
``{"detail": <that>}``. Neither SDK reads ``detail``:

* OpenAI clients (``/v1/*``) read a top-level ``error`` object
  ``{"message", "type", "param", "code"}``.
* Ollama clients (``/api/*``) read a top-level ``error`` string.

The exception handlers in ``main.py`` build their bodies with
:func:`error_content`: the family's ``error`` is added at the top level and
``detail`` is kept exactly as before, so clients that already read ``detail``
(VHS-transcripts, the frontend, federation peers parsing an older node) keep
working. ``detail`` is deprecated on ``/v1`` and ``/api`` and will be retired
later. Every other path (``/ocabra/*``, ``/health``…) is left untouched.

Out of scope: responses built directly by middlewares (admission, federation
rules), which already use the family's shape, and errors sent inside a
stream that has already started.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any, Literal

ErrorFamily = Literal["openai", "ollama"]


def error_family(path: str) -> ErrorFamily | None:
    """The API family whose error format applies to *path*, or ``None``."""
    if path == "/v1" or path.startswith("/v1/"):
        return "openai"
    if path == "/api" or path.startswith("/api/"):
        return "ollama"
    return None


def request_path(request: Any) -> str:
    """``request.url.path``, or ``""`` for a request without a URL."""
    url = getattr(request, "url", None)
    return str(getattr(url, "path", "") or "")


def _default_type(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def _status_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return f"HTTP {status_code}"


def _validation_param(err: dict[str, Any]) -> str | None:
    loc = [str(p) for p in err.get("loc") or () if p not in ("body", "query", "path", "header")]
    return ".".join(loc) or None


def _validation_message(errors: list[Any]) -> str:
    parts: list[str] = []
    for err in errors:
        if isinstance(err, dict):
            where = _validation_param(err)
            msg = str(err.get("msg") or "invalid value")
            parts.append(f"{where}: {msg}" if where else msg)
        else:
            parts.append(str(err))
    return "; ".join(parts) or "Invalid request"


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def openai_error(detail: Any, status_code: int) -> dict[str, Any]:
    """OpenAI's ``{"message", "type", "param", "code"}`` from any ``detail``.

    * ``{"error": {...}}`` — that object, completed with the missing keys
      (extra keys such as ``retry_after`` are kept).
    * ``{"error": "text"}`` or ``{"message": ..., ...}`` — mapped over.
    * a list (``RequestValidationError.errors()``) — joined, with the first
      field as ``param``.
    * a string — the message.
    """
    obj: dict[str, Any]
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        obj = dict(detail["error"])
    elif isinstance(detail, dict) and isinstance(detail.get("error"), str):
        obj = {k: v for k, v in detail.items() if k in ("type", "param", "code")}
        obj["message"] = detail["error"]
    elif isinstance(detail, dict) and "message" in detail:
        obj = {k: v for k, v in detail.items() if k in ("message", "type", "param", "code")}
    elif isinstance(detail, list):
        first = next((e for e in detail if isinstance(e, dict)), None)
        obj = {
            "message": _validation_message(detail),
            "type": "invalid_request_error",
            "param": _validation_param(first) if first else None,
            "code": None,
        }
    elif detail is None or detail == "":
        obj = {"message": _status_phrase(status_code)}
    else:
        obj = {"message": _as_text(detail)}

    message = obj.get("message")
    if message is None or message == "":
        obj["message"] = _status_phrase(status_code)
    elif not isinstance(message, str):
        obj["message"] = _as_text(message)
    obj.setdefault("type", _default_type(status_code))
    obj.setdefault("param", None)
    obj.setdefault("code", None)
    return obj


def ollama_error(detail: Any, status_code: int) -> str:
    """Ollama's ``error`` string from any ``detail``."""
    if isinstance(detail, dict):
        err = detail.get("error")
        if isinstance(err, str):
            return err
        if isinstance(err, dict) and err.get("message") is not None:
            return _as_text(err["message"])
        if detail.get("message") is not None:
            return _as_text(detail["message"])
        return _as_text(detail)
    if isinstance(detail, list):
        return _validation_message(detail)
    if detail is None or detail == "":
        return _status_phrase(status_code)
    return _as_text(detail)


def error_content(path: str, status_code: int, detail: Any) -> dict[str, Any]:
    """The JSON body for an error on *path*: ``detail`` as today, plus the
    family's top-level ``error`` on ``/v1`` and ``/api``."""
    content: dict[str, Any] = {"detail": detail}
    family = error_family(path)
    if family == "openai":
        content["error"] = openai_error(detail, status_code)
    elif family == "ollama":
        content["error"] = ollama_error(detail, status_code)
    return content
