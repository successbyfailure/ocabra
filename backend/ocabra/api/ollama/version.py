"""
GET /api/version — Ollama-compatible version probe.

Part of the Ollama API that oCabra was missing: clients (and the ``ollama``
CLI itself) ask for it to decide what the server supports before sending
anything else, so its absence looked like "this is not an Ollama server".
Reported 2026-11-08 by a benchmark harness that could not record which
version produced each measurement.

Deliberately unauthenticated, like ``/health`` and like Ollama itself: it is
how a client identifies the server, and it reveals nothing a request to
``/health`` does not.
"""

from __future__ import annotations

from fastapi import APIRouter

from ocabra.config import settings

router = APIRouter()


@router.get("/version", summary="Server version (Ollama-compatible)")
async def version() -> dict[str, str]:
    """Return the running oCabra version, in Ollama's shape.

    Response: ``{"version": "<semver>"}``.
    """

    return {"version": settings.app_version}
