"""Request-body reading and client-disconnect helpers for the request path."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any

from fastapi import Request

from ocabra.core.inference.errors import _openai_error


async def read_json_object(request: Request) -> dict:
    """The request body as a JSON object, or a 400 (BUG-18).

    ``await request.json()`` raised on malformed JSON, and a valid JSON that is
    not an object (``[]``, ``"x"``) broke on the first ``body.get``: either
    way the client got a 500 for its own mistake.
    """
    try:
        body = await request.json()
    except ValueError as exc:
        raise _openai_error(
            f"The request body is not valid JSON: {exc}",
            "invalid_request_error",
            code="invalid_json",
        ) from exc
    if not isinstance(body, dict):
        raise _openai_error(
            "The request body must be a JSON object.",
            "invalid_request_error",
            code="invalid_json",
        )
    return body


class ClientGoneError(Exception):
    """El cliente cerro la conexion antes de que llegase la respuesta."""


async def await_unless_client_gone(
    request: Request,
    coro: Coroutine[Any, Any, Any],
    *,
    poll_seconds: float = 1.0,
) -> Any:
    """Espera ``coro``, cancelandola si el cliente se desconecta antes.

    En la via de streaming Starlette ya cancela la tarea al desconectarse el
    cliente, y esa cancelacion cierra la conexion con el worker, lo que hace
    que el motor aborte la generacion y libere la GPU (ver el contrato en
    ``WorkerPool._raw_stream``). En la via NO de streaming no ocurre: FastAPI
    deja correr el handler hasta el final.

    Consecuencia medida el 2026-11-07: un cliente se rindio a los ~900 s y
    oCabra siguio generando hasta su propio timeout de 1500 s, con el slot de
    admision y la GPU ocupados diez minutos de mas para una respuesta que ya no
    iba a leer nadie. Cancelar al detectar la desconexion propaga lo mismo que
    en streaming: el ``async with gate.slot(...)`` suelta la admision y httpx
    cierra la conexion aguas arriba.
    """
    tarea = asyncio.ensure_future(coro)
    try:
        while True:
            hecho, _ = await asyncio.wait({tarea}, timeout=poll_seconds)
            if hecho:
                return tarea.result()
            try:
                desconectado = await request.is_disconnected()
            except Exception:  # noqa: BLE001 -- nunca romper por la sonda
                desconectado = False
            if desconectado:
                tarea.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await tarea
                raise ClientGoneError("client disconnected before the response was ready")
    except asyncio.CancelledError:
        tarea.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await tarea
        raise
