"""Helper compartido por los routers de compatibilidad de nombres OpenAI.

``openai_image_compat.py`` y ``openai_tts_compat.py`` son gemelos: cada uno
recupera su registro desde ``app.state`` y da 503 si todavía no está listo.
"""

from __future__ import annotations

from typing import TypeVar

from fastapi import HTTPException, Request

T = TypeVar("T")


def get_compat_registry(request: Request, *, attr: str, label: str) -> T:
    """Devuelve el registro de compatibilidad guardado en ``app.state.<attr>``.

    Args:
        attr: nombre del atributo en ``app.state`` (``openai_tts_compat``...).
        label: etiqueta para el mensaje de error (``image``, ``TTS``...).

    Raises:
        HTTPException 503: el registro aún no se ha inicializado.
    """
    registry = getattr(request.app.state, attr, None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI {label} compat registry not available",
            headers={"Retry-After": "5"},
        )
    return registry
