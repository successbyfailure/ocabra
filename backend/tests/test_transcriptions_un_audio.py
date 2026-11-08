"""Mandar varios audios en una peticion no puede salir con un 200 a medias.

``/v1/audio/transcriptions`` acepta un ``file``, como el endpoint de OpenAI.
Lo que hacia antes con varios era quedarse con el ultimo y devolver un 200 sin
mencionar los otros: el cliente creia haber transcrito cuatro audios y tenia
uno. Reportado el 2026-11-08 por un arnes que mandaba cuatro WAV y recibia una
sola transcripcion, y reproducido con dos: volvia el segundo, en silencio.

Transcribir por lotes va por ``/v1/batches``, una linea por audio.

Estos tests pasan por el handler a proposito. El arreglo del idioma de este
mismo dia se metio en una funcion que las peticiones normales no visitan, con
sus tests en verde y produccion sin arreglar; la leccion es probar la capa por
la que entra la peticion.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.api.openai import router as openai_router
from ocabra.api.openai._deps import get_openai_user
from ocabra.core.profile_registry import ProfileRegistry

_WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


@pytest.fixture
def cliente() -> TestClient:
    admin = UserContext(
        user_id=None,
        username=None,
        role="system_admin",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )

    async def _usuario():
        return admin

    app = FastAPI()
    app.include_router(openai_router, prefix="/v1")
    for dep in (get_current_user, get_openai_user):
        app.dependency_overrides[dep] = _usuario

    mm = MagicMock()
    mm.get_state = AsyncMock(return_value=None)
    mm.list_states = AsyncMock(return_value=[])
    mm.touch_last_request_at = AsyncMock()
    app.state.model_manager = mm
    app.state.worker_pool = MagicMock()
    # Un registro de verdad, vacio: con un MagicMock la resolucion revienta
    # con un TypeError al await-ear sus atributos, y el test del caso normal no
    # distinguiria un 500 de un 400.
    app.state.profile_registry = ProfileRegistry()
    return TestClient(app)


def test_dos_audios_se_rechazan_en_vez_de_transcribir_uno(cliente):
    respuesta = cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", ("uno.wav", _WAV, "audio/wav")), ("file", ("dos.wav", _WAV, "audio/wav"))],
        data={"model": "whisper-large-v3-turbo"},
    )
    assert respuesta.status_code == 400
    error = respuesta.json()["detail"]["error"]
    assert error["code"] == "multiple_files"
    assert error["param"] == "file"
    # El mensaje tiene que decir cuantos venian y por donde va un lote.
    assert "2" in error["message"]
    assert "/v1/batches" in error["message"]


def test_cuatro_audios_tambien(cliente):
    """El caso del informe."""
    respuesta = cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", (f"{i}.wav", _WAV, "audio/wav")) for i in range(4)],
        data={"model": "whisper-large-v3-turbo"},
    )
    assert respuesta.status_code == 400
    assert "4" in respuesta.json()["detail"]["error"]["message"]


def test_un_solo_audio_pasa_del_corte(cliente):
    """No puede quedar un 400 para el caso normal.

    Mas adelante fallara por no haber modelo que resolver, pero la guarda de
    varios ficheros no es la que tiene que cortarlo.
    """
    respuesta = cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", ("uno.wav", _WAV, "audio/wav"))],
        data={"model": "whisper-large-v3-turbo"},
    )
    cuerpo = respuesta.json()
    codigo = (cuerpo.get("detail", {}) or {}).get("error", {}).get("code")
    assert codigo != "multiple_files"


def test_sin_fichero_no_lo_corta_esta_guarda(cliente):
    """Cero ficheros es cosa de la validacion de FastAPI, no de la guarda."""
    respuesta = cliente.post("/v1/audio/transcriptions", data={"model": "whisper-large-v3-turbo"})
    assert respuesta.status_code == 422
