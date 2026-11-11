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
    mm.make_room_for_whisper = AsyncMock()
    mm.get_expected_load_seconds = AsyncMock(return_value=None)
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


# ── Timeout del worker (2026-11-10) ──────────────────────────────────────────


def _cliente_con_worker(monkeypatch, post_error: Exception):
    """Handler real con un perfil y un worker de whisper simulados; el POST al
    worker lanza ``post_error``."""
    from types import SimpleNamespace

    import httpx

    from ocabra.api.openai import audio as audio_mod
    from ocabra.backends.base import BackendCapabilities

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

    profile = SimpleNamespace(
        profile_id="whisper-large-v3-diarized",
        base_model_id="whisper/whisper-large-v3",
        load_overrides={"whisper": {"diarization_enabled": True}},
        request_defaults={"diarize": True},
    )
    state = SimpleNamespace(
        model_id="whisper/whisper-large-v3",
        backend_model_id="openai/whisper-large-v3",
        backend_type="whisper",
        capabilities=BackendCapabilities(audio_transcription=True),
    )
    monkeypatch.setattr(audio_mod, "resolve_profile", AsyncMock(return_value=(profile, state)))

    class _Cliente:
        def __init__(self, *a, **k):
            self.timeout = k.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **k):
            outcome = post_error.pop(0) if isinstance(post_error, list) else post_error
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(audio_mod.httpx, "AsyncClient", _Cliente)

    app = FastAPI()
    app.include_router(openai_router, prefix="/v1")
    for dep in (get_current_user, get_openai_user):
        app.dependency_overrides[dep] = _usuario
    mm = MagicMock()
    mm.begin_request = MagicMock(return_value="req-1")
    mm.end_request = MagicMock()
    mm.unload = AsyncMock()
    mm.load = AsyncMock()
    mm.touch_last_request_at = AsyncMock()
    mm.get_state = AsyncMock(return_value=state)
    mm.make_room_for_whisper = AsyncMock()
    mm.get_expected_load_seconds = AsyncMock(return_value=None)
    app.state.model_manager = mm
    backend = SimpleNamespace(health_check=AsyncMock(return_value=True))
    app.state.worker_pool = SimpleNamespace(
        get_worker=MagicMock(return_value=SimpleNamespace(port=18004)),
        get_backend=AsyncMock(return_value=backend),
    )
    app.state.profile_registry = ProfileRegistry()
    _ = httpx
    return TestClient(app), mm


def test_un_worker_lento_da_504_y_no_se_recarga(monkeypatch):
    """Un ReadTimeout es un worker vivo y ocupado. Antes se trataba como caido:
    se mataba, se recargaba desalojando a otro modelo y se repetia el mismo
    audio para volver a cortarse a los 300 s (lote de escriba, 2026-11-10)."""
    import httpx

    cliente, mm = _cliente_con_worker(monkeypatch, httpx.ReadTimeout("slow"))
    respuesta = cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", ("largo.wav", _WAV, "audio/wav"))],
        data={"model": "whisper-large-v3-diarized"},
    )

    assert respuesta.status_code == 504
    assert respuesta.json()["detail"]["error"]["code"] == "transcription_timeout"
    mm.unload.assert_not_awaited()
    mm.load.assert_not_awaited()


def test_un_worker_caido_se_sigue_recargando_una_vez(monkeypatch):
    import httpx

    cliente, mm = _cliente_con_worker(monkeypatch, httpx.ConnectError("refused"))
    respuesta = cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", ("a.wav", _WAV, "audio/wav"))],
        data={"model": "whisper-large-v3-diarized"},
    )

    assert respuesta.status_code == 503
    mm.unload.assert_awaited_once()
    mm.load.assert_awaited_once()


def test_el_timeout_del_worker_sigue_al_de_inferencia(monkeypatch):
    from ocabra.api.openai import audio as audio_mod

    monkeypatch.setattr(audio_mod.settings, "inference_request_timeout_seconds", 1500)
    assert audio_mod._worker_timeout() == 1500.0


def _diarizacion_sin_memoria():
    import httpx

    peticion = httpx.Request("POST", "http://127.0.0.1:18004/transcribe")
    respuesta = httpx.Response(
        503,
        json={"detail": "diarization_out_of_memory: no GPU memory even at batch size 4"},
        request=peticion,
    )
    return httpx.HTTPStatusError("503", request=peticion, response=respuesta)


def _transcripcion_ok():
    import httpx

    peticion = httpx.Request("POST", "http://127.0.0.1:18004/transcribe")
    return httpx.Response(200, json={"text": "hola"}, request=peticion)


def test_sin_memoria_para_diarizar_expulsa_y_reintenta_una_vez(monkeypatch):
    """El worker ya bajo el lote de pyannote hasta 1 y reducir el contexto no
    basto (qwen3.8 en las dos tarjetas con vision, 2026-11-11): se expulsa lo
    que hay en la tarjeta de whisper, como en una carga bajo presion."""
    cliente, mm = _cliente_con_worker(
        monkeypatch, [_diarizacion_sin_memoria(), _transcripcion_ok()]
    )
    mm.evict_for_whisper = AsyncMock(return_value=True)
    mm.get_state = AsyncMock(return_value=None)  # sin cabeceras X-Ocabra
    respuesta = cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", ("a.wav", _WAV, "audio/wav"))],
        data={"model": "whisper-large-v3-diarized"},
    )

    assert respuesta.status_code == 200
    assert respuesta.json()["text"] == "hola"
    mm.evict_for_whisper.assert_awaited_once()


def test_si_ni_expulsando_cabe_da_503_claro_con_retry_after(monkeypatch):
    cliente, mm = _cliente_con_worker(
        monkeypatch, [_diarizacion_sin_memoria(), _diarizacion_sin_memoria()]
    )
    mm.evict_for_whisper = AsyncMock(return_value=True)
    respuesta = cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", ("a.wav", _WAV, "audio/wav"))],
        data={"model": "whisper-large-v3-diarized"},
    )

    assert respuesta.status_code == 503
    assert respuesta.json()["detail"]["error"]["code"] == "diarization_out_of_memory"
    assert respuesta.headers["Retry-After"] == "30"
    mm.evict_for_whisper.assert_awaited_once()


def test_si_no_hay_nada_que_expulsar_no_reintenta(monkeypatch):
    cliente, mm = _cliente_con_worker(monkeypatch, [_diarizacion_sin_memoria()])
    mm.evict_for_whisper = AsyncMock(return_value=False)
    respuesta = cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", ("a.wav", _WAV, "audio/wav"))],
        data={"model": "whisper-large-v3-diarized"},
    )

    assert respuesta.status_code == 503
    assert respuesta.json()["detail"]["error"]["code"] == "diarization_out_of_memory"


# ── Atribucion de la peticion en curso (2026-11-11) ─────────────────────────


def test_la_transcripcion_en_curso_lleva_su_usuario(monkeypatch):
    """La seccion Peticiones del dashboard mostraba "anónimo" en las
    transcripciones: el handler marcaba la peticion en curso sin usuario."""
    import httpx

    from ocabra.core.model_manager import ModelManager

    cliente, mm = _cliente_con_worker(monkeypatch, httpx.ReadTimeout("slow"))
    vistos: list[dict] = []
    real = ModelManager(worker_pool=None)

    def begin_request(model_id, **kwargs):
        rid = real.begin_request(model_id, **kwargs)
        active = real._active_requests[rid]
        vistos.append({"username": active.username, "api_key_name": active.api_key_name})
        return rid

    mm.begin_request = MagicMock(side_effect=begin_request)
    usuario = UserContext(
        user_id="u-1",
        username="revilla",
        role="user",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
        api_key_name="portatil",
    )

    async def _usuario():
        return usuario

    for dep in (get_current_user, get_openai_user):
        cliente.app.dependency_overrides[dep] = _usuario

    cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", ("a.wav", _WAV, "audio/wav"))],
        data={"model": "whisper-large-v3-diarized"},
    )

    assert vistos == [{"username": "revilla", "api_key_name": "portatil"}]


def test_un_anonimo_sigue_sin_usuario():
    from ocabra.core.model_manager import ModelManager

    real = ModelManager(worker_pool=None)
    anon = UserContext(
        user_id=None,
        username=None,
        role="user",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=True,
    )
    rid = real.begin_request("whisper/x", user=anon)
    assert real._active_requests[rid].username is None


# ── Una peticion sin diarizar usa el worker diarizado ya cargado (2026-11-11) ─
#
# faster-whisper-large-v3 y su -diarized son dos workers con los mismos pesos.
# En la 3060, junto al turbo fijado, no caben a la vez, y un cliente que
# alternaba peticiones normales y diarizadas hacia que cada una desalojase a la
# otra (8 veces en 45 min).

from ocabra.api.openai._deps import compute_worker_key  # noqa: E402
from ocabra.core.model_manager import ModelStatus  # noqa: E402

_BASE = "whisper/faster-whisper-large-v3"
_DIAR = {"whisper": {"diarization_enabled": True}, "diarization_enabled": True}
_KEY_DIAR = compute_worker_key(_BASE, _DIAR)


def _gemelos(monkeypatch, *, cargado_normal: bool, cargado_diar: bool, acceso=None):
    from types import SimpleNamespace

    from ocabra.api.openai import audio as audio_mod
    from ocabra.backends.base import BackendCapabilities

    usuario = UserContext(
        user_id="u-1",
        username="rogelio",
        role="user",
        group_ids=[],
        accessible_model_ids=set(acceso or ()),
        is_anonymous=False,
    )
    if acceso is None:
        usuario.role = "system_admin"

    async def _usuario():
        return usuario

    normal = SimpleNamespace(
        profile_id="faster-whisper-large-v3",
        base_model_id=_BASE,
        load_overrides=None,
        request_defaults=None,
        enabled=True,
    )
    diarizado = SimpleNamespace(
        profile_id="faster-whisper-large-v3-diarized",
        base_model_id=_BASE,
        load_overrides=_DIAR,
        request_defaults={"diarize": True},
        enabled=True,
    )

    class _Registry:
        async def get(self, pid):
            return {p.profile_id: p for p in (normal, diarizado)}.get(pid)

        async def list_by_model(self, base):
            return [normal, diarizado] if base == _BASE else []

    def _st(model_id, loaded):
        return SimpleNamespace(
            model_id=model_id,
            backend_model_id="Systran/faster-whisper-large-v3",
            backend_type="whisper",
            status=ModelStatus.LOADED if loaded else ModelStatus.UNLOADED,
            capabilities=BackendCapabilities(audio_transcription=True),
        )

    estados = {
        _BASE: _st(_BASE, cargado_normal),
        _KEY_DIAR: _st(_KEY_DIAR, cargado_diar),
    }
    resueltos: list[str] = []

    async def fake_resolve(pid, *a, **k):
        resueltos.append(pid)
        prof = normal if pid == normal.profile_id else diarizado
        # resolve_profile deja el worker cargado
        return prof, _st(_BASE, True)

    monkeypatch.setattr(audio_mod, "resolve_profile", fake_resolve)
    enviados: list[dict] = []

    class _Cliente:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, *a, **k):
            enviados.append(dict(k.get("data") or {}))
            import httpx

            return httpx.Response(200, json={"text": "hola"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(audio_mod.httpx, "AsyncClient", _Cliente)

    app = FastAPI()
    app.include_router(openai_router, prefix="/v1")
    for dep in (get_current_user, get_openai_user):
        app.dependency_overrides[dep] = _usuario
    mm = MagicMock()
    mm.begin_request = MagicMock(return_value="req-1")
    mm.end_request = MagicMock()
    mm.unload = AsyncMock()
    mm.load = AsyncMock()
    mm.touch_last_request_at = AsyncMock()
    mm.make_room_for_whisper = AsyncMock()
    mm.get_expected_load_seconds = AsyncMock(return_value=None)
    mm.get_state = AsyncMock(side_effect=lambda mid: estados.get(mid))
    app.state.model_manager = mm
    backend = SimpleNamespace(health_check=AsyncMock(return_value=True))
    app.state.worker_pool = SimpleNamespace(
        get_worker=MagicMock(return_value=SimpleNamespace(port=18000)),
        get_backend=AsyncMock(return_value=backend),
    )
    app.state.profile_registry = _Registry()
    return TestClient(app), mm, resueltos, enviados


def _transcribe(cliente, model="faster-whisper-large-v3", **data):
    return cliente.post(
        "/v1/audio/transcriptions",
        files=[("file", ("a.wav", _WAV, "audio/wav"))],
        data={"model": model, **data},
    )


def test_sin_diarizar_va_al_diarizado_cargado_y_le_pide_no_diarizar(monkeypatch):
    cliente, mm, resueltos, enviados = _gemelos(
        monkeypatch, cargado_normal=False, cargado_diar=True
    )
    respuesta = _transcribe(cliente)

    assert respuesta.status_code == 200
    assert resueltos == ["faster-whisper-large-v3-diarized"]
    assert enviados[0]["diarize"] == "false"
    assert mm.begin_request.call_args.args[0] == _KEY_DIAR
    assert mm.make_room_for_whisper.await_args.args[0] == _KEY_DIAR
    assert mm.make_room_for_whisper.await_args.kwargs["diarize"] is False
    assert mm.end_request.call_args.args[0] == _KEY_DIAR


def test_con_su_worker_cargado_no_se_desvia(monkeypatch):
    cliente, _mm, resueltos, enviados = _gemelos(
        monkeypatch, cargado_normal=True, cargado_diar=True
    )
    _transcribe(cliente)
    assert resueltos == ["faster-whisper-large-v3"]
    assert "diarize" not in enviados[0]


def test_sin_el_diarizado_cargado_carga_el_suyo(monkeypatch):
    cliente, _mm, resueltos, _ = _gemelos(monkeypatch, cargado_normal=False, cargado_diar=False)
    _transcribe(cliente)
    assert resueltos == ["faster-whisper-large-v3"]


def test_pedir_diarizacion_no_se_desvia(monkeypatch):
    cliente, _mm, resueltos, _ = _gemelos(monkeypatch, cargado_normal=False, cargado_diar=True)
    _transcribe(cliente, diarize="true")
    assert resueltos == ["faster-whisper-large-v3"]


def test_sin_acceso_al_diarizado_no_se_desvia(monkeypatch):
    cliente, _mm, resueltos, _ = _gemelos(
        monkeypatch,
        cargado_normal=False,
        cargado_diar=True,
        acceso={"faster-whisper-large-v3"},
    )
    _transcribe(cliente)
    assert resueltos == ["faster-whisper-large-v3"]
