"""Tres fallos de producción del 2026-11-16.

1. Realtime en modo conversación: el ruido que disparaba el VAD llegaba a
   Whisper sin el filtro de alucinaciones y salía texto inventado (a menudo
   en otro alfabeto), al que además el LLM contestaba.
2. Un proyector de visión cargado a demanda encogió ornith de 229k a 143k y
   las peticiones de texto de 146k respondían 503 sin fin: llama-server no
   puede descargar el proyector.
3. La reserva de arranque de los presets de dos tarjetas no cubría a whisper
   más el margen del vigilante de VRAM: nemotron se desalojaba y recargaba.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.core.model_manager import ModelManager, ModelStatus
from ocabra.core.realtime_session import RealtimeSession
from ocabra.core.runtime_coordinator import multi_gpu_launch_context


def _session(monkeypatch) -> RealtimeSession:
    monkeypatch.setattr("ocabra.config.settings.realtime_default_stt_model", "")
    monkeypatch.setattr("ocabra.config.settings.realtime_default_tts_model", "")
    ws = MagicMock()
    ws.send_text = AsyncMock()
    mm = MagicMock()
    mm.list_states = AsyncMock(return_value=[])
    mm.begin_request = MagicMock(return_value="r")
    mm.end_request = MagicMock()
    s = RealtimeSession(ws=ws, model_id="llm", worker_pool=MagicMock(), model_manager=mm)
    s.stt_model_id = "whisper/turbo"
    s._ensure_stt_worker = AsyncMock(return_value=SimpleNamespace(port=18000))
    s._record_request = AsyncMock()
    s._should_use_native_audio = AsyncMock(return_value=False)
    return s


def _whisper(segments):
    text = " ".join(seg["text"] for seg in segments)

    async def _post(self, url, **kw):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"text": text, "segments": segments},
        )

    return _post


@pytest.mark.asyncio
async def test_el_ruido_transcrito_como_alucinacion_no_llega_al_llm(monkeypatch):
    s = _session(monkeypatch)
    s._generate_response = AsyncMock()
    s.audio_buffer.extend(b"\x00\x01" * 8000)
    ruido = [{"text": "字幕由Amara.org社区提供", "start": 0.0, "end": 0.5, "no_speech_prob": 0.9}]
    with patch("httpx.AsyncClient.post", new=_whisper(ruido)):
        await s._commit_and_respond()
    s._generate_response.assert_not_awaited()
    tipos = [c.args[0] for c in s.ws.send_text.await_args_list]
    assert not any("input_audio_transcription.completed" in t for t in tipos)


@pytest.mark.asyncio
async def test_la_voz_real_si_llega(monkeypatch):
    s = _session(monkeypatch)
    s._generate_response = AsyncMock()
    s.audio_buffer.extend(b"\x00\x01" * 16000)
    voz = [{"text": "¿Qué hora es?", "start": 0.0, "end": 1.2, "no_speech_prob": 0.02}]
    with patch("httpx.AsyncClient.post", new=_whisper(voz)):
        await s._commit_and_respond()
    s._generate_response.assert_awaited_once()
    assert s.conversation[-1]["content"] == "¿Qué hora es?"


# ── proyector de visión ─────────────────────────────────────────────────


def _manager(*, vision_mb=True, active=True, busy=False):
    mm = ModelManager(worker_pool=None)
    state = SimpleNamespace(
        model_id="m::h",
        status=ModelStatus.LOADED,
        backend_type="llamacpp_mks",
        backend_model_id="m",
        live_context={"n_ctx": 143360, **({"vision_mb": {"1": 1144}} if vision_mb else {})},
    )
    mm._states = {"m::h": state}
    backend = SimpleNamespace(
        vision_is_active=lambda _m: active,
        _contains_image=lambda body: "image_url" in str(body),
    )
    mm._registered_backend = lambda _t: backend
    mm.is_busy = lambda _m: busy
    mm.unload = AsyncMock()
    return mm


@pytest.mark.asyncio
async def test_una_peticion_de_texto_recarga_sin_proyector():
    mm = _manager()
    assert await mm.release_vision_by_reload("m::h", {"messages": [{"content": "hola"}]})
    mm.unload.assert_awaited_once_with("m::h", reason="vision_release")


@pytest.mark.parametrize(
    ("kwargs", "body"),
    [
        ({"busy": True}, {"messages": []}),  # tiene peticiones en curso
        ({"vision_mb": False}, {"messages": []}),  # el encogido no fue por el proyector
        ({"active": False}, {"messages": []}),
        ({}, {"messages": [{"content": [{"type": "image_url"}]}]}),  # trae imagen
    ],
)
@pytest.mark.asyncio
async def test_no_recarga_cuando_no_toca(kwargs, body):
    mm = _manager(**kwargs)
    assert not await mm.release_vision_by_reload("m::h", body)
    mm.unload.assert_not_awaited()


# ── reserva de whisper en la 3060 ────────────────────────────────────────


def test_la_reserva_por_defecto_deja_a_nemotron_bajo_el_umbral_del_vigilante():
    from ocabra.config import Settings

    reserve = Settings().llama_cpp_multi_gpu_reserve_mb
    probe = {  # sonda real de nemotron-3.5-lightning (2026-11-15)
        "used_mb": {"0": 9350, "1": 21644},
        "limit_mb": {"0": 11382, "1": 23601},
        "kv_mb_per_1k": {"0": 3.1, "1": 4.15},
    }
    launch, at_launch = multi_gpu_launch_context(1048576, probe, reserve)
    whisper_mb = 2838  # lo que ocupa whisper en la 3060 en producción
    assert at_launch["0"] + whisper_mb <= 12288 * 0.96
