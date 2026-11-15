"""Realtime API: dialecto GA, frecuencias reales y voz (2026-11-16).

OpenClaw habla el dialecto GA (``session.audio.input``/``.output``) y manda
PCM16 a 24 kHz. oCabra solo leía el beta y trataba toda entrada como 16 kHz:
el audio llegaba a VAD y STT a 0,67× de velocidad, y la voz, el idioma y los
parámetros de turno se ignoraban.
"""

from __future__ import annotations

import base64
import io
import json
import math
import struct
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.core.realtime_audio import (
    _ALAW_DECODE,
    _ULAW_DECODE,
    ALAW,
    PCM16,
    ULAW,
    InputAudioDecoder,
    OutputAudioEncoder,
    StreamResampler,
    _linear_to_alaw,
    _linear_to_ulaw,
    openai_vad_threshold_to_rms,
    parse_audio_format,
    split_wav,
)
from ocabra.core.realtime_session import RealtimeSession
from ocabra.core.vad import SimpleVAD

# ── utilidades ──────────────────────────────────────────────────────────


def _sine(rate: int, seconds: float, freq: float = 440.0, amp: float = 0.5) -> list[int]:
    n = int(rate * seconds)
    return [int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]


def _pcm(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _zero_crossings(samples: list[int]) -> int:
    return sum(1 for a, b in zip(samples, samples[1:], strict=False) if (a < 0) != (b < 0))


def _wav(samples: list[int], rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(_pcm(samples))
    return buf.getvalue()


# ── formatos ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "ga", "rate", "esperado"),
    [
        ("pcm16", False, None, (PCM16, 16000)),  # lo documentado por oCabra
        ("pcm16", False, 24000, (PCM16, 24000)),  # extensión input_audio_sample_rate
        ({"type": "audio/pcm"}, False, None, (PCM16, 24000)),  # GA: 24 kHz
        ({"type": "audio/pcm", "rate": 24000}, True, None, (PCM16, 24000)),
        ({"type": "audio/pcmu"}, True, None, (ULAW, 8000)),
        ({"type": "audio/pcma"}, True, None, (ALAW, 8000)),
        ("g711_ulaw", False, None, (ULAW, 8000)),
        ("mp3", False, None, None),
    ],
)
def test_formatos_de_los_dos_dialectos(value, ga, rate, esperado):
    assert parse_audio_format(value, ga=ga, rate=rate) == esperado


def test_remuestreo_24k_a_16k_conserva_duracion_y_tono():
    src = _sine(24000, 1.0)
    out = StreamResampler(24000, 16000).process(src)
    assert abs(len(out) - 16000) <= 2
    # 440 Hz siguen siendo 440 Hz: ~880 cruces por cero en 1 s.
    assert abs(_zero_crossings(out) - 880) <= 9  # 1 %: el arranque del filtro


def test_remuestreo_por_trozos_igual_que_de_una_vez():
    src = _sine(24000, 0.5)
    whole = StreamResampler(24000, 16000).process(src)
    chunked = []
    r = StreamResampler(24000, 16000)
    for i in range(0, len(src), 480):  # trozos de 20 ms
        chunked += r.process(src[i : i + 480])
    assert abs(len(whole) - len(chunked)) <= 1
    assert max(abs(a - b) for a, b in zip(whole, chunked, strict=False)) <= 2


def test_g711_ida_y_vuelta():
    for s in (-32000, -1000, -10, 0, 10, 1000, 32000):
        assert abs(_ULAW_DECODE[_linear_to_ulaw(s)] - s) <= max(4, abs(s) * 0.04)
        assert abs(_ALAW_DECODE[_linear_to_alaw(s)] - s) <= max(16, abs(s) * 0.04)


def test_decodificador_de_entrada_24k_da_16k_y_aguanta_bytes_sueltos():
    raw = _pcm(_sine(24000, 1.0))
    dec = InputAudioDecoder(PCM16, 24000)
    out = dec.feed(raw[:1001]) + dec.feed(raw[1001:])  # corte a mitad de muestra
    assert abs(len(out) // 2 - 16000) <= 3


def test_decodificador_mulaw_8k_da_16k():
    ulaw = bytes(_linear_to_ulaw(s) for s in _sine(8000, 1.0))
    out = InputAudioDecoder(ULAW, 8000).feed(ulaw)
    assert abs(len(out) // 2 - 16000) <= 3


def test_wav_de_piper_a_22050_sale_a_24k():
    pcm, rate = split_wav(_wav(_sine(22050, 1.0), 22050))
    assert rate == 22050
    out = OutputAudioEncoder(PCM16, 24000, rate).feed(pcm)
    assert abs(len(out) // 2 - 24000) <= 3


def test_umbral_de_openai_se_traduce_a_energia():
    base = SimpleVAD.SPEECH_THRESHOLD
    assert openai_vad_threshold_to_rms(0.5, base) == pytest.approx(base)
    assert openai_vad_threshold_to_rms(0.8, base) == pytest.approx(base * 1.6)
    assert openai_vad_threshold_to_rms(0.03, base) == pytest.approx(0.03)  # RMS de oCabra
    assert openai_vad_threshold_to_rms(None, base) is None


# ── sesión ──────────────────────────────────────────────────────────────


class _Compat:
    def resolve(self, model, voice):
        return {"echo": "piper-es-es-davefx-medium", "alloy": "piper-es-mx-claude-high"}.get(voice)


def _session(monkeypatch) -> RealtimeSession:
    monkeypatch.setattr("ocabra.config.settings.realtime_default_stt_model", "")
    monkeypatch.setattr("ocabra.config.settings.realtime_default_tts_model", "")
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ws.app = SimpleNamespace(state=SimpleNamespace(openai_tts_compat=_Compat()))
    known = {
        "whisper/turbo",
        "piper/es_ES-davefx-medium",
        "piper/es_MX-claude-high",
    }
    mm = MagicMock()
    mm.list_states = AsyncMock(return_value=[])
    mm.get_state = AsyncMock(side_effect=lambda mid: SimpleNamespace() if mid in known else None)
    pool = MagicMock()
    pool.get_worker = MagicMock(return_value=None)
    registry = MagicMock()
    profiles = {
        "piper-es-es-davefx-medium": SimpleNamespace(
            profile_id="piper-es-es-davefx-medium",
            base_model_id="piper/es_ES-davefx-medium",
            enabled=True,
            profile_kind="execution",
        ),
    }
    registry.get = AsyncMock(side_effect=lambda pid: profiles.get(pid))
    s = RealtimeSession(
        ws=ws, model_id="llm", worker_pool=pool, model_manager=mm, profile_registry=registry
    )
    s.stt_model_id = "whisper/turbo"
    s.tts_model_id = "piper/es_MX-claude-high"
    s._should_use_native_audio = AsyncMock(return_value=False)
    s._load_model_with_progress = AsyncMock()
    s._load_remaining_models = AsyncMock()
    return s


def _events(session) -> list[dict]:
    return [json.loads(c.args[0]) for c in session.ws.send_text.await_args_list]


GA_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "realtime",
        "model": "recomendado-ha-chatbots",
        "instructions": "Eres un asistente.",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "noise_reduction": None,
                "transcription": {"model": "gpt-4o-transcribe", "language": "es"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "echo"},
        },
        "tools": [{"type": "function", "name": "f", "description": "d", "parameters": {}}],
        "tool_choice": "auto",
    },
}


@pytest.mark.asyncio
async def test_session_update_ga_se_aplica_y_se_refleja(monkeypatch):
    s = _session(monkeypatch)
    await s._handle_session_update(GA_UPDATE)

    assert (s.input_audio_format, s.input_audio_rate) == (PCM16, 24000)
    assert s._stt_language == "es"
    # "gpt-4o-transcribe" no existe aquí: se queda el STT de oCabra.
    assert s.stt_model_id == "whisper/turbo"
    # "echo" → la voz fijada para echo en el modo compatibilidad.
    assert s.voice == "echo"
    assert s.tts_model_id == "piper/es_ES-davefx-medium"
    assert s._vad.SILENCE_DURATION_MS == 500
    assert s._vad.SPEECH_THRESHOLD == pytest.approx(SimpleVAD.SPEECH_THRESHOLD)
    assert len(s.tools) == 1

    updated = [e for e in _events(s) if e["type"] == "session.updated"][-1]["session"]
    assert updated["type"] == "realtime"
    assert updated["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert updated["audio"]["input"]["transcription"]["language"] == "es"
    assert updated["audio"]["output"]["voice"] == "echo"
    assert updated["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}


@pytest.mark.asyncio
async def test_el_dialecto_beta_sigue_igual(monkeypatch):
    s = _session(monkeypatch)
    await s._handle_session_update(
        {
            "session": {
                "input_audio_format": "pcm16",
                "voice": "alloy",
                "input_audio_transcription": {"model": "whisper/turbo", "language": "es"},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 600},
            }
        }
    )
    assert (s.input_audio_format, s.input_audio_rate) == (PCM16, 16000)
    updated = [e for e in _events(s) if e["type"] == "session.updated"][-1]["session"]
    assert "audio" not in updated and updated["input_audio_format"] == "pcm16"
    assert s._dialect_ga is False


@pytest.mark.asyncio
async def test_entrada_24k_llega_a_16k_y_el_vad_mide_tiempo_real(monkeypatch):
    s = _session(monkeypatch)
    await s._handle_session_update(GA_UPDATE)
    s._commit_and_respond = AsyncMock()
    s._start_partial = MagicMock()

    voz = _sine(24000, 1.2, amp=0.3)
    silencio = [0] * int(24000 * 1.5)
    audio = _pcm(voz + silencio)
    for i in range(0, len(audio), 960):  # trozos de 20 ms a 24 kHz
        await s._handle_audio_append({"audio": base64.b64encode(audio[i : i + 960]).decode()})

    ev = _events(s)
    started = next(e for e in ev if e["type"] == "input_audio_buffer.speech_started")
    stopped = next(e for e in ev if e["type"] == "input_audio_buffer.speech_stopped")
    # A 16 kHz leído como 16 kHz: el silencio de 500 ms se detecta a ~1,7 s
    # de audio real, no a ~2,5 s como cuando 24 kHz se leía como 16 kHz.
    assert started["audio_start_ms"] < 400
    assert 1600 <= stopped["audio_end_ms"] <= 1900
    s._commit_and_respond.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_response_false_solo_confirma(monkeypatch):
    s = _session(monkeypatch)
    update = json.loads(json.dumps(GA_UPDATE))
    update["session"]["audio"]["input"]["turn_detection"]["create_response"] = False
    await s._handle_session_update(update)
    s._commit_and_respond = AsyncMock()
    s._commit_audio_only = AsyncMock()
    s._start_partial = MagicMock()
    audio = _pcm(_sine(24000, 1.0, amp=0.3) + [0] * 24000)
    for i in range(0, len(audio), 960):
        await s._handle_audio_append({"audio": base64.b64encode(audio[i : i + 960]).decode()})
    s._commit_and_respond.assert_not_awaited()
    s._commit_audio_only.assert_awaited_once()


@pytest.mark.asyncio
async def test_la_respuesta_sale_a_24k_con_nombres_ga(monkeypatch):
    s = _session(monkeypatch)
    await s._handle_session_update(GA_UPDATE)
    s._tts_ready.set()
    s._record_request = AsyncMock()
    s._worker_pool.get_worker = MagicMock(return_value=SimpleNamespace(port=18999))
    piper_wav = _wav(_sine(22050, 1.0), 22050)

    posted = {}

    async def _post(self, url, json=None, **kw):
        posted.update(json or {})
        return SimpleNamespace(content=piper_wav, raise_for_status=lambda: None)

    with patch("httpx.AsyncClient.post", new=_post):
        await s._synthesize_and_send("Hola", "resp_1", "item_1")

    assert posted["response_format"] == "wav"
    deltas = [e for e in _events(s) if e["type"] == "response.output_audio.delta"]
    assert deltas and not any(e["type"] == "response.audio.delta" for e in _events(s))
    total = sum(len(base64.b64decode(e["delta"])) for e in deltas)
    assert abs(total // 2 - 24000) <= 5  # 1 s a 24 kHz, no 0,92 s


@pytest.mark.asyncio
async def test_salida_pcmu_para_telefonia(monkeypatch):
    s = _session(monkeypatch)
    update = json.loads(json.dumps(GA_UPDATE))
    update["session"]["audio"]["output"]["format"] = {"type": "audio/pcmu"}
    await s._handle_session_update(update)
    s._tts_ready.set()
    s._record_request = AsyncMock()
    s._worker_pool.get_worker = MagicMock(return_value=SimpleNamespace(port=18999))

    async def _post(self, url, json=None, **kw):
        return SimpleNamespace(
            content=_wav(_sine(22050, 1.0), 22050), raise_for_status=lambda: None
        )

    with patch("httpx.AsyncClient.post", new=_post):
        await s._synthesize_and_send("Hola", "resp_1", "item_1")
    deltas = [e for e in _events(s) if e["type"] == "response.output_audio.delta"]
    assert abs(sum(len(base64.b64decode(e["delta"])) for e in deltas) - 8000) <= 5


@pytest.mark.asyncio
async def test_el_flujo_clasico_emite_la_transcripcion_del_usuario(monkeypatch):
    """OpenClaw no mostraba lo que dijo el usuario: el flujo Whisper → texto
    → LLM solo lo metía dentro de conversation.item.created."""
    s = _session(monkeypatch)
    s.audio_buffer.extend(b"\x00\x01" * 16000)
    s._transcribe = AsyncMock(return_value="¿Cuál es la capital de Francia?")
    await s._commit_audio_only()
    ev = [e for e in _events(s) if e["type"] == "conversation.item.input_audio_transcription.completed"]
    assert ev and ev[0]["transcript"] == "¿Cuál es la capital de Francia?"
    creado = next(e for e in _events(s) if e["type"] == "conversation.item.created")
    assert ev[0]["item_id"] == creado["item"]["id"]
