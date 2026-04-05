# Plan: Voice Pipeline — oCabra

**Fecha:** 2026-04-05  
**Estado:** Activo  
**Objetivo:** Que oCabra sirva como backend completo de voz para asistentes, en dos fases:
- Fase 1 — Los tres endpoints oficiales OpenAI (`/v1/audio/transcriptions`, `/v1/chat/completions`, `/v1/audio/speech`) funcionan correctamente y con baja latencia.
- Fase 2 — Implementar la [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) via WebSocket (`GET /v1/realtime`), que unifica los tres endpoints en un único canal bidireccional.

**Contexto:** El cliente es una app Android (Assistant) que usa los tres endpoints en secuencia. El flujo es: grabar audio → STT → LLM streaming → TTS → reproducir. La app también tiene modo VAD (detección de silencio) y activación por palabra clave.

---

## Fase 1 — Tres endpoints oficiales funcionando correctamente

### F1-1: TTS — soporte real de `response_format`

**Problema:** El `tts_worker.py` real (`backend/workers/tts_worker.py`) siempre devuelve WAV con `media_type="audio/wav"`, ignorando el campo `response_format` del request. El cliente Android guarda el archivo con extensión `.mp3` y Android `MediaPlayer` falla o reproduce audio corrupto.

**Archivos afectados:**
- `backend/workers/tts_worker.py` — `/synthesize` endpoint
- `backend/ocabra/backends/tts_backend.py` — `forward_stream()` (pasa `response_format` al worker)

**Cambios en `backend/workers/tts_worker.py`:**

```python
# En SynthesizeRequest añadir:
response_format: str = "mp3"   # mp3 | wav | pcm | opus | flac

# En /synthesize, tras generar audio_bytes (WAV interno):
audio_bytes, content_type = _encode_audio(raw_wav_bytes, body.response_format)

# Nueva función de encoding:
def _encode_audio(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    """Convierte WAV interno al formato solicitado."""
    fmt = fmt.lower()
    if fmt == "wav":
        return wav_bytes, "audio/wav"
    if fmt == "pcm":
        return _wav_to_pcm(wav_bytes), "audio/pcm"
    if fmt in ("mp3", "opus", "flac", "aac"):
        return _wav_to_container(wav_bytes, fmt), FORMAT_CONTENT_TYPES[fmt]
    return wav_bytes, "audio/wav"
```

Para la conversión real usar `soundfile` + `pydub` (ya deben estar disponibles en el entorno TTS). Si no están, usar `ffmpeg` via subprocess como fallback.

El worker tiene tres variantes de modelo (`base`, `custom_voice`, `placeholder`). Los tres deben pasar por la misma función `_encode_audio`.

**Stub en `workers/tts_worker.py` (raíz):** Este archivo es un stub que genera tonos sintéticos. Actualizar igualmente para que respete `response_format`, ya que se usa en tests y dev sin GPU.

---

### F1-2: TTS — streaming real por frases (sentence streaming)

**Problema actual:** El worker genera TODO el audio antes de responder. Para textos largos (respuesta del LLM) esto añade 2-5 segundos antes del primer byte de audio.

**Solución:** Añadir endpoint `/synthesize/stream` en el worker que:
1. Recibe el texto completo
2. Lo divide en frases (split por `.`, `!`, `?`, `\n`)
3. Sintetiza frase a frase
4. Yield chunks de audio conforme se generan

```python
@app.post("/synthesize/stream")
async def synthesize_stream(body: SynthesizeRequest) -> StreamingResponse:
    async def _generate():
        for sentence in _split_sentences(body.input):
            if not sentence.strip():
                continue
            wav = await asyncio.to_thread(_synthesize, runtime, sentence, ...)
            encoded, _ = _encode_audio(wav, body.response_format)
            yield encoded

    return StreamingResponse(_generate(), media_type=content_type)
```

El endpoint `/v1/audio/speech` de oCabra ya hace streaming via `_stream_audio()` — solo necesita apuntar a `/synthesize/stream` en lugar de `/synthesize`.

**Latencia esperada:** Primera frase de audio en ~300-800ms desde recibir el texto.

---

### F1-3: STT — verificar compatibilidad de formatos con Android

El cliente Android puede enviar:
- **M4A/AAC** (via `MediaRecorder`) — cuando STT es remoto
- **WAV PCM 16kHz** (via `AudioRecord`) — cuando STT es local pero se usa remoto como fallback

El endpoint `/v1/audio/transcriptions` pasa el archivo tal cual al worker Whisper via `files={"file": (...)}`. Verificar:

1. El worker `backend/workers/whisper_worker.py` acepta M4A/AAC sin error
2. El campo `Content-Type` del multipart se transmite correctamente
3. El campo `model` en el request del Android puede ser cualquier string (e.g. "whisper-1") — oCabra debe resolver esto al model_id real configurado, o el usuario configura el model_id correcto en la app

**Acción:** Añadir en `backend/ocabra/api/openai/audio.py` un log de warning si `model_id` no existe en el model manager, con mensaje claro: "El model_id enviado por el cliente no coincide con ningún modelo configurado en oCabra."

---

### F1-4: TTS — endpoint de voces disponibles

La app Android muestra voces hardcodeadas ("alloy", etc). Añadir o documentar que `GET /v1/audio/speech/voices` (extensión propia) devuelve las voces disponibles para un modelo TTS, para que la app pueda listarlas dinámicamente.

El worker ya expone `GET /voices` — exponerlo via oCabra:

```python
# En backend/ocabra/api/openai/audio.py
@router.get("/audio/voices")
async def list_voices(model: str, request: Request, user: ...):
    """Lista voces disponibles para un modelo TTS."""
    ...
```

---

### F1-5: Chat — streaming con TTS en paralelo desde el cliente

No es un cambio en oCabra, sino documentar el patrón correcto para que el cliente Android lo implemente (ver plan del Assistant):

```
LLM SSE stream → buffer por frases → POST /v1/audio/speech por frase
```

oCabra ya lo soporta. El cambio es en el cliente.

---

### Tabla de archivos — Fase 1

| Archivo | Cambio | Prioridad |
|---------|--------|-----------|
| `backend/workers/tts_worker.py` | Encoding real MP3/WAV/PCM + streaming por frases | **Alta** |
| `workers/tts_worker.py` (stub raíz) | Mismo: respetar `response_format` | Media |
| `backend/ocabra/api/openai/audio.py` | Apuntar TTS a `/synthesize/stream`, añadir `/audio/voices` | Alta |
| `backend/ocabra/backends/tts_backend.py` | Asegurar que `forward_stream` pasa `response_format` | Media |
| `backend/workers/whisper_worker.py` | Verificar aceptación de M4A/AAC | Media |

---

## Fase 2 — OpenAI Realtime API (`GET /v1/realtime`)

### Protocolo de referencia

La Realtime API de OpenAI usa WebSocket. El cliente se conecta a `wss://.../v1/realtime?model=<model>` con header `OpenAI-Beta: realtime=v1`.

**Eventos del cliente → servidor:**
```
session.update           — configura VAD, voice, instrucciones, modalities
input_audio_buffer.append  — chunk de audio PCM16 base64-encoded
input_audio_buffer.commit  — fin del turno de usuario
input_audio_buffer.clear   — descarta buffer
response.create          — fuerza respuesta (sin VAD)
response.cancel          — cancela respuesta en curso
```

**Eventos del servidor → cliente:**
```
session.created          — confirmación de conexión
session.updated          — confirmación de session.update
input_audio_buffer.speech_started   — VAD detectó voz
input_audio_buffer.speech_stopped   — VAD detectó silencio
input_audio_buffer.committed        — buffer procesado
conversation.item.created           — transcripción disponible
response.created                    — respuesta iniciada
response.audio.delta                — chunk de audio PCM16 base64
response.audio.done                 — audio completo
response.audio_transcript.delta     — chunk de transcripción del TTS
response.audio_transcript.done      — transcripción TTS completa
response.done                       — respuesta completa con usage
error                               — error
```

### F2-1: Endpoint WebSocket

**Archivo nuevo:** `backend/ocabra/api/openai/realtime.py`

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
router = APIRouter()

@router.websocket("/realtime")
async def realtime(websocket: WebSocket, model: str, user: UserContext = Depends(...)):
    await websocket.accept()
    session = RealtimeSession(websocket, model, user)
    await session.run()
```

Montar en `main.py` junto con los demás routers de `/v1/`.

### F2-2: RealtimeSession

**Archivo nuevo:** `backend/ocabra/core/realtime_session.py`

Responsabilidades:
1. Gestionar estado de sesión (audio buffer, historial de conversación, config VAD/voz)
2. Recibir eventos del cliente y despacharlos
3. Coordinar el pipeline: audio → Whisper → LLM → TTS → audio
4. Emitir eventos al cliente según protocolo

```python
class RealtimeSession:
    def __init__(self, ws, model_id, user):
        self.ws = ws
        self.llm_model_id = model_id
        self.stt_model_id = None   # configurado via session.update
        self.tts_model_id = None   # configurado via session.update
        self.voice = "alloy"
        self.instructions = ""
        self.audio_buffer = bytearray()
        self.conversation: list[dict] = []
        self.vad_enabled = True

    async def run(self):
        await self._send_session_created()
        async for message in self._receive_events():
            await self._dispatch(message)
```

### F2-3: VAD en servidor

Para la modalidad `server_vad` (detección automática de fin de turno):

**Librería recomendada:** `silero-vad` (PyTorch, ~1MB, muy preciso) o lógica simple por energía RMS (sin dependencias extra, suficiente para empezar).

```python
class SimpleVAD:
    """VAD por energía RMS. Sin dependencias extra."""
    SPEECH_THRESHOLD = 0.02    # RMS > this = speech
    SILENCE_DURATION_MS = 800  # ms de silencio para cortar turno
    MIN_SPEECH_MS = 200        # ms mínimos de voz para procesar

    def process_chunk(self, pcm16_bytes: bytes) -> VadEvent | None:
        # Retorna: SPEECH_STARTED, SPEECH_STOPPED, o None
        ...
```

El audio del cliente llega como PCM16 a 16kHz, mono (base64). El VAD procesa chunk a chunk y emite `input_audio_buffer.speech_started` / `speech_stopped`.

### F2-4: Pipeline interno

```
audio_buffer (PCM16) → Whisper worker (HTTP interno)
                            ↓ texto transcrito
                   LLM worker (HTTP streaming SSE)
                            ↓ tokens streaming
                   SentenceSplitter
                            ↓ frases completas
                   TTS worker (/synthesize/stream)
                            ↓ audio WAV chunks
                   encode → PCM16 base64
                            ↓
                   WebSocket → cliente
```

El pipeline es asíncrono con back-pressure: si el cliente no consume el audio rápido, el generador TTS hace pausa.

### F2-5: Configuración de la sesión

El cliente puede configurar en `session.update`:
```json
{
  "type": "session.update",
  "session": {
    "modalities": ["text", "audio"],
    "instructions": "Eres un asistente conciso...",
    "voice": "alloy",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "input_audio_transcription": {
      "model": "whisper-large-v3"
    },
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "prefix_padding_ms": 300,
      "silence_duration_ms": 800
    },
    "tools": [...],
    "tool_choice": "auto"
  }
}
```

oCabra necesita mapear `voice` al modelo TTS cargado y `input_audio_transcription.model` al modelo STT.

### F2-6: Tool calls en Realtime

El protocolo soporta tool calls con eventos específicos:
- `response.function_call_arguments.delta` — streaming de argumentos
- `response.function_call_arguments.done` — llamada lista para ejecutar
- `conversation.item.create` con `type: "function_call_output"` — resultado de la herramienta

Para la primera implementación, omitir tool calls y documentarlo. Añadir en Fase 2.1 si se necesita.

### Tabla de archivos — Fase 2

| Archivo | Cambio |
|---------|--------|
| `backend/ocabra/api/openai/realtime.py` | **Nuevo** — endpoint WebSocket `/v1/realtime` |
| `backend/ocabra/core/realtime_session.py` | **Nuevo** — gestión de sesión, pipeline, VAD |
| `backend/ocabra/core/vad.py` | **Nuevo** — SimpleVAD (RMS) con interfaz para Silero futuro |
| `backend/ocabra/main.py` | Incluir `realtime.router` |
| `backend/pyproject.toml` | Añadir `websockets` si no está (FastAPI usa `starlette` que ya lo incluye) |
| `backend/workers/tts_worker.py` | Requiere F1-1 y F1-2 completados |

---

## Orden de ejecución recomendado

```
Fase 1:
  [1] F1-1 TTS encoding real (MP3/WAV/PCM) — BLOQUEANTE para que Android funcione
  [2] F1-2 TTS streaming por frases — latencia baja de primer audio
  [3] F1-3 STT verificar M4A — puede ser solo tests/validación
  [4] F1-4 Voices endpoint — nice-to-have para la app

Fase 2:
  [5] F2-1 + F2-3 WebSocket endpoint + VAD simple — base del protocolo
  [6] F2-2 + F2-4 RealtimeSession + pipeline completo
  [7] F2-5 session.update config
  [8] F2-6 Tool calls (opcional)
```

---

## Notas de compatibilidad

- **Android `MediaPlayer`** soporta MP3, AAC/M4A, WAV, OGG. No soporta PCM raw sin cabecera WAV.
- **Formato recomendado para Realtime:** PCM16 raw (sin cabecera) a 16kHz, mono — que es el formato nativo de Whisper y el más eficiente para streaming.
- **Formato recomendado para 3-API pipeline:** MP3 — comprimido, soportado nativamente por MediaPlayer.
- La Realtime API oficial de OpenAI también soporta `g711_ulaw` y `g711_alaw` — no implementar en primera fase.

---

## Dependencias de infra

- `soundfile` o `pydub` para encoding de audio en el TTS worker (verificar disponibilidad en el entorno Docker)
- `ffmpeg` como fallback via subprocess para conversión de formatos
- `websockets` — ya incluido via `starlette[full]` que usa FastAPI
- Silero VAD (opcional, Fase 2.1): `silero-vad` pip package, ~2MB
