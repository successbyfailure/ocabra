# Briefing: Stream 2-C — Audio Backend (Whisper + TTS)

**Prerequisito: Stream 1-B completado.**
**Rama:** `feat/2-C-audio`

## Objetivo

Implementar dos backends de audio:
1. **Whisper** — transcripción de audio (compatible OpenAI `/v1/audio/transcriptions`)
2. **TTS** — síntesis de voz (compatible OpenAI `/v1/audio/speech`)

## Ficheros propios

```
backend/ocabra/backends/whisper_backend.py
backend/ocabra/backends/tts_backend.py
workers/whisper_worker.py
workers/tts_worker.py
backend/tests/test_whisper_backend.py
backend/tests/test_tts_backend.py
```

---

## Whisper Backend

### Modelos soportados

| Modelo | VRAM | Idiomas |
|--------|------|---------|
| `openai/whisper-tiny` | ~0.5 GB | multilingual |
| `openai/whisper-base` | ~1 GB | multilingual |
| `openai/whisper-small` | ~2 GB | multilingual |
| `openai/whisper-medium` | ~5 GB | multilingual |
| `openai/whisper-large-v3` | ~10 GB | multilingual |
| `openai/whisper-large-v3-turbo` | ~6 GB | multilingual |
| `systran/faster-whisper-*` | igual | multilingual, faster-whisper |

### workers/whisper_worker.py

```python
"""
FastAPI worker usando faster-whisper para transcripción eficiente.
Endpoints:
  POST /transcribe  → multipart/form-data con "file" (audio)
  GET  /health
  GET  /info
"""

from faster_whisper import WhisperModel

@app.post("/transcribe")
async def transcribe(
    file: UploadFile,
    language: str | None = None,
    response_format: str = "json",  # json | text | srt | vtt | verbose_json
    timestamp_granularities: list[str] = ["segment"]
) -> TranscriptionResponse:
    ...
```

### whisper_backend.py

```python
class WhisperBackend(BackendInterface):

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(audio_transcription=True)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        # Lookup por nombre de modelo conocido, o estima por tamaño de ficheros
        KNOWN_SIZES = {
            "whisper-tiny": 300, "whisper-base": 500,
            "whisper-small": 1200, "whisper-medium": 3000,
            "whisper-large-v3": 6000, "whisper-large-v3-turbo": 3500,
        }
        ...

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        """
        El body para /transcribe viene como multipart desde el cliente OpenAI.
        Reenvía el fichero de audio al worker via httpx multipart.
        Traduce respuesta al formato OpenAI Transcription.
        """
```

### Formato OpenAI Transcription

```python
# Input: multipart/form-data
# - file: audio (mp3, mp4, mpeg, mpga, m4a, wav, webm)
# - model: str
# - language?: str (ISO 639-1)
# - prompt?: str (hint de contexto)
# - response_format?: json|text|srt|vtt|verbose_json
# - temperature?: float
# - timestamp_granularities[]?: word|segment

# Output:
{ "text": "Transcripción completa aquí." }

# verbose_json output:
{
  "text": "...",
  "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": "..."}],
  "language": "es"
}
```

---

## TTS Backend

### Modelos soportados

| Modelo | VRAM | Notas |
|--------|------|-------|
| `Qwen/Qwen3-TTS` | ~8 GB | multilingüe, alta calidad |
| `hexgrad/Kokoro-82M` | ~1 GB | rápido, calidad buena |
| `suno/bark` | ~6 GB | expresivo, lento |

### workers/tts_worker.py

```python
"""
FastAPI worker para síntesis de voz.
Endpoints:
  POST /synthesize  → retorna audio binario
  GET  /voices      → lista de voces disponibles
  GET  /health
  GET  /info
"""

@app.post("/synthesize")
async def synthesize(
    input: str,               # texto a sintetizar
    voice: str = "default",   # voz seleccionada
    response_format: str = "mp3",  # mp3|opus|aac|flac|wav|pcm
    speed: float = 1.0
) -> StreamingResponse:
    # Retorna audio binario con Content-Type correcto
    ...

@app.get("/voices")
async def list_voices() -> list[VoiceInfo]:
    ...
```

### tts_backend.py

```python
class TTSBackend(BackendInterface):

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(tts=True)

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        """
        Traduce formato OpenAI /v1/audio/speech al worker:
        OpenAI input: {model, input, voice, response_format, speed}
        Worker input: {input, voice, response_format, speed}
        Retorna audio binario con el Content-Type correcto.
        """
```

### Formato OpenAI Speech

```python
# Input JSON:
{
  "model": "qwen3-tts",
  "input": "Hola, esto es una prueba de síntesis de voz.",
  "voice": "alloy",      # alloy|echo|fable|onyx|nova|shimmer (mapear a voces del modelo)
  "response_format": "mp3",
  "speed": 1.0
}

# Output: audio binario (streaming)
# Content-Type: audio/mpeg (mp3) | audio/opus | audio/aac | audio/flac | audio/wav
```

### Mapa de voces OpenAI → modelo

Cada modelo TTS tiene sus propias voces. Definir un mapa de compatibilidad:

```python
VOICE_MAPPINGS = {
    "qwen3-tts": {
        "alloy": "zh-CN-XiaoxiaoNeural",
        "nova": "en-US-AriaNeural",
        # ...
    },
    "kokoro": {
        "alloy": "af",
        "echo": "am",
        # ...
    }
}
```

---

## Notas comunes

- Ambos workers usan FastAPI con Uvicorn, igual que el diffusers worker.
- El audio se puede procesar de forma síncrona (run_in_executor) para no bloquear el event loop.
- Los ficheros de audio temporales deben borrarse después de procesarlos.
- Para Whisper, usar `compute_type="float16"` en GPU para velocidad óptima.

## Tests requeridos

- Test de WhisperBackend: mock de worker, verificar traducción de formato
- Test de TTSBackend: mock de worker, verificar Content-Type correcto
- Test de mapeo de voces: todas las voces OpenAI tienen un equivalente

## Dependencias adicionales (añadir a pyproject.toml)

```toml
audio = [
    "faster-whisper>=1.1",
    "soundfile>=0.12",
]
# TTS: transformers ya está en deps base
```

## Estado

- [ ] En progreso
- [x] Completado
