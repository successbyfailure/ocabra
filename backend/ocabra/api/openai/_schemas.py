"""OpenAI-compatible request/response schemas for API documentation.

These schemas are used ONLY for OpenAPI spec generation (Swagger/ReDoc).
The actual endpoints parse request.json() directly for flexibility.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Chat Completions ────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, assistant, tool", examples=["user"])
    content: str | list[Any] | None = Field(
        None, description="Message content (string or multimodal array)", examples=["Hello!"]
    )
    name: str | None = Field(None, description="Optional name for the participant")
    tool_calls: list[Any] | None = Field(None, description="Tool calls (assistant only)")
    tool_call_id: str | None = Field(None, description="Tool call ID (tool role only)")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Profile ID o model ID", examples=["qwen3-8b"])
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float | None = Field(0.7, description="Sampling temperature (0.0-2.0)")
    top_p: float | None = Field(1.0, description="Nucleus sampling threshold")
    max_tokens: int | None = Field(None, description="Max tokens to generate")
    stream: bool = Field(False, description="Stream response via SSE")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    seed: int | None = Field(None, description="Reproducibility seed")
    frequency_penalty: float | None = Field(0.0, description="Frequency penalty (-2.0 to 2.0)")
    presence_penalty: float | None = Field(0.0, description="Presence penalty (-2.0 to 2.0)")
    tools: list[Any] | None = Field(None, description="Tool definitions for function calling")
    tool_choice: str | Any | None = Field(None, description="Tool choice strategy")
    response_format: dict | None = Field(
        None, description='Response format, e.g. {"type": "json_object"}'
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "model": "qwen3-8b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in Spanish"},
            ],
            "max_tokens": 256,
            "temperature": 0.7,
        }
    ]}}


# ── Text Completions ───────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str = Field(..., description="Profile ID o model ID", examples=["qwen3-8b"])
    prompt: str | list[str] = Field(..., description="Input prompt(s)")
    max_tokens: int | None = Field(256, description="Max tokens to generate")
    temperature: float | None = Field(0.7, description="Sampling temperature")
    stream: bool = Field(False, description="Stream response via SSE")
    stop: str | list[str] | None = Field(None, description="Stop sequences")


# ── Embeddings ──────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="Profile ID o model ID", examples=["qwen3-embedding-8b"])
    input: str | list[str] = Field(..., description="Text(s) to embed", examples=["Hello world"])
    encoding_format: str | None = Field("float", description="Encoding format: float or base64")


# ── TTS ─────────────────────────────────────────────────────────

class SpeechRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo TTS", examples=["kokoro-82m"])
    input: str = Field(..., description="Texto a sintetizar", examples=["Hola desde oCabra"])
    voice: str = Field("alloy", description="Voz (depende del modelo). Consulta /v1/audio/voices")
    response_format: str = Field(
        "mp3", description="Formato de audio: mp3, wav, opus, flac, pcm, aac"
    )
    speed: float = Field(1.0, description="Velocidad (0.25-4.0)")
    language: str = Field("Auto", description="Idioma (Auto para deteccion automatica)")
    reference_audio: str | None = Field(
        None, description="Audio de referencia para voice cloning (base64 WAV, min 5s recomendado)"
    )
    reference_text: str | None = Field(
        None, description="Transcripcion del audio de referencia (mejora calidad)"
    )
    speaker: str | None = Field(
        None, description="Speaker name para modelos CustomVoice (ryan, vivian, etc.)"
    )
    instruct: str | None = Field(
        None, description="Instruccion de estilo (ej: 'Speak calmly and slowly')"
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "model": "kokoro-82m",
            "input": "Hello, this is a test of text to speech.",
            "voice": "af_heart",
            "response_format": "mp3",
            "speed": 1.0,
        }
    ]}}


# ── STT ─────────────────────────────────────────────────────────
# (multipart, documented via description only)


# ── Images ──────────────────────────────────────────────────────

class ImageGenerationRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo de imagen")
    prompt: str = Field(..., description="Texto descriptivo de la imagen", examples=["A cat on a rainbow"])
    n: int = Field(1, description="Numero de imagenes a generar")
    size: str = Field("512x512", description="Tamano: 256x256, 512x512, 1024x1024")
    num_inference_steps: int | None = Field(None, description="Pasos de inferencia")
    guidance_scale: float | None = Field(None, description="Guidance scale (CFG)")
    negative_prompt: str | None = Field(None, description="Negative prompt")


# ── Rerank ──────────────────────────────────────────────────────

class RerankRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo de reranking")
    query: str = Field(..., description="Query de busqueda")
    documents: list[str] = Field(..., description="Documentos a reordenar por relevancia")
    top_n: int | None = Field(None, description="Numero de resultados a retornar")


# ── Music ───────────────────────────────────────────────────────

class MusicGenerationRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo de musica")
    prompt: str = Field(..., description="Descripcion de la musica a generar")
    duration_seconds: int = Field(30, description="Duracion en segundos")
    lyrics: str | None = Field(None, description="Letra de la cancion (opcional)")
