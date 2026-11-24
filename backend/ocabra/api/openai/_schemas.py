"""OpenAI/Ollama-compatible request schemas, for OpenAPI docs and validation.

These schemas feed the OpenAPI spec (Swagger/ReDoc) *and*, via
:func:`validate_body`, give the request handlers a permissive early gate on
the body they read with :func:`ocabra.core.inference.body.read_json_object`
(REF-11, closes BUG-18).

The validation is deliberately **permissive**: every model sets
``extra="allow"`` so unknown fields a real OpenAI/Ollama client sends pass
through untouched, and optional fields stay loosely typed. :func:`validate_body`
only rejects input that is genuinely malformed — a wrong JSON type for a known
field, or a missing truly-required field (``model``/``messages``/``input``/…) —
and returns the original ``dict`` unchanged so the backend body is identical to
the raw-dict path it replaced.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocabra.core.inference.errors import _openai_error

# ── Chat Completions ────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, assistant, tool", examples=["user"])
    content: str | list[Any] | None = Field(
        None, description="Message content (string or multimodal array)", examples=["Hello!"]
    )
    name: str | None = Field(None, description="Optional name for the participant")
    tool_calls: list[Any] | None = Field(None, description="Tool calls (assistant only)")
    tool_call_id: str | None = Field(None, description="Tool call ID (tool role only)")

    # Clients send extra per-message keys (``refusal``, ``function_call``,
    # Ollama's ``images``, ``thinking``…); keep them.
    model_config = ConfigDict(extra="allow")


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
    reasoning_effort: str | None = Field(
        None,
        description=(
            "Thinking level for reasoning-capable models: `low`, `medium` or "
            "`high` (`off`/`none`/`disabled` turn thinking off; `default` keeps "
            "the model profile's launch default). "
            "oCabra translates the value per backend: vLLM gets it as "
            "`extra_body.chat_template_kwargs.reasoning_effort`, Ollama ≥0.5 "
            "receives `think: 'low'|'medium'|'high'|False`, llama.cpp-native "
            "models (e.g. gemma4:26b, qwen3.8:27b) get it forwarded directly "
            "as `reasoning_effort`/`reasoning: 'off'`, and backends with no "
            "native control ignore it. Non-reasoning models silently accept "
            "the field."
        ),
    )
    reasoning_budget: int | None = Field(
        None,
        description=(
            "Hard token cap on thinking, for llama.cpp-native models only "
            "(e.g. gemma4:26b, qwen3.8:27b). Forces the model to cut "
            "reasoning short once the cap is hit and write the final "
            "answer, unlike `reasoning_effort` (a soft hint the chat "
            "template may ignore) — guarantees non-empty `content`. "
            "Verified live 2026-10-29 against qwen3.8:27b: budget=200 "
            "produced 674 chars of reasoning and a clean `finish_reason: "
            '"stop"` (vs ~16.6k chars unconstrained). Server default is '
            "unrestricted (-1) if you don't set this. No native "
            "equivalent on other backends — silently ignored there."
        ),
    )
    response_format: dict | None = Field(
        None,
        description=(
            "Structured output. Two shapes supported, forwarded verbatim to "
            "the backend:\n"
            '- `{"type": "json_object"}` — any valid JSON. Supported by '
            "vLLM, Ollama ≥0.1.33 and llama.cpp.\n"
            '- `{"type": "json_schema", "json_schema": {"name": "...", '
            '"strict": true, "schema": {...}}}` — constrained decoding '
            "against a JSON Schema. Supported by vLLM ≥0.26, Ollama ≥0.4 "
            "and llama.cpp (grammar). The model is forced to produce output "
            "that validates against the schema."
        ),
    )

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {
                    "model": "qwen3-8b",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Say hello in Spanish"},
                    ],
                    "max_tokens": 256,
                    "temperature": 0.7,
                },
                {
                    "model": "mistral-7b",
                    "messages": [
                        {"role": "user", "content": "Give me a fictional person"},
                    ],
                    "max_tokens": 300,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "person",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "age": {"type": "integer"},
                                    "hobbies": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["name", "age", "hobbies"],
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            ]
        }
    }


# ── Text Completions ───────────────────────────────────────────


class CompletionRequest(BaseModel):
    model: str = Field(..., description="Profile ID o model ID", examples=["qwen3-8b"])
    # ``list[Any]`` also covers OpenAI's token-id arrays (``list[int]`` /
    # ``list[list[int]]``); backends that reject them answer the client.
    prompt: str | list[Any] = Field(..., description="Input prompt(s)")
    max_tokens: int | None = Field(256, description="Max tokens to generate")
    temperature: float | None = Field(0.7, description="Sampling temperature")
    stream: bool = Field(False, description="Stream response via SSE")
    stop: str | list[str] | None = Field(None, description="Stop sequences")

    model_config = ConfigDict(extra="allow")


# ── Embeddings ──────────────────────────────────────────────────


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="Profile ID o model ID", examples=["qwen3-embedding-8b"])
    # ``list[Any]`` also covers OpenAI's token-id arrays for embeddings.
    input: str | list[Any] = Field(..., description="Text(s) to embed", examples=["Hello world"])
    encoding_format: str | None = Field("float", description="Encoding format: float or base64")

    model_config = ConfigDict(extra="allow")


# ── TTS ─────────────────────────────────────────────────────────


class SpeechRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo TTS", examples=["kokoro-82m"])
    input: str = Field(..., description="Texto a sintetizar", examples=["Hola desde oCabra"])
    text: str | None = Field(
        None,
        description="Alias de compatibilidad de input para clientes TTS no estrictos",
    )
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "kokoro-82m",
                    "input": "Hello, this is a test of text to speech.",
                    "voice": "af_heart",
                    "response_format": "mp3",
                    "speed": 1.0,
                }
            ]
        }
    }


# ── STT ─────────────────────────────────────────────────────────
# (multipart, documented via description only)


# ── Images ──────────────────────────────────────────────────────


class ImageGenerationRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo de imagen")
    prompt: str = Field(
        ..., description="Texto descriptivo de la imagen", examples=["A cat on a rainbow"]
    )
    n: int = Field(1, description="Numero de imagenes a generar")
    size: str = Field("512x512", description="Tamano: 256x256, 512x512, 1024x1024")
    num_inference_steps: int | None = Field(None, description="Pasos de inferencia")
    guidance_scale: float | None = Field(None, description="Guidance scale (CFG)")
    negative_prompt: str | None = Field(None, description="Negative prompt")

    model_config = ConfigDict(extra="allow")


# ── Rerank ──────────────────────────────────────────────────────


class RerankRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo de reranking")
    query: str = Field(..., description="Query de busqueda")
    documents: list[str] = Field(..., description="Documentos a reordenar por relevancia")
    top_n: int | None = Field(None, description="Numero de resultados a retornar")


# ── Pooling / Score / Classify ──────────────────────────────────


class PoolingRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo con soporte pooling")
    input: str | list[str] | None = Field(
        None,
        description="Texto(s) de entrada. El resto del body se reenvia verbatim al backend.",
    )

    model_config = ConfigDict(extra="allow")


class ScoreRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo de scoring/similaridad")
    queries: str | list[str] = Field(
        ..., description="Query(s). Acepta tambien los alias 'query' o 'text_1'."
    )
    documents: str | list[str] = Field(
        ...,
        description=(
            "Documento(s) emparejados con 'queries'. Acepta tambien 'document' o "
            "'text_2'. Si ambos son listas, deben tener la misma longitud."
        ),
    )


class ClassifyRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo de clasificacion")
    input: str | list[str] = Field(..., description="Texto(s) a clasificar")


# ── Music ───────────────────────────────────────────────────────


class MusicGenerationRequest(BaseModel):
    model: str = Field(..., description="Profile ID del modelo de musica (ACE-Step)")
    prompt: str = Field(..., description="Descripcion de la musica a generar")
    lyrics: str | None = Field(None, description="Letra de la cancion (opcional)")
    audio_duration: int | None = Field(
        None, description="Duracion en segundos (10-600, default segun el modelo)"
    )
    bpm: int | None = Field(None, description="Tempo en BPM (30-300)")
    key_scale: str | None = Field(
        None, description="Tonalidad, p.ej. 'C Major'", examples=["C Major"]
    )
    time_signature: str | None = Field(None, description="Compas, p.ej. '4'", examples=["4"])
    inference_steps: int | None = Field(
        None, description="Pasos de difusion (default 8 en el modelo turbo)"
    )
    thinking: bool | None = Field(
        None, description="Planificacion de la cancion asistida por LM (default false)"
    )
    vocal_language: str | None = Field(
        None, description="Idioma de la letra, p.ej. 'en'/'es' (default 'en')"
    )
    seed: int | None = Field(None, description="Semilla; -1 para aleatoria")
    response_format: str | None = Field(
        None,
        json_schema_extra={"enum": ["mp3", "wav", "flac"]},
        description="Formato de audio de salida (default 'mp3')",
    )

    model_config = ConfigDict(extra="allow")


# ── Ollama-compatible requests (/api/*) ─────────────────────────
# These are not part of the OpenAI OpenAPI tables; they exist so the Ollama
# handlers get the same permissive early validation. Only ``model`` is truly
# required — Ollama's own API treats a body with just ``model`` (no
# ``messages``/``prompt``) as a model-load/keepalive call, so those stay
# optional. ``extra="allow"`` keeps ``options``, ``think``, ``format``,
# ``keep_alive``, ``tools``, ``images``… untouched.


class OllamaChatRequest(BaseModel):
    model: str = Field(..., description="Profile ID or Ollama model name")
    messages: list[Any] | None = Field(None, description="Conversation messages")
    stream: bool | None = Field(None, description="Stream the response as NDJSON")

    model_config = ConfigDict(extra="allow")


class OllamaGenerateRequest(BaseModel):
    model: str = Field(..., description="Profile ID or Ollama model name")
    prompt: str | None = Field(None, description="Prompt text")
    stream: bool | None = Field(None, description="Stream the response as NDJSON")

    model_config = ConfigDict(extra="allow")


class OllamaEmbedRequest(BaseModel):
    model: str = Field(..., description="Profile ID or Ollama model name")
    input: str | list[Any] | None = Field(None, description="Text(s) to embed")
    prompt: str | None = Field(None, description="Legacy single-text input alias")

    model_config = ConfigDict(extra="allow")


# ── Permissive validation gate (REF-11 / BUG-18) ────────────────


def _first_validation_error(exc: ValidationError) -> tuple[str | None, str]:
    """A ``(param, message)`` pair describing the first validation failure."""
    errors = exc.errors()
    if not errors:
        return None, "The request body is invalid."
    first = errors[0]
    loc = [str(p) for p in first.get("loc", ()) if p != "body"]
    param = ".".join(loc) or None
    msg = str(first.get("msg") or "invalid value")
    return param, (f"{param}: {msg}" if param else msg)


def validate_body(model_cls: type[BaseModel], body: dict[str, Any]) -> dict[str, Any]:
    """Validate *body* against *model_cls* and return it **unchanged**.

    The model is used purely as a gate: on success the original ``body`` dict
    is returned verbatim (same object), so unknown/extra fields and
    loosely-typed values a working client sends reach the backend exactly as
    before. On a genuinely malformed body — a known field with the wrong JSON
    type, or a missing required field — it raises an OpenAI-shaped
    :class:`fastapi.HTTPException` (``status_code=422``). ``main.py``'s
    per-family error handler (REF-10) renders it as a top-level ``error``
    object on ``/v1`` and an ``error`` string on ``/api``.
    """
    try:
        model_cls.model_validate(body)
    except ValidationError as exc:
        param, message = _first_validation_error(exc)
        raise _openai_error(
            message,
            "invalid_request_error",
            param=param,
            code="invalid_body",
            status_code=422,
        ) from exc
    return body
