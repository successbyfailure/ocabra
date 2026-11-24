"""The ``/docs`` API description and the OpenAPI schema customization.

``_API_DESCRIPTION`` is the prose shown at the top of Swagger; a test
(``test_la_referencia_documenta_todos_los_endpoints_de_batches``) reads it and
the Batches table. ``build_openapi_schema(app)`` enriches the auto-generated
spec with the typed request schemas and the multipart / WebSocket endpoints
Swagger can't infer, and caches the result on ``app.openapi_schema``.
"""

from fastapi import FastAPI

_API_DESCRIPTION = """
# oCabra — Multi-GPU AI Model Server

Compatible con las APIs de **OpenAI** (`/v1/`) y **Ollama** (`/api/`).

Esta pagina documenta la **forma** de cada endpoint. Lo que no cubre esta en
dos paginas aparte, pensadas para quien construye sobre oCabra:

| Pagina | Para que sirve |
|--------|----------------|
| [**Changelog**](/docs/changelog) | Cambios de comportamiento entre versiones, parametros nuevos y errores corregidos. Marca con **Accion** lo que requiere tocar tu codigo. Leelo si algo dejo de funcionar, o antes de subir de version. |
| [**Modelos**](/docs/models) | Como rinde cada modelo en este despliegue: latencia, contexto util, soporte de herramientas y razonamiento. |

Ambas se sirven tambien en Markdown crudo añadiendo `.md`
([`/docs/changelog.md`](/docs/changelog.md),
[`/docs/models.md`](/docs/models.md)) — misma fuente, sin maquetar, pensado
para agentes y para `curl`. Un navegador recibe ahí la página maquetada; añade
`?raw=1` para ver el Markdown también desde él.

## Autenticacion

Todas las peticiones requieren API key en la cabecera:

```
Authorization: Bearer sk-...
```

Las API keys se crean desde el dashboard de oCabra o via la API interna.

## Modelos

El campo `model` acepta el **profile_id** del modelo (ej: `qwen3-8b`, `kokoro-82m`).
Consulta `GET /v1/models` para ver los modelos disponibles.

## Capacidades

| Capacidad | Endpoint | Backends |
|-----------|----------|----------|
| **Chat / Completions** | `/v1/chat/completions` | vllm, sglang, llama_cpp, ollama, bitnet, tensorrt_llm |
| **Embeddings** | `/v1/embeddings` | vllm, sglang, llama_cpp, ollama |
| **Text-to-Speech** | `/v1/audio/speech` | tts (Kokoro, Bark, Qwen3-TTS), chatterbox, voxtral, piper |
| **Speech-to-Text** | `/v1/audio/transcriptions` | whisper (faster-whisper, Whisper) |
| **Generacion de imagenes** | `/v1/images/generations` | diffusers |
| **Edicion de imagenes** | `/v1/images/edits` | diffusers |
| **Escalado de fotos** | `/v1/images/upscale` (multipart; devuelve la imagen o `b64_json`) | upscaler (Real-ESRGAN, UltraSharp, Remacri…) |
| **Generacion de musica** | `/v1/audio/generate` | acestep |
| **Generacion de video (con audio)** | `/v1/videos` (asincrono) | comfyui_video (LTX-2.5) |
| **Reranking** | `/v1/rerank` | vllm |
| **Scoring de pares** | `/v1/score` | vllm |
| **Pooling** | `/v1/pooling` | vllm |
| **Clasificacion de texto** | `/v1/classify` | vllm |

## Carga no interactiva: usa Batches, no `/v1/*` directo

Si no hay un humano esperando la respuesta en pantalla (automatizaciones,
backfills, transcripcion masiva, generacion de datos, agentes que lanzan
rafagas) usa **`POST /v1/batches`** en vez de pegarle directo a `/v1/chat/completions`,
`/v1/embeddings`, etc. Las peticiones sincronas compiten por los mismos slots
de GPU que usuarios reales; un script que manda miles de peticiones por esa
via puede degradar la latencia de quien si esta esperando en vivo. Los lotes
se procesan en background con prioridad mas baja que el trafico interactivo
(una peticion interactiva puede desalojar una linea de lote en curso; se
reintenta sola, no cuenta como fallo).

Flujo: sube un JSONL con `POST /v1/files` (`purpose="batch"`, una linea por
peticion: `{"custom_id", "method": "POST", "url", "body"}`) y crea el lote con
`POST /v1/batches` (`input_file_id`, `endpoint`). Consulta el progreso con
`GET /v1/batches/{id}`.

El objeto que devuelven `POST /v1/batches`, `GET /v1/batches/{id}`, `GET
/v1/batches` (lista) y `.../cancel` es el `Batch` de OpenAI, con su esquema en
la sección Schemas: `status` (`validating`, `in_progress`, `finalizing`,
`completed`, `failed`, `expired`, `cancelling`, `cancelled`),
`request_counts` (`total`, `completed`, `failed`) y, al terminar,
`output_file_id` (una línea por petición correcta) y `error_file_id` (una por
cada fallo). Los dos se leen con `GET /v1/files/{id}/content`.

**Por qué espera un lote** (extensión de oCabra). Mientras no ha terminado, el
`Batch` trae `ocabra`, con `phase` y `detail` (una frase lista para una barra de
progreso):

| `phase` | Qué pasa |
|---------|----------|
| `queued` | Hay otros lotes delante (`queue_position`); se procesan de uno en uno. |
| `starting` | Recién cogido, leyendo la entrada. |
| `loading_model` | Sus líneas esperan a que cargue el modelo; `models[].expected_wait_seconds` estima lo que queda con su tiempo de carga habitual (lo mismo que `X-Ocabra-Expected-Wait-Seconds` en una petición normal). |
| `running` | Hay líneas sirviéndose (`in_flight`). |
| `yielding_to_interactive` | Su modelo se cedió a tráfico interactivo; las líneas (`waiting`) se reintentan solas en `retry_in_seconds`. |
| `waiting_for_vram` | No había memoria de GPU libre; se reintenta enseguida. |
| `finalizing` / `cancelling` | Escribiendo los resultados / parando. |

**Límites.** Un lote admite como mucho **50 000 peticiones** por defecto
(`BATCH_MAX_REQUESTS`; cuentan las líneas no
vacías). Por encima, `POST /v1/batches` responde 400 con
`code: "batch_too_large"`, y si no hay ninguna, `code: "empty_batch"`. El fichero
de entrada tiene el tope de cualquier subida, **512 MB** (`POST /v1/files` → 413).
Una línea mal formada hace que se rechace el lote entero con un 400 que lista
hasta 20 errores (`error.errors`, con `line`, `code` y `message`). El lote
caduca a las 24 h (`completion_window`); lo no despachado para entonces sale en
`error_file_id` como `batch_expired`.

### Endpoints admitidos en lote

**oCabra admite en lotes bastante mas que el estandar de Batches de OpenAI.**
Los tres primeros son los clasicos de ese estandar; los otros nueve son
extension de oCabra. Cinco de ellos ni siquiera existen como endpoint en la API
de OpenAI (`rerank`, `score`, `pooling`, `classify` y `audio/generate`), y los
cuatro restantes son endpoints suyos que aqui ademas se pueden lotear: imagenes,
voz, transcripciones y edicion de imagenes.

| `endpoint` | Entrada de la linea | Salida de la linea |
|------------|---------------------|--------------------|
| `/v1/chat/completions` | JSON | JSON |
| `/v1/completions` | JSON | JSON |
| `/v1/embeddings` | JSON | JSON |
| `/v1/rerank` | JSON | JSON |
| `/v1/score` | JSON | JSON |
| `/v1/pooling` | JSON | JSON |
| `/v1/classify` | JSON | JSON |
| `/v1/images/generations` | JSON | **`file_id`** (imagen) |
| `/v1/audio/speech` | JSON | **`file_id`** (audio) |
| `/v1/audio/generate` | JSON | **`file_id`** (audio/musica) |
| `/v1/audio/transcriptions` | **`file_id`** (audio) | JSON |
| `/v1/images/edits` | **`file_id`** (imagen) | **`file_id`** (imagen) |

Las dos mecanicas que se salen del JSON plano:

- **Entrada binaria por referencia.** `transcriptions` y `images/edits` son
  multipart en su forma sincrona, y un binario no cabe en una linea JSON. Sube
  el fichero antes con `POST /v1/files` usando
  `purpose="batch_input_audio"` o `purpose="batch_input_image"`, y pon su id en
  el `body` de la linea: `file_id` para transcripciones; `image_file_id` y,
  opcionalmente, `mask_file_id` o `image_ref_{1,2,3}_file_id` para edicion.
- **Salida binaria por referencia.** Las respuestas de imagen, voz y musica no
  se inlinean en el JSONL de resultados: cada linea trae un `file_id` que
  descargas con `GET /v1/files/{id}/content`.

## Routers "recomendado-*": modelos que el servidor mantiene calientes

Ademas de perfiles normales (`profile_kind: execution`, un modelo concreto),
`model=` tambien acepta perfiles **router** (`profile_kind: router`): no
ejecutan nada por si mismos, enumeran `routing_targets` por prioridad y
delegan en el primero que ya este cargado. Este despliegue expone varios con
el prefijo `recomendado-` (p.ej. `recomendado-sota`, `recomendado-ha-chatbots`,
`recomendado-stt`) que apuntan a los modelos que el servidor mantiene
calientes habitualmente (`load_policy: pin`/`warm`) — usalos cuando no
necesitas un modelo concreto y prefieres "el mejor disponible ahora mismo"
sin pagar un cold-start. Consulta `GET /v1/models` para ver los disponibles y
sus `routing_targets` actuales. Las cabeceras `X-Ocabra-Router` /
`X-Ocabra-Router-Target` indican a que perfil apuntaste y cual respondio
realmente.

Un router solo elige entre los destinos que saben hacer lo que pide la
peticion. En chat (`/v1/chat/completions`, `/api/chat`, `/api/generate`): si
lleva imagenes, solo destinos con vision; si lleva `tools`, solo con
herramientas; lo mismo con audio y video. Un destino cargado que no sirve no
gana por estar cargado. Si ningun destino del router puede, la respuesta es
400 `model_not_capable`, igual que con un modelo concreto. En `/v1/models` un
router anuncia `vision`, `tools`, `audio_input`, `video_input`,
`image_generation` e `image_editing` si alguno de sus destinos la tiene.
`recomendado-imagen` sirve para los dos endpoints de imagen: genera con un
modelo en `/v1/images/generations` y edita con otro en `/v1/images/edits`.

## Video al estilo OpenAI

`/v1/videos` funciona como la API de video de OpenAI: es **asincrona**.

1. `POST /v1/videos` con `model`, `prompt`, `seconds` y `size` (JSON o
   multipart) devuelve el objeto `video` en `queued`.
2. `GET /v1/videos/{id}` hasta que `status` sea `completed` o `failed`;
   `progress` va de 0 a 100 y `queue_position` dice cuantos tienes delante.
3. `GET /v1/videos/{id}/content` descarga el MP4 (H.264 + AAC: el modelo
   genera tambien el sonido; describelo en el prompt). `?variant=thumbnail`
   da un JPEG del primer fotograma.

**Partir de imagenes.** `input_reference` (como en OpenAI) es el primer
fotograma: el video anima esa imagen siguiendo el prompt. Va como fichero en
multipart, o en JSON como `data:` URL, URL `http(s)` publica o
`{"image_url": ...}`. Sin `size`, el video toma la proporcion de la imagen.
`last_frame` (extension, necesita `input_reference`) es el ultimo fotograma:
el video va de una imagen a la otra. Hasta 20 MB por imagen.

Diferencias: `seconds` es cualquier entero de 1 a 10; `size` se redondea a
multiplos de 64 (`1280x720` sale `1280x704`); `seed`, `negative_prompt` y
`last_frame` son extensiones. Se genera un video a la vez en todo el nodo; uno de 5 s a 720p
tarda ~65 s con el modelo cargado y ~85 s si hay que cargarlo. El fichero se
borra a las 24 h. No disponible en nodos `zero_data`.

## Imagen al estilo OpenAI

Un cliente escrito para la API de imagenes de OpenAI funciona sin cambios
salvo la URL base y la clave:

- `model`: `gpt-image-1`, `gpt-image-1-mini`, `dall-e-3` o `dall-e-2` se
  resuelven al perfil que el administrador haya fijado en el modo
  compatibilidad (normalmente `recomendado-imagen`). Sin `model` se trata
  como `dall-e-2`, igual que OpenAI. Un nombre sin fijar da 404.
- Ediciones: las imagenes pueden ir en `image` o en `image[]` (como las manda
  el SDK); la primera es la base y el resto, referencias. `mask` es opcional.
- `size`: cualquier `AnchoxAlto` (`1024x1024`, `1536x1024`, `1024x1536`...);
  `auto` o un valor no reconocido es 1024x1024.
- `quality`: `low` usa la mitad de pasos de inferencia que el perfil, `high` y
  `hd` la vez y media; `medium`, `standard` y `auto` los del perfil. Si mandas
  `num_inference_steps`, gana ese.
- `response_format`: `url` por defecto con `dall-e-*` (o sin `model`),
  `b64_json` con el resto, como OpenAI. Se acepta siempre el que pidas.
- `background`, `output_format`, `output_compression`, `moderation` y
  `style` se aceptan y se ignoran: la imagen sale siempre en PNG.

## Formatos de audio (TTS)

`mp3` (default), `wav`, `opus`, `flac`, `pcm`, `aac`

## Voces TTS

Dependen del modelo. Usa `GET /v1/audio/voices?model=<profile_id>` para consultar las voces disponibles.

## Carga en frío y estado del modelo

Los modelos se cargan **bajo demanda**. Una petición a un modelo frío mantiene la conexión abierta mientras carga (puede tardar minutos en modelos grandes):

- La respuesta incluye los headers `X-Ocabra-Model-Status`, `X-Ocabra-Model-Id` y, si aún no está cargado, `X-Ocabra-Expected-Wait-Seconds`.
- Durante la carga, el stream SSE emite **comentarios** (`: {...}`) como keepalive y pista de estado. Las líneas SSE que empiezan por `:` son comentarios y las ignora todo parser conforme (el SDK de OpenAI incluido) — **no las parsees como chunks**.
- Para consultar el estado sin lanzar la petición: `GET /v1/models/{id}` → el campo `ocabra` incluye `status`, `loaded` y `expected_wait_seconds`.

## Transcripción (STT)

`POST /v1/audio/transcriptions` acepta, además de los campos OpenAI, estos form-params propios de oCabra (**no estándar OpenAI** — un cliente/SDK oficial no los reconoce):

| Campo | Default | Efecto |
|-------|---------|--------|
| `vad_filter` | `true` | Descarta los tramos sin voz (VAD) antes de transcribir (mitiga alucinaciones de Whisper). |
| `condition_on_previous_text` | `false` | Evita que un bucle de repetición se propague vía el contexto del decoder (como WhisperX). |
| `diarize` | `false` | Identificación de hablantes (pyannote); cada segmento de `verbose_json` lleva `speaker`. Soportado por perfiles de los backends `whisper` (faster-whisper, pyannote en el propio proceso) y `whisper_cpp` (vía un sidecar de diarización lanzado de forma perezosa — requiere el backend `whisper` instalado, pyannote vive en su venv). Primera petición paga el coste de carga de pyannote (coste único) y requiere `HF_TOKEN` accesible en el entorno del worker/sidecar. `vibeasr` no implementa diarización: `diarize=true` contra ese perfil responde 400 (`diarization_not_supported`), no se ignora en silencio. |

Límite de subida: 256 MB por parte (configurable). Para audio largo, sube comprimido (`mp3`/`opus`/`flac`) en vez de WAV.

### Transcripción en streaming (baja latencia)

Dos formas de transcribir en vivo:

- **WebSocket transcription-only** (recomendado): `GET /v1/realtime?model=<whisper>&intent=transcription`. Envía audio PCM16 (16 kHz mono) con `input_audio_buffer.append`; el servidor segmenta por VAD y emite:
  - `conversation.item.input_audio_transcription.delta` — hipótesis **parciales** mientras se habla (prefijo estable, LocalAgreement), para baja latencia al primer token.
  - `conversation.item.input_audio_transcription.completed` — transcript **final** del segmento, con extensiones oCabra: `segments` y `words` (timestamps absolutos de sesión) y `speakers` (con perfiles `*-diarized`).

  Configurable por `session.update`: `{ "input_audio_transcription": { "language": "es", "prompt": "...", "diarize": true }, "turn_detection": {...} }`. Con un perfil `*-diarized` (o `diarize:true`) cada segmento lleva `speaker`, y los ids (`speaker_1`, `speaker_2`, …) son **consistentes entre segmentos de la sesión** (emparejamiento por embeddings, no las etiquetas locales de pyannote). Filtra por defecto las alucinaciones de Whisper sobre silencio (rellenos tipo «Gracias/Adiós» que el modelo inventa sobre no-voz), usando el `no_speech_prob` del propio modelo — desactivable con `input_audio_transcription.suppress_hallucinations:false`. Para sesiones largas, pasa `num_speakers` (query o `input_audio_transcription.num_speakers`) con el nº esperado de hablantes — **acota el namespace y evita la deriva de ids**. Para reconexiones, `conversation_id` (query) **reanuda el mismo namespace de hablantes** entre sesiones consecutivas del mismo audio. La sesión sirve `session.created` de inmediato: puedes empezar a enviar audio al momento, y el audio capturado **durante el cold-start del modelo se bufferiza y se transcribe al cargar** (no se pierde). Sin LLM ni TTS. Compatible con clientes de la Realtime Transcription API de OpenAI.
- **Batch sobre modelo caliente**: con el perfil whisper en `load_policy: pin`, `POST /v1/audio/transcriptions` sobre clips de 2–5 s responde en ~200 ms (modelo `base`); sirve para emular streaming con ventanas solapadas.

## Modo agente

Los modelos con id `agent/<slug>` ejecutan un bucle de herramientas en el servidor. Son compatibles con `/v1/chat/completions` (streaming y no-streaming). Para recibir el progreso de herramientas como eventos SSE de oCabra (`event: ocabra.tool_started` / `ocabra.tool_result`), envía la cabecera `X-Ocabra-Stream-Events: true`; sin ella el stream es **OpenAI-estándar** (solo chunks de chat).
"""


def build_openapi_schema(app: FastAPI) -> dict:
    """Enrich the auto-generated OpenAPI spec with typed request schemas."""
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    from ocabra.api.openai._schemas import (
        ChatCompletionRequest,
        ClassifyRequest,
        CompletionRequest,
        EmbeddingRequest,
        ImageGenerationRequest,
        MusicGenerationRequest,
        PoolingRequest,
        RerankRequest,
        ScoreRequest,
        SpeechRequest,
    )
    from ocabra.api.openai.batches import ImageEditsBatchLineBody, TranscriptionBatchLineBody

    # Inject Pydantic schemas into components
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    for model_cls in (
        ChatCompletionRequest,
        CompletionRequest,
        EmbeddingRequest,
        SpeechRequest,
        ImageGenerationRequest,
        RerankRequest,
        PoolingRequest,
        ScoreRequest,
        ClassifyRequest,
        MusicGenerationRequest,
        # Not wired to any requestBody below — these describe the shape of
        # a line's "body" inside a Batches JSONL input file, which isn't
        # itself part of any HTTP request schema. Injected here purely so
        # they're discoverable in the Swagger "Schemas" section instead of
        # living only in prose (see POST /v1/batches' description).
        TranscriptionBatchLineBody,
        ImageEditsBatchLineBody,
    ):
        schemas[model_cls.__name__] = model_cls.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )

    # Map endpoints → schemas
    endpoint_schemas = {
        "/v1/chat/completions": "ChatCompletionRequest",
        "/v1/completions": "CompletionRequest",
        "/v1/embeddings": "EmbeddingRequest",
        "/v1/audio/speech": "SpeechRequest",
        "/v1/images/generations": "ImageGenerationRequest",
        "/v1/rerank": "RerankRequest",
        "/v1/pooling": "PoolingRequest",
        "/v1/score": "ScoreRequest",
        "/v1/classify": "ClassifyRequest",
        "/v1/audio/generate": "MusicGenerationRequest",
    }

    for path, schema_name in endpoint_schemas.items():
        if path in schema.get("paths", {}):
            for method_info in schema["paths"][path].values():
                if isinstance(method_info, dict) and "summary" in method_info:
                    method_info["requestBody"] = {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                            }
                        },
                    }

    # Transcriptions is multipart/form-data (read via request.form), so its
    # fields aren't auto-generated — describe them for the interactive docs.
    tr_methods = schema.get("paths", {}).get("/v1/audio/transcriptions", {})
    for method_info in tr_methods.values():
        if isinstance(method_info, dict) and "summary" in method_info:
            method_info["requestBody"] = {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file", "model"],
                            "properties": {
                                "file": {"type": "string", "format": "binary", "description": "Audio (wav/mp3/m4a/ogg/flac/webm)."},
                                "model": {"type": "string", "description": "Whisper profile_id."},
                                "language": {"type": "string"},
                                "prompt": {"type": "string"},
                                "response_format": {"type": "string", "enum": ["json", "text", "srt", "vtt", "verbose_json"], "default": "json"},
                                "temperature": {"type": "number", "default": 0.0},
                                "timestamp_granularities[]": {"type": "array", "items": {"type": "string", "enum": ["segment", "word"]}},
                                "diarize": {"type": "boolean", "description": "Identificación de hablantes (pyannote, extensión oCabra). Soportado por whisper (en proceso) y whisper_cpp (sidecar lazy, requiere el backend whisper instalado). Carga pyannote de forma perezosa en la primera petición. Contra vibeasr responde 400."},
                                "vad_filter": {"type": "boolean", "default": True, "description": "Descarta tramos sin voz (VAD) antes de transcribir."},
                                "condition_on_previous_text": {"type": "boolean", "default": False, "description": "Evita que un bucle de alucinación se propague vía el contexto del decoder."},
                            },
                        }
                    }
                },
            }

    # Same reasoning as transcriptions above: /v1/images/edits is
    # multipart/form-data (read via request.form), not auto-generated.
    edit_methods = schema.get("paths", {}).get("/v1/images/edits", {})
    for method_info in edit_methods.values():
        if isinstance(method_info, dict) and "summary" in method_info:
            method_info["requestBody"] = {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["prompt"],
                            "properties": {
                                "image": {"type": "string", "format": "binary", "description": "Imagen base PNG a editar. Obligatoria, aquí o en image[]."},
                                "image[]": {"type": "array", "items": {"type": "string", "format": "binary"}, "description": "Como la manda el SDK de OpenAI: una o varias imágenes; la primera es la base y el resto, referencias."},
                                "mask": {"type": "string", "format": "binary", "description": "PNG opcional; pixeles transparentes marcan el area a repintar. Sin mask, edicion img2img sobre la imagen completa."},
                                "image_ref_1": {"type": "string", "format": "binary", "description": "Imagen de referencia extra opcional (pipelines multi-imagen, hasta 3)."},
                                "image_ref_2": {"type": "string", "format": "binary"},
                                "image_ref_3": {"type": "string", "format": "binary"},
                                "model": {"type": "string", "description": "Profile ID, router (recomendado-imagen) o nombre de OpenAI fijado en el modo compatibilidad (gpt-image-1, dall-e-2...)."},
                                "quality": {"type": "string", "enum": ["low", "medium", "high", "auto", "standard", "hd"], "description": "Se traduce a pasos de inferencia sobre los del perfil: low x0,5, high/hd x1,5. Si mandas num_inference_steps, gana ese."},
                                "prompt": {"type": "string", "description": "Descripcion de la edicion."},
                                "response_format": {"type": "string", "enum": ["b64_json", "url"], "description": "Por defecto b64_json; url si el modelo pedido es dall-e-* o no se manda, como OpenAI."},
                                "n": {"type": "integer", "default": 1},
                                "size": {"type": "string"},
                                "negative_prompt": {"type": "string"},
                                "num_inference_steps": {"type": "integer"},
                                "guidance_scale": {"type": "number"},
                                "strength": {"type": "number"},
                                "seed": {"type": "integer"},
                                "user": {"type": "string"},
                            },
                        }
                    }
                },
            }

    # Add WebSocket endpoints (Swagger can't auto-generate these)
    paths = schema.setdefault("paths", {})
    paths["/v1/realtime"] = {
        "get": {
            "tags": ["OpenAI Realtime"],
            "summary": "Realtime audio session (WebSocket)",
            "description": (
                "**WebSocket** endpoint for bidirectional audio streaming.\n\n"
                "Establishes a session that coordinates STT → LLM → TTS in real-time.\n\n"
                "### Connection\n"
                "```\n"
                "ws://<host>/v1/realtime?model=<profile_id>\n"
                "Authorization: Bearer <api_key>\n"
                "```\n\n"
                "### Client → Server events\n"
                "| Event | Descripcion |\n"
                "|-------|-------------|\n"
                "| `session.update` | Configurar sesion (modalities, voice, instructions, "
                "turn_detection, input_audio_format, output_audio_format) |\n"
                "| `input_audio_buffer.append` | Enviar chunk de audio PCM16 (base64) |\n"
                "| `input_audio_buffer.commit` | Forzar procesamiento del buffer |\n"
                "| `input_audio_buffer.clear` | Limpiar buffer de audio |\n"
                "| `response.create` | Solicitar respuesta del LLM |\n"
                "| `response.cancel` | Cancelar respuesta en curso |\n\n"
                "### Server → Client events\n"
                "| Event | Descripcion |\n"
                "|-------|-------------|\n"
                "| `session.created` | Sesion iniciada (devuelve session config) |\n"
                "| `session.updated` | Config de sesion actualizada |\n"
                "| `input_audio_buffer.speech_started` | VAD detecto inicio de habla |\n"
                "| `input_audio_buffer.speech_stopped` | VAD detecto fin de habla |\n"
                "| `input_audio_buffer.committed` | Buffer procesado |\n"
                "| `conversation.item.created` | Nuevo item en conversacion |\n"
                "| `response.created` | Respuesta iniciada |\n"
                "| `response.audio.delta` | Chunk de audio TTS (base64 PCM16) |\n"
                "| `response.audio.done` | Audio TTS completado |\n"
                "| `response.audio_transcript.delta` | Texto parcial de la respuesta |\n"
                "| `response.audio_transcript.done` | Texto completo de la respuesta |\n"
                "| `response.done` | Respuesta completada |\n"
                "| `error` | Error en la sesion |\n\n"
                "### Dialectos\n"
                "Acepta el `session.update` **beta** (`input_audio_format`, `voice`, "
                "`input_audio_transcription`, `turn_detection` en la raiz) y el **GA** "
                "(`session.type: \"realtime\"`, `session.audio.input.{format,transcription,"
                "turn_detection}`, `session.audio.output.{format,voice}`, `output_modalities`). "
                "Con GA, `session.updated` devuelve tambien la forma GA y los eventos de audio "
                "usan los nombres GA (`response.output_audio.delta`, "
                "`response.output_audio_transcript.delta`...).\n\n"
                "### Audio formats\n"
                "- Input: PCM16 mono little-endian, G.711 u-law o A-law. La frecuencia es la "
                "declarada: GA `{\"type\":\"audio/pcm\",\"rate\":24000}` (24 kHz si no se "
                "indica), `audio/pcmu`/`audio/pcma` (8 kHz); beta `pcm16` = **16 kHz** salvo "
                "`input_audio_sample_rate`, `g711_ulaw`/`g711_alaw` = 8 kHz. Se remuestrea a 16 kHz "
                "para VAD y STT; tiempos y offsets son de audio real.\n"
                "- Output: PCM16 a 24 kHz por defecto, o el formato/frecuencia pedidos "
                "(`audio/pcmu` para telefonia).\n"
                "- `voice`: un nombre de OpenAI (`alloy`, `echo`...) usa la voz fijada para el "
                "en la compatibilidad de TTS.\n"
                "- `turn_detection.threshold` en la escala de OpenAI (0-1, 0,5 por defecto); "
                "`create_response:false` confirma el turno sin contestar; "
                "`interrupt_response:false` desactiva el corte de la respuesta cuando el "
                "usuario habla.\n"
            ),
            "parameters": [
                {
                    "name": "model",
                    "in": "query",
                    "required": True,
                    "description": "Profile ID del modelo LLM para la sesion",
                    "schema": {"type": "string", "example": "qwen3-8b"},
                }
            ],
            "responses": {
                "101": {"description": "WebSocket upgrade exitoso"},
                "1008": {"description": "Autenticacion requerida (WebSocket close)"},
            },
        }
    }

    app.openapi_schema = schema
    return schema
