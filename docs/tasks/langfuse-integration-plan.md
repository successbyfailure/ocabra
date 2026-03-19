# Plan de integración: Langfuse

**Fecha:** 2026-03-19
**Objetivo:** Integrar oCabra con una instancia Langfuse externa (self-hosted en otro host) para observabilidad de llamadas LLM. Completamente opcional; desactivado por defecto.

---

## Contexto

Langfuse es una plataforma open-source de observabilidad para LLMs. La instancia está en otro host (no forma parte de este stack Docker). oCabra simplemente envía trazas via HTTP al endpoint de Langfuse cuando está configurado.

**Lo que aporta sobre las stats internas de oCabra (`request_stats`):**

| Capacidad | `request_stats` actual | Con Langfuse |
|---|---|---|
| Duración, modelo, GPU | ✅ | ✅ |
| Tokens input/output | ⚠️ solo non-streaming | ✅ ambos modos |
| Coste estimado (€/token) | ❌ solo energía | ✅ |
| Trazas por sesión/usuario | ❌ | ✅ |
| Contenido de mensajes (opt-in) | ❌ | ✅ |
| Evaluaciones / feedback humano | ❌ | ✅ |
| Dashboard interactivo | ❌ | ✅ |

---

## Modelo de privacidad

```
LANGFUSE_CAPTURE_CONTENT=false  (default — privado)
  → Langfuse recibe: model_id, duration_ms, input_tokens, output_tokens,
                     error, user_id (campo "user" del body OpenAI)

LANGFUSE_CAPTURE_CONTENT=true
  → Además: messages[] / prompt completo + texto de la completion
```

El contenido de los mensajes nunca sale de oCabra salvo configuración explícita.

---

## Arquitectura de integración

El hook está en **`WorkerPool.forward_request` y `forward_stream`** — un único punto que cubre todos los backends y rutas sin tocar ningún handler individual.

Para streaming, el wrapper **no añade latencia al usuario**: los chunks se yieldan inmediatamente. El parsing SSE y el envío a Langfuse ocurren en el `finally` del generador, una vez entregado el último chunk.

```
Request → FastAPI → WorkerPool.forward_request/stream
                         ↓
                    [yield chunks to client inmediatamente]
                         ↓ (finally, post-stream)
                    LangfuseTracer.trace_generation()
                         ↓
                    Langfuse HTTP API (fire-and-forget, async)
```

---

## Ficheros a crear / modificar

| Fichero | Cambio |
|---|---|
| `backend/ocabra/integrations/langfuse_tracer.py` | **Nuevo** — singleton, `trace_generation()`, `wrap_stream()` |
| `backend/ocabra/core/worker_pool.py` | Hook tracer en `forward_request` y `forward_stream` |
| `backend/ocabra/stats/collector.py` | Fix paralelo: extraer tokens de chunks SSE (beneficia también a `request_stats`) |
| `backend/ocabra/config.py` | Settings `langfuse_*` |
| `backend/pyproject.toml` | `langfuse>=2.0` como dependencia opcional |
| `docker-compose.yml` | Variables `LANGFUSE_*` en servicio `api` (todas vacías por defecto) |
| `.env.example` | Documentar las variables |
| `backend/ocabra/main.py` | `langfuse_tracer.shutdown()` en lifespan shutdown |

---

## Paso 1 — Settings (`config.py`)

```python
# Langfuse observability (desactivado por defecto)
langfuse_enabled: bool = False
langfuse_public_key: str = ""
langfuse_secret_key: str = ""
langfuse_host: str = "https://cloud.langfuse.com"  # apuntar a instancia self-hosted
langfuse_capture_content: bool = False   # privado por defecto
langfuse_sample_rate: float = 1.0        # 0.0–1.0, para reducir volumen en producción
langfuse_flush_interval_s: float = 2.0   # el SDK acumula y envía en batches
```

Añadir `langfuse_public_key` y `langfuse_secret_key` al validador `_empty_string_to_none`.

---

## Paso 2 — `LangfuseTracer`

**Fichero:** `backend/ocabra/integrations/langfuse_tracer.py`

Responsabilidades:
- Inicializar `langfuse.Langfuse` una vez (singleton lazy, thread-safe)
- `trace_generation()` — para respuestas no-streaming
- `wrap_stream()` — generador transparente que parsea SSE y envía traza al terminar
- Respetar `capture_content` y `sample_rate`
- **Nunca propagar excepciones al caller** — fallos de Langfuse son silenciosos (log warning)

```python
"""
Langfuse tracer — optional LLM observability integration.

Disabled by default (LANGFUSE_ENABLED=false).
When enabled, records generation traces for all /v1/* and /api/* inference calls.
Content (messages + completions) is only sent if LANGFUSE_CAPTURE_CONTENT=true.
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from ocabra.config import settings

logger = structlog.get_logger(__name__)

_client = None  # Langfuse singleton


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("langfuse_disabled_missing_keys")
        return None
    try:
        import langfuse
        _client = langfuse.Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            flush_interval=settings.langfuse_flush_interval_s,
        )
        logger.info("langfuse_initialized", host=settings.langfuse_host)
    except Exception as exc:
        logger.warning("langfuse_init_failed", error=str(exc))
        return None
    return _client


def _should_sample() -> bool:
    return random.random() < settings.langfuse_sample_rate


def _extract_usage(response_body: dict) -> tuple[int | None, int | None]:
    usage = response_body.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def _extract_completion_text(response_body: dict) -> str | None:
    choices = response_body.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    message = choice.get("message") or {}
    if message.get("content"):
        return message["content"]
    if choice.get("text"):
        return choice["text"]
    return None


async def trace_generation(
    *,
    model_id: str,
    path: str,
    request_body: dict,
    response_body: dict,
    duration_ms: float,
    error: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Record a non-streaming generation trace. Fire-and-forget."""
    client = _get_client()
    if client is None or not _should_sample():
        return
    try:
        input_tokens, output_tokens = _extract_usage(response_body)
        trace = client.trace(
            name=f"ocabra:{path.split('/')[-1]}",
            user_id=user_id,
            session_id=session_id,
            metadata={"path": path},
        )
        trace.generation(
            name=model_id,
            model=model_id,
            input=_build_input(path, request_body),
            output=_extract_completion_text(response_body) if settings.langfuse_capture_content else None,
            usage={"input": input_tokens, "output": output_tokens, "unit": "TOKENS"},
            metadata={"duration_ms": duration_ms},
            level="ERROR" if error else "DEFAULT",
            status_message=error,
        )
    except Exception as exc:
        logger.warning("langfuse_trace_failed", error=str(exc))


async def wrap_stream(
    generator: AsyncIterator[bytes],
    *,
    model_id: str,
    path: str,
    request_body: dict,
    user_id: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[bytes]:
    """
    Transparent wrapper around a streaming SSE generator.

    Yields all chunks immediately (zero added latency to the client).
    Parses SSE in-flight to extract token counts from the final usage chunk.
    Sends the Langfuse trace in the finally block, after the last chunk is delivered.
    """
    client = _get_client()
    if client is None or not _should_sample():
        async for chunk in generator:
            yield chunk
        return

    start = time.monotonic()
    input_tokens: int | None = None
    output_tokens: int | None = None
    text_parts: list[str] = []
    error: str | None = None

    try:
        async for chunk in generator:
            yield chunk  # entrega inmediata — sin buffering
            _parse_sse_chunk_into(
                chunk,
                input_tokens_box=[input_tokens],
                output_tokens_box=[output_tokens],
                text_parts=text_parts if settings.langfuse_capture_content else None,
            )
            # Actualizar desde las cajas mutables
            input_tokens = _parse_sse_chunk_into.last_input  # ver implementación real abajo
            output_tokens = _parse_sse_chunk_into.last_output
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        try:
            trace = client.trace(
                name=f"ocabra:{path.split('/')[-1]}",
                user_id=user_id,
                session_id=session_id,
                metadata={"path": path, "stream": True},
            )
            trace.generation(
                name=model_id,
                model=model_id,
                input=_build_input(path, request_body),
                output="".join(text_parts) if settings.langfuse_capture_content else None,
                usage={"input": input_tokens, "output": output_tokens, "unit": "TOKENS"},
                metadata={"duration_ms": duration_ms},
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            )
        except Exception as exc:
            logger.warning("langfuse_stream_trace_failed", error=str(exc))


def _build_input(path: str, body: dict) -> Any:
    """Respects capture_content: only metadata when false, full messages when true."""
    if not settings.langfuse_capture_content:
        return {
            "model": body.get("model"),
            "temperature": body.get("temperature"),
            "max_tokens": body.get("max_tokens"),
            "stream": body.get("stream"),
        }
    if "messages" in body:
        return body["messages"]
    if "prompt" in body:
        return body["prompt"]
    return body


def parse_sse_chunk(
    chunk: bytes,
    *,
    input_tokens_ref: list,
    output_tokens_ref: list,
    text_parts: list[str] | None,
) -> None:
    """
    Parse a raw SSE chunk to extract usage data and optionally delta text.
    Mutates the mutable ref lists in-place.

    vLLM emits usage in the last data chunk when stream_options.include_usage=true.
    llama-server (BitNet) follows the same convention.
    """
    try:
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str in ("[DONE]", ""):
                continue
            data = json.loads(data_str)
            usage = data.get("usage") or {}
            if usage.get("prompt_tokens"):
                input_tokens_ref[0] = usage["prompt_tokens"]
            if usage.get("completion_tokens"):
                output_tokens_ref[0] = usage["completion_tokens"]
            if text_parts is not None:
                for choice in data.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text_parts.append(delta["content"])
    except Exception:
        pass  # chunk mal formado — silencioso


async def shutdown() -> None:
    """Flush pending traces on process exit. Called from lifespan shutdown."""
    client = _get_client()
    if client:
        try:
            client.flush()
        except Exception:
            pass
```

---

## Paso 3 — Hook en `WorkerPool`

**Fichero:** `backend/ocabra/core/worker_pool.py`

En `forward_request` (no-streaming):
```python
async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
    start = time.monotonic()
    result = await self._do_forward_request(model_id, path, body)
    if settings.langfuse_enabled:
        from ocabra.integrations.langfuse_tracer import trace_generation
        asyncio.create_task(trace_generation(
            model_id=model_id,
            path=path,
            request_body=body,
            response_body=result,
            duration_ms=(time.monotonic() - start) * 1000,
            user_id=body.get("user"),
        ))
    return result
```

En `forward_stream`:
```python
async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
    raw = self._raw_forward_stream(model_id, path, body)
    if settings.langfuse_enabled:
        from ocabra.integrations.langfuse_tracer import wrap_stream
        async for chunk in wrap_stream(raw, model_id=model_id, path=path,
                                       request_body=body, user_id=body.get("user")):
            yield chunk
    else:
        async for chunk in raw:
            yield chunk
```

---

## Paso 4 — Fix de tokens en streaming (`stats/collector.py`)

**Este fix es independiente de Langfuse** pero usa la misma lógica de parsing SSE. Actualmente `input_tokens` y `output_tokens` son `NULL` en `request_stats` para todas las peticiones streaming.

Refactorizar `StatsMiddleware` para que, en streaming, inyecte un wrapper del `StreamingResponse` que parsea chunks SSE y actualiza el stat con los tokens al finalizar. Reutilizar `parse_sse_chunk` de `langfuse_tracer.py` (o moverlo a `stats/sse_parser.py` para que ambos lo importen).

---

## Paso 5 — Variables de entorno y dependencia

**`docker-compose.yml`** — añadir en servicio `api`:
```yaml
LANGFUSE_ENABLED: ${LANGFUSE_ENABLED:-false}
LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:-}
LANGFUSE_SECRET_KEY: ${LANGFUSE_SECRET_KEY:-}
LANGFUSE_HOST: ${LANGFUSE_HOST:-https://cloud.langfuse.com}
LANGFUSE_CAPTURE_CONTENT: ${LANGFUSE_CAPTURE_CONTENT:-false}
LANGFUSE_SAMPLE_RATE: ${LANGFUSE_SAMPLE_RATE:-1.0}
```

**`.env.example`**:
```bash
# Langfuse observability (opcional, desactivado por defecto)
# Apunta a tu instancia self-hosted o Langfuse cloud
# LANGFUSE_ENABLED=false
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=http://langfuse.tudominio.com  # o https://cloud.langfuse.com
# LANGFUSE_CAPTURE_CONTENT=false   # true para registrar contenido de mensajes
# LANGFUSE_SAMPLE_RATE=1.0         # reducir a 0.1 en producción con mucho tráfico
```

**`pyproject.toml`** — añadir `langfuse>=2.0` en dependencias (es liviano, ~2 MB, sin impacto si `langfuse_enabled=false`).

**`main.py`** — en el bloque shutdown del lifespan:
```python
if settings.langfuse_enabled:
    from ocabra.integrations.langfuse_tracer import shutdown as langfuse_shutdown
    await langfuse_shutdown()
```

---

## Estructura de una traza

```
Trace: ocabra:chat_completions
├── user_id: "user-123"        ← campo "user" del body OpenAI (si se envía)
├── session_id: ...            ← para multi-turn (no disponible aún sin session middleware)
├── metadata: { path, stream }
└── Generation: "mistral-7b-instruct"
    ├── model: "mistral-7b-instruct"
    ├── input:  { model, temperature, max_tokens }   ← capture_content=false
    │   ó      [{"role":"user","content":"..."}]     ← capture_content=true
    ├── output: null                                  ← capture_content=false
    │   ó      "La respuesta completa..."             ← capture_content=true
    ├── usage:  { input: 128, output: 312, unit: "TOKENS" }
    ├── duration_ms: 1843
    └── level: DEFAULT | ERROR
```

---

## Tests

**`tests/integrations/test_langfuse_tracer.py`:**
- `test_disabled_no_sdk_call` — `langfuse_enabled=False` → sin llamadas al SDK
- `test_missing_keys_no_call` — keys vacíos → warning, sin crash
- `test_trace_no_content` — `capture_content=False` → input sin mensajes
- `test_trace_with_content` — `capture_content=True` → input contiene mensajes completos
- `test_wrap_stream_passthrough_disabled` — sin Langfuse, generador pasa sin modificar
- `test_wrap_stream_zero_latency` — el primer chunk se yielda antes de que termine el generador
- `test_wrap_stream_parses_usage` — último chunk SSE con usage → tokens extraídos correctamente
- `test_wrap_stream_captures_delta_text` — con `capture_content=True`, text parts acumulados
- `test_sample_rate_zero_never_traces` — `sample_rate=0.0` → nunca envía trazas
- `test_shutdown_calls_flush`
- `test_exception_in_stream_still_sends_error_trace` — si el generator falla, se envía traza con `level=ERROR`
