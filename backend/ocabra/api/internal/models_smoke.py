"""Smoke-test endpoint for a single model (REF-07 split).

``POST /ocabra/models/{id}/smoke-test`` loads a model (its base config, or an
LLM's default runtime preset, exactly like Load) and sends one minimal request
matching its capabilities. Extracted verbatim from ``api/internal/models.py``;
behaviour, response shape and the ``model_manager`` ACL are unchanged. The
router is mounted onto the main models router via ``include_router`` so the
public path is identical.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ocabra.api._deps_auth import UserContext, require_role

router = APIRouter(tags=["models"])


class SmokeTestRequest(BaseModel):
    """Body of ``POST /models/{model_id}/smoke-test``."""

    model_config = ConfigDict(extra="forbid")

    cold: bool = Field(
        default=False,
        description="Unload the model first when it is loaded, to measure a cold load.",
    )


class SmokeTestResult(BaseModel):
    """Outcome of a smoke test. Failures are reported in ``error``, never as 5xx."""

    loaded: bool
    load_ms: int | None = None
    request_kind: str | None = None
    request_ok: bool | None = None
    request_ms: int | None = None
    tokens_per_second: float | None = None
    error: str | None = None


def _silence_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A mono 16-bit PCM WAV of silence."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


def _smoke_request_kind(state) -> str | None:
    """Which minimal request exercises a loaded model, by its capabilities."""
    caps = getattr(state, "capabilities", None)
    for kind in ("chat", "embeddings", "rerank", "audio_transcription", "tts"):
        if getattr(caps, kind, False):
            return kind
    return None


def _describe_error(exc: Exception) -> str:
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        body = (exc.response.text or "")[:300]
        return f"HTTP {exc.response.status_code}: {body}".strip()
    detail = getattr(exc, "detail", None)
    text = str(detail if detail is not None else exc) or type(exc).__name__
    return text[:500]


async def _send_smoke_request(request: Request, kind: str, worker_key: str, state) -> dict:
    """Send the minimal request of ``kind`` to the loaded worker.

    Returns the parsed JSON (``{}`` for audio output). Raises on failure.
    """
    import httpx

    from ocabra.api.openai._deps import to_backend_body

    pool = request.app.state.worker_pool
    if kind == "chat":
        # Same call the preset benchmark makes.
        body = {
            "model": state.backend_model_id,
            "messages": [{"role": "user", "content": "Di hola."}],
            "max_tokens": 8,
            "stream": False,
        }
        return await pool.forward_request(
            worker_key, "/v1/chat/completions", to_backend_body(state, body)
        )
    if kind == "embeddings":
        body = {"model": state.backend_model_id, "input": "hola"}
        return await pool.forward_request(
            worker_key, "/v1/embeddings", to_backend_body(state, body)
        )
    if kind == "rerank":
        body = {
            "model": state.backend_model_id,
            "query": "¿Qué come una cabra?",
            "documents": ["La cabra come hierba.", "El cielo es azul."],
        }
        return await pool.forward_request(worker_key, "/rerank", to_backend_body(state, body))
    worker = pool.get_worker(worker_key)
    if worker is None:
        raise RuntimeError(f"no worker running for '{worker_key}'")
    base_url = f"http://127.0.0.1:{worker.port}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        if kind == "audio_transcription":
            resp = await client.post(
                f"{base_url}/transcribe",
                files={"file": ("silence.wav", _silence_wav(), "audio/wav")},
                data={"response_format": "json", "temperature": "0.0"},
            )
        else:  # tts
            resp = await client.post(
                f"{base_url}/synthesize",
                json={
                    "input": "hola",
                    "voice": "alloy",
                    "response_format": "wav",
                    "speed": 1.0,
                    "language": "Auto",
                },
            )
        resp.raise_for_status()
        if kind == "tts" and not resp.content:
            raise RuntimeError("the worker returned no audio")
        try:
            return resp.json() if kind == "audio_transcription" else {}
        except ValueError:
            return {}


@router.post(
    "/models/{model_id:path}/smoke-test",
    summary="Load a model and send it one minimal request",
    description=(
        "Check that a model works with its current configuration: make sure it "
        "is loaded (its base config, or for an LLM with runtime presets its "
        "default preset, exactly like Load) and send one tiny request matching "
        "its capabilities — chat, embeddings, rerank, transcription (1 s of "
        "silence) or speech. With `cold: true` a loaded model is unloaded first "
        "to time a cold load. Failures are reported in `error`, not as 5xx."
    ),
    response_model=SmokeTestResult,
    responses={404: {"description": "Model not found"}},
)
async def smoke_test_model(
    model_id: str,
    request: Request,
    body: SmokeTestRequest | None = None,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Smoke-test one model.

    Args:
        model_id: canonical model id.
        body: ``{"cold": bool}`` (default false).

    Returns:
        ``{"loaded", "load_ms", "request_kind", "request_ok", "request_ms",
        "tokens_per_second", "error"}``. ``request_kind`` is null (and
        ``request_ok`` null) for models without a supported minimal request.
    """
    # Lazy import avoids a cycle: ``models`` imports this module's router.
    from ocabra.api.internal.models import _load_manual_plan, _resolve_manual_load_plan
    from ocabra.core.model_manager import ModelStatus

    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    cold = bool(body.cold) if body is not None else False
    result = SmokeTestResult(loaded=False)

    try:
        plan = await _resolve_manual_load_plan(request, mm, model_id)
    except Exception as exc:  # noqa: BLE001 — reported, never a 500
        result.error = f"no se pudo resolver la carga: {_describe_error(exc)}"
        return result.model_dump(mode="json")
    worker_key = plan[0]

    if cold:
        current = await mm.get_state(worker_key)
        if current is not None and current.status == ModelStatus.LOADED:
            if mm.is_busy(worker_key):
                result.loaded = True
                result.error = "el modelo está atendiendo peticiones; no se descarga en frío"
                return result.model_dump(mode="json")
            try:
                await mm.unload(worker_key, reason="smoke_test")
            except Exception as exc:  # noqa: BLE001
                result.error = f"no se pudo descargar: {_describe_error(exc)}"
                return result.model_dump(mode="json")

    started = time.monotonic()
    try:
        await _load_manual_plan(mm, model_id, plan)
    except Exception as exc:  # noqa: BLE001 — reported, never a 500
        result.load_ms = int((time.monotonic() - started) * 1000)
        result.error = f"carga fallida: {_describe_error(exc)}"
        return result.model_dump(mode="json")
    result.load_ms = int((time.monotonic() - started) * 1000)

    worker_state = await mm.get_state(worker_key)
    result.loaded = bool(worker_state is not None and worker_state.status == ModelStatus.LOADED)
    if not result.loaded:
        status = getattr(getattr(worker_state, "status", None), "value", None)
        result.error = f"el modelo no quedó cargado (estado: {status})"
        return result.model_dump(mode="json")

    kind = _smoke_request_kind(worker_state)
    result.request_kind = kind
    if kind is None:
        return result.model_dump(mode="json")
    request_started = time.monotonic()
    try:
        response = await _send_smoke_request(request, kind, worker_key, worker_state)
    except Exception as exc:  # noqa: BLE001 — reported, never a 500
        result.request_ms = max(1, int((time.monotonic() - request_started) * 1000))
        result.request_ok = False
        result.error = f"petición {kind} fallida: {_describe_error(exc)}"
        return result.model_dump(mode="json")
    result.request_ms = max(1, int((time.monotonic() - request_started) * 1000))
    result.request_ok = True
    if kind == "chat" and isinstance(response, dict):
        usage = response.get("usage") or {}
        output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if isinstance(output_tokens, int) and output_tokens > 0:
            result.tokens_per_second = round(output_tokens * 1000 / result.request_ms, 2)
    return result.model_dump(mode="json")
