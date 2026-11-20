"""Request admission: in-flight cap, maintenance mode and circuit breaker (REF-08).

One place decides whether an inference request may start, and one counter
tracks it while it runs:

* :class:`AdmissionMiddleware` (pure ASGI, just outside ``StatsMiddleware``)
  resolves the worker a request will use, rejects it under maintenance (503)
  or at the worker's cap (429), and otherwise marks it in flight with
  ``ModelManager.try_begin_request`` — check and count in one step under the
  manager's lock, so a burst cannot overshoot the cap. It releases the slot
  when the response has been sent, the stream has ended, the client has hung
  up or the request was cancelled. It also turns the dispatch errors of the
  circuit breaker and the generation timeout into their 503/504.
* Multipart uploads are admitted the same way: their ``model`` is read from
  the form without reading the file (BUG-37) and the body is replayed
  untouched. Before, they were only *checked* here and counted by the
  endpoint after the whole file had arrived, so simultaneous uploads all
  passed (BUG-38).
* Endpoints that used to count themselves (transcriptions, image
  generations and edits) take the admitted slot with :class:`EndpointInflight`
  instead of opening a second one; it only opens its own when the request
  did not come through admission (e.g. an untracked path).
* A ``/v1/realtime`` WebSocket is checked at the handshake and not counted:
  the session can last for hours and each of its turns marks itself in
  flight (``RealtimeSession``).

``StatsMiddleware`` only records: it gets the parsed JSON body, the start
time and the in-flight id from ``request.state``, and admission hands it its
rejections through the ``on_reject`` callback.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.requests import HTTPConnection

from ocabra.config import settings
from ocabra.core.model_manager_helpers import compute_worker_key
from ocabra.core.request_telemetry import current_inflight_request_id
from ocabra.core.runtime_coordinator import _text_size
from ocabra.core.worker_pool import InferenceTimeoutError, WorkerDegradedError

logger = structlog.get_logger(__name__)

Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

# Media types whose body is never JSON: the middleware does not read them.
_BINARY_PREFIXES = ("multipart/", "audio/", "video/", "image/", "application/octet-stream")

# How much of a multipart body may be read looking for the ``model`` field
# before giving up. The text fields come first in what the usual clients send
# (httpx/openai-python put ``data`` before ``files``); a field that only shows
# up after a file is not looked for, the file is never read here.
MULTIPART_PEEK_LIMIT = 64 * 1024
_MAX_FIELD_VALUE = 512

_BOUNDARY_RE = re.compile(r'boundary=(?:"([^"]+)"|([^;\s]+))', re.IGNORECASE)
_NAME_RE = re.compile(rb'\bname="([^"]*)"', re.IGNORECASE)
_FILENAME_RE = re.compile(rb"\bfilename\*?=", re.IGNORECASE)


def body_is_binary(content_type: str) -> bool:
    """True for bodies the middleware must not buffer (uploads, media)."""
    ctype = (content_type or "").strip().lower()
    return ctype.startswith(_BINARY_PREFIXES)


def is_multipart(content_type: str) -> bool:
    return (content_type or "").strip().lower().startswith("multipart/form-data")


def maintenance_response(model_id: str) -> JSONResponse:
    """503 for a model under maintenance (``ModelManager.set_maintenance``)."""
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": "30"},
        content={
            "error": {
                "message": f"Model '{model_id}' is under maintenance. Please retry shortly.",
                "type": "server_error",
                "code": "model_maintenance",
            }
        },
    )


def capacity_retry_after(effective_cap: int) -> int:
    """A low-concurrency worker (e.g. a single-slot Ollama model) drains its
    queue far slower than the flat 2 s this used to suggest unconditionally —
    tell low-slot callers to back off longer."""
    return 2 if effective_cap > 4 else 10


def capacity_response(model_id: str, effective_cap: int) -> JSONResponse:
    """429 for a worker already at its in-flight cap."""
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(capacity_retry_after(effective_cap))},
        content={
            "error": {
                "message": (
                    f"Model '{model_id}' is at capacity "
                    f"({effective_cap} concurrent requests for this "
                    "worker). Please retry shortly."
                ),
                "type": "rate_limit_error",
                "code": "model_at_capacity",
            }
        },
    )


def _multipart_boundary(content_type: str) -> bytes | None:
    match = _BOUNDARY_RE.search(content_type or "")
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return value.encode("latin-1", errors="ignore") or None


def _scan_multipart(data: bytes, boundary: bytes, field: str) -> tuple[str | None, bool]:
    """Look for ``field`` in the parts already received.

    Returns ``(value, done)``: ``done`` once the value is found, a file part
    starts before it, or the body is malformed; ``(None, False)`` means "need
    more bytes".
    """
    delimiter = b"--" + boundary
    wanted = field.encode()
    pos = data.find(delimiter)
    if pos < 0:
        return None, False
    while True:
        pos += len(delimiter)
        if data[pos : pos + 2] == b"--":
            return None, True  # closing delimiter: no such field
        headers_end = data.find(b"\r\n\r\n", pos)
        if headers_end < 0:
            return None, False
        headers = data[pos:headers_end]
        name = _NAME_RE.search(headers)
        if _FILENAME_RE.search(headers):
            # A file part: what follows is the upload itself.
            return None, True
        value_start = headers_end + 4
        next_delim = data.find(b"\r\n" + delimiter, value_start)
        if name is not None and name.group(1) == wanted:
            if next_delim < 0:
                if len(data) - value_start > _MAX_FIELD_VALUE:
                    return None, True
                return None, False
            raw = data[value_start:next_delim]
            if len(raw) > _MAX_FIELD_VALUE:
                return None, True
            value = raw.decode("utf-8", errors="ignore").strip()
            return (value or None), True
        if next_delim < 0:
            return None, False
        pos = next_delim + 2


async def peek_multipart_field(
    receive: Receive,
    content_type: str,
    field: str = "model",
    *,
    limit: int = MULTIPART_PEEK_LIMIT,
) -> tuple[str | None, Receive]:
    """Read just enough of a multipart body to find one text field.

    Returns the value (or None) and a ``receive`` that replays the messages
    already read, unchanged, before handing over to the original one — the
    endpoint parses the full upload as if nothing had looked at it. Reads at
    most ``limit`` bytes plus one ASGI message, and stops at the first file
    part, so an audio or image upload is never buffered here.
    """
    boundary = _multipart_boundary(content_type)
    if boundary is None:
        return None, receive

    buffered: list[Message] = []
    data = bytearray()
    value: str | None = None
    while True:
        message = await receive()
        buffered.append(message)
        if message.get("type") != "http.request":
            break  # disconnect: replay it and let the endpoint see it
        data += message.get("body", b"") or b""
        found, done = _scan_multipart(bytes(data), boundary, field)
        if done:
            value = found
            break
        if not message.get("more_body", False) or len(data) >= limit:
            break
    del data

    async def replay() -> Message:
        if buffered:
            return buffered.pop(0)
        return await receive()

    return value, replay


async def send_websocket_denial(
    response: JSONResponse, scope: dict, receive: Receive, send: Send
) -> None:
    """Reject a WebSocket handshake with ``response`` (status, headers, body).

    Uses the ASGI WebSocket Denial Response extension (uvicorn has it), so the
    client gets the same 503/429 and ``Retry-After`` an HTTP request would.
    Without the extension the handshake is closed before ``accept``, which
    the server answers with a 403.
    """
    if "websocket.http.response" in (scope.get("extensions") or {}):
        await response(scope, receive, send)
        return
    await send({"type": "websocket.close", "code": 1013, "reason": "try again later"})


# ── What a request is (shared with the stats) ────────────────────────────


TRACKED_API_PATHS = {
    "/api/chat",
    "/api/generate",
    "/api/embeddings",
    "/api/embed",
}


def should_track(path: str) -> bool:
    if path.startswith("/v1/videos"):
        # Asíncrono: el POST solo encola. Cada trabajo registra su propia
        # fila al terminar (core/video_jobs.py), con la duración de verdad.
        return False
    if path.startswith("/v1/"):
        return True
    return path in TRACKED_API_PATHS


def classify_request_kind(path: str) -> str:
    mapping = {
        "/v1/chat/completions": "chat",
        "/v1/completions": "completion",
        # Responses API: text generation with the same usage block
        # (input_tokens/output_tokens); it was recorded as "other".
        "/v1/responses": "responses",
        "/v1/audio/generate": "music",
        "/v1/embeddings": "embedding",
        "/v1/images/generations": "image_generation",
        "/v1/images/edits": "image_generation",
        # Not "image_generation": no prompt, no steps, and its cost is the
        # pixels of the photo, so its stats must not mix with diffusion's.
        "/v1/images/upscale": "image_upscale",
        "/v1/audio/transcriptions": "audio_transcription",
        "/v1/audio/speech": "tts",
        "/v1/rerank": "rerank",
        "/v1/pooling": "pooling",
        "/v1/score": "score",
        "/v1/classify": "classify",
        "/api/chat": "ollama_chat",
        "/api/generate": "ollama_generate",
        "/api/embeddings": "ollama_embedding",
        "/api/embed": "ollama_embedding",
    }
    return mapping.get(path, "other")


def approx_text_tokens(text: object) -> int | None:
    """Rough token estimate (~word count) for endpoints without a usage payload."""
    if isinstance(text, str) and text.strip():
        return len(text.split())
    return None


def count_input_tokens(request_payload: dict | None, request_kind: str) -> int | None:
    """Approx input tokens (~word count) from the request, for text-gen kinds whose
    streamed response lacked a usage payload.

    Includes ``tools`` (function-calling schemas), same as the context-preset
    estimator (``estimate_request_token_budget`` / ``_text_size`` in
    runtime_coordinator.py). 2026-10-29: without this, a tool-heavy request
    (e.g. Home Assistant's large function schemas) showed as "~128 tok
    entrada" live while the real prompt — confirmed via the preset
    selector's own ``required_tokens`` — was 18,240. That gap made a request
    legitimately busy on a huge prompt look suspiciously idle."""
    if not isinstance(request_payload, dict):
        return None
    if request_kind in ("chat", "ollama_chat"):
        total = 0
        for m in request_payload.get("messages") or []:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, str):
                total += len(content.split())
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += len(part["text"].split())
            # Mismo agujero que tenian las ``tools`` antes de 2026-10-29, en el
            # otro lado de la conversacion: lo que el asistente pide a una
            # herramienta viaja en ``tool_calls``, no en ``content``. Con un
            # cliente de codigo eso es la mitad del prompt, y la tarjeta
            # mostraba "~35.383 tok entrada" para una peticion que el motor
            # conto en 196.549 (flint:40b, 2026-11-09).
            for call in m.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                if isinstance(fn, dict):
                    _chars, words = _text_size(fn.get("arguments"))
                    total += words
                    if isinstance(fn.get("name"), str):
                        total += 1
        _tools_chars, tools_words = _text_size(request_payload.get("tools"))
        total += tools_words
        return total or None
    if request_kind in ("completion", "ollama_generate"):
        return approx_text_tokens(request_payload.get("prompt"))
    return None


def payload_streaming(payload: object, path: str = "") -> bool | None:
    """Si la petición va en streaming.

    Sin ``stream`` en el cuerpo, la API de OpenAI responde de una vez y la
    nativa de Ollama (``/api/*``) hace streaming. Antes la ausencia se tomaba
    como "desconocido" y la UI trataba como streaming peticiones que no lo
    eran: les salía "sin actividad del worker" a los 15 s porque no emiten
    nada hasta el final (2026-11-10, tráfico real de benchmark).
    """
    if not isinstance(payload, dict):
        return None
    if "stream" in payload:
        return bool(payload.get("stream"))
    if path.startswith("/api/"):
        return True
    if path.startswith("/v1/"):
        return False
    return None


def payload_max_output_tokens(payload: object) -> int | None:
    """Tope de salida pedido, en cualquiera de sus nombres (OpenAI y Ollama)."""
    if not isinstance(payload, dict):
        return None
    candidates = [
        payload.get("max_completion_tokens"),
        payload.get("max_tokens"),
        payload.get("max_output_tokens"),
    ]
    options = payload.get("options")
    if isinstance(options, dict):
        candidates.append(options.get("num_predict"))
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return int(value)
    return None


async def resolve_inflight_auth_meta(request: Request) -> dict:
    """Best-effort auth lookup for admission.

    Runs before the endpoint, so ``request.state.auth_user`` isn't populated
    yet (that happens inside endpoint handlers). We re-resolve the bearer
    token here solely so ``/stats/recent`` can show *who* is driving the
    currently in-flight request instead of a bare "anónimo" row. Failures
    are swallowed — this never gates the request.
    """
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return {}
    token = header[7:].strip()
    if not token:
        return {}
    try:
        from ocabra.api._deps_auth import _resolve_api_key
        from ocabra.database import AsyncSessionLocal
    except Exception:  # noqa: BLE001
        return {}
    try:
        async with AsyncSessionLocal() as session:
            ctx = await _resolve_api_key(token, session)
    except Exception:  # noqa: BLE001
        return {}
    if ctx is None or ctx.is_anonymous:
        return {}
    return {
        "user_id": ctx.user_id,
        "username": ctx.username,
        "api_key_name": ctx.api_key_name,
    }


def apply_peer_rules(request: Request, auth_meta: dict) -> Callable[[], None] | None:
    """If the caller is a federation peer served under rules, carry them along.

    ``request.state.federation_rules`` (read by ``is_batch_origin_request`` for
    «me first») and the ``current_peer_rules`` contextvar (read by the model
    loader for «may it evict»). The identity is the authenticated service user
    of the peer, never a header. Runs in the task the endpoint runs in, before
    it loads anything. Returns what undoes the contextvar once the request is
    over (None when nothing was set).
    """
    fm = getattr(getattr(request.app, "state", None), "federation_manager", None)
    if fm is None or not auth_meta.get("user_id"):
        return None
    try:
        rules = fm.rules_for_user(auth_meta.get("user_id"))
    except Exception:  # noqa: BLE001 — never block a request on this
        return None
    from ocabra.core.federation_rules import ALL_ALLOWED, current_peer_rules

    if rules is None or rules == ALL_ALLOWED:
        return None
    request.state.federation_rules = rules
    token = current_peer_rules.set(rules)
    return lambda: current_peer_rules.reset(token)


async def request_bound_for_peer(request: Request, mm: Any, model_id: str) -> bool:
    """True when this request is headed for a federation peer rather than a
    local worker.

    Mirrors step 2 of ``ocabra.core.federation.resolve_federated``: the model
    is not LOADED here but at least one online peer has it loaded, so the
    handler will proxy it out. That matters for admission because the local
    slot-derived cap in ``effective_inflight_cap`` describes a local worker's
    KV cache — a worker this request will never touch. Left unchecked, a
    generative model that only ever runs on a peer inherits the 1-slot
    assumption and gets throttled to ``max_queued_per_slot`` (4) concurrent
    requests, no matter how much headroom the peer actually has. The peer runs
    its own admission control, so the flat ceiling is the right local guard.

    Admission runs *before* the handler resolves
    federation, so this is necessarily a prediction rather than an
    observation. It is deliberately conservative: when the model is loaded
    locally we never skip the cap, even though ``select_target`` could still
    pick a peer. Best effort — any failure falls back to local accounting.
    """
    from ocabra.core.model_manager import ModelStatus

    if not settings.federation_enabled:
        return False
    federation_manager = getattr(request.app.state, "federation_manager", None)
    if federation_manager is None:
        return False
    try:
        state = await mm.get_state(model_id)
        if state is not None and state.status == ModelStatus.LOADED:
            return False
        return bool(federation_manager.get_remote_models().get(model_id))
    except Exception:  # noqa: BLE001 -- federation stays best effort
        return False


async def predict_router_worker(request: Request, registry: Any, profile: Any) -> str | None:
    """Worker key the router will most likely serve this request with.

    Admission runs before the endpoint asks ``RouterResolver.pick``, so this
    is a prediction. It used to be the first local target of the list, and
    the cap was checked against it even when the router would pick a
    different one (BUG-38): a router whose first target was cold but whose
    second was warm and saturated admitted everything, and one whose first
    target was saturated rejected requests the router would have sent to an
    idle warm one. ``pick`` sends a request to the first target that is
    already LOADED, wherever it is in the list, and only cold-starts in list
    order when nothing is warm; this mirrors that (without its session veto,
    capability and context checks). The in-flight count is moved to the
    worker actually chosen afterwards (``rebind_request`` in
    ``api/openai/_deps.py``).
    """
    from ocabra.core.model_manager import ModelStatus

    fm = getattr(request.app.state, "federation_manager", None)
    mm = getattr(request.app.state, "model_manager", None)
    first: str | None = None
    for target_id in profile.routing_targets or []:
        target = await registry.get(target_id)
        if target is not None and target.enabled and target.base_model_id:
            key = compute_worker_key(target.base_model_id, target.load_overrides)
            if first is None:
                first = key
            if mm is None:
                break
            try:
                state = await mm.get_state(key)
                if state is None and key != target.base_model_id:
                    state = await mm.get_state(target.base_model_id)
            except Exception:  # noqa: BLE001 - fall back to list order
                state = None
            if state is not None and getattr(state, "status", None) == ModelStatus.LOADED:
                return key
            continue
        if target is None and fm is not None and first is None:
            # A target only a peer has (a pod first in
            # ``recomendado-sota``): its own id, not the next
            # local target's — that charged the pod's requests
            # to ornith here (in-flight, admission, stats).
            remote_id = fm.remote_model_for(str(target_id))
            if remote_id is not None:
                return remote_id
    return first


async def resolve_inflight_model_id(request: Request, requested: str | None) -> str | None:
    """Resolve a public profile/model id to the worker key used by ModelManager.

    Statistics retain the public id from the request body. Only lifecycle and
    admission tracking use this resolved key.
    """
    if not requested:
        return None

    registry = getattr(request.app.state, "profile_registry", None)
    if registry is not None:
        try:
            profile = await registry.get(requested)
            if profile is None:
                # Hidden compatibility aliases (e.g. "gemma4:26b" -> the
                # renamed "gemma4:26b-llamacpp" profile after a backend
                # migration) aren't profile_ids themselves, so a plain
                # registry.get() misses them. Without this, stats blamed
                # the alias's OLD backend_model_id match in the fallback
                # below — attributing every request to the wrong backend
                # even though the real request correctly used the new one.
                # See resolve_profile() in api/openai/_deps.py for the
                # same alias-then-direct-lookup order this mirrors.
                alias_resolver = getattr(registry, "resolve_alias", None)
                alias_target = (
                    await alias_resolver(requested) if alias_resolver is not None else None
                )
                profile = alias_target[0] if alias_target is not None else None
            if profile is not None and profile.enabled:
                if getattr(profile, "profile_kind", "execution") == "router":
                    return await predict_router_worker(request, registry, profile)
                return compute_worker_key(profile.base_model_id, profile.load_overrides)

            # Legacy canonical model ids resolve through their default profile.
            if "/" in requested:
                profiles = await registry.list_by_model(requested)
                profile = next(
                    (item for item in profiles if item.enabled and item.is_default),
                    None,
                )
                if profile is None:
                    profile = next((item for item in profiles if item.enabled), None)
                if profile is not None:
                    return compute_worker_key(
                        profile.base_model_id,
                        profile.load_overrides,
                    )
        except Exception as exc:  # noqa: BLE001 - tracking must not block inference
            logger.warning(
                "inflight_model_resolution_failed",
                requested_model_id=requested,
                error=str(exc),
            )

    model_manager = getattr(request.app.state, "model_manager", None)
    if model_manager is not None:
        try:
            state = await model_manager.get_state(requested)
            if state is not None:
                return state.model_id
        except Exception:  # noqa: BLE001 - fall back to the public id
            pass

    return requested


# ── Dispatch errors: generation timeout and circuit breaker ───────────────


def dispatch_error_response(exc: BaseException, path: str) -> JSONResponse | None:
    """The response for an error a worker call raised, or None if it has none.

    * ``InferenceTimeoutError`` → 504 ``generation_timeout``.
    * ``WorkerDegradedError`` (the circuit breaker tripped: the worker failed
      several times in a row in a way that means its process died, see
      ``worker_pool._looks_like_dead_backend``) → 503 ``worker_degraded`` with
      the breaker's cooldown as ``Retry-After``, instead of letting the request
      sit behind a dead runner for a full upstream timeout.

    Ollama paths (``/api/*``) get Ollama's ``{"error": "text"}``.
    """
    if isinstance(exc, InferenceTimeoutError):
        status, code, headers = 504, "generation_timeout", None
    elif isinstance(exc, WorkerDegradedError):
        status, code = 503, "worker_degraded"
        headers = {"Retry-After": str(settings.worker_circuit_breaker_cooldown_s)}
    else:
        return None
    if path.startswith("/api/"):
        content: dict[str, Any] = {"error": str(exc)}
    else:
        content = {"error": {"message": str(exc), "type": "server_error", "code": code}}
    return JSONResponse(status_code=status, headers=headers, content=content)


# ── The admission layer ─────────────────────────────────────────────────────

# What admission leaves in ``request.state`` (Starlette keeps it in
# ``scope["state"]``, shared by every layer of the request):
#   inflight_model_id / inflight_request_id — the admitted slot (the endpoint
#     moves it with ``rebind_request`` when the runtime differs);
#   admission_payload — the JSON body, parsed once (None for uploads and
#     non-JSON), so the stats do not read the body again;
#   admission_started — (monotonic, wall clock) when the request arrived.
PAYLOAD_STATE_KEY = "admission_payload"
STARTED_STATE_KEY = "admission_started"


@dataclass(frozen=True)
class AdmissionRejection:
    """A request admission turned away, for the stats to record."""

    model_id: str
    status_code: int
    error_code: str
    request_kind: str
    path: str
    started_at: datetime
    duration_ms: float


RejectHook = Callable[[HTTPConnection, AdmissionRejection], None]


async def read_json_body(receive: Receive) -> tuple[dict | None, Receive]:
    """Read the whole body and parse it as a JSON object (None otherwise).

    Returns the object and a ``receive`` that replays the body, in one
    message, before handing over to the original one. What the stats
    middleware did with ``request.json()``, without its own copy of the body.
    """
    chunks: list[bytes] = []
    tail: Message | None = None
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            tail = message  # disconnect before the end: replay it after the body
            break
        chunks.append(message.get("body", b"") or b"")
        if not message.get("more_body", False):
            break
    body = b"".join(chunks)
    del chunks
    payload: dict | None = None
    if body:
        try:
            parsed = json.loads(body)
        except Exception:  # noqa: BLE001 - not JSON: the endpoint answers it
            parsed = None
        payload = parsed if isinstance(parsed, dict) else None
    pending: list[Message] = [{"type": "http.request", "body": body, "more_body": tail is not None}]
    if tail is not None:
        pending.append(tail)

    async def replay() -> Message:
        if pending:
            return pending.pop(0)
        return await receive()

    return payload, replay


def _model_manager(conn: HTTPConnection) -> Any:
    try:
        return getattr(conn.app.state, "model_manager", None)
    except (AttributeError, KeyError):
        return None


class AdmissionMiddleware:
    """Pure ASGI: admit, reject or count every tracked inference request.

    Sits just outside ``StatsMiddleware`` and inside
    ``FederationHopMiddleware`` (it reads the ``federation_info`` that one
    leaves in the scope). Pure ASGI on purpose: it runs in the request's own
    task, so the slot it opens belongs to that task (``cancel_request`` from
    the dashboard cancels it) and the contextvars it sets
    (``current_inflight_request_id``, ``current_peer_rules``) are seen by the
    endpoint; and it releases the slot only when the inner app has finished
    sending, so streams keep it until their last chunk or the hang-up.
    """

    def __init__(self, app: Any, *, on_reject: RejectHook | None = None) -> None:
        self.app = app
        self._on_reject = on_reject

    async def __call__(self, scope: dict, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            if should_track(HTTPConnection(scope).url.path):
                if await self._deny_websocket(scope, receive, send):
                    return
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http" or not should_track(HTTPConnection(scope).url.path):
            await self.app(scope, receive, send)
            return
        await self._admit(scope, receive, send)

    async def _admit(self, scope: dict, receive: Receive, send: Send) -> None:
        start = time.monotonic()
        started_at = datetime.now(UTC)
        state = scope.setdefault("state", {})
        state[STARTED_STATE_KEY] = (start, started_at)

        content_type = Headers(scope=scope).get("content-type", "")
        payload: dict | None = None
        peeked: str | None = None
        if is_multipart(content_type):
            try:
                peeked, receive = await peek_multipart_field(receive, content_type)
            except Exception as exc:  # noqa: BLE001 - admission must not break uploads
                logger.warning("multipart_model_peek_failed", error=str(exc))
                peeked = None
        elif not body_is_binary(content_type):
            payload, receive = await read_json_body(receive)
        state[PAYLOAD_STATE_KEY] = payload

        request = Request(scope, receive)
        path = request.url.path
        request_kind = classify_request_kind(path)
        requested = str(payload["model"]) if payload and payload.get("model") else None
        # A multipart upload has no JSON body; its ``model`` was peeked from
        # the form. The stats still take the name the endpoint leaves in
        # ``request.state.stats_model_id``.
        public_model = requested or peeked
        mm = _model_manager(request)
        model_key = await resolve_inflight_model_id(request, public_model)

        request_id: str | None = None
        context_token = None
        undo_peer_rules: Callable[[], None] | None = None
        if mm and model_key:
            if mm.is_in_maintenance(model_key):
                # Maintenance window: reject new requests fast and clear so
                # whoever is testing against this model doesn't have real
                # traffic contending for the same GPU/worker. Requests already
                # admitted before the window opened are untouched.
                await self._reject(
                    request, scope, receive, send,
                    admission=AdmissionRejection(
                        model_key, 503, "model_maintenance", request_kind, path,
                        started_at, (time.monotonic() - start) * 1000,
                    ),
                    response=maintenance_response(model_key),
                )  # fmt: skip
                return
            auth_meta = await resolve_inflight_auth_meta(request)
            undo_peer_rules = apply_peer_rules(request, auth_meta)
            remote_bound = await request_bound_for_peer(request, mm, model_key)
            request_id = mm.try_begin_request(
                model_key,
                settings.max_inflight_per_model,
                user_id=auth_meta.get("user_id"),
                username=auth_meta.get("username"),
                api_key_name=auth_meta.get("api_key_name"),
                # The public ``model`` the client sent — a profile id
                # ("qwen3.8:27b") or router id ("recomendado-ha-chatbots") —
                # next to the worker key, so the UI shows the friendly name.
                profile_id=public_model,
                # Approximate, available immediately (exact prompt_tokens
                # isn't known until the backend responds).
                input_tokens_estimate=count_input_tokens(payload, request_kind),
                # Encoder endpoints are exempt from the generative slot
                # scaling whether or not the model is loaded locally.
                request_kind=request_kind,
                remote_bound=remote_bound,
                streaming=payload_streaming(payload, path),
                max_output_tokens=payload_max_output_tokens(payload),
                federation_info=state.get("federation_info"),
            )
            if request_id is None:
                # The worker is already at capacity. Reject rather than pile
                # on so one client's burst can't starve others.
                effective_cap = mm.effective_inflight_cap(
                    model_key,
                    settings.max_inflight_per_model,
                    request_kind=request_kind,
                    remote_bound=remote_bound,
                )
                try:
                    await self._reject(
                        request, scope, receive, send,
                        admission=AdmissionRejection(
                            model_key, 429, "model_at_capacity", request_kind, path,
                            started_at, (time.monotonic() - start) * 1000,
                        ),
                        response=capacity_response(model_key, effective_cap),
                    )  # fmt: skip
                finally:
                    _undo(undo_peer_rules)
                return
            state["inflight_model_id"] = model_key
            state["inflight_request_id"] = request_id
            # WorkerPool / FairSlotGate read it to know which request a lease
            # or a prefill progress report belongs to.
            context_token = current_inflight_request_id.set(request_id)

        response_started = False

        async def send_tracking(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_tracking)
        except Exception as exc:
            response = dispatch_error_response(exc, path)
            if response is None or response_started:
                raise
            await response(scope, receive, send)
        finally:
            if request_id is not None:
                # The endpoint may have moved the slot to the runtime it
                # resolved (``rebind_request``): release it where it is now.
                mm.end_request(state.get("inflight_model_id") or model_key, request_id)
            if context_token is not None:
                _undo(lambda: current_inflight_request_id.reset(context_token))
            _undo(undo_peer_rules)

    async def _reject(
        self,
        conn: HTTPConnection,
        scope: dict,
        receive: Receive,
        send: Send,
        *,
        admission: AdmissionRejection,
        response: JSONResponse,
    ) -> None:
        if self._on_reject is not None:
            try:
                self._on_reject(conn, admission)
            except Exception as exc:  # noqa: BLE001 - recording must not change the answer
                logger.warning("admission_reject_hook_failed", error=str(exc))
        if scope["type"] == "websocket":
            await send_websocket_denial(response, scope, receive, send)
        else:
            await response(scope, receive, send)

    async def _deny_websocket(self, scope: dict, receive: Receive, send: Send) -> bool:
        """Maintenance mode and in-flight cap for a WebSocket handshake (BUG-38).

        The model is the ``model`` query parameter (``/v1/realtime?model=``).
        Same answers as an HTTP request: 503 + ``Retry-After`` under
        maintenance, 429 + ``Retry-After`` when the worker is at its cap. The
        session is not counted in flight: it can last for hours, and each of
        its turns marks itself in flight (``RealtimeSession`` →
        ``begin_request``); the check only stops a new session from piling
        onto a saturated worker. Returns True when the handshake was denied.
        Best effort: a failure lets the session through.
        """
        conn = HTTPConnection(scope)
        requested = (conn.query_params.get("model") or "").strip()
        if not requested:
            return False
        mm = _model_manager(conn)
        if mm is None:
            return False
        path = conn.url.path
        request_kind = (
            "realtime_transcription"
            if (conn.query_params.get("intent") or "").strip().lower() == "transcription"
            else "realtime_session"
        )
        try:
            model_id = await resolve_inflight_model_id(conn, requested)  # type: ignore[arg-type]
            if not model_id:
                return False
            response = None
            if mm.is_in_maintenance(model_id):
                response = maintenance_response(model_id)
                error_code, status = "model_maintenance", 503
            else:
                remote_bound = await request_bound_for_peer(conn, mm, model_id)  # type: ignore[arg-type]
                effective_cap = mm.effective_inflight_cap(
                    model_id,
                    settings.max_inflight_per_model,
                    request_kind=request_kind,
                    remote_bound=remote_bound,
                )
                if effective_cap > 0 and mm.inflight_count(model_id) >= effective_cap:
                    response = capacity_response(model_id, effective_cap)
                    error_code, status = "model_at_capacity", 429
        except Exception as exc:  # noqa: BLE001 - never block a session on this
            logger.warning("realtime_admission_check_failed", model=requested, error=str(exc))
            return False
        if response is None:
            return False
        await self._reject(
            conn, scope, receive, send,
            admission=AdmissionRejection(
                model_id, status, error_code, request_kind, path, datetime.now(UTC), 0.0
            ),
            response=response,
        )  # fmt: skip
        return True


def _undo(fn: Callable[[], None] | None) -> None:
    if fn is None:
        return
    try:
        fn()
    except ValueError:  # token from another context: nothing to undo here
        pass


# ── Endpoints ───────────────────────────────────────────────────────────────


class EndpointInflight:
    """The in-flight slot of the request an endpoint is serving.

    For the handlers that used to call ``begin_request`` themselves
    (transcriptions, image generations and edits): the request already holds
    the slot admission opened, so this takes that one — moved to
    ``worker_key`` if the endpoint resolved a different runtime than the one
    admission predicted — instead of counting the request a second time. The
    slot is released by admission once the response is out. Only when the
    request did not come through admission (an untracked path, a direct
    call) does it open its own slot, released by :meth:`release`.

    Usage::

        inflight = EndpointInflight(request, model_manager, worker_key, user=user)
        request_id = inflight.acquire()
        try:
            ...
        finally:
            inflight.release()
    """

    def __init__(self, request: Any, model_manager: Any, worker_key: str, *, user: Any = None):
        self._request = request
        self._mm = model_manager
        self._worker_key = worker_key
        self._user = user
        self._own: str | None = None

    def acquire(self) -> str:
        state = getattr(self._request, "state", None)
        admitted = getattr(state, "inflight_request_id", None) if state is not None else None
        if isinstance(admitted, str) and admitted:
            current = getattr(state, "inflight_model_id", None)
            if isinstance(current, str) and current and current != self._worker_key:
                self._mm.rebind_request(admitted, current, self._worker_key)
                state.inflight_model_id = self._worker_key
            annotate = getattr(self._mm, "annotate_request", None)
            if self._user is not None and callable(annotate):
                annotate(admitted, self._user)
            return admitted
        self._own = self._mm.begin_request(self._worker_key, user=self._user)
        return self._own

    def release(self) -> None:
        if self._own is not None:
            own, self._own = self._own, None
            self._mm.end_request(self._worker_key, own)
