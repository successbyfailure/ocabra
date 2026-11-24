"""Un único flujo de petición para la capa de inferencia (REF-01, fase A).

Cada endpoint declara un :class:`RequestDescriptor` —qué clase de petición es,
qué capacidad exige, si se federa, si hace streaming, si aplica razonamiento, si
honra la desconexión del cliente y la ruta de despacho— y delega en
:func:`run_request`, que ejecuta los pasos SIEMPRE en el mismo orden::

    authorize → route → plan_runtime → federate_or_local → ensure_loaded
      → admit → dispatch → map_error

Ningún paso se reimplementa aquí: cada uno llama al helper que ya existe en
``core/inference/``. El objetivo de la fase A es montar el motor **sin cambiar
el comportamiento** de ningún endpoint que declare exactamente los pasos que hoy
ejecuta; en esta fase solo se migran ``/v1/embeddings`` y ``/v1/pooling`` como
prueba, rellenando a propósito sus dos huecos (cancelación por desconexión y la
formalización —sin regresión— del paso de admisión).

Layering (REF-09): este módulo vive en ``core`` y **no** importa de ``api``. La
federación —cuyo helper ``try_proxy_json`` vive en la capa ``api``— se inyecta
como un ``callable`` (parámetro ``federate``) en vez de importarse, para no
reintroducir la inversión de capas que REF-09 cerró.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ocabra.core.inference.admission import check_capability
from ocabra.core.inference.body import ClientGoneError, await_unless_client_gone
from ocabra.core.inference.errors import (
    raise_inference_dispatch_error,
    raise_upstream_http_error,
    stream_error_details,
)
from ocabra.core.inference.loading import ocabra_response_headers
from ocabra.core.inference.reasoning import to_backend_body
from ocabra.core.inference.resolution import merge_profile_defaults, resolve_profile

if TYPE_CHECKING:
    from ocabra.api._deps_auth import UserContext


@dataclass(frozen=True)
class RequestDescriptor:
    """Qué hace un endpoint, de forma declarativa, para que el motor lo ejecute.

    Args:
        request_kind: Etiqueta de la clase de petición (``"embeddings"``,
            ``"pooling"``, ``"chat"``…). Hace el paso ``admit`` consciente de la
            clase: solo las clases generativas que lo declaran
            (``applies_fair_share``) pasan por la reja de reparto justo; el resto
            —embeddings/rerank/pooling— NO, y conservan su concurrencia alta
            gobernada solo por ``AdmissionMiddleware`` (ver
            ``project_embeddings_inflight_cap``).
        capability: Capacidad que el modelo resuelto debe tener; se comprueba
            tras la resolución con :func:`check_capability`.
        dispatch_path: Ruta interna a la que se despacha (``"/v1/embeddings"``).
        federates: Si intenta servir desde un peer antes que en local.
        streaming: Si la respuesta es un stream (no soportado aún en fase A).
        applies_reasoning: Si traduce los controles de razonamiento del cliente.
            Informativo: ``to_backend_body`` aplica sus transformaciones (inocuas
            cuando no hay ``reasoning_effort``) en cualquier caso.
        honors_disconnect: Si cancela el despacho cuando el cliente se desconecta
            antes de la respuesta (``await_unless_client_gone``).
        applies_fair_share: Si pasa por la reja de reparto justo por slots/KV.
        router_capability: Capacidad por la que un router debe filtrar destinos
            en la resolución. ``None`` = no filtrar (comportamiento histórico de
            embeddings/pooling).
        routes_request_body: Si el cuerpo de la petición viaja a
            :func:`resolve_profile` (y de ahí al planificador de runtime/router).
            ``True`` por defecto —chat/embeddings estiman tokens del cuerpo para
            elegir preset y reserva de KV—. Los endpoints de audio lo ponen a
            ``False`` para conservar su resolución histórica, que NUNCA pasó
            cuerpo: un cuerpo de TTS/transcripción (texto a sintetizar, forma
            multipart) no debe re-seleccionar preset ni mover el worker_key (ver
            REF-01 fase C).
    """

    request_kind: str
    capability: str
    dispatch_path: str
    federates: bool = True
    streaming: bool = False
    applies_reasoning: bool = False
    honors_disconnect: bool = False
    applies_fair_share: bool = False
    router_capability: str | tuple[str, ...] | None = None
    routes_request_body: bool = True
    capability_label: str | None = None
    """Etiqueta humana del paso ``capability`` en el mensaje de error (``The
    model '…' does not support <label>``). ``None`` = usa ``capability`` como
    etiqueta (comportamiento histórico de embeddings/pooling). Chat la fija a
    ``"chat completions"`` para que el 400 sea byte a byte el de hoy."""


# El ``federate`` inyectado: recibe ``(model_id, body)`` y devuelve la respuesta
# del peer o ``None`` para procesar en local (ver ``api/openai/_federation.py``).
FederateStep = Callable[[str, dict], Awaitable["Response | None"]]

# Hook opcional entre ``merge`` y ``dispatch``: puede validar y lanzar (p. ej. el
# 400 de ``dimensions`` de embeddings) con el cuerpo ya fusionado a la vista.
PreDispatchHook = Callable[["PipelineContext"], Awaitable[None]]

# Hook opcional de error de despacho (fase B): recibe la excepción cruda y el
# contexto ANTES del mapeo por defecto. Si devuelve una ``Response`` se devuelve
# tal cual (el reenrutado de router de chat tras ``WorkerGoneError`` /
# ``VisionNoRoomError``); si devuelve ``None`` se aplica el mapeo de siempre
# (``raise_inference_dispatch_error``). Ningún endpoint de embeddings/pooling lo
# usa: su comportamiento no cambia.
OnDispatchError = Callable[[Exception, "PipelineContext"], Awaitable["Response | None"]]

# Costura de despacho inyectado (fase C): cuando se pasa, el motor ejecuta
# federate→route→cap→merge y cede el paso de despacho —con su manejo de errores y
# su construcción de respuesta— al endpoint, que recibe el contexto y devuelve una
# ``Response`` ya lista. La usan los endpoints de audio, cuyos workers hablan
# protocolos nativos (``/synthesize``, ``/transcribe``) con cuerpos multipart o
# respuestas binarias que ``forward_request`` (JSON→JSON) y el relay SSE de
# ``_dispatch_stream`` no saben transportar. Ningún endpoint JSON la usa: su
# camino —``to_backend_body``, admisión, ``forward_request``, ``map_error``—
# queda intacto.
DispatchStep = Callable[["PipelineContext"], Awaitable["Response"]]

# Centinela para distinguir "no se pasó override de router_capability" de
# "se pasó None a propósito" (None = no filtrar).
_UNSET: Any = object()


@dataclass
class PipelineContext:
    """Estado que se va rellenando a medida que avanza :func:`run_request`."""

    request: Request
    descriptor: RequestDescriptor
    body: dict
    user: UserContext | None
    model_manager: Any
    profile_registry: Any
    worker_pool: Any
    profile: Any = None
    state: Any = None
    worker_key: str | None = None
    merged_body: dict = field(default_factory=dict)


@dataclass
class RequestOutcome:
    """Resultado crudo del despacho más el contexto que el endpoint necesita.

    El endpoint hace su post-proceso específico (p. ej. ``dimensions`` o
    ``mark_prefill_since`` en embeddings) sobre ``result`` y devuelve la
    respuesta HTTP con ``response_headers``.
    """

    result: Any
    profile: Any
    state: Any
    worker_key: str
    merged_body: dict
    response_headers: dict[str, str]
    forward_started: float
    queue_state: dict | None = None
    """Estado de la cola de reparto justo sondeado ANTES del despacho (solo
    clases generativas no-batch; ``None`` para el resto). Chat lo usa para los
    heads-up ``X-Ocabra-Slot-Queue-Saturated`` / ``X-Ocabra-KV-*`` de la
    respuesta no-streaming."""


def _client_disconnected_error() -> HTTPException:
    """499 «client closed request» — el mismo código que usa el colector para
    las cancelaciones de streaming."""
    return HTTPException(
        status_code=499,
        detail={
            "error": {
                "message": "Client disconnected before the response was ready",
                "type": "server_error",
                "code": "client_disconnected",
            }
        },
    )


def resolve_admission_kwargs(
    descriptor: RequestDescriptor,
    request: Request,
    state: Any,
    user: UserContext | None,
) -> dict[str, Any]:
    """Los kwargs de admisión que recibe ``worker_pool.forward_request``.

    **Consciente de la clase de petición (crítico).** Para las clases que NO
    declaran reparto justo —embeddings, rerank, score, pooling, classify—
    devuelve ``{}``: ``forward_request`` entonces va directo al worker (vía
    ``_forward_request_unfair``) sin engancharse a ``FairSlotGate``, exactamente
    como hoy. Su concurrencia la sigue gobernando solo ``AdmissionMiddleware``
    (``effective_inflight_cap(..., request_kind=...)``), que da a embeddings su
    tope propio —NO la reja generativa de 1 slot— (incidente real documentado en
    ``project_embeddings_inflight_cap``; la concurrencia se resuelve por
    ``request_kind`` vía ``resolve_worker_slots``, ver ``project_slot_resolution_drift``
    y ``project_federated_encoder_inflight_cap``).

    Solo las clases generativas (``applies_fair_share=True``) obtienen la reja:
    se arman ``user_key``/``slots``/``is_batch`` (y el presupuesto de KV cuando
    el backend lo publica) a partir de los mismos helpers que usan hoy chat y
    completions. Ningún endpoint de la fase A activa esta rama.
    """
    if not descriptor.applies_fair_share or user is None:
        return {}

    from ocabra.core.inference.admission import (
        fairness_kv_budget,
        fairness_slots,
        fairness_user_key,
        is_batch_origin_request,
    )

    request_state = request.state
    kv_capacity, kv_tokens = fairness_kv_budget(
        state,
        required_tokens=getattr(request_state, "runtime_required_tokens", None),
        effective_context=getattr(request_state, "runtime_effective_context", None),
        reservation_tokens=getattr(request_state, "runtime_kv_reservation_tokens", None),
    )
    kwargs: dict[str, Any] = {
        "user_key": fairness_user_key(user),
        "slots": fairness_slots(state),
        "is_batch": is_batch_origin_request(request),
    }
    # Telemetría de runtime (tokens/s, cold start…): chat y completions la pasan
    # al worker cuando el middleware la colgó de ``request.state``. Inocua si no
    # está. Solo para clases generativas (esta rama), nunca para embeddings.
    telemetry = getattr(request_state, "runtime_telemetry", None)
    if telemetry is not None:
        kwargs["telemetry"] = telemetry
    if kv_capacity is not None:
        kwargs["kv_capacity_tokens"] = kv_capacity
        kwargs["kv_tokens"] = kv_tokens
    return kwargs


async def run_request(
    request: Request,
    *,
    descriptor: RequestDescriptor,
    body: dict,
    user: UserContext | None,
    model_manager: Any,
    profile_registry: Any,
    worker_pool: Any,
    federate: FederateStep | None = None,
    pre_dispatch: PreDispatchHook | None = None,
    on_dispatch_error: OnDispatchError | None = None,
    dispatch: DispatchStep | None = None,
    router_capability: Any = _UNSET,
) -> Response | RequestOutcome:
    """Ejecuta el flujo único de petición, paso a paso, en orden.

    Pasos y helper al que delega cada uno:

    - ``authorize`` + ``route`` + ``plan_runtime`` + ``ensure_loaded`` →
      :func:`resolve_profile` (hace router, ACL por usuario, elección de preset
      de runtime y disparo de carga).
    - ``federate_or_local`` → el ``callable`` ``federate`` inyectado
      (``try_proxy_json`` / ``try_proxy_stream``): corto-circuita si un peer
      sirve la petición.
    - ``admit`` → :func:`resolve_admission_kwargs` (consciente de la clase).
    - ``dispatch`` → ``worker_pool.forward_request`` (no-streaming, envuelto en
      ``await_unless_client_gone`` cuando el descriptor honra la desconexión) o
      ``worker_pool.forward_stream`` (streaming, con peek del primer chunk).
    - ``map_error`` → :func:`raise_upstream_http_error` /
      :func:`raise_inference_dispatch_error`; ``on_dispatch_error`` puede
      reenrutar antes del mapeo (router de chat tras ``WorkerGoneError`` /
      ``VisionNoRoomError``).

    Para una petición no-streaming devuelve un :class:`RequestOutcome` con el
    resultado crudo; para streaming devuelve una ``StreamingResponse`` ya lista;
    en ambos modos devuelve la ``Response`` del peer cuando la federación sirvió
    la petición.

    ``router_capability``: override por petición de la capacidad por la que el
    router filtra destinos (chat la calcula del cuerpo + partes multimodales);
    sin pasar = la del descriptor.

    ``dispatch``: costura de despacho inyectado (fase C). Si se pasa, tras
    federate→route→cap→merge el motor cede el despacho al endpoint (audio, con
    workers nativos multipart/binarios que no caben en ``forward_request``) y
    devuelve su ``Response`` directamente, sin tocar ``to_backend_body`` ni la
    admisión. ``None`` = camino de despacho estándar.
    """
    model_id = str(body.get("model") or "")
    effective_router_capability = (
        router_capability if router_capability is not _UNSET else descriptor.router_capability
    )

    # Paso federate_or_local: si un peer tiene el modelo caliente, sirve allí.
    if descriptor.federates and federate is not None:
        fed_resp = await federate(model_id, body)
        if fed_resp is not None:
            return fed_resp

    # Pasos authorize + route + plan_runtime + ensure_loaded.
    router_resolver = getattr(request.app.state, "router_resolver", None)
    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
        request_body=(body if descriptor.routes_request_body else None),
        router_resolver=router_resolver,
        request_state=request.state,
        required_capability=effective_router_capability,
    )

    # Paso capability-check (tras la resolución, como hoy). La etiqueta humana
    # del mensaje se conserva: chat dice "chat completions", no "chat".
    check_capability(
        state, descriptor.capability, descriptor.capability_label or descriptor.capability
    )

    merged_body = merge_profile_defaults(profile, body)
    # El worker que ``resolve_profile`` cargó: su clave incluye el preset de
    # runtime, que ``compute_worker_key(base, load_overrides)`` no ve.
    worker_key = state.model_id

    ctx = PipelineContext(
        request=request,
        descriptor=descriptor,
        body=body,
        user=user,
        model_manager=model_manager,
        profile_registry=profile_registry,
        worker_pool=worker_pool,
        profile=profile,
        state=state,
        worker_key=worker_key,
        merged_body=merged_body,
    )
    if pre_dispatch is not None:
        # Validación específica del endpoint con el cuerpo ya fusionado a la
        # vista (el 400 de ``dimensions`` de embeddings; en chat, las
        # capacidades extra —tools/vision/audio/video— y el tope de salida).
        await pre_dispatch(ctx)

    # Paso dispatch inyectado (fase C): el endpoint se encarga del despacho, el
    # mapeo de errores y la construcción de la respuesta. El motor ya hizo
    # federate→route→cap→merge. Se usa en audio, cuyo despacho a workers nativos
    # (multipart/binario) no cabe en ``forward_request``/relay SSE. Devuelve la
    # ``Response`` del endpoint tal cual; no se toca to_backend_body ni admisión.
    if dispatch is not None:
        return await dispatch(ctx)

    backend_body = to_backend_body(state, merged_body)
    admission_kwargs = resolve_admission_kwargs(descriptor, request, state, user)

    # Paso dispatch streaming: su propio peek del primer chunk y su relay con
    # error en banda. Devuelve la ``StreamingResponse`` directamente.
    if descriptor.streaming:
        return await _dispatch_stream(
            worker_pool=worker_pool,
            worker_key=worker_key,
            dispatch_path=descriptor.dispatch_path,
            backend_body=backend_body,
            admission_kwargs=admission_kwargs,
        )

    from ocabra.core.fair_slot_gate import KVCapacityError
    from ocabra.core.worker_pool import (
        InferenceTimeoutError,
        VisionNoRoomError,
        WorkerDegradedError,
        WorkerGoneError,
    )

    # Heads-up de cola de reparto justo, sondeado ANTES del despacho (solo clases
    # generativas no-batch; chat lo refleja en cabeceras ``X-Ocabra-*`` de la
    # respuesta no-streaming).
    queue_state: dict | None = None
    if descriptor.applies_fair_share and not admission_kwargs.get("is_batch", False):
        from ocabra.core.inference.admission import fairness_kv_budget

        _rs = request.state
        _, _kv_tokens = fairness_kv_budget(
            state,
            required_tokens=getattr(_rs, "runtime_required_tokens", None),
            effective_context=getattr(_rs, "runtime_effective_context", None),
            reservation_tokens=getattr(_rs, "runtime_kv_reservation_tokens", None),
        )
        queue_state = worker_pool.peek_queue_state(worker_key, kv_tokens=_kv_tokens)

    forward_started = time.monotonic()
    coro = worker_pool.forward_request(
        worker_key,
        descriptor.dispatch_path,
        backend_body,
        **admission_kwargs,
    )
    # Paso dispatch + map_error. La desconexión del cliente cancela el despacho
    # en curso (mismo mecanismo que chat) cuando el descriptor lo honra.
    try:
        if descriptor.honors_disconnect:
            result = await await_unless_client_gone(request, coro)
        else:
            result = await coro
    except ClientGoneError:
        raise _client_disconnected_error() from None
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
    except (
        WorkerGoneError,
        VisionNoRoomError,
        KVCapacityError,
        InferenceTimeoutError,
        WorkerDegradedError,
    ) as exc:
        # El hook puede reenrutar (router de chat) antes del mapeo por defecto.
        if on_dispatch_error is not None:
            rerouted = await on_dispatch_error(exc, ctx)
            if rerouted is not None:
                return rerouted
        raise_inference_dispatch_error(exc)

    response_headers = await ocabra_response_headers(
        request, model_manager, worker_key, profile.base_model_id
    )
    return RequestOutcome(
        result=result,
        profile=profile,
        state=state,
        worker_key=worker_key,
        merged_body=merged_body,
        response_headers=response_headers,
        forward_started=forward_started,
        queue_state=queue_state,
    )


async def _peek_first_chunk(
    gen: AsyncIterator[bytes],
) -> tuple[bytes | None, Exception | None]:
    """Avanza *gen* una vez, antes de devolver ninguna respuesta HTTP.

    Permite distinguir "el backend rechazó la petición de entrada" (nada
    enviado aún → aún se puede responder con un 4xx/5xx real) de "el backend
    empezó a emitir y falló a mitad" (estado 200 ya comprometido → error SSE en
    banda). Es el mismo peek que chat introdujo en v0.87.3 (BUG-17) para que un
    rechazo no llegue como 200 con ``data: {"error": …}`` dentro del stream.
    """
    try:
        first_chunk = await gen.__anext__()
        return first_chunk, None
    except StopAsyncIteration:
        return None, None
    except Exception as exc:  # noqa: BLE001 — cualquier fallo del primer intento merece un estado real
        return None, exc


async def _relay_stream(
    gen: AsyncIterator[bytes],
    first_chunk: bytes | None,
) -> AsyncIterator[bytes]:
    """Continúa un generador de ``forward_stream`` cuyo primer chunk ya es
    conocido-bueno (:func:`_peek_first_chunk`), con el mismo fallback de error
    SSE en banda que chat: una vez enviadas las cabeceras (200), un fallo a
    mitad solo se puede comunicar como ``data: {"error": …}`` + ``[DONE]``."""
    try:
        if first_chunk is not None:
            yield first_chunk
        async for chunk in gen:
            yield chunk
    except Exception as exc:  # noqa: BLE001 — el error va en banda, no como estado HTTP
        message, code = stream_error_details(exc)
        error_payload = json.dumps(
            {"error": {"message": message, "type": "server_error", "code": code}}
        )
        yield f"data: {error_payload}\n\n".encode()
        yield b"data: [DONE]\n\n"


async def _dispatch_stream(
    *,
    worker_pool: Any,
    worker_key: str,
    dispatch_path: str,
    backend_body: dict,
    admission_kwargs: dict[str, Any],
) -> StreamingResponse:
    """Despacho streaming genérico: ``forward_stream`` + peek del primer chunk.

    Reproduce la orquestación de streaming genérica de chat (su camino de
    fallback ``_peek_first_chunk`` + ``_stream_chat_from``): el primer chunk se
    espera ANTES de devolver ninguna respuesta —aún no hay cabeceras—, así que
    un rechazo se traduce a un estado HTTP real (``raise``), igual que chat
    (BUG-17, v0.87.3). Solo cuando el backend ya empezó a emitir se devuelve una
    ``StreamingResponse`` cuyo relay serializa un fallo posterior como error SSE
    en banda. NO incluye el camino de carga perezosa con keepalives ni el
    reenrutado de router de chat, que viven en la capa ``api`` (dependen de
    ``api/sse.py`` y del ``RouterResolver``) y se quedan en ``chat.py``.
    """
    from ocabra.core.fair_slot_gate import KVCapacityError
    from ocabra.core.worker_pool import WorkerGoneError

    gen = worker_pool.forward_stream(
        worker_key,
        dispatch_path,
        backend_body,
        **admission_kwargs,
    )
    first_chunk, first_exc = await _peek_first_chunk(gen)
    if first_exc is not None:
        # Nada enviado aún: estado HTTP real como hace chat (líneas 448-458).
        if isinstance(first_exc, httpx.HTTPStatusError):
            raise_upstream_http_error(first_exc)
        if isinstance(first_exc, (KVCapacityError, WorkerGoneError)):
            raise_inference_dispatch_error(first_exc)
        message, code = stream_error_details(first_exc)
        raise HTTPException(
            status_code=502,
            detail={"error": {"message": message, "type": "server_error", "code": code}},
        )
    return StreamingResponse(
        _relay_stream(gen, first_chunk),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
