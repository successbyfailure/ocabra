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

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response

from ocabra.core.inference.admission import check_capability
from ocabra.core.inference.body import ClientGoneError, await_unless_client_gone
from ocabra.core.inference.errors import (
    raise_inference_dispatch_error,
    raise_upstream_http_error,
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


# El ``federate`` inyectado: recibe ``(model_id, body)`` y devuelve la respuesta
# del peer o ``None`` para procesar en local (ver ``api/openai/_federation.py``).
FederateStep = Callable[[str, dict], Awaitable["Response | None"]]

# Hook opcional entre ``merge`` y ``dispatch``: puede validar y lanzar (p. ej. el
# 400 de ``dimensions`` de embeddings) con el cuerpo ya fusionado a la vista.
PreDispatchHook = Callable[["PipelineContext"], Awaitable[None]]


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
) -> Response | RequestOutcome:
    """Ejecuta el flujo único de petición, paso a paso, en orden.

    Pasos y helper al que delega cada uno:

    - ``authorize`` + ``route`` + ``plan_runtime`` + ``ensure_loaded`` →
      :func:`resolve_profile` (hace router, ACL por usuario, elección de preset
      de runtime y disparo de carga).
    - ``federate_or_local`` → el ``callable`` ``federate`` inyectado
      (``try_proxy_json``): corto-circuita si un peer sirve la petición.
    - ``admit`` → :func:`resolve_admission_kwargs` (consciente de la clase).
    - ``dispatch`` → ``worker_pool.forward_request`` (envuelto en
      ``await_unless_client_gone`` cuando el descriptor honra la desconexión).
    - ``map_error`` → :func:`raise_upstream_http_error` /
      :func:`raise_inference_dispatch_error`.

    Devuelve un :class:`RequestOutcome` con el resultado crudo, o la
    ``Response`` del peer cuando la federación sirvió la petición.
    """
    if descriptor.streaming:
        # La fase A solo migra endpoints no-streaming (embeddings/pooling). El
        # camino de streaming (chat/completions) se añadirá a propósito en la
        # fase B, con su peek del primer chunk y su reja de reparto justo.
        raise NotImplementedError(
            "run_request streaming path is not implemented yet (REF-01 phase B)"
        )

    model_id = str(body.get("model") or "")

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
        request_body=body,
        router_resolver=router_resolver,
        request_state=request.state,
        required_capability=descriptor.router_capability,
    )

    # Paso capability-check (tras la resolución, como hoy).
    check_capability(state, descriptor.capability, descriptor.capability)

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
        # vista (p. ej. el 400 de ``dimensions``), antes de gastar el despacho.
        await pre_dispatch(ctx)

    backend_body = to_backend_body(state, merged_body)
    admission_kwargs = resolve_admission_kwargs(descriptor, request, state, user)

    from ocabra.core.worker_pool import (
        InferenceTimeoutError,
        WorkerDegradedError,
        WorkerGoneError,
    )

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
    except (InferenceTimeoutError, WorkerDegradedError, WorkerGoneError) as exc:
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
    )
