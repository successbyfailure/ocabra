"""Mutable per-request runtime telemetry shared by API and stats layers."""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass

# Id de la peticion en vuelo (``ModelManager.try_begin_request``) de la
# request HTTP actual. Lo fija el middleware de stats ANTES de ``call_next``:
# BaseHTTPMiddleware copia el contexto al lanzar la app interna, asi que llega
# hasta ``WorkerPool`` y ``FairSlotGate`` (incluido el cuerpo de un
# ``StreamingResponse``). La propagacion inversa (de la app al middleware) no
# funciona, por eso nada debe ESCRIBIRLO fuera del middleware.
# Sirve para atribuir a una peticion concreta lo que solo ve la capa de
# transporte: su lease o su puesto en la cola del gate y el progreso de
# prefill que emite llama.cpp con ``return_progress``.
current_inflight_request_id: ContextVar[str | None] = ContextVar(
    "ocabra_current_inflight_request_id", default=None
)


@dataclass
class RequestRuntimeTelemetry:
    """Timings and admission facts collected while one inference is served.

    The object is attached to ``request.state`` before endpoint dispatch.  It
    stays mutable so streaming generators can add facts after the middleware
    has returned the ``StreamingResponse`` but before the final stats row is
    persisted.
    """

    started_monotonic: float
    prompt_tokens_estimate: int | None = None
    kv_tokens_reserved: int | None = None
    queue_ms: int | None = None
    load_ms: int | None = None
    ttft_ms: int | None = None
    prefill_ms: int | None = None
    slot_id: int | None = None
    physical_slots: int | None = None
    admission_limit: int | None = None
    admission_concurrency: int | None = None
    admission_lease_id: int | None = None

    def mark_load(self, duration_ms: int) -> None:
        self.load_ms = max(0, int(duration_ms))

    def mark_prefill_since(self, started: float, now: float | None = None) -> None:
        """Prefill = lo que tardó el worker desde ``started`` (``time.monotonic``).

        Para quien solo hace prefill (embeddings) y cuyo motor no informa de
        ``timings.prompt_ms``: la llamada al worker ya no incluye ni la cola
        de oCabra ni la carga del modelo, que pasan antes de ``started``.
        """
        if self.prefill_ms is None:
            observed = time.monotonic() if now is None else now
            self.prefill_ms = max(0, int((observed - started) * 1000))

    def mark_first_token(self, now: float | None = None) -> None:
        if self.ttft_ms is None:
            observed = time.monotonic() if now is None else now
            self.ttft_ms = max(0, int((observed - self.started_monotonic) * 1000))

    def as_dict(self) -> dict[str, int]:
        values = {
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "kv_tokens_reserved": self.kv_tokens_reserved,
            "queue_ms": self.queue_ms,
            "load_ms": self.load_ms,
            "ttft_ms": self.ttft_ms,
            "prefill_ms": self.prefill_ms,
            "slot_id": self.slot_id,
            "physical_slots": self.physical_slots,
            "admission_limit": self.admission_limit,
            "admission_concurrency": self.admission_concurrency,
            "admission_lease_id": self.admission_lease_id,
        }
        return {key: int(value) for key, value in values.items() if value is not None}


def mark_prefill_since(request_state: object, started: float) -> None:
    """Anota el prefill en la telemetría de la petición, si la hay.

    Lo usan los endpoints de embeddings justo después de que el worker
    responde: es el tiempo con el que el middleware de stats calcula los
    tokens/s de prefill, que el worker no devuelve.
    """
    telemetry = getattr(request_state, "runtime_telemetry", None)
    if isinstance(telemetry, RequestRuntimeTelemetry):
        telemetry.mark_prefill_since(started)
