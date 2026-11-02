"""Mutable per-request runtime telemetry shared by API and stats layers."""

from __future__ import annotations

import time
from dataclasses import dataclass


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

    def mark_load(self, duration_ms: int) -> None:
        self.load_ms = max(0, int(duration_ms))

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
        }
        return {key: int(value) for key, value in values.items() if value is not None}
