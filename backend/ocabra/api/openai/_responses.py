"""OpenAPI ``responses=`` / header documentation for the model-loading
endpoints.

Pure metadata consumed by the FastAPI route decorators (``X-Ocabra-*`` headers,
the shared 409/429/503/504 load-failure block and the streaming vs
non-streaming response shapes). Kept together so every endpoint documents the
same contract.
"""

from __future__ import annotations

# OpenAPI: canonical X-Ocabra-* header set emitted by every endpoint that
# resolves a model (chat, completions, embeddings, pooling/score/rerank/
# classify, audio/transcriptions, audio/speech, audio/generate,
# images/generations, images/edits). Streaming endpoints flush these before
# the load blocks; non-streaming attach them to the final response.
OCABRA_MODEL_HEADERS: dict = {
    "X-Ocabra-Model-Status": {
        "description": (
            "Worker status at emit time (``configured``, ``loading``, "
            "``loaded``, ``unloaded``, ``error``). Always present."
        ),
        "schema": {"type": "string"},
    },
    "X-Ocabra-Model-Id": {
        "description": ("Canonical model id resolved from the request (post-router)."),
        "schema": {"type": "string"},
    },
    "X-Ocabra-Expected-Wait-Seconds": {
        "description": (
            "Median load time (seconds) over the last 5 successful loads of "
            "this model. Emitted only when the worker isn't ``loaded`` yet "
            "and historical samples exist."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Load-Queue-Depth": {
        "description": (
            "Total loads currently queued (waiting + active) behind the "
            "cross-backend load gate. Emitted only under pressure."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Load-Active": {
        "description": (
            "Loads actually running right now. With the current mutex this "
            "is at most 1; kept explicit so clients can distinguish "
            "queued-behind vs. loading-now."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Was-Cold-Start": {
        "description": (
            "``1`` when the request paid for a model load. Omitted when the "
            "model was already warm. Requires the endpoint to have "
            "captured the pre-load state (all resource endpoints do)."
        ),
        "schema": {"type": "string", "enum": ["1"]},
    },
    "X-Ocabra-Load-Duration-Ms": {
        "description": (
            "Milliseconds the request spent waiting for the model to be "
            "ready. Only emitted alongside ``X-Ocabra-Was-Cold-Start``."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Router": {
        "description": (
            "When the requested model was a router profile, this is the "
            "``profile_id`` the client asked for. See "
            "``X-Ocabra-Router-Target`` for what actually served."
        ),
        "schema": {"type": "string"},
    },
    "X-Ocabra-Router-Target": {
        "description": (
            "``profile_id`` of the profile that actually served the request "
            "after the router picked. Present only alongside "
            "``X-Ocabra-Router``."
        ),
        "schema": {"type": "string"},
    },
    "X-Ocabra-Slot-Queue-Saturated": {
        "description": (
            "``1`` (non-streaming only) when the request arrived to find "
            "every one of the worker's fair-share concurrency slots "
            "already occupied (see ``FairSlotGate``) and had to wait its "
            "turn — covers queueing on an *already-loaded* model, distinct "
            "from ``X-Ocabra-Expected-Wait-Seconds``/``model_loading`` "
            "which cover a cold model load. No queue *position* is given: "
            "the gate alternates by least-recently-served user, not strict "
            "FIFO, so a numeric position would be unstable. Streaming "
            "clients get the equivalent ``ocabra.slot_queued`` SSE comment "
            "instead (see the streaming response description above)."
        ),
        "schema": {"type": "string", "enum": ["1"]},
    },
    "X-Ocabra-Slot-In-Flight": {
        "description": (
            "Requests in flight on that worker at arrival time. Present "
            "only alongside ``X-Ocabra-Slot-Queue-Saturated``."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-Slot-Count": {
        "description": (
            "Configured concurrency slots for that worker. Present only "
            "alongside ``X-Ocabra-Slot-Queue-Saturated``."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-KV-Queue-Saturated": {
        "description": (
            "``1`` when the request had to wait because its conservative KV "
            "reservation did not fit in the unified pool remainder. Distinct "
            "from physical slot saturation."
        ),
        "schema": {"type": "string", "enum": ["1"]},
    },
    "X-Ocabra-KV-Capacity-Tokens": {
        "description": (
            "Logical unified-pool capacity used by oCabra admission. Present "
            "alongside ``X-Ocabra-KV-Queue-Saturated``."
        ),
        "schema": {"type": "integer"},
    },
    "X-Ocabra-KV-Reserved-Tokens": {
        "description": (
            "Logical KV tokens already reserved when the request arrived. "
            "This is an oCabra estimate, not observed llama.cpp cell usage."
        ),
        "schema": {"type": "integer"},
    },
}


#: Shared by every 503/429 ``responses`` entry below — ``Retry-After`` is
#: only present on genuinely transient failures (worker still loading,
#: registry not ready yet, circuit breaker, at-capacity). Permanent/config
#: failures (bad router target, federation disabled, a model stuck in
#: ``ModelStatus.ERROR``) return the same status codes without this header.
RETRY_AFTER_HEADER_DOC: dict = {
    "Retry-After": {
        "description": (
            "Seconds to wait before retrying. Present only when the "
            "failure is transient (model/worker loading, registry warming "
            "up post-restart, circuit breaker cooldown, at-capacity). "
            "Absent for permanent/config failures — retrying won't help."
        ),
        "schema": {"type": "integer"},
    },
}

#: Reused across every OpenAI-compatible endpoint that loads a model
#: on-demand — 503 covers "model/worker not available yet" (transient, see
#: ``RETRY_AFTER_HEADER_DOC``), 429 covers the per-worker concurrency cap.
#: 409/504 (2026-10-31) cover failures from the request-dispatch path
#: itself (``raise_inference_dispatch_error``) — previously undocumented
#: here even though the codes (``insufficient_vram``, ``generation_timeout``,
#: ``worker_degraded``) already existed and were used by the streaming
#: in-band error path (``stream_error_details``).
LOAD_FAILURE_RESPONSES_DOC: dict = {
    409: {
        "description": (
            "Not enough free VRAM to load the model even after waiting "
            "for scheduler eviction (``insufficient_vram``). Permanent "
            "given the server's current state — no ``Retry-After``."
        ),
    },
    429: {
        "description": (
            "The resolved worker is already at its concurrency cap "
            "(``model_at_capacity``) — admission control rejected this "
            "request rather than queue it indefinitely."
        ),
        "headers": RETRY_AFTER_HEADER_DOC,
    },
    503: {
        "description": (
            "Model/worker not available — still loading, failed to load "
            "(``model_load_failed``), an internal registry isn't ready yet "
            "post-restart, or the per-worker circuit breaker is open after "
            "repeated failures (``worker_degraded``). See ``Retry-After`` "
            "below for whether retrying makes sense."
        ),
        "headers": RETRY_AFTER_HEADER_DOC,
    },
    504: {
        "description": (
            "The worker accepted the request but didn't respond within "
            "oCabra's inference timeout (``generation_timeout``) — "
            "transient, see ``Retry-After``."
        ),
        "headers": RETRY_AFTER_HEADER_DOC,
    },
}


STREAMING_LOAD_RESPONSE_DOC: dict = {
    200: {
        "description": (
            "Successful response. When ``stream=true`` the body is "
            "``text/event-stream``; oCabra pre-flushes the ``X-Ocabra-*`` "
            "headers below **before** triggering any model load, and "
            "interleaves two oCabra-aware SSE **comment** frames carrying "
            "load progress:\n\n"
            '``: {"event":"ocabra.model_loading","model_id":...,'
            '"worker_key":...,"status":...,'
            '"expected_wait_seconds":...}\\n\\n`` on stream open, and\n\n'
            '``: {"event":"ocabra.model_ready","model_id":...,'
            '"worker_key":...,"load_duration_ms":...,'
            '"was_cold_start":...}\\n\\n`` once the model is ready.\n\n'
            "A third, unrelated to cold-start, is emitted when the model "
            "is already loaded but every fair-share concurrency slot is "
            "occupied (see ``X-Ocabra-Slot-Queue-Saturated`` below for the "
            "non-streaming equivalent and why there's no queue position):\n\n"
            '``: {"event":"ocabra.slot_queued","model_id":...,'
            '"in_flight":...,"slots":...}\\n\\n``\n\n'
            "The leading ``:`` makes these SSE **comments**, ignored by "
            "strict OpenAI SDK parsers. oCabra-aware clients may parse the "
            "JSON after the colon. If the load fails after headers have "
            'been flushed, the error travels as ``data: {"error": ...}`` '
            "in the same stream (not as an HTTP status)."
        ),
        "headers": OCABRA_MODEL_HEADERS,
    },
    **LOAD_FAILURE_RESPONSES_DOC,
}


NON_STREAMING_LOAD_RESPONSE_DOC: dict = {
    200: {
        "description": (
            "Successful response. Non-streaming endpoints attach the "
            "``X-Ocabra-*`` headers below to the final response — clients "
            "then learn retrospectively whether the request paid for a cold "
            "start (``X-Ocabra-Was-Cold-Start`` + ``-Load-Duration-Ms``) "
            "and whether the server is under load (``-Load-Queue-Depth`` / "
            "``-Load-Active``)."
        ),
        "headers": OCABRA_MODEL_HEADERS,
    },
    **LOAD_FAILURE_RESPONSES_DOC,
}
