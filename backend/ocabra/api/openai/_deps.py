"""Compatibility barrel for the OpenAI endpoint dependencies (REF-06).

The request-path logic that used to live here was split into cohesive modules:

* ``ocabra.api.openai.deps``      — the ``Depends`` providers.
* ``ocabra.core.inference.errors`` — OpenAI-shaped error constructors.
* ``ocabra.core.inference.body``   — request-body reading, client-disconnect.
* ``ocabra.core.inference.reasoning`` — reasoning/think + backend-body build.
* ``ocabra.core.inference.admission`` — fair-slot/KV sizing, capabilities.
* ``ocabra.core.inference.loading``   — on-demand loading + ``X-Ocabra-*``.
* ``ocabra.core.inference.resolution``— model/profile/router resolution.
* ``ocabra.api.sse``               — SSE keepalive + comment frames.
* ``ocabra.api.openai._responses`` — OpenAPI response/header docs.

This module re-exports every public symbol so existing handler imports
(``from ._deps import ...``) keep working unchanged. The inference modules
live under ``core`` so ``core`` code no longer imports ``api`` on the request
path (REF-09). New code should import from the modules above directly.

``asyncio`` is imported here on purpose: some tests patch
``ocabra.api.openai._deps.asyncio.sleep`` (which is the shared ``asyncio``
module), so the attribute must resolve through this module.
"""

from __future__ import annotations

import asyncio  # noqa: F401 — test patch target (ocabra.api.openai._deps.asyncio.sleep)

from ocabra.api.openai._responses import (
    LOAD_FAILURE_RESPONSES_DOC,
    NON_STREAMING_LOAD_RESPONSE_DOC,
    OCABRA_MODEL_HEADERS,
    RETRY_AFTER_HEADER_DOC,
    STREAMING_LOAD_RESPONSE_DOC,
)
from ocabra.api.openai.deps import (
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
)
from ocabra.api.sse import (
    SSE_KEEPALIVE_INTERVAL_S,
    keepalive_until_done,
    sse_ocabra_event,
)
from ocabra.core.inference.admission import (
    chat_required_capabilities,
    check_capability,
    fairness_kv_budget,
    fairness_slots,
    fairness_user_key,
    is_batch_origin_request,
    ollama_required_capabilities,
)
from ocabra.core.inference.body import (
    ClientGoneError,
    await_unless_client_gone,
    read_json_object,
)
from ocabra.core.inference.errors import (
    _UPSTREAM_HEADERS_KEPT,
    EngineSuffixedModelName,
    _openai_error,
    engine_suffixed_error,
    raise_inference_dispatch_error,
    raise_upstream_http_error,
    router_pick_error,
    stream_error_details,
)
from ocabra.core.inference.loading import (
    _LOAD_WAIT_NOTE_MIN_MS,
    _do_ensure_loaded,
    _do_ensure_loaded_inner,
    _ensure_load_timeout_s,
    _ensure_worker_loaded,
    _snapshot_pre_load,
    _wait_for_eviction_claim,
    _wait_for_service_gpu_and_retry_load,
    _wait_for_unloading_transition,
    build_model_status_headers,
    ensure_worker_loaded,
    ocabra_response_headers,
    prepare_worker_state,
)
from ocabra.core.inference.reasoning import (
    _REASONING_EFFORT_DEFAULT,
    _REASONING_EFFORT_LEVELS,
    _REASONING_EFFORT_OFF,
    DEFAULT_REASONING_EFFORT,
    _apply_reasoning_budget,
    _apply_reasoning_effort,
    _apply_thinking_controls,
    _flatten_text_only_content,
    _normalize_messages_for_backend,
    _normalize_tool_call_entry,
    _output_allowance,
    _reasoning_budget_capped,
    _reasoning_budget_for_level,
    _resolve_default_reasoning_effort,
    _thinking_disable_intent,
    to_backend_body,
)
from ocabra.core.inference.resolution import (
    _ROUTER_DEFAULTS,
    ensure_loaded,
    lookup_profile,
    merge_profile_defaults,
    note_router_resolution,
    resolve_model,
    resolve_profile,
    resolve_profile_runtime_plan,
)

# Previously re-exported from here (helpers that already live in core); kept
# for back-compat.
from ocabra.core.model_manager_helpers import (
    compute_worker_key,
    strip_diarization_overrides,
)

__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "LOAD_FAILURE_RESPONSES_DOC",
    "NON_STREAMING_LOAD_RESPONSE_DOC",
    "OCABRA_MODEL_HEADERS",
    "RETRY_AFTER_HEADER_DOC",
    "SSE_KEEPALIVE_INTERVAL_S",
    "STREAMING_LOAD_RESPONSE_DOC",
    "_LOAD_WAIT_NOTE_MIN_MS",
    "_REASONING_EFFORT_DEFAULT",
    "_REASONING_EFFORT_LEVELS",
    "_REASONING_EFFORT_OFF",
    "_ROUTER_DEFAULTS",
    "_UPSTREAM_HEADERS_KEPT",
    "ClientGoneError",
    "EngineSuffixedModelName",
    "_apply_reasoning_budget",
    "_apply_reasoning_effort",
    "_apply_thinking_controls",
    "_do_ensure_loaded",
    "_do_ensure_loaded_inner",
    "_ensure_load_timeout_s",
    "_ensure_worker_loaded",
    "_flatten_text_only_content",
    "_normalize_messages_for_backend",
    "_normalize_tool_call_entry",
    "_openai_error",
    "_output_allowance",
    "_reasoning_budget_capped",
    "_reasoning_budget_for_level",
    "_resolve_default_reasoning_effort",
    "_snapshot_pre_load",
    "_thinking_disable_intent",
    "_wait_for_eviction_claim",
    "_wait_for_service_gpu_and_retry_load",
    "_wait_for_unloading_transition",
    "await_unless_client_gone",
    "build_model_status_headers",
    "chat_required_capabilities",
    "check_capability",
    "compute_worker_key",
    "engine_suffixed_error",
    "ensure_loaded",
    "ensure_worker_loaded",
    "fairness_kv_budget",
    "fairness_slots",
    "fairness_user_key",
    "get_federation_manager",
    "get_model_manager",
    "get_openai_user",
    "get_profile_registry",
    "is_batch_origin_request",
    "keepalive_until_done",
    "lookup_profile",
    "merge_profile_defaults",
    "note_router_resolution",
    "ocabra_response_headers",
    "ollama_required_capabilities",
    "prepare_worker_state",
    "raise_inference_dispatch_error",
    "raise_upstream_http_error",
    "read_json_object",
    "resolve_model",
    "resolve_profile",
    "resolve_profile_runtime_plan",
    "router_pick_error",
    "sse_ocabra_event",
    "stream_error_details",
    "strip_diarization_overrides",
    "to_backend_body",
]
