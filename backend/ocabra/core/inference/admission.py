"""Admission-gate sizing (fair-share slots, KV budget), request-origin
classification and capability checks for the request path.

The admission *middleware* itself lives in :mod:`ocabra.core.admission`; this
module holds the per-request helpers that size its gate and answer whether a
request is capable / batch-originated.
"""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from fastapi import Request

from ocabra.core.inference.errors import _openai_error
from ocabra.core.model_ref import is_llama_cpp_family

if TYPE_CHECKING:
    from ocabra.api._deps_auth import UserContext
    from ocabra.core.model_manager import ModelState


def is_batch_origin_request(request: Request) -> bool:
    """True when this call was dispatched by the batch processor
    (``ocabra.core.batch_processor``), not a real client.

    The batch processor calls back into this same app over
    ``httpx.ASGITransport`` carrying the per-process internal token (the
    same one ``get_current_user`` uses to impersonate the batch owner, see
    ``_deps_auth.py``) plus a dedicated origin header. Only trust the origin
    header with that token: an external caller can send any header it likes,
    but cannot know a token that never leaves this process.
    """

    # «Mi servidor, mis reglas» — «me first»: a peer's requests get the batch
    # class, so local ones go ahead and may preempt them. Set from the
    # authenticated service user (not a header a client could send).
    peer_rules = getattr(getattr(request, "state", None), "federation_rules", None)
    if peer_rules is not None and getattr(peer_rules, "local_first", False):
        return True

    from ocabra.api._deps_auth import INTERNAL_TOKEN, INTERNAL_TOKEN_HEADER

    internal = request.headers.get(INTERNAL_TOKEN_HEADER, "")
    if not internal or not hmac.compare_digest(internal, INTERNAL_TOKEN):
        return False
    return request.headers.get("X-Internal-Request-Origin", "") == "batch"


def fairness_user_key(user: UserContext) -> str:
    """Identity key for the per-worker fair-share admission gate
    (``ocabra.core.fair_slot_gate``). Prefers the stable user_id; falls
    back to username, then a shared "anonymous" bucket when auth is
    disabled (``REQUIRE_API_KEY_*=false``) — every anonymous caller then
    fair-shares as a single "user", which is strictly better than no
    fairness at all but can't tell two anonymous callers apart.
    """
    return user.user_id or user.username or "anonymous"


def fairness_slots(state: ModelState) -> int:
    """How many requests this model's worker can run concurrently, for
    sizing the fair-share gate.

    Thin wrapper over ``resolve_worker_slots``, which is also what admission
    (``ModelManager.effective_inflight_cap``) and the dashboard card use. They
    were three separate implementations and two of them disagreed: admission
    assumed a single slot whenever the runtime-preset coordinator had not
    written ``_runtime.parallelism``, so a llama.cpp model with
    ``parallel_slots=4`` ran on a 4-slot gate while being admitted as if it had
    one. Unknown backends stay conservatively one-at-a-time.
    """
    from ocabra.core.model_manager_helpers import resolve_worker_slots

    return resolve_worker_slots(state) or 1


def fairness_kv_budget(
    state: ModelState,
    *,
    required_tokens: int | None,
    effective_context: int | None = None,
    reservation_tokens: int | None = None,
) -> tuple[int | None, int]:
    """Return ``(pool_capacity, request_reservation)`` for fair admission.

    KV weighting is deliberately opt-in at runtime: only llama.cpp workers
    that explicitly enabled the unified cache publish a capacity. Every other
    backend receives ``(None, 0)`` and keeps the historical slot-only gate.
    """
    if not is_llama_cpp_family(state.backend_type) or required_tokens is None:
        return None, 0

    from ocabra.core.model_manager_helpers import resolve_llama_cpp_option

    enabled = resolve_llama_cpp_option(state, "kv_unified", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None, 0

    runtime = state.extra_config.get("_runtime") if isinstance(state.extra_config, dict) else None
    raw_capacity = effective_context
    if raw_capacity is None and isinstance(runtime, dict):
        raw_capacity = runtime.get("effective_context")
    if raw_capacity is None:
        raw_capacity = resolve_llama_cpp_option(state, "ctx_size", None)
    if raw_capacity is None and isinstance(state.extra_config, dict):
        use_case = state.extra_config.get("use_case")
        if isinstance(use_case, dict):
            raw_capacity = use_case.get("context")
    try:
        capacity = int(raw_capacity or 0)
        # Legacy runtime presets express context per slot. Adaptive unified-KV
        # presets instead expose the aggregate physical pool so one request can
        # use all of it; those must not be multiplied by parallelism.
        if (
            isinstance(runtime, dict)
            and runtime.get("preset_key")
            and not runtime.get("context_is_pool")
        ):
            capacity *= max(1, int(runtime.get("parallelism") or 1))
        # El planificador puede separar las dos cifras: ``required_tokens``
        # elige ventana y se infla a proposito cuando no se ha podido medir el
        # prompt; reservar eso bloquearia el pool entero (ver ``_reserva_kv``).
        reservation = max(
            0, int(reservation_tokens if reservation_tokens is not None else required_tokens)
        )
    except (TypeError, ValueError):
        return None, 0
    # A context changed in place holds the pool llama-server really has now;
    # the preset and extra_config still describe the one it was loaded with.
    live = getattr(state, "live_context", None)
    if isinstance(live, dict):
        try:
            live_pool = int(live.get("n_ctx") or 0)
        except (TypeError, ValueError):
            live_pool = 0
        if live_pool > 0:
            capacity = min(capacity, live_pool) if capacity > 0 else live_pool
    if capacity <= 0:
        return None, 0
    return capacity, reservation


def chat_required_capabilities(body: dict, content_kinds: set[str]) -> tuple[str, ...]:
    """What a chat request needs from the model, for a router to choose by.

    The same checks ``check_capability`` makes after resolution, but known
    before it: a router can then skip a warm target that would only answer
    400 (2026-11-15: ``recomendado-sota`` sent images to whichever target
    was loaded, with or without vision).
    """
    wanted: list[str] = []
    if body.get("tools"):
        wanted.append("tools")
    if "image_url" in content_kinds:
        wanted.append("vision")
    if "input_audio" in content_kinds:
        wanted.append("audio_input")
    if "video_url" in content_kinds:
        wanted.append("video_input")
    return tuple(wanted)


def ollama_required_capabilities(body: dict) -> tuple[str, ...]:
    """``chat_required_capabilities`` for Ollama-native bodies: images travel
    as base64 lists in ``images`` (per message in /api/chat, top-level in
    /api/generate)."""
    wanted: list[str] = []
    if body.get("tools"):
        wanted.append("tools")
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    if body.get("images") or any(isinstance(m, dict) and m.get("images") for m in messages):
        wanted.append("vision")
    return tuple(wanted)


def check_capability(state: ModelState, capability: str, endpoint: str) -> None:
    """Raise 400 if the model lacks the required capability."""
    if not getattr(state.capabilities, capability, False):
        raise _openai_error(
            f"The model '{state.model_id}' does not support {endpoint}.",
            "invalid_request_error",
            param="model",
            code="model_not_capable",
        )
