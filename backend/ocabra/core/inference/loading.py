"""On-demand worker loading, load-wait handling and ``X-Ocabra-*`` status
headers for the request path.

Holds the public :func:`ensure_worker_loaded` wrapper (used by streaming
endpoints and other callers), its internals, the derived-worker state prep
(:func:`prepare_worker_state`) and the cold-start / load-pressure header
builders. Moved out of ``api`` (REF-09) so ``core`` callers — the realtime
session, context-probe service and PIN autoload — no longer import from
``api``; the ``api`` layer re-exports these via ``_deps``.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from ocabra.core.inference.errors import _openai_error
from ocabra.core.model_manager_helpers import strip_diarization_overrides
from ocabra.core.request_telemetry import current_runtime_telemetry

if TYPE_CHECKING:
    from ocabra.core.model_manager import ModelManager, ModelState

logger = structlog.get_logger(__name__)


def _ensure_load_timeout_s() -> int:
    from ocabra.config import settings

    return max(60, int(settings.model_load_wait_timeout_s))


async def _wait_for_service_gpu_and_retry_load(
    model_manager: ModelManager,
    model_id: str,
    state: ModelState | None,
) -> ModelState | None:
    """After a load fails with InsufficientVRAMError, check whether the cause is
    an external service (Hunyuan, TRELLIS.2, ...) holding a temporary VRAM
    reservation on this model's preferred GPU (see ServiceManager.reserve_gpu_vram)
    — as opposed to genuine, indefinite capacity exhaustion. If so, poll until the
    reservation clears (or we time out) and retry the load, instead of failing
    the request immediately. Plain models competing for VRAM with each other are
    unaffected: this only activates when a service reservation is the reason.

    Returns the freshly loaded state on success, or None if the caller should
    fall through to the normal InsufficientVRAMError → 409 response.
    """
    from ocabra.core.model_manager import ModelStatus
    from ocabra.core.scheduler import InsufficientVRAMError

    gpu_manager = model_manager.gpu_manager
    target_gpu = state.preferred_gpu if state else None
    if gpu_manager is None or target_gpu is None:
        return None
    if not gpu_manager.has_service_lock(target_gpu):
        return None

    blockers = gpu_manager.service_locks(target_gpu)
    model_manager.mark_service_gpu_wait(model_id, ", ".join(blockers) or "?")
    try:
        for _ in range(_ensure_load_timeout_s()):
            await asyncio.sleep(1)
            if gpu_manager.has_service_lock(target_gpu):
                continue
            try:
                await model_manager.load(model_id)
            except InsufficientVRAMError:
                # Reservation cleared but something else (a genuinely concurrent
                # load) grabbed the VRAM first — keep waiting for the timeout.
                continue
            new_state = await model_manager.get_state(model_id)
            if new_state and new_state.status == ModelStatus.LOADED:
                return new_state
        return None
    finally:
        model_manager.clear_service_gpu_wait(model_id)


async def _wait_for_unloading_transition(
    model_manager: ModelManager,
    model_id: str,
    state: ModelState,
) -> ModelState:
    """Wait for an in-progress unload before attempting a new load.

    An unload is a transient state, not a terminal model failure.  Treating it
    as unavailable makes requests race with idle/manual eviction and produces a
    needless 503 even though the model can be loaded again immediately after.
    """
    from ocabra.core.model_manager import ModelStatus

    if state.status != ModelStatus.UNLOADING:
        return state

    deadline = asyncio.get_running_loop().time() + _ensure_load_timeout_s()
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.1)
        current = await model_manager.get_state(model_id)
        if current is None:
            break
        if current.status != ModelStatus.UNLOADING:
            return current

    raise _openai_error(
        f"Model '{model_id}' did not finish unloading in time.",
        "server_error",
        code="model_transition_timeout",
        status_code=503,
        headers={"Retry-After": "2"},
    )


async def _wait_for_eviction_claim(
    model_manager: ModelManager,
    model_id: str,
    state: ModelState,
) -> ModelState:
    """Wait while an eviction has claimed the model (still LOADED, about to go).

    Without this a request admitted between "the drain saw it idle" and the
    unload went to a worker killed a second later (502 mid-stream). Afterwards
    the model is UNLOADING or UNLOADED, which the normal path handles: wait,
    then load it again.
    """
    check = getattr(model_manager, "is_eviction_claimed", None)

    def claimed() -> bool:
        # ``is True``: a mocked manager answers with a truthy mock.
        return callable(check) and check(model_id) is True

    if not claimed():
        return state
    # Admitted (in flight) but not on the worker: a context change made under
    # the claim must not wait for this request (see begin_claim_wait).
    begin = getattr(model_manager, "begin_claim_wait", None)
    end = getattr(model_manager, "end_claim_wait", None)
    if callable(begin):
        begin(model_id)
    try:
        deadline = asyncio.get_running_loop().time() + _ensure_load_timeout_s()
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
            if not claimed():
                break
    finally:
        if callable(end):
            end(model_id)
    if claimed():
        raise _openai_error(
            f"Model '{model_id}' is being unloaded; retry shortly.",
            "server_error",
            code="model_transition_timeout",
            status_code=503,
            headers={"Retry-After": "2"},
        )
    current = await model_manager.get_state(model_id)
    return current if current is not None else state


async def ocabra_response_headers(
    request: object,
    model_manager: ModelManager,
    worker_key: str,
    base_model_id: str,
) -> dict[str, str]:
    """Compact wrapper for post-load endpoints.

    Reads ``pre_load_status`` and ``load_started_wall`` off ``request.state``
    (populated by :func:`_snapshot_pre_load`) and calls
    :func:`build_model_status_headers` with the cold-start + router fields
    already filled in. Endpoints get a one-liner for the whole header set.
    """
    st = getattr(request, "state", None)
    pre_status = getattr(st, "pre_load_status", None) if st is not None else None
    load_started = getattr(st, "load_started_wall", None) if st is not None else None
    via_router = getattr(st, "via_router_profile_id", None) if st is not None else None
    router_target = getattr(st, "resolved_model_id", None) if st is not None else None
    elapsed_ms: int | None = None
    if load_started is not None:
        elapsed_ms = int(max(0.0, time.monotonic() - load_started) * 1000)
    return await build_model_status_headers(
        model_manager,
        worker_key,
        base_model_id,
        pre_status=pre_status,
        elapsed_ms=elapsed_ms,
        via_router_profile_id=via_router,
        router_target_profile_id=router_target,
    )


async def _snapshot_pre_load(
    model_manager: ModelManager,
    worker_key: str,
    request_state: object | None,
) -> None:
    """Stamp ``pre_load_status`` + ``load_started_wall`` on ``request.state``.

    Called right before ``_ensure_worker_loaded`` so downstream endpoints can
    tell whether the request paid for a cold start and how long the load
    took. No-op when ``request_state`` isn't provided.
    """
    if request_state is None:
        return
    try:
        if hasattr(request_state, "pre_load_status"):
            return  # first snapshot wins (e.g. router fallback path)
        state = await model_manager.get_state(worker_key)
        request_state.pre_load_status = state.status.value if state is not None else "unknown"
        request_state.load_started_wall = time.monotonic()
    except Exception:  # noqa: BLE001 — never block resolution on telemetry
        pass


async def build_model_status_headers(
    model_manager: ModelManager,
    worker_key: str,
    base_model_id: str,
    *,
    pre_status: str | None = None,
    elapsed_ms: int | None = None,
    via_router_profile_id: str | None = None,
    router_target_profile_id: str | None = None,
) -> dict[str, str]:
    """Return the canonical ``X-Ocabra-*`` header set for any request.

    Base fields (always emitted when the state is known):
      - ``X-Ocabra-Model-Status`` — worker state at header-flush time.
      - ``X-Ocabra-Model-Id`` — canonical model id after router resolution.

    Pressure hints (emitted only when the worker isn't yet ``loaded``):
      - ``X-Ocabra-Expected-Wait-Seconds`` — median of the last 5 loads.
      - ``X-Ocabra-Load-Queue-Depth`` / ``-Load-Active`` — how many other
        loads are pending/running behind the backend gate.

    Cold-start flags (post-load; require ``pre_status`` from *before* the
    load call):
      - ``X-Ocabra-Was-Cold-Start`` — ``"1"`` when the client paid for a
        load. Omitted when the model was already warm.
      - ``X-Ocabra-Load-Duration-Ms`` — how long the request had to wait
        for the model to be ready. Requires ``elapsed_ms``.

    Router attribution (chat/completions routers):
      - ``X-Ocabra-Router`` — the router profile the client asked for.
      - ``X-Ocabra-Router-Target`` — the target profile that actually ran.

    Streaming endpoints call this **before** the load with no cold-start /
    router kwargs; non-streaming ones call it **after** and pass
    ``pre_status``/``elapsed_ms`` so the response fully describes what
    happened.
    """
    state = await model_manager.get_state(worker_key)
    if state is None and worker_key != base_model_id:
        state = await model_manager.get_state(base_model_id)

    if state is None:
        return {}

    headers: dict[str, str] = {
        "X-Ocabra-Model-Status": state.status.value,
        "X-Ocabra-Model-Id": state.model_id,
    }
    if state.status.value != "loaded":
        # Try the worker key first (covers per-profile workers); fall back to
        # the base model so we still surface a hint when this is the first
        # load of an override variant.
        expected = await model_manager.get_expected_load_seconds(worker_key)
        if expected is None and worker_key != base_model_id:
            expected = await model_manager.get_expected_load_seconds(base_model_id)
        if expected is not None:
            headers["X-Ocabra-Expected-Wait-Seconds"] = str(expected)
    # Cross-request load pressure. Clients render this as "queued behind N
    # cold starts" so the wait doesn't feel silent.
    queue_waiters = int(getattr(model_manager, "_load_queue_waiters", 0) or 0)
    in_progress = len(getattr(model_manager, "_load_in_progress", set()) or ())
    if queue_waiters or in_progress:
        headers["X-Ocabra-Load-Queue-Depth"] = str(queue_waiters + in_progress)
        headers["X-Ocabra-Load-Active"] = str(in_progress)

    # Cold-start attribution — set only when the caller captured the state
    # before triggering the load. ``pre_status="loaded"`` is treated as
    # "warm request" and both fields are omitted.
    if pre_status is not None and pre_status != "loaded":
        headers["X-Ocabra-Was-Cold-Start"] = "1"
        if elapsed_ms is not None and elapsed_ms >= 0:
            headers["X-Ocabra-Load-Duration-Ms"] = str(int(elapsed_ms))

    # Router attribution — same shape as chat/completions streaming today.
    if via_router_profile_id:
        headers["X-Ocabra-Router"] = str(via_router_profile_id)
        if router_target_profile_id:
            headers["X-Ocabra-Router-Target"] = str(router_target_profile_id)
    return headers


async def ensure_worker_loaded(
    model_manager: ModelManager,
    base_model_id: str,
    worker_key: str,
    load_overrides: dict | None,
    *,
    preferred_gpu: int | None = None,
    load_policy: str | None = None,
    required_tokens: int | None = None,
) -> ModelState:
    """Public wrapper around :func:`_ensure_worker_loaded`.

    Exposed for streaming endpoints that pre-flush headers and trigger the
    actual load from inside the SSE generator.
    """
    return await _ensure_worker_loaded(
        model_manager,
        base_model_id,
        worker_key,
        load_overrides,
        preferred_gpu=preferred_gpu,
        load_policy=load_policy,
        required_tokens=required_tokens,
    )


async def _ensure_worker_loaded(
    model_manager: ModelManager,
    base_model_id: str,
    worker_key: str,
    load_overrides: dict | None,
    *,
    preferred_gpu: int | None = None,
    load_policy: str | None = None,
    required_tokens: int | None = None,
) -> ModelState:
    """Ensure the worker identified by *worker_key* is loaded.

    When *worker_key* differs from *base_model_id* (non-empty overrides),
    we check if a virtual model entry already exists in ModelManager; if not,
    we create one by cloning the base model's state and applying
    ``load_overrides`` as extra config.

    Returns the :class:`ModelState` of the loaded worker.
    """

    async def _mark_runtime(state: ModelState) -> None:
        marker = getattr(model_manager, "mark_runtime_request", None)
        if marker is None:
            return
        result = marker(state.model_id, required_tokens=required_tokens)
        if inspect.isawaitable(result):
            await result

    # If worker_key == base_model_id, just use the normal path
    if worker_key == base_model_id:
        state = await model_manager.get_state(base_model_id)
        if state is None:
            raise _openai_error(
                f"Base model '{base_model_id}' is not configured.",
                "invalid_request_error",
                param="model",
                code="model_not_found",
                status_code=404,
            )
        # Delegate to the existing ensure_loaded for actual loading
        state = await _do_ensure_loaded(model_manager, base_model_id)
        await _mark_runtime(state)
        return state

    # Clone from the current base configuration. Derived states are retained in
    # memory after unload, while the base configuration remains mutable. Always
    # resolve the base first so an unloaded derived worker cannot restart with a
    # stale option such as vision_mode or mmproj_model_id.
    state = await prepare_worker_state(
        model_manager,
        base_model_id,
        worker_key,
        load_overrides,
        preferred_gpu=preferred_gpu,
        load_policy=load_policy,
    )
    state = await _do_ensure_loaded(model_manager, worker_key)
    await _mark_runtime(state)
    return state


async def prepare_worker_state(
    model_manager: ModelManager,
    base_model_id: str,
    worker_key: str,
    load_overrides: dict | None,
    *,
    preferred_gpu: int | None = None,
    load_policy: str | None = None,
) -> ModelState | None:
    """Create (or refresh, when not running) the derived state of
    ``worker_key`` without loading it. Shared by :func:`_ensure_worker_loaded`
    and the PIN autoload, which loads it through its own polite path."""
    base_state = await model_manager.get_state(base_model_id)
    if base_state is None:
        raise _openai_error(
            f"Base model '{base_model_id}' is not configured.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    # Without the diarization switch, like the worker key: diarizing is decided
    # per request (see strip_diarization_overrides).
    # Deep merge: a profile that overrides one option of a backend section
    # (``{"vllm": {"max_model_len": 8192}}``) keeps the rest of the base
    # section instead of replacing it wholesale. A null drops the base key.
    from ocabra.core.model_manager import _deep_merge_dicts

    merged_extra = _deep_merge_dicts(
        dict(base_state.extra_config or {}), strip_diarization_overrides(load_overrides)
    )

    # Dedicated worker: reuse an existing transient state, but refresh it when
    # no process is running. Mutating a loaded/loading state would make its
    # metadata disagree with the arguments of the live backend process.
    state = await model_manager.get_state(worker_key)
    if state is not None:
        from ocabra.core.model_manager import LoadPolicy, ModelStatus

        if state.status not in (ModelStatus.LOADED, ModelStatus.LOADING):
            state.extra_config = merged_extra
            state.preferred_gpu = (
                preferred_gpu if preferred_gpu is not None else base_state.preferred_gpu
            )
            state.load_policy = LoadPolicy(load_policy or base_state.load_policy.value)
            state.pin_permanent = base_state.pin_permanent
        return state

    try:
        # Derived workers are in-memory only: the profile that owns the
        # overrides is already persisted in ``model_profiles`` and knows how
        # to reconstruct this state on demand, so a second ``model_configs``
        # row would just show up as a phantom "(override)" entry in the
        # model list. Restart drops these — that's fine, they get recreated
        # by the next request that resolves the profile.
        await model_manager.add_model(
            model_id=worker_key,
            backend_type=base_state.backend_type,
            display_name=f"{base_state.display_name} (override)",
            load_policy=load_policy or base_state.load_policy.value,
            pin_permanent=base_state.pin_permanent,
            preferred_gpu=(
                preferred_gpu if preferred_gpu is not None else base_state.preferred_gpu
            ),
            extra_config=merged_extra,
            persist=False,
            backend_model_id_override=base_state.backend_model_id,
        )
    except Exception:
        # May already exist from concurrent request
        pass
    return await model_manager.get_state(worker_key)


# Por debajo de esto, la vuelta de un modelo ya cargado (leer su estado y
# tocar last_request_at); por encima, la peticion espero a que cargase.
_LOAD_WAIT_NOTE_MIN_MS = 200


async def _do_ensure_loaded(
    model_manager: ModelManager,
    model_id: str,
) -> ModelState:
    """Core loading logic extracted from ``ensure_loaded``.

    Triggers on-demand loading, waits for LOADING state, and touches
    ``last_request_at``. The wait goes to the request's telemetry as
    ``load_ms`` (all routes, batches included): the stats row keeps the total
    the client waited, and the duration estimator subtracts it so a cold start
    does not pass for a slow request.
    """
    started = time.monotonic()
    state = await _do_ensure_loaded_inner(model_manager, model_id)
    waited_ms = int((time.monotonic() - started) * 1000)
    telemetry = current_runtime_telemetry.get()
    if telemetry is not None and waited_ms >= _LOAD_WAIT_NOTE_MIN_MS:
        telemetry.note_load_wait(waited_ms)
    return state


async def _do_ensure_loaded_inner(
    model_manager: ModelManager,
    model_id: str,
) -> ModelState:
    from ocabra.core.model_manager import ModelStatus

    state = await model_manager.get_state(model_id)
    if state is None:
        logger.warning("ensure_loaded_state_missing", model_id=model_id)
        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    state = await _wait_for_eviction_claim(model_manager, model_id, state)
    state = await _wait_for_unloading_transition(model_manager, model_id, state)

    async def _touch(resolved_id: str, request_at: datetime) -> None:
        state.last_request_at = request_at
        touch = getattr(model_manager, "touch_last_request_at", None)
        if touch is not None:
            result = touch(resolved_id, request_at)
            if inspect.isawaitable(result):
                await result

    if state.status == ModelStatus.LOADED:
        await _touch(model_id, datetime.now(UTC))
        return state

    if state.status in (ModelStatus.CONFIGURED, ModelStatus.UNLOADED, ModelStatus.ERROR):
        try:
            await model_manager.load(model_id)
        except Exception as exc:
            from ocabra.core.scheduler import InsufficientVRAMError

            if isinstance(exc, InsufficientVRAMError):
                retried_state = await _wait_for_service_gpu_and_retry_load(
                    model_manager, model_id, state
                )
                if retried_state is not None:
                    await _touch(model_id, datetime.now(UTC))
                    return retried_state
                raise _openai_error(
                    str(exc),
                    "invalid_request_error",
                    code="insufficient_vram",
                    status_code=409,
                    # Transient: memory frees as other models go idle (BUG-19).
                    headers={"Retry-After": "30"},
                ) from exc
            raise _openai_error(
                f"Failed to load model '{model_id}': {exc}",
                "server_error",
                code="model_load_failed",
                status_code=503,
                headers={"Retry-After": "10"},
            ) from exc

        state = await model_manager.get_state(model_id)
        if state and state.status == ModelStatus.LOADED:
            await _touch(model_id, datetime.now(UTC))
            return state

    if state and state.status == ModelStatus.LOADING:
        for _ in range(_ensure_load_timeout_s()):
            await asyncio.sleep(1)
            state = await model_manager.get_state(model_id)
            if state and state.status == ModelStatus.LOADED:
                await _touch(model_id, datetime.now(UTC))
                return state
            if state is None or state.status != ModelStatus.LOADING:
                break
        if state and state.status == ModelStatus.LOADING:
            raise _openai_error(
                f"Model '{model_id}' did not finish loading in time.",
                "server_error",
                code="model_load_timeout",
                status_code=503,
                headers={"Retry-After": "10"},
            )

    detail_suffix = ""
    if state and state.error_message:
        detail_suffix = f" detail: {state.error_message}"

    # A model stuck in ERROR needs a human to look at it (bad weights, repeated
    # crash, etc.) — retrying on a timer won't help, so skip the hint there.
    # Any other non-terminal status (CONFIGURED, UNLOADING, ...) is a normal
    # transitional state a short wait can resolve.
    retry_headers = None if state and state.status == ModelStatus.ERROR else {"Retry-After": "10"}
    raise _openai_error(
        (
            f"Model '{model_id}' is not available "
            f"(status: {state.status.value if state else 'unknown'}).{detail_suffix}"
        ),
        "server_error",
        code="model_unavailable",
        status_code=503,
        headers=retry_headers,
    )
