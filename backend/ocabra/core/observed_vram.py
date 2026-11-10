"""VRAM that llama.cpp workers were measured to hold, learned after each load.

The load-time estimate (``plan_llama_cpp_vram_mb``) models weights, KV and the
recurrent-state cache, but not llama.cpp's compute buffer, which depends on the
model, the context, ``ubatch`` and on how many GPUs the layers are split over.
Measured 2026-11-10 on flint:40b: the two-GPU 262k preset held 27,168 MiB with
``ubatch=128`` against an estimate of 25,787, and 28,898 with ``ubatch=512``.
No formula fitted two models, so instead the real footprint of each worker is
recorded and used next time the same configuration loads.

Only ever raises a requirement: the admission takes the larger of its estimate
and what was observed. A low reading never shrinks a reservation, because some
workers grow after load (gemma4:26b reaches the edge of the card only with
prompts near its maximum context).

A reading is tied to a fingerprint of everything that changes the footprint
(the llama.cpp section and the placement), so editing ``ubatch`` or the context
discards it instead of applying a stale figure. Values live in Redis: if they
are lost, the next load measures again.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

from ocabra.redis_client import get_key, set_key

logger = structlog.get_logger(__name__)

_KEY_PREFIX = "llama_cpp:observed_vram:"
#: Readings older than this are dropped, so a model that is reconfigured by
#: hand (a different GGUF behind the same path, a driver change) relearns.
_TTL_SECONDS = 30 * 24 * 3600
#: Added on top of a reading. Taken right after load, before any request, so it
#: misses the buffers that grow on the first large prefill (+24-44 MiB measured
#: on flint:40b with 134k prompts) and leaves room for allocator fragmentation.
OBSERVED_VRAM_MARGIN_MB = 256


def footprint_fingerprint(extra_config: dict[str, Any] | None) -> str:
    """Hash of the configuration that determines a worker's VRAM.

    Covers the whole ``llama_cpp`` section (context, ubatch, slots, cache types,
    tensor split, disabled GPUs...), the use case and the GPUs a preset forces.
    It cannot include the GPUs the scheduler will pick, because admission needs
    the fingerprint before choosing them; without a forced placement only the
    total of a reading is used. The rest of ``_runtime`` is left out because it
    carries bookkeeping that does not change memory use.
    """
    config = extra_config if isinstance(extra_config, dict) else {}
    llama = config.get("llama_cpp")
    runtime = config.get("_runtime") if isinstance(config.get("_runtime"), dict) else {}
    payload = {
        "llama_cpp": llama if isinstance(llama, dict) else {},
        "use_case": config.get("use_case"),
        "runtime_gpu_indices": runtime.get("gpu_indices"),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


async def record(
    model_id: str,
    *,
    fingerprint: str,
    per_gpu_mb: dict[int, int],
) -> None:
    """Store what a freshly loaded worker holds on each GPU."""
    if not per_gpu_mb or any(mb <= 0 for mb in per_gpu_mb.values()):
        return
    data = {
        "fingerprint": fingerprint,
        "per_gpu_mb": {str(gpu): int(mb) for gpu, mb in per_gpu_mb.items()},
        "total_mb": int(sum(per_gpu_mb.values())),
        "recorded_at": time.time(),
    }
    try:
        await set_key(_KEY_PREFIX + model_id, data, ttl=_TTL_SECONDS)
    except Exception as exc:  # noqa: BLE001 - learning is best effort
        logger.warning("observed_vram_record_failed", model_id=model_id, error=str(exc))
        return
    logger.info(
        "observed_vram_recorded",
        model_id=model_id,
        total_mb=data["total_mb"],
        per_gpu_mb=data["per_gpu_mb"],
    )


async def lookup(model_id: str, *, fingerprint: str) -> dict[int, int] | None:
    """Per-GPU reading for this exact configuration, or ``None``."""
    try:
        data = await get_key(_KEY_PREFIX + model_id)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict) or data.get("fingerprint") != fingerprint:
        return None
    raw = data.get("per_gpu_mb")
    if not isinstance(raw, dict):
        return None
    try:
        return {int(gpu): int(mb) for gpu, mb in raw.items() if int(mb) > 0} or None
    except (TypeError, ValueError):
        return None


def raise_to_observed(
    vram_needed: int,
    required_gpu_vram_mb: dict[int, int] | None,
    observed: dict[int, int],
) -> tuple[int, dict[int, int] | None]:
    """Apply a reading to the admission figures; never lowers either of them.

    The total grows to the observed sum plus the margin. A per-GPU split, when
    the admission has one, grows GPU by GPU, sharing the margin evenly.
    """
    margin = OBSERVED_VRAM_MARGIN_MB
    total = max(int(vram_needed), sum(observed.values()) + margin)
    if not required_gpu_vram_mb:
        return total, required_gpu_vram_mb
    share = margin // max(1, len(required_gpu_vram_mb))
    per_gpu = {
        gpu: max(int(mb), observed[gpu] + share) if gpu in observed else int(mb)
        for gpu, mb in required_gpu_vram_mb.items()
    }
    return total, per_gpu
