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
    ceilings: dict[int, int | None],
) -> tuple[int, dict[int, int] | None]:
    """Apply a reading to the admission figures; never lowers either of them.

    The total grows to the observed sum plus the margin, and a per-GPU split,
    when the admission has one, GPU by GPU with the margin shared evenly. Every
    raise stops at ``ceilings``: the most the admission can find free on that
    GPU. Asking for more makes a model unloadable even on an empty card, which
    is what 0.63.0 did to qwen3-coder:30b (24,024 MiB held, 24,280 asked, 24,159
    possible). A GPU whose ceiling is unknown is not raised at all.
    """
    margin = OBSERVED_VRAM_MARGIN_MB

    total = int(vram_needed)
    total_cap = (
        sum(ceilings[gpu] for gpu in observed)  # type: ignore[misc]
        if observed and all(ceilings.get(gpu) is not None for gpu in observed)
        else None
    )
    if total_cap is not None:
        total = max(total, min(sum(observed.values()) + margin, total_cap))

    if not required_gpu_vram_mb:
        return total, required_gpu_vram_mb
    share = margin // max(1, len(required_gpu_vram_mb))
    per_gpu: dict[int, int] = {}
    for gpu, mb in required_gpu_vram_mb.items():
        cap = ceilings.get(gpu)
        if gpu in observed and cap is not None:
            per_gpu[gpu] = max(int(mb), min(observed[gpu] + share, cap))
        else:
            per_gpu[gpu] = int(mb)
    return total, per_gpu


# ── Curva medida: VRAM frente a contexto (2026-11-13) ────────────────────────
#
# El planificador de presets estima la VRAM con una fórmula (pesos + KV por
# token + sobrecargas) y varias constantes calibradas a mano con otro modelo u
# otra configuración: el coste del borrador MTP medido en qwen3.8, el de gemma4
# medido con 1 slot y el proyector residente... Apiladas, decían que
# gemma4:26b solo cabía hasta 129.564 tokens en la 3090 mientras el modelo
# corría a 262.144 con 2 GB libres, y la guarda rechazaba presets que caben.
#
# Cada carga y cada cambio de contexto en vivo miden lo que el worker ocupa de
# verdad. Aquí se guardan esos puntos por configuración (todo lo que cambia la
# huella salvo el contexto) y se ajusta una recta, que el planificador usa en
# lugar de la fórmula. Solo dentro de lo medido: más allá del mayor contexto
# medido no se extrapola.

_CURVE_PREFIX = "llama_cpp:vram_curve:"
_CURVE_MAX_POINTS = 16
#: Separación mínima entre el menor y el mayor contexto medidos para ajustar una
#: pendiente con sentido.
CURVE_MIN_SPAN_TOKENS = 16384
_CURVE_KEY_OPTIONS = (
    "parallel_slots",
    "cache_type_k",
    "cache_type_v",
    "ubatch_size",
    "batch_size",
    "kv_unified",
    "flash_attn",
    "gpu_layers",
    "n_cpu_moe",
    "split_mode",
    "tensor_split",
)


def _norm_cache_type(value: Any) -> str:
    text = str(value or "f16").lower()
    return {"q8": "q8_0", "q4": "q4_0", "fp16": "f16"}.get(text, text)


def curve_key(llama: dict[str, Any] | None, gpus: list[int] | None) -> str:
    """Identity of a configuration for the curve: everything that changes the
    footprint except the context. ``llama`` is the effective ``llama_cpp``
    section (with the live ``parallel_slots``/``ubatch_size`` if they were
    changed in place); ``gpus`` the GPUs the worker occupies."""
    section = llama if isinstance(llama, dict) else {}
    payload: dict[str, Any] = {k: section.get(k) for k in _CURVE_KEY_OPTIONS}
    payload["cache_type_k"] = _norm_cache_type(section.get("cache_type_k"))
    payload["cache_type_v"] = _norm_cache_type(section.get("cache_type_v"))
    payload["ubatch_size"] = int(section.get("ubatch_size") or 512)
    payload["parallel_slots"] = int(section.get("parallel_slots") or 1)
    spec = section.get("speculative")
    payload["spec_type"] = str(spec.get("spec_type") or "") if isinstance(spec, dict) else ""
    # Resident projector or not: on_demand and disabled start without it.
    payload["vision_resident"] = str(section.get("vision_mode") or "") == "enabled"
    payload["gpus"] = sorted(int(g) for g in (gpus or []))
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


async def record_curve_point(
    model_id: str, *, key: str, context_tokens: int, total_mb: int
) -> None:
    """Add one (context, measured VRAM) reading. Per context the largest reading
    is kept: a sample taken before a buffer grew must not lower it."""
    if context_tokens <= 0 or total_mb <= 0:
        return
    redis_key = f"{_CURVE_PREFIX}{model_id}:{key}"
    try:
        data = await get_key(redis_key)
        points = dict(data.get("points") or {}) if isinstance(data, dict) else {}
        ctx = str(int(context_tokens))
        points[ctx] = max(int(points.get(ctx) or 0), int(total_mb))
        if len(points) > _CURVE_MAX_POINTS:
            # Keep the extremes (they fix the slope) and the most recent rest.
            ordered = sorted(points, key=int)
            keep = {ordered[0], ordered[-1], ctx}
            for c in reversed(ordered):
                if len(keep) >= _CURVE_MAX_POINTS:
                    break
                keep.add(c)
            points = {c: points[c] for c in keep}
        await set_key(redis_key, {"points": points, "updated_at": time.time()}, ttl=_TTL_SECONDS)
    except Exception as exc:  # noqa: BLE001 - learning is best effort
        logger.warning("vram_curve_record_failed", model_id=model_id, error=str(exc))


class VramCurve:
    """Recta ajustada a las medidas de una configuración."""

    def __init__(self, points: dict[int, int]) -> None:
        self.points = dict(sorted(points.items()))
        xs = list(self.points)
        ys = [self.points[x] for x in xs]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        var = sum((x - mean_x) ** 2 for x in xs)
        self.slope_mb_per_token = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / var
            if var > 0
            else 0.0
        )
        self.intercept_mb = mean_y - self.slope_mb_per_token * mean_x
        self.min_context = xs[0]
        self.max_context = xs[-1]

    def predict_mb(self, context_tokens: int) -> int:
        """VRAM at ``context_tokens``, plus the margin. Never below a reading
        already taken at that context or a smaller one."""
        fitted = self.intercept_mb + self.slope_mb_per_token * context_tokens
        floor = max((mb for c, mb in self.points.items() if c <= context_tokens), default=0)
        return int(max(fitted, floor)) + OBSERVED_VRAM_MARGIN_MB

    def max_context_within(self, capacity_mb: int) -> int:
        """Largest measured-range context whose prediction fits ``capacity_mb``
        (0 when not even the smallest measured one does)."""
        if self.predict_mb(self.max_context) <= capacity_mb:
            return self.max_context
        if self.slope_mb_per_token <= 0 or self.predict_mb(self.min_context) > capacity_mb:
            return 0
        lo, hi = self.min_context, self.max_context
        while hi - lo > 256:
            mid = (lo + hi) // 2
            if self.predict_mb(mid) <= capacity_mb:
                lo = mid
            else:
                hi = mid
        return lo


async def lookup_curve(model_id: str, *, key: str) -> VramCurve | None:
    """The measured curve for this configuration, or ``None`` with fewer than
    two readings spread over at least ``CURVE_MIN_SPAN_TOKENS``."""
    try:
        data = await get_key(f"{_CURVE_PREFIX}{model_id}:{key}")
    except Exception:  # noqa: BLE001
        return None
    raw = data.get("points") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return None
    try:
        points = {int(c): int(mb) for c, mb in raw.items() if int(c) > 0 and int(mb) > 0}
    except (TypeError, ValueError):
        return None
    if len(points) < 2 or max(points) - min(points) < CURVE_MIN_SPAN_TOKENS:
        return None
    curve = VramCurve(points)
    if curve.slope_mb_per_token < 0:
        return None
    return curve
