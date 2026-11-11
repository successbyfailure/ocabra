"""Measure the largest context a llama.cpp preset fits on its GPUs.

The worker is loaded once with a small context and grown in place (fork
endpoint ``POST /props {"context"}``, see ``ModelManager.resize_context``),
reading the process's real VRAM on every GPU after each step. The search
stops at the safety margin: ``vram_buffer_mb`` below what an admission can
ever find free on each card (``GPUManager.admission_ceiling_mb``). The vision
projector is deliberately left out of that budget: it is loaded on demand,
and when it does not fit the context is reduced at that moment instead.

Three layers, so the search can be tested without GPUs:

* :func:`next_context` — pure step: given the measurements so far, which
  context to try next, or ``None`` when the search has converged.
* :func:`run_probe` — the loop, with ``resize`` and ``measure`` injected.
* :class:`ContextProbeService` (``context_probe_service.py``) — prepares the
  GPUs, loads the worker, runs the loop and puts everything back.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

#: Contexts are tried in multiples of this many tokens.
GRANULARITY = 4096


@dataclass
class ProbeStep:
    context: int
    used_mb: dict[int, int]
    fits: bool
    note: str = ""


@dataclass
class ProbeResult:
    max_context: int
    used_mb: dict[int, int]
    limit_mb: dict[int, int]
    kv_mb_per_1k: dict[int, float]
    steps: list[ProbeStep] = field(default_factory=list)
    capped_by: str = "memory"  # "memory" or "train"


def _floor(tokens: float) -> int:
    return max(GRANULARITY, int(tokens) // GRANULARITY * GRANULARITY)


def _slopes(fitting: list[ProbeStep]) -> dict[int, float]:
    """MB per token on each GPU, from the two largest contexts that fit.

    Using the top pair, not the first one, keeps the estimate honest when the
    compute buffer grows with the context too (it does, per model)."""
    if len(fitting) < 2:
        return {}
    a, b = fitting[-2], fitting[-1]
    span = b.context - a.context
    if span <= 0:
        return {}
    return {
        gpu: max(0.0, (b.used_mb.get(gpu, 0) - a.used_mb.get(gpu, 0)) / span) for gpu in b.used_mb
    }


def next_context(
    steps: list[ProbeStep],
    limit_mb: dict[int, int],
    cap: int,
) -> int | None:
    """Next context to try, or ``None`` when the largest fitting one is found.

    * Until there are two fitting points, double the context.
    * Then jump to the context the per-GPU slope predicts would reach the
      limit on the first card to fill.
    * A step that does not fit becomes the upper bound; from there, bisect.
    """
    fitting = sorted((s for s in steps if s.fits), key=lambda s: s.context)
    failing = sorted(s.context for s in steps if not s.fits)
    lo = fitting[-1].context if fitting else 0
    hi = min(failing[0] if failing else cap + GRANULARITY, cap + GRANULARITY)
    if lo >= cap or hi - lo <= GRANULARITY:
        return None

    if len(fitting) < 2:
        target = lo * 2 if lo else GRANULARITY
    else:
        slopes = _slopes(fitting)
        last = fitting[-1]
        predictions = [
            last.context + (limit_mb[gpu] - last.used_mb.get(gpu, 0)) / slope
            for gpu, slope in slopes.items()
            if slope > 0 and gpu in limit_mb
        ]
        target = min(predictions) if predictions else lo * 2

    if failing:
        # Once something did not fit, never ask for more than the midpoint:
        # the prediction already failed once in this range.
        target = min(target, (lo + hi) / 2)
    target = min(_floor(target), cap)
    if target >= hi:
        target = _floor((lo + hi) / 2)
    if target <= lo:
        return None
    return target


async def run_probe(
    *,
    start_context: int,
    cap: int,
    limit_mb: dict[int, int],
    resize: Callable[[int], Awaitable[bool]],
    measure: Callable[[], Awaitable[dict[int, int]]],
    on_step: Callable[[ProbeStep], None] | None = None,
    max_steps: int = 24,
) -> ProbeResult:
    """Grow from ``start_context`` until the next step would cross a limit.

    ``resize(ctx)`` returns False when the worker could not take that context
    (the fork restores the previous one). ``measure()`` returns the worker's
    MB on each GPU. The worker is left at the largest context that fit.
    """
    steps: list[ProbeStep] = []

    def record(step: ProbeStep) -> None:
        steps.append(step)
        if on_step is not None:
            on_step(step)

    used = await measure()
    record(ProbeStep(start_context, used, _within(used, limit_mb)))
    if not steps[0].fits:
        raise ProbeError(
            f"the starting context ({start_context}) already exceeds the limit: "
            f"used {used} MB, limit {limit_mb} MB"
        )

    for _ in range(max_steps):
        target = next_context(steps, limit_mb, cap)
        if target is None:
            break
        if not await resize(target):
            record(
                ProbeStep(target, {}, False, "did not fit: llama.cpp restored the previous context")
            )
            continue
        used = await measure()
        fits = _within(used, limit_mb)
        record(ProbeStep(target, used, fits, "" if fits else "over the safety margin"))

    best = max((s for s in steps if s.fits), key=lambda s: s.context)
    current = steps[-1]
    if current.context != best.context or not current.fits:
        if not await resize(best.context):
            raise ProbeError(f"could not return to the largest fitting context {best.context}")
        best = ProbeStep(best.context, await measure(), True, "back to the largest fitting context")
        record(best)

    fitting = sorted((s for s in steps if s.fits), key=lambda s: s.context)
    slopes = _slopes(fitting)
    return ProbeResult(
        max_context=best.context,
        used_mb=dict(best.used_mb),
        limit_mb=dict(limit_mb),
        kv_mb_per_1k={gpu: round(slope * 1024, 2) for gpu, slope in slopes.items()},
        steps=steps,
        capped_by="train" if best.context >= cap else "memory",
    )


def _within(used: dict[int, int], limit_mb: dict[int, int]) -> bool:
    return all(used.get(gpu, 0) <= limit for gpu, limit in limit_mb.items())


class ProbeError(RuntimeError):
    """The probe could not produce a trustworthy maximum."""
