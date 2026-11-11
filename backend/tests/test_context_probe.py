"""Search for the largest context a llama.cpp preset fits (ocabra.core.context_probe)."""

import pytest

from ocabra.core.context_probe import (
    GRANULARITY,
    ProbeError,
    ProbeStep,
    next_context,
    run_probe,
)


class FakeWorker:
    """A worker whose VRAM per GPU follows its context.

    ``kv`` is MB per 1k tokens per GPU; ``growth`` adds a compute buffer that
    grows with the context (MB per 1k tokens, squared-ish), which makes a
    straight-line prediction overshoot. ``physical`` is where llama.cpp itself
    fails to allocate and restores the previous context.
    """

    def __init__(self, base, kv, *, growth=None, physical=None, start=32768):
        self.base, self.kv, self.growth = base, kv, growth or {}
        self.physical = physical or {}
        self.ctx = start
        self.resizes: list[int] = []

    def usage(self, ctx):
        k = ctx / 1024
        return {
            g: int(self.base[g] + self.kv[g] * k + self.growth.get(g, 0) * k * k / 256)
            for g in self.base
        }

    async def resize(self, ctx):
        self.resizes.append(ctx)
        if any(self.usage(ctx)[g] > cap for g, cap in self.physical.items()):
            return False
        self.ctx = ctx
        return True

    async def measure(self):
        return self.usage(self.ctx)


def true_max(worker, limit, cap):
    best = 0
    for ctx in range(GRANULARITY, cap + 1, GRANULARITY):
        if all(worker.usage(ctx)[g] <= limit[g] for g in limit):
            best = ctx
    return best


@pytest.mark.asyncio
async def test_single_card_linear_kv_lands_within_one_step_of_the_real_maximum():
    # qwen3.8-like on the 3090: 16 GB of weights, ~17 MB of KV per 1k tokens.
    worker = FakeWorker({1: 16000}, {1: 17.0})
    limit = {1: 23600}

    result = await run_probe(
        start_context=32768,
        cap=262144,
        limit_mb=limit,
        resize=worker.resize,
        measure=worker.measure,
    )

    assert true_max(worker, limit, 262144) - result.max_context <= GRANULARITY
    assert worker.usage(result.max_context)[1] <= limit[1]
    assert worker.ctx == result.max_context
    assert len(result.steps) <= 8
    assert result.kv_mb_per_1k[1] == pytest.approx(17.0, abs=0.1)


@pytest.mark.asyncio
async def test_two_cards_stop_at_the_first_card_to_fill():
    # tensor split: the 3060 fills first although the 3090 still has room.
    worker = FakeWorker({0: 6000, 1: 15000}, {0: 9.0, 1: 22.0})
    limit = {0: 11000, 1: 23600}

    result = await run_probe(
        start_context=32768,
        cap=1048576,
        limit_mb=limit,
        resize=worker.resize,
        measure=worker.measure,
    )

    expected = true_max(worker, limit, 1048576)
    assert expected - result.max_context <= GRANULARITY
    assert all(worker.usage(result.max_context)[g] <= limit[g] for g in limit)


@pytest.mark.asyncio
async def test_a_compute_buffer_that_grows_with_context_never_ends_over_the_limit():
    worker = FakeWorker({1: 14000}, {1: 12.0}, growth={1: 6.0})
    limit = {1: 23600}

    result = await run_probe(
        start_context=16384,
        cap=524288,
        limit_mb=limit,
        resize=worker.resize,
        measure=worker.measure,
    )

    assert worker.usage(result.max_context)[1] <= limit[1]
    assert true_max(worker, limit, 524288) - result.max_context <= GRANULARITY
    assert any(not s.fits for s in result.steps)  # the straight line overshot once


@pytest.mark.asyncio
async def test_the_trained_context_caps_the_search():
    worker = FakeWorker({1: 8000}, {1: 4.0})
    result = await run_probe(
        start_context=32768,
        cap=131072,
        limit_mb={1: 23600},
        resize=worker.resize,
        measure=worker.measure,
    )
    assert result.max_context == 131072
    assert result.capped_by == "train"


@pytest.mark.asyncio
async def test_a_size_llama_cpp_cannot_allocate_is_treated_as_too_big():
    # llama.cpp refuses before our margin is reached (fragmentation, etc.).
    worker = FakeWorker({1: 16000}, {1: 17.0}, physical={1: 19000})
    limit = {1: 23600}

    result = await run_probe(
        start_context=32768,
        cap=262144,
        limit_mb=limit,
        resize=worker.resize,
        measure=worker.measure,
    )

    assert worker.usage(result.max_context)[1] <= 19000
    assert worker.ctx == result.max_context
    assert any("did not fit" in s.note for s in result.steps)


@pytest.mark.asyncio
async def test_a_start_already_over_the_limit_is_an_error():
    worker = FakeWorker({1: 23500}, {1: 17.0})
    with pytest.raises(ProbeError, match="starting context"):
        await run_probe(
            start_context=32768,
            cap=262144,
            limit_mb={1: 23600},
            resize=worker.resize,
            measure=worker.measure,
        )


def test_next_context_doubles_until_it_has_a_slope_then_predicts():
    first = [ProbeStep(32768, {1: 16544}, True)]
    assert next_context(first, {1: 23600}, 262144) == 65536
    two = first + [ProbeStep(65536, {1: 17088}, True)]
    # 17 MB per 1k tokens: room for (23600-17088)/17*1024 more ≈ 392k → capped
    assert next_context(two, {1: 23600}, 262144) == 262144
