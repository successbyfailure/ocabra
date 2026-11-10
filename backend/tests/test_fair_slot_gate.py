from __future__ import annotations

import asyncio

import pytest

from ocabra.core.fair_slot_gate import (
    FairSlotGate,
    KVCapacityError,
    WorkerPoolFairGates,
)


@pytest.mark.asyncio
async def test_single_user_within_slots_never_waits():
    gate = FairSlotGate(slots=2)
    await gate.acquire("alice")
    await gate.acquire("alice")  # second slot, still free — must not block
    await gate.release()
    await gate.release()


@pytest.mark.asyncio
async def test_new_user_is_interleaved_not_starved_behind_a_backlog():
    """Alice fires 10 requests against a 1-slot gate; while her first is
    still running, Bob sends one. Bob must NOT wait for Alice's entire
    backlog — he's served on the very next free slot, ahead of any more of
    Alice's queued backlog, because he hasn't been served at all while
    Alice already has. This is strict alternation regardless of burst size
    (2026-10-28 design request): 'las peticiones de multiples usuarios se
    vayan alternando en los slots disponibles, independientemente de que
    uno haya mandado una rafaga y tenga peticiones encoladas' — explicitly
    confirmed to apply even when Alice's queued request was technically
    queued before Bob arrived, and even for slots=1."""
    gate = FairSlotGate(slots=1)

    # Alice's first request gets the only slot immediately.
    await gate.acquire("alice")

    # Alice queues 9 more while her first is "running".
    alice_waiters = [asyncio.ensure_future(gate.acquire("alice")) for _ in range(9)]
    await asyncio.sleep(0)  # let them register in the queue

    # Bob arrives after Alice already has a backlog queued.
    bob_waiter = asyncio.ensure_future(gate.acquire("bob"))
    await asyncio.sleep(0)

    served_order = ["alice"]  # alice's first request, already granted above

    async def release_and_record(expected_next_owner_waiter, owner_label):
        await gate.release()
        await expected_next_owner_waiter
        served_order.append(owner_label)

    # Bob has never been served while Alice already has (her first request)
    # — Bob wins the very next free slot, cutting ahead of Alice's already-
    # queued backlog entirely.
    await release_and_record(bob_waiter, "bob")

    assert served_order == ["alice", "bob"]

    # From here on, only alice has pending work — subsequent releases keep
    # serving her without anyone to alternate with (nothing starves since
    # there's no one else waiting).
    await release_and_record(alice_waiters[0], "alice")
    assert served_order[-1] == "alice"

    # Cleanup: drain remaining waiters so the test doesn't leak tasks.
    for fut in alice_waiters[1:]:
        await gate.release()
        await fut
    await gate.release()


@pytest.mark.asyncio
async def test_two_users_alternate_once_both_have_backlogs():
    gate = FairSlotGate(slots=1)
    await gate.acquire("alice")  # immediate

    alice_more = [asyncio.ensure_future(gate.acquire("alice")) for _ in range(3)]
    bob_more = [asyncio.ensure_future(gate.acquire("bob")) for _ in range(3)]
    await asyncio.sleep(0)

    order: list[str] = []

    async def step(label_hint: str):
        await gate.release()
        done, _ = await asyncio.wait([*alice_more, *bob_more], return_when=asyncio.FIRST_COMPLETED)
        for fut in done:
            if fut in alice_more:
                order.append("alice")
                alice_more.remove(fut)
            else:
                order.append("bob")
                bob_more.remove(fut)

    for _ in range(6):
        await step("x")
    await gate.release()

    # Once both users have a backlog, service strictly alternates between
    # them — no two consecutive grants to the same user while the other
    # still has pending work.
    for i in range(1, len(order)):
        if order[i] == order[i - 1]:
            # Allowed only once the other user's queue has fully drained.
            assert order[i - 1] not in order[i:], (
                f"same user served twice in a row too early: {order}"
            )


@pytest.mark.asyncio
async def test_burst_from_one_user_does_not_outrank_a_never_served_newcomer():
    """2026-10-28 follow-up to the design discussion: with slots=2, Alice
    fires 3 requests (2 get free immediate slots, 1 queues) and Bob arrives
    right after with 1. Bob must be served ahead of Alice's lone queued
    straggler — Alice already consumed 2 free slots, Bob has had none —
    even though Alice's straggler was technically queued before Bob
    arrived. This is the case that a naive "stay in rotation only with >=2
    queued" rule gets wrong (it would let Alice's straggler go first);
    least-recently-served-first gets it right."""
    gate = FairSlotGate(slots=2)

    await gate.acquire("alice")  # immediate, slot 1/2
    await gate.acquire("alice")  # immediate, slot 2/2

    alice3 = asyncio.ensure_future(gate.acquire("alice"))  # queues, slots full
    await asyncio.sleep(0)
    bob1 = asyncio.ensure_future(gate.acquire("bob"))  # arrives after, also queues
    await asyncio.sleep(0)

    served_order: list[str] = []

    async def release_and_record(expected_next_owner_waiter, owner_label):
        await gate.release()
        await expected_next_owner_waiter
        served_order.append(owner_label)

    await release_and_record(bob1, "bob")
    assert served_order == ["bob"]

    await release_and_record(alice3, "alice")
    assert served_order == ["bob", "alice"]

    await gate.release()
    await gate.release()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_a_slot():
    gate = FairSlotGate(slots=1)
    await gate.acquire("alice")  # holds the only slot

    task = asyncio.ensure_future(gate.acquire("bob"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Alice releases; since bob's wait was cancelled and cleaned up, a
    # fresh acquire for a third user must be granted immediately, proving
    # no slot was leaked to the cancelled waiter.
    await gate.release()
    await gate.acquire("carol")  # must not hang


def test_worker_pool_fair_gates_resizes_on_slot_count_change():
    registry = WorkerPoolFairGates()
    gate_a = registry.get("worker-1", slots=2)
    assert gate_a.slots == 2
    gate_same = registry.get("worker-1", slots=2)
    assert gate_same is gate_a
    gate_resized = registry.get("worker-1", slots=4)
    assert gate_resized is not gate_a
    assert gate_resized.slots == 4


@pytest.mark.asyncio
async def test_kv_reservation_blocks_until_both_dimensions_fit():
    gate = FairSlotGate(slots=3, kv_capacity_tokens=100)
    await gate.acquire("alice", kv_tokens=70)

    bob = asyncio.create_task(gate.acquire("bob", kv_tokens=40))
    await asyncio.sleep(0)
    assert not bob.done()  # A physical slot is free, but only 30 KV tokens are.

    carol = asyncio.create_task(gate.acquire("carol", kv_tokens=20))
    await asyncio.sleep(0)
    assert carol.done()  # A fitting head may use otherwise-idle capacity.
    assert gate.snapshot()["kv_reserved_tokens"] == 90

    await gate.release(kv_tokens=20)
    assert not bob.done()
    await gate.release(kv_tokens=70)
    await bob
    assert gate.snapshot()["kv_reserved_tokens"] == 40
    await gate.release(kv_tokens=40)


@pytest.mark.asyncio
async def test_request_larger_than_kv_pool_is_rejected_not_queued():
    gate = FairSlotGate(slots=4, kv_capacity_tokens=96_000)

    with pytest.raises(KVCapacityError) as exc_info:
        await gate.acquire("alice", kv_tokens=100_000)

    assert exc_info.value.requested_tokens == 100_000
    assert exc_info.value.capacity_tokens == 96_000
    assert gate.snapshot()["in_flight"] == 0


@pytest.mark.asyncio
async def test_batch_kv_reservation_can_block_with_a_physical_slot_free():
    gate = FairSlotGate(slots=2, kv_capacity_tokens=100)
    await gate.acquire("batch", is_batch=True, kv_tokens=70)

    assert gate.snapshot()["slot_saturated"] is False
    assert gate.batch_blocks(40) is True
    assert gate.batch_blocks(20) is False

    await gate.release(is_batch=True, kv_tokens=70)
    assert gate.batch_blocks(40) is False


@pytest.mark.asyncio
async def test_cancelling_granted_weighted_slot_releases_kv_reservation():
    gate = FairSlotGate(slots=1, kv_capacity_tokens=100)
    entered = asyncio.Event()
    keep_running = asyncio.Event()

    async def run_request() -> None:
        async with gate.slot("alice", kv_tokens=80):
            entered.set()
            await keep_running.wait()

    task = asyncio.create_task(run_request())
    await entered.wait()
    assert gate.snapshot()["kv_reserved_tokens"] == 80

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert gate.snapshot()["in_flight"] == 0
    assert gate.snapshot()["kv_reserved_tokens"] == 0


@pytest.mark.asyncio
async def test_context_lease_release_is_idempotent():
    gate = FairSlotGate(slots=2, kv_capacity_tokens=100)
    admission = await gate.acquire("alice", kv_tokens=40)

    await gate.release(kv_tokens=40, lease_id=admission.lease_id)
    await gate.release(kv_tokens=40, lease_id=admission.lease_id)

    snapshot = gate.snapshot()
    assert snapshot["in_flight"] == 0
    assert snapshot["active_leases"] == 0
    assert snapshot["kv_reserved_tokens"] == 0


@pytest.mark.asyncio
async def test_finished_owner_task_is_reaped_before_next_admission():
    gate = FairSlotGate(slots=1, kv_capacity_tokens=100)

    async def abandon_after_acquire():
        return await gate.acquire("alice", kv_tokens=80, _track_owner=True)

    abandoned = await asyncio.create_task(abandon_after_acquire())
    assert gate.snapshot()["in_flight"] == 1

    replacement = await asyncio.wait_for(
        gate.acquire("bob", kv_tokens=20),
        timeout=0.5,
    )
    assert replacement.admission_concurrency == 1
    assert gate.snapshot()["kv_reserved_tokens"] == 20

    # A late close from the abandoned generator cannot decrement Bob's lease.
    await gate.release(kv_tokens=80, lease_id=abandoned.lease_id)
    assert gate.snapshot()["in_flight"] == 1
    await gate.release(kv_tokens=20, lease_id=replacement.lease_id)


def test_worker_pool_fair_gates_resizes_on_kv_capacity_change():
    registry = WorkerPoolFairGates()
    gate_a = registry.get("worker-1", slots=4, kv_capacity_tokens=96_000)
    gate_same = registry.get("worker-1", slots=4, kv_capacity_tokens=96_000)
    gate_resized = registry.get("worker-1", slots=4, kv_capacity_tokens=128_000)

    assert gate_same is gate_a
    assert gate_resized is not gate_a
    assert gate_resized.kv_capacity_tokens == 128_000


@pytest.mark.asyncio
async def test_worker_pool_fair_gates_defers_resize_while_active():
    """An active gate is never replaced (that would orphan its reservations);
    the lookup keeps it and the new sizes apply once it drains."""
    registry = WorkerPoolFairGates()
    gate = registry.get("worker-1", slots=4, kv_capacity_tokens=96_000)
    await gate.acquire("alice", kv_tokens=32_000)

    assert registry.get("worker-1", slots=2, kv_capacity_tokens=64_000) is gate
    assert gate.snapshot()["kv_reserved_tokens"] == 32_000

    await gate.release(kv_tokens=32_000)
    resized = registry.get("worker-1", slots=2, kv_capacity_tokens=64_000)
    assert resized is not gate
    assert (resized.slots, resized.kv_capacity_tokens) == (2, 64_000)


@pytest.mark.asyncio
async def test_large_prefill_reduces_effective_concurrency_without_resizing_slots():
    gate = FairSlotGate(
        slots=4,
        medium_prefill_tokens=32_000,
        medium_prefill_max_concurrency=3,
        large_prefill_tokens=64_000,
        large_prefill_max_concurrency=2,
    )
    release = asyncio.Event()
    entered: list[str] = []

    async def run(label: str, prompt_tokens: int) -> None:
        async with gate.slot(label, prompt_tokens=prompt_tokens) as admission:
            entered.append(label)
            assert admission.physical_slots == 4
            await release.wait()

    tasks = [
        asyncio.create_task(run("large", 80_000)),
        asyncio.create_task(run("small-1", 4_000)),
        asyncio.create_task(run("small-2", 4_000)),
    ]
    await asyncio.sleep(0.01)

    assert entered == ["large", "small-1"]
    assert gate.snapshot()["admission_limit"] == 2
    assert gate.snapshot()["compute_saturated"] is True

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_medium_prefill_allows_three_of_four_physical_slots():
    gate = FairSlotGate(
        slots=4,
        medium_prefill_tokens=32_000,
        medium_prefill_max_concurrency=3,
        large_prefill_tokens=64_000,
        large_prefill_max_concurrency=2,
    )
    release = asyncio.Event()
    entered: list[str] = []

    async def run(label: str) -> None:
        async with gate.slot(label, prompt_tokens=40_000):
            entered.append(label)
            await release.wait()

    tasks = [asyncio.create_task(run(f"request-{index}")) for index in range(4)]
    await asyncio.sleep(0.01)

    assert len(entered) == 3
    assert gate.snapshot()["admission_limit"] == 3

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_large_stream_prefill_blocks_small_until_first_chunk_then_allows_decode():
    gate = FairSlotGate(
        slots=4,
        medium_prefill_tokens=32_000,
        medium_prefill_max_concurrency=3,
        large_prefill_tokens=64_000,
        large_prefill_max_concurrency=2,
    )
    large = await gate.acquire(
        "large",
        prompt_tokens=80_000,
        _track_prefill=True,
    )
    small_task = asyncio.create_task(
        gate.acquire("small", prompt_tokens=4_000, _track_prefill=True)
    )
    await asyncio.sleep(0)
    assert not small_task.done()

    await gate.mark_prefill_complete(large.lease_id)
    small = await asyncio.wait_for(small_task, timeout=0.5)
    assert small.admission_concurrency == 2

    await gate.release(lease_id=small.lease_id)
    await gate.release(lease_id=large.lease_id)


@pytest.mark.asyncio
async def test_small_stream_prefills_can_still_run_in_parallel():
    gate = FairSlotGate(
        slots=4,
        medium_prefill_tokens=32_000,
        medium_prefill_max_concurrency=3,
    )
    first = await gate.acquire("first", prompt_tokens=4_000, _track_prefill=True)
    second = await asyncio.wait_for(
        gate.acquire("second", prompt_tokens=8_000, _track_prefill=True),
        timeout=0.5,
    )

    assert second.admission_concurrency == 2
    await gate.release(lease_id=second.lease_id)
    await gate.release(lease_id=first.lease_id)
