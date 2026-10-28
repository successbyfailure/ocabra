from __future__ import annotations

import asyncio

import pytest

from ocabra.core.fair_slot_gate import FairSlotGate, WorkerPoolFairGates


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
    backlog — only for whatever was already running/queued ahead of him at
    the moment he arrived, then the gate alternates. This is the exact
    scenario from the user's design request (2026-10-28): 'si un usuario
    tiene 10 peticiones en cola y otro empieza a mandar... se reparten los
    slots según van quedando libres', including explicitly for slots=1."""
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

    # Alice's 1st queued (her 2nd overall) was already ahead of Bob when he
    # arrived — it's reasonable for it to go first. But Bob must be served
    # on the round AFTER that, not after Alice's whole backlog.
    await release_and_record(alice_waiters[0], "alice")
    await release_and_record(bob_waiter, "bob")

    assert served_order == ["alice", "alice", "bob"]

    # From here on, only alice has pending work — subsequent releases keep
    # serving her without anyone to alternate with (nothing starves since
    # there's no one else waiting).
    await release_and_record(alice_waiters[1], "alice")
    assert served_order[-1] == "alice"

    # Cleanup: drain remaining waiters so the test doesn't leak tasks.
    for fut in alice_waiters[2:]:
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
        done, _ = await asyncio.wait(
            [*alice_more, *bob_more], return_when=asyncio.FIRST_COMPLETED
        )
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
            assert order[i - 1] not in order[i:], f"same user served twice in a row too early: {order}"


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
