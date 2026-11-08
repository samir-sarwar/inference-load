import asyncio

import pytest

from inference_load_lab.admission import (
    AdmissionClosed,
    FifoAdmission,
    LengthAwareAdmission,
    QueueFull,
    QueueTimedOut,
)


async def wait_for_queued(gate, count):
    async def wait():
        while (await gate.snapshot())["queued_requests"] != count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


@pytest.mark.asyncio
async def test_fifo_admission_and_new_arrivals_cannot_bypass_waiters():
    gate = FifoAdmission(concurrency=1, max_queue=3)
    first = await gate.acquire(1)
    second = asyncio.create_task(gate.acquire(1))
    await wait_for_queued(gate, 1)
    third = asyncio.create_task(gate.acquire(1))
    await wait_for_queued(gate, 2)
    await first.release()
    second_lease = await second
    fourth = asyncio.create_task(gate.acquire(1))
    await wait_for_queued(gate, 2)
    assert not third.done() and not fourth.done()
    await second_lease.release()
    third_lease = await third
    assert not fourth.done()
    await third_lease.release()
    fourth_lease = await fourth
    assert fourth_lease.queue_wait_ms >= 0
    await fourth_lease.release()
    assert (await gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [FifoAdmission, LengthAwareAdmission])
async def test_limit_and_queue_capacity_and_idempotent_release(factory):
    gate = factory(concurrency=2, max_queue=1)
    one, two = await gate.acquire(1), await gate.acquire(1)
    waiting = asyncio.create_task(gate.acquire(1))
    await wait_for_queued(gate, 1)
    with pytest.raises(QueueFull):
        await gate.acquire(1)
    await one.release()
    next_lease = await waiting
    assert (await gate.snapshot())["active_requests"] == 2
    await one.release()
    assert (await gate.snapshot())["active_requests"] == 2
    await two.release()
    await next_lease.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [FifoAdmission, LengthAwareAdmission])
async def test_timeout_and_cancellation_remove_waiting_tickets(factory):
    gate = factory(concurrency=1, max_queue=1)
    active = await gate.acquire(1)
    with pytest.raises(QueueTimedOut):
        await gate.acquire(0.01)
    assert (await gate.snapshot())["queued_requests"] == 0
    waiting = asyncio.create_task(gate.acquire(1))
    await wait_for_queued(gate, 1)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert (await gate.snapshot())["queued_requests"] == 0
    await active.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [FifoAdmission, LengthAwareAdmission])
async def test_closed_gate_wakes_waiters_and_rejects_new_work(factory):
    gate = factory(concurrency=1, max_queue=1)
    active = await gate.acquire(1)
    waiting = asyncio.create_task(gate.acquire(1))
    await wait_for_queued(gate, 1)
    await gate.close()
    with pytest.raises(AdmissionClosed):
        await waiting
    with pytest.raises(AdmissionClosed):
        await gate.acquire(1)
    assert (await gate.snapshot())["queued_requests"] == 0
    await active.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [FifoAdmission, LengthAwareAdmission])
async def test_zero_queue_capacity_rejects_excess_work_immediately(factory):
    gate = factory(concurrency=1, max_queue=0)
    active = await gate.acquire(1)
    with pytest.raises(QueueFull):
        await gate.acquire(1)
    await active.release()


@pytest.mark.asyncio
async def test_length_aware_selects_lowest_cost_and_breaks_ties_by_arrival():
    gate = LengthAwareAdmission(1, 4)
    active = await gate.acquire(1)
    long = asyncio.create_task(gate.acquire(1, estimated_cost=1000))
    await wait_for_queued(gate, 1)
    short = asyncio.create_task(gate.acquire(1, estimated_cost=10))
    await wait_for_queued(gate, 2)
    tied = asyncio.create_task(gate.acquire(1, estimated_cost=10))
    await wait_for_queued(gate, 3)
    await active.release()
    lease = await short
    assert lease.arrival_order == 3 and lease.admission_order == 2
    assert not long.done() and not tied.done()
    await lease.release()
    lease = await tied
    assert lease.arrival_order == 4
    await lease.release()
    lease = await long
    assert lease.arrival_order == 2
    await lease.release()
    assert (await gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_aged_long_request_cannot_be_bypassed_by_new_short_requests(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("inference_load_lab.admission.perf_counter", lambda: now[0])
    gate = LengthAwareAdmission(1, 4, aging_seconds=5)
    active = await gate.acquire(1)
    old = asyncio.create_task(gate.acquire(1, estimated_cost=1000))
    await wait_for_queued(gate, 1)
    now[0] = 6
    short = asyncio.create_task(gate.acquire(1, estimated_cost=1))
    await wait_for_queued(gate, 2)
    await active.release()
    lease = await old
    assert lease.aged_promotion and lease.queue_wait_ms == 6000
    assert not short.done()
    await lease.release()
    lease = await short
    assert not lease.aged_promotion
    await lease.release()
