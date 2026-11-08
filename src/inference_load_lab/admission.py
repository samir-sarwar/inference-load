"""Bounded FIFO and length-aware admission for a single asyncio process."""

import asyncio
import math
from dataclasses import dataclass
from time import perf_counter


class QueueFull(RuntimeError):
    pass


class QueueTimedOut(RuntimeError):
    pass


class AdmissionClosed(RuntimeError):
    pass


@dataclass
class Lease:
    gate: "AdmissionQueue"
    queue_wait_ms: float
    released: bool = False
    arrival_order: int = 0
    admission_order: int = 0
    aged_promotion: bool = False

    async def release(self):
        async with self.gate.condition:
            if not self.released:
                self.released = True
                self.gate.active -= 1
                self.gate.condition.notify_all()


@dataclass(eq=False)
class Ticket:
    arrival_order: int
    queued_at: float
    estimated_cost: float


class AdmissionQueue:
    def __init__(
        self, concurrency: int, max_queue: int, *, policy="fifo", aging_seconds=10.0
    ):
        if concurrency < 1 or max_queue < 0:
            raise ValueError("Concurrency must be positive and queue size nonnegative.")
        if policy not in {"fifo", "length-aware"}:
            raise ValueError("Unknown admission policy.")
        if not math.isfinite(aging_seconds) or aging_seconds <= 0:
            raise ValueError("Aging threshold must be finite and positive.")
        self.concurrency = concurrency
        self.max_queue = max_queue
        self.policy = policy
        self.aging_seconds = aging_seconds
        self.active = 0
        self.waiters: list[Ticket] = []
        self.condition = asyncio.Condition()
        self.closed = False
        self.arrivals = 0
        self.admissions = 0

    def _next_ticket(self, now):
        if not self.waiters:
            return None
        if self.policy == "fifo":
            return self.waiters[0]
        aged = [t for t in self.waiters if now - t.queued_at >= self.aging_seconds]
        if aged:
            return min(aged, key=lambda t: t.arrival_order)
        return min(self.waiters, key=lambda t: (t.estimated_cost, t.arrival_order))

    def _admit(self, ticket, started):
        self.active += 1
        self.admissions += 1
        now = perf_counter()
        return Lease(
            self,
            (now - started) * 1000,
            arrival_order=ticket.arrival_order,
            admission_order=self.admissions,
            aged_promotion=self.policy == "length-aware"
            and now - ticket.queued_at >= self.aging_seconds,
        )

    async def acquire(self, timeout: float, *, estimated_cost: float = 1) -> Lease:
        if (
            isinstance(estimated_cost, bool)
            or not math.isfinite(estimated_cost)
            or estimated_cost <= 0
        ):
            raise ValueError("Estimated request cost must be finite and positive.")
        started = perf_counter()
        async with self.condition:
            if self.closed:
                raise AdmissionClosed("The request manager is shutting down.")
            self.arrivals += 1
            ticket = Ticket(self.arrivals, perf_counter(), estimated_cost)
            if self.active < self.concurrency and not self.waiters:
                return self._admit(ticket, started)
            if len(self.waiters) >= self.max_queue:
                raise QueueFull("The request queue is full.")

            self.waiters.append(ticket)
            self.condition.notify_all()
            try:
                async with asyncio.timeout(timeout):
                    await self.condition.wait_for(
                        lambda: (
                            self.closed
                            or (
                                self.active < self.concurrency
                                and self._next_ticket(perf_counter()) is ticket
                            )
                        )
                    )
                if self.closed:
                    raise AdmissionClosed("The request manager is shutting down.")
                self.waiters.remove(ticket)
                self.condition.notify_all()
                return self._admit(ticket, started)
            except TimeoutError:
                raise QueueTimedOut(
                    "Timed out waiting for an active request slot."
                ) from None
            finally:
                if ticket in self.waiters:
                    self.waiters.remove(ticket)
                    self.condition.notify_all()

    async def snapshot(self) -> dict:
        async with self.condition:
            return {
                "policy": self.policy,
                "concurrency_limit": self.concurrency,
                "queue_capacity": self.max_queue,
                "active_requests": self.active,
                "queued_requests": len(self.waiters),
            }

    async def close(self):
        async with self.condition:
            self.closed = True
            self.condition.notify_all()


class FifoAdmission(AdmissionQueue):
    def __init__(self, concurrency, max_queue):
        super().__init__(concurrency, max_queue)


class LengthAwareAdmission(AdmissionQueue):
    def __init__(self, concurrency, max_queue, aging_seconds=10.0):
        super().__init__(
            concurrency, max_queue, policy="length-aware", aging_seconds=aging_seconds
        )
