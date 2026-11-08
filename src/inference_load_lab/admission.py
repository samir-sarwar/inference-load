"""Bounded FIFO admission for a single asyncio process."""

import asyncio
from collections import deque
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
    gate: "FifoAdmission"
    queue_wait_ms: float
    released: bool = False

    async def release(self):
        async with self.gate.condition:
            if not self.released:
                self.released = True
                self.gate.active -= 1
                self.gate.condition.notify_all()


class FifoAdmission:
    def __init__(self, concurrency: int, max_queue: int):
        if concurrency < 1 or max_queue < 0:
            raise ValueError("Concurrency must be positive and queue size nonnegative.")
        self.concurrency = concurrency
        self.max_queue = max_queue
        self.active = 0
        self.waiters = deque()
        self.condition = asyncio.Condition()
        self.closed = False

    async def acquire(self, timeout: float) -> Lease:
        started = perf_counter()
        async with self.condition:
            if self.closed:
                raise AdmissionClosed("The request manager is shutting down.")
            if self.active < self.concurrency and not self.waiters:
                self.active += 1
                return Lease(self, (perf_counter() - started) * 1000)
            if len(self.waiters) >= self.max_queue:
                raise QueueFull("The request queue is full.")

            ticket = object()
            self.waiters.append(ticket)
            try:
                async with asyncio.timeout(timeout):
                    await self.condition.wait_for(
                        lambda: (
                            self.closed
                            or (
                                self.waiters[0] is ticket
                                and self.active < self.concurrency
                            )
                        )
                    )
                if self.closed:
                    raise AdmissionClosed("The request manager is shutting down.")
                self.waiters.popleft()
                self.active += 1
                self.condition.notify_all()
                return Lease(self, (perf_counter() - started) * 1000)
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
                "policy": "fifo",
                "concurrency_limit": self.concurrency,
                "queue_capacity": self.max_queue,
                "active_requests": self.active,
                "queued_requests": len(self.waiters),
            }

    async def close(self):
        async with self.condition:
            self.closed = True
            self.condition.notify_all()
