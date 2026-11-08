"""Deterministic arrival schedules independent of response completion."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from time import perf_counter


def arrival_offsets(
    mode: str,
    requests: int,
    *,
    request_rate: float | None = None,
    burst_size: int = 16,
    burst_interval: float = 1.0,
) -> list[float]:
    if requests < 1:
        raise ValueError("The request count must be positive.")
    if mode == "rate":
        if request_rate is None or not math.isfinite(request_rate) or request_rate <= 0:
            raise ValueError("Rate mode requires a finite positive --request-rate.")
        return [i / request_rate for i in range(requests)]
    if mode == "burst":
        if burst_size < 1 or not math.isfinite(burst_interval) or burst_interval <= 0:
            raise ValueError("Burst size and interval must be positive.")
        return [(i // burst_size) * burst_interval for i in range(requests)]
    raise ValueError("Arrival schedules require rate or burst mode.")


async def dispatch_arrivals(
    offsets: list[float],
    max_in_flight: int,
    send: Callable[[int, float, float], Awaitable[None]],
    drop: Callable[[int, float, float], None],
) -> dict:
    """Drop excess arrivals rather than waiting for a driver-side request slot."""
    if max_in_flight < 1:
        raise ValueError("--max-in-flight must be positive.")
    started = perf_counter()
    active = set()
    peak = 0
    async with asyncio.TaskGroup() as group:
        for request_id, offset in enumerate(offsets):
            delay = started + offset - perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            active.difference_update([task for task in active if task.done()])
            if len(active) >= max_in_flight:
                drop(request_id, offset, started)
            else:
                task = group.create_task(send(request_id, offset, started))
                active.add(task)
                task.add_done_callback(active.discard)
                peak = max(peak, len(active))
            # Give I/O and completed-task callbacks a turn, including within a burst.
            await asyncio.sleep(0)
    return {
        "duration_s": perf_counter() - started,
        "arrival_span_s": offsets[-1] if offsets else 0.0,
        "peak_in_flight": peak,
    }
