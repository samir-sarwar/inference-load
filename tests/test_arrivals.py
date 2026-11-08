import asyncio
import json

import httpx
import pytest

from inference_load_lab.arrivals import arrival_offsets
from inference_load_lab.benchmark import (
    BenchmarkConfig,
    run_arrival_batch,
    run_suite,
    summarize,
)

SSE = 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n'


def test_rate_and_burst_schedules_are_deterministic():
    assert arrival_offsets("rate", 4, request_rate=4) == [0, 0.25, 0.5, 0.75]
    assert arrival_offsets("burst", 5, burst_size=2, burst_interval=0.5) == [
        0,
        0,
        0.5,
        0.5,
        1,
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "rate"},
        {"mode": "rate", "request_rate": 0},
        {"mode": "rate", "request_rate": float("inf")},
        {"mode": "burst", "burst_size": 0},
        {"mode": "burst", "burst_interval": -1},
        {"mode": "burst", "max_in_flight": 0},
        {"mode": "concurrency", "request_rate": 2},
    ],
)
def test_invalid_arrival_settings_are_rejected(options):
    with pytest.raises(ValueError):
        BenchmarkConfig("https://example.test", **options)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "offsets"),
    [
        ({"mode": "rate", "request_rate": 100}, [0, 0.01, 0.02, 0.03]),
        (
            {"mode": "burst", "burst_size": 2, "burst_interval": 0.03},
            [0, 0, 0.03, 0.03],
        ),
    ],
)
async def test_arrivals_continue_while_every_earlier_response_is_blocked(
    tmp_path, settings, offsets
):
    release = asyncio.Event()
    all_started = asyncio.Event()
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        if calls == 4:
            all_started.set()
        await release.wait()
        return httpx.Response(200, text=SSE)

    config = BenchmarkConfig(
        "https://example.test",
        requests=4,
        warmup=0,
        concurrencies=(),
        max_in_flight=4,
        **settings,
    )
    output = tmp_path / "run"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(run_suite(client, config, "test-key", output))
        try:
            await asyncio.wait_for(all_started.wait(), timeout=2)
            assert calls == 4
            release.set()
            report = await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert report["status"] == "ok"
    summary = report["runs"][0]
    assert summary["peak_in_flight"] == 4
    assert summary["dispatch_lag_ms"]["samples"] == 4
    assert summary["arrival_e2e_ms"]["samples"] == 4
    assert summary["http_status_counts"] == {"200": 4}
    records = sorted(
        [
            json.loads(line)
            for line in (output / "requests.jsonl").read_text().splitlines()
        ],
        key=lambda row: row["request_id"],
    )
    assert [row["scheduled_offset_s"] for row in records] == pytest.approx(offsets)
    assert all(row["dispatch_lag_ms"] >= 0 for row in records)
    assert all(row["arrival_e2e_ms"] >= row["e2e_ms"] for row in records)
    assert json.loads((output / "config.json").read_text())["load_model"] == "open_loop"


@pytest.mark.asyncio
async def test_driver_cap_drops_excess_arrivals_without_waiting_or_sending_them():
    release = asyncio.Event()
    calls = 0
    records = []

    async def respond(request):
        nonlocal calls
        calls += 1
        await release.wait()
        return httpx.Response(200, text=SSE)

    config = BenchmarkConfig(
        "https://example.test",
        requests=6,
        warmup=0,
        mode="burst",
        burst_size=6,
        max_in_flight=2,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(
            run_arrival_batch(client, config, "test-key", [0] * 6, records)
        )
        try:

            async def wait_for_drops():
                while sum(row["status"] == "dropped" for row in records) < 4:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_drops(), timeout=2)
            assert calls == 2
            release.set()
            timing = await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    summary = summarize(records, timing["duration_s"])
    assert summary["recorded_requests"] == 6
    assert summary["completed_requests"] == summary["successful_requests"] == 2
    assert summary["failed_requests"] == 0
    assert summary["dropped_requests"] == 4
    assert summary["error_types"] == {"ClientCapacityExceeded": 4}
    assert timing["peak_in_flight"] == 2
    for record in records:
        if record["status"] == "dropped":
            assert record["started_offset_s"] is None
            assert record["e2e_ms"] is None
            assert record["http_status"] is None


@pytest.mark.asyncio
async def test_server_429_is_a_failure_separate_from_driver_drops(tmp_path):
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if calls % 2:
            return httpx.Response(
                429, json={"error": {"type": "queue_full", "message": "full"}}
            )
        return httpx.Response(200, text=SSE)

    config = BenchmarkConfig(
        "https://example.test", requests=4, warmup=0, mode="burst", burst_size=4
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        report = await run_suite(client, config, "test-key", tmp_path / "run")
    summary = report["runs"][0]
    assert report["status"] == "failed"
    assert summary["failed_requests"] == 2
    assert summary["dropped_requests"] == 0
    assert summary["http_status_counts"] == {"429": 2, "200": 2}
    assert summary["error_types"] == {"queue_full": 2}


@pytest.mark.asyncio
async def test_cancelling_during_future_arrivals_preserves_completed_records(tmp_path):
    first = asyncio.Event()

    def respond(request):
        first.set()
        return httpx.Response(200, text=SSE)

    config = BenchmarkConfig(
        "https://example.test", requests=3, warmup=0, mode="rate", request_rate=1
    )
    output = tmp_path / "run"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(run_suite(client, config, "test-key", output))
        await asyncio.wait_for(first.wait(), timeout=2)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert json.loads((output / "summary.json").read_text())["status"] == "interrupted"
    records = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1 and records[0]["status"] == "ok"
