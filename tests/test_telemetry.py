import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from inference_load_lab.benchmark import BenchmarkConfig, run_suite
from inference_load_lab.telemetry import RunpodMonitor, percentage


def response(utilization=75):
    return {
        "data": {
            "pod": {
                "id": "pod-test",
                "runtime": {"uptimeInSeconds": 42},
                "latestTelemetry": {
                    "time": datetime.now(UTC).isoformat(),
                    "individualGpuMetrics": [
                        {
                            "id": "GPU-test",
                            "percentUtilization": utilization,
                            "memoryUtilization": 30,
                        }
                    ],
                },
            }
        }
    }


@pytest.mark.asyncio
async def test_benchmark_samples_only_measured_load_and_keeps_credentials_private(
    tmp_path,
):
    def gpu_response(request):
        assert request.headers["Authorization"] == "Bearer private-runpod-key"
        assert "private-runpod-key" not in str(request.url)
        assert json.loads(request.content)["variables"] == {"podId": "pod-test"}
        return httpx.Response(200, json=response())

    async def inference_response(request):
        await asyncio.sleep(0.025)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":3}}\n\ndata: [DONE]\n\n',
        )

    directory = tmp_path / "run"
    config = BenchmarkConfig(
        "https://example.test", requests=4, warmup=2, concurrencies=(2,)
    )
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(gpu_response)) as gpu,
        httpx.AsyncClient(
            transport=httpx.MockTransport(inference_response)
        ) as inference,
    ):
        monitor = RunpodMonitor(
            gpu, "private-runpod-key", "pod-test", directory / "gpu.jsonl", 0.01
        )
        report = await run_suite(
            inference, config, "private-inference-key", directory, gpu_monitor=monitor
        )
    summary = report["runs"][0]
    assert summary["gpu"]["status"] == "ok"
    assert summary["gpu"]["successful_polls"] >= 2
    assert summary["gpu"]["unique_samples"] >= 2
    assert summary["gpu"]["gpus"][0]["mean_util_percent"] == 75
    assert summary["tpot_ms"]["samples"] == 4
    rows = [
        json.loads(line) for line in (directory / "gpu.jsonl").read_text().splitlines()
    ]
    assert all(r["phase"] == "measured" and r["concurrency"] == 2 for r in rows)
    for path in directory.iterdir():
        assert "private-runpod-key" not in path.read_text()
        assert "private-inference-key" not in path.read_text()


@pytest.mark.asyncio
async def test_sample_from_after_the_window_is_excluded_from_utilization(tmp_path):
    entered = asyncio.Event()

    async def slow(request):
        entered.set()
        await asyncio.sleep(0.03)
        return httpx.Response(200, json=response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        monitor = RunpodMonitor(client, "private", "pod-test", tmp_path / "gpu.jsonl")
        async with monitor.measure(1) as observation:
            await entered.wait()
        summary = observation.summary()
    assert summary["status"] == "unavailable"
    assert summary["outside_window_polls_excluded"] == 1
    assert summary["gpus"] == []


@pytest.mark.asyncio
async def test_late_response_can_contain_a_valid_sample_from_inside_the_window(
    tmp_path,
):
    entered = asyncio.Event()

    async def slow(request):
        body = response()
        entered.set()
        await asyncio.sleep(0.03)
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        monitor = RunpodMonitor(client, "private", "pod-test", tmp_path / "gpu.jsonl")
        async with monitor.measure(1) as observation:
            await entered.wait()
    assert observation.summary()["unique_samples"] == 1
    assert observation.records[0]["sample_finished_offset_s"] > observation.duration


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [False, True])
async def test_cached_gpu_samples_are_deduplicated_and_stale_samples_excluded(
    tmp_path, stale
):
    enough = asyncio.Event()
    frozen = None
    calls = 0

    def cached(request):
        nonlocal frozen, calls
        if frozen is None:
            frozen = response()
            if stale:
                frozen["data"]["pod"]["latestTelemetry"]["time"] = (
                    "2000-01-01T00:00:00Z"
                )
        calls += 1
        if calls == 3:
            enough.set()
        return httpx.Response(200, json=frozen)

    async with httpx.AsyncClient(transport=httpx.MockTransport(cached)) as client:
        monitor = RunpodMonitor(
            client, "private", "pod-test", tmp_path / "gpu.jsonl", 0.001
        )
        async with monitor.measure(1) as observation:
            await asyncio.wait_for(enough.wait(), timeout=2)
            await asyncio.sleep(0)
    summary = observation.summary()
    assert summary["successful_polls"] >= 3
    if stale:
        assert summary["unique_samples"] == 0
        assert summary["status"] == "unavailable"
    else:
        assert summary["unique_samples"] == 1
        assert summary["duplicate_polls_excluded"] >= 2


@pytest.mark.asyncio
async def test_missing_gpu_samples_are_errors_instead_of_zero_utilization(tmp_path):
    called = asyncio.Event()

    def missing(request):
        called.set()
        return httpx.Response(
            200, json={"errors": [{"message": "private account error"}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(missing)) as client:
        monitor = RunpodMonitor(client, "private", "pod-test", tmp_path / "gpu.jsonl")
        async with monitor.measure(1) as observation:
            await called.wait()
            await asyncio.sleep(0)
    summary = observation.summary()
    assert summary["status"] == "unavailable"
    assert summary["failed_polls"] == 1
    assert "private account error" not in (tmp_path / "gpu.jsonl").read_text()


@pytest.mark.parametrize("value", [-1, 101, float("nan"), float("inf"), True, None])
def test_invalid_gpu_percentages_are_rejected(value):
    with pytest.raises(ValueError):
        percentage(value)
