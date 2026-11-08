import asyncio
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from inference_load_lab.benchmark import (
    BenchmarkConfig,
    load_prompts,
    percentile,
    run_suite,
    summarize,
)


def stream(tokens=3):
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "local answer"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    if tokens is not None:
        events.append({"choices": [], "usage": {"completion_tokens": tokens}})
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def read_records(output_dir):
    return [
        json.loads(line)
        for line in (output_dir / "requests.jsonl").read_text().splitlines()
    ]


@pytest.mark.asyncio
async def test_bounded_workers_send_exact_request_count_and_exclude_warmup(
    tmp_path, capsys
):
    active = peak = calls = 0

    async def respond(request):
        nonlocal active, peak, calls
        active += 1
        calls += 1
        peak = max(peak, active)
        await asyncio.sleep(0.002)
        active -= 1
        return httpx.Response(200, text=stream(tokens=3))

    config = BenchmarkConfig(
        "https://example.test", requests=7, concurrencies=(3,), warmup=2
    )
    output = tmp_path / "run"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        report = await run_suite(client, config, "test-key", output)

    assert calls == 9
    assert peak == 3
    records = read_records(output)
    assert Counter(r["phase"] for r in records) == {"warmup": 2, "measured": 7}
    assert sorted(r["request_id"] for r in records if r["phase"] == "measured") == list(
        range(7)
    )
    run = report["runs"][0]
    assert run["completed_requests"] == run["successful_requests"] == 7
    assert run["total_output_tokens"] == 21
    assert run["output_tokens_per_second"] == pytest.approx(21 / run["duration_s"])
    assert not capsys.readouterr().out


@pytest.mark.asyncio
async def test_sweep_replays_same_seeded_prompt_sequence_and_saves_no_api_key(tmp_path):
    config = BenchmarkConfig(
        "https://example.test",
        requests=8,
        concurrencies=(1, 3),
        warmup=3,
        prompts=("prompt A", "prompt B", "prompt C"),
        seed=42,
    )
    output = tmp_path / "run"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=stream())
        )
    ) as client:
        report = await run_suite(
            client,
            config,
            "private-test-key",
            output,
            server_config={"environment": {"VLLM_API_KEY": "private-test-key"}},
        )

    assert report["status"] == "ok"
    records = read_records(output)
    schedules = [
        [
            r["prompt_index"]
            for r in sorted(records, key=lambda r: r["request_id"])
            if r["concurrency"] == concurrency and r["phase"] == "measured"
        ]
        for concurrency in config.concurrencies
    ]
    assert schedules[0] == schedules[1]
    assert set(schedules[0]) == {0, 1, 2}
    settings = json.loads((output / "config.json").read_text())
    assert settings["seed"] == 42
    assert settings["prompts"] == list(config.prompts)
    for path in output.iterdir():
        assert "private-test-key" not in path.read_text()


@pytest.mark.asyncio
async def test_failed_requests_are_retained_and_excluded_from_success_metrics(tmp_path):
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            return httpx.Response(503, text="Unavailable: private-test-key")
        return httpx.Response(200, text=stream(tokens=4))

    output = tmp_path / "run"
    config = BenchmarkConfig(
        "https://example.test", requests=4, concurrencies=(2,), warmup=0
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        report = await run_suite(client, config, "private-test-key", output)

    run = report["runs"][0]
    assert report["status"] == "failed"
    assert run["successful_requests"] == 3
    assert run["failed_requests"] == 1
    assert run["ttft_ms"]["samples"] == 3
    assert run["total_output_tokens"] == 12
    assert run["output_tokens_per_second"] == pytest.approx(12 / run["duration_s"])
    errors = [r for r in read_records(output) if r["status"] == "error"]
    assert len(errors) == 1
    assert errors[0]["e2e_ms"] >= 0
    assert "HTTP 503" in errors[0]["error"]["message"]
    assert "private-test-key" not in (output / "requests.jsonl").read_text()


@pytest.mark.asyncio
async def test_missing_usage_is_visible_instead_of_reporting_incomplete_throughput(
    tmp_path,
):
    output = tmp_path / "run"
    config = BenchmarkConfig(
        "https://example.test", requests=1, concurrencies=(1,), warmup=0
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=stream(tokens=None))
        )
    ) as client:
        report = await run_suite(client, config, "test-key", output)
    run = report["runs"][0]
    assert run["successful_requests"] == 1
    assert run["missing_token_usage"] == 1
    assert run["total_output_tokens"] is None
    assert run["output_tokens_per_second"] is None


@pytest.mark.asyncio
async def test_failed_warmup_stops_before_measured_load(tmp_path):
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="Invalid key")

    output = tmp_path / "run"
    config = BenchmarkConfig(
        "https://example.test", requests=100, concurrencies=(1, 2), warmup=1
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        report = await run_suite(client, config, "test-key", output)
    assert calls == 1
    assert report["status"] == "failed"
    assert report["runs"][0]["status"] == "warmup_failed"
    assert report["runs"][0]["completed_requests"] == 0
    assert [r["phase"] for r in read_records(output)] == ["warmup"]


@pytest.mark.asyncio
async def test_total_request_deadline_records_timeout(tmp_path):
    async def respond(request):
        await asyncio.sleep(1)
        return httpx.Response(200, text=stream())

    output = tmp_path / "run"
    config = BenchmarkConfig(
        "https://example.test",
        requests=1,
        concurrencies=(1,),
        warmup=0,
        timeout=0.01,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        report = await run_suite(client, config, "test-key", output)
    assert report["runs"][0]["failed_requests"] == 1
    assert read_records(output)[0]["error"]["type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_existing_results_are_not_overwritten(tmp_path):
    marker = tmp_path / "existing.txt"
    marker.write_text("keep me")
    async with httpx.AsyncClient() as client:
        with pytest.raises(FileExistsError):
            await run_suite(
                client, BenchmarkConfig("https://example.test"), "test-key", tmp_path
            )
    assert marker.read_text() == "keep me"


@pytest.mark.asyncio
async def test_interrupted_run_preserves_completed_requests(tmp_path):
    second_started = asyncio.Event()
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            second_started.set()
            await asyncio.Event().wait()
        return httpx.Response(200, text=stream())

    output = tmp_path / "run"
    config = BenchmarkConfig(
        "https://example.test",
        requests=3,
        concurrencies=(1,),
        warmup=0,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(run_suite(client, config, "test-key", output))
        await asyncio.wait_for(second_started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "interrupted"
    records = read_records(output)
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert records[0]["request_id"] == 0


def test_percentiles_and_run_throughput_use_correct_denominators():
    assert percentile([], 0.95) is None
    assert percentile([7], 0.95) == 7
    assert percentile([100, 200, 300, 400, 500], 0.5) == 300
    assert percentile([100, 200, 300, 400, 500], 0.95) == 480
    rows = [
        {"status": "ok", "e2e_ms": 100, "ttft_ms": 20, "output_tokens": 10},
        {"status": "ok", "e2e_ms": 300, "ttft_ms": 40, "output_tokens": 30},
        {"status": "error", "e2e_ms": 1000, "ttft_ms": None, "output_tokens": None},
    ]
    result = summarize(rows, duration=2)
    assert result["output_tokens_per_second"] == 20
    assert result["successful_requests_per_second"] == 1
    assert result["e2e_ms"] == {"p50": 200, "p95": 290}


def test_prompt_file_supports_strings_and_prompt_objects(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('"one"\n\n{"prompt":"two"}\n')
    assert load_prompts(path) == ("one", "two")
    path.write_text('"valid"\n{"bad": true}\n')
    with pytest.raises(ValueError, match=r":2: expected"):
        load_prompts(path)


@pytest.mark.parametrize(
    "settings",
    [
        {"requests": 0},
        {"concurrencies": (0,)},
        {"concurrencies": (1, 1)},
        {"warmup": -1},
        {"timeout": float("nan")},
        {"prompts": ("",)},
    ],
)
def test_invalid_workloads_are_rejected(settings):
    with pytest.raises(ValueError):
        BenchmarkConfig("https://example.test", **settings)


def test_cli_runs_against_local_streaming_server(tmp_path):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            body = stream(tokens=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "cli-run"
    env = os.environ.copy()
    env["VLLM_API_KEY"] = "local-test-key"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from inference_load_lab import main; main()",
                "bench",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--requests",
                "4",
                "--concurrency",
                "2",
                "--warmup",
                "2",
                "--output-dir",
                str(output),
            ],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 6
    assert "OK=4 failed=0" in result.stdout
    assert "local answer" not in result.stdout
    assert "local-test-key" not in result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["runs"][0]["total_output_tokens"] == 8
