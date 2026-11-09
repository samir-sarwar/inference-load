import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from inference_load_lab.analysis import analyze_experiment
from inference_load_lab.analysis import main as analyze_main
from inference_load_lab.experiment import route, trial_order


def test_trial_order_balances_positions():
    names = ["direct", "fifo-c4", "fifo-c16"]
    orders = [trial_order(names, i) for i in range(3)]
    for position in range(3):
        assert {order[position] for order in orders} == set(names)
    assert trial_order(names, 3) == list(reversed(names))


@pytest.mark.asyncio
async def test_cancellation_shuts_down_temporary_gateway(tmp_path):
    url = None
    with pytest.raises(asyncio.CancelledError):
        async with route(
            {"name": "fifo-c1", "concurrency": 1},
            {"queue_size": 4, "queue_timeout": 5, "workload": {"timeout": 5}},
            "http://127.0.0.1:9",
            "upstream-key",
            "frontend-key",
            tmp_path / "gateway.log",
        ) as url:
            raise asyncio.CancelledError
    async with httpx.AsyncClient(timeout=1) as client:
        with pytest.raises(httpx.RequestError):
            await client.get(url + "/health")


@pytest.fixture(scope="module")
def completed_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("controlled-experiment")
    prompts = root / "prompts.jsonl"
    prompts.write_text('"Explain caching."\n"Explain batching."\n')
    server_config = root / "server.json"
    server_config.write_text('{"model":"local-test","revision":"fixed-test-revision"}')
    spec = {
        "trials": 3,
        "baseline": "fifo-c1",
        "candidate": "fifo-c2",
        "variants": [
            {"name": "direct", "concurrency": None},
            {"name": "fifo-c1", "concurrency": 1},
            {"name": "fifo-c2", "concurrency": 2},
        ],
        "queue_size": 8,
        "queue_timeout": 5,
        "prompts_file": str(prompts),
        "server_config": str(server_config),
        "require_gpu_telemetry": False,
        "workload": {
            "model": "local-test",
            "requests": 4,
            "concurrencies": [2],
            "warmup": 1,
            "max_tokens": 3,
            "ignore_eos": True,
            "seed": 0,
            "timeout": 5,
        },
    }
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(spec))
    calls = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            value = (
                b"# HELP local_test Test metric\nlocal_test 1\n"
                if self.path == "/metrics"
                else b'{"data":[{"id":"local-test"}],"version":"local-test"}'
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.headers["Authorization"], body))
            first = b'data: {"choices":[{"delta":{"content":"local result"}}]}\n\n'
            last = (
                "data: "
                + json.dumps(
                    {"choices": [], "usage": {"completion_tokens": body["max_tokens"]}}
                )
                + "\n\ndata: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(first) + len(last)))
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            time.sleep(0.01)
            self.wfile.write(last)
            self.wfile.flush()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = root / "experiment"
    env = os.environ.copy()
    env.update(
        VLLM_API_KEY="private-experiment-key", GATEWAY_API_KEY="private-frontend-key"
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from inference_load_lab import main; main()",
                "experiment",
                "--config",
                str(spec_path),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--output-dir",
                str(output),
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        mixed = root / "mixed.jsonl"
        mixed.write_text(
            '{"prompt":"short","class":"short","max_tokens":3}\n{"prompt":"long","class":"long","max_tokens":6}\n'
        )
        scheduling = {
            **spec,
            "trials": 4,
            "target_class": "short",
            "candidate": "length-aware-c1",
            "variants": [
                {"name": "fifo-c1", "concurrency": 1},
                {
                    "name": "length-aware-c1",
                    "concurrency": 1,
                    "policy": "length-aware",
                    "aging_seconds": 1,
                },
            ],
            "prompts_file": str(mixed),
            "workload": {
                **spec["workload"],
                "mode": "burst",
                "requests": 8,
                "burst_size": 8,
                "max_in_flight": 16,
            },
        }
        scheduling_path = root / "scheduling.json"
        scheduling_path.write_text(json.dumps(scheduling))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from inference_load_lab import main; main()",
                "experiment",
                "--config",
                str(scheduling_path),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--output-dir",
                str(root / "scheduling"),
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert (
        len(calls) == 117
    )  # 9 * (1 + 4) original requests, plus 8 * (1 + 8) mixed-workload requests.
    assert all(
        key == "Bearer private-experiment-key" and body["ignore_eos"]
        for key, body in calls
    )
    metadata = json.loads((output / "experiment.json").read_text())
    assert metadata["status"] == "complete"
    assert len(metadata["runs"]) == 9
    for entry in metadata["runs"]:
        config = json.loads((output / entry["output_dir"] / "config.json").read_text())
        if entry["variant"] != "direct":
            with pytest.raises(httpx.RequestError):
                httpx.get(config["base_url"] + "/health", timeout=0.5)
    for path in output.rglob("*"):
        if path.is_file():
            assert "private-experiment-key" not in path.read_text()
            assert "private-frontend-key" not in path.read_text()
    return output


def test_full_experiment_can_be_analyzed_and_exports_raw_trial_data(
    completed_experiment,
):
    report = analyze_experiment(completed_experiment)
    assert report["eligible_comparison"] is True
    assert len(report["paired_changes"]) == 3
    assert set(report["variants"]) == {"direct", "fifo-c1", "fifo-c2"}
    analyze_main([str(completed_experiment), "--no-plots"])
    assert (
        len((completed_experiment / "analysis/trials.csv").read_text().splitlines())
        == 10
    )


def test_mixed_burst_experiment_compares_policies_and_reports_target_and_tradeoffs(
    completed_experiment,
):
    path = completed_experiment.parent / "scheduling"
    report = analyze_experiment(path)
    assert report["eligible_comparison"] is True
    assert report["target_class"] == "short"
    assert len(report["paired_changes"]) == 4
    assert "target_e2e_p95_ms_reduction_percent" in report["paired_changes"][0]
    assert "long_e2e_p95_increase_percent" in report["paired_changes"][0]
    assert all(
        row["short_successful_requests"] == row["long_successful_requests"] == 4
        for row in report["runs"]
    )
    assert report["variants"]["length-aware-c1"]["long_e2e_p95_ms"] is not None


def test_changed_workload_cannot_be_compared(completed_experiment, tmp_path):
    output = tmp_path / "experiment"
    shutil.copytree(completed_experiment, output)
    path = output / "runs/trial-01/fifo-c1/config.json"
    config = json.loads(path.read_text())
    config["max_tokens"] += 1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="changed between runs"):
        analyze_experiment(output)


@pytest.mark.parametrize("cause", ["drops", "tokens", "gateway", "gpu", "truncated"])
def test_invalid_comparison_never_claims_improvement(
    completed_experiment, tmp_path, cause
):
    output = tmp_path / "experiment"
    shutil.copytree(completed_experiment, output)
    path = output / "runs/trial-01/fifo-c2/summary.json"
    data = json.loads(path.read_text())
    row = data["runs"][0]
    if cause == "drops":
        row["dropped_requests"] = 1
    elif cause == "gateway":
        row["request_manager_configs"][0]["concurrency_limit"] = 999
    elif cause == "tokens":
        requests = path.parent / "requests.jsonl"
        rows = [json.loads(line) for line in requests.read_text().splitlines()]
        next(r for r in rows if r["phase"] == "measured")["output_tokens"] = 2
        requests.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif cause == "gpu":
        metadata_path = output / "experiment.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["spec"]["require_gpu_telemetry"] = True
        metadata_path.write_text(json.dumps(metadata))
    else:
        for requests in output.glob("runs/*/*/requests.jsonl"):
            rows = [json.loads(line) for line in requests.read_text().splitlines()]
            rows = [r for r in rows if r["phase"] != "measured" or r["request_id"] != 3]
            requests.write_text("".join(json.dumps(r) + "\n" for r in rows))
    path.write_text(json.dumps(data))
    report = analyze_experiment(output)
    assert report["eligible_comparison"] is False
    assert report["consistent_improvement_observed"] is False
    assert report["paired_changes"] == []


def test_missing_trial_is_refused(completed_experiment, tmp_path):
    output = tmp_path / "experiment"
    shutil.copytree(completed_experiment, output)
    path = output / "experiment.json"
    data = json.loads(path.read_text())
    data["runs"].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing or duplicate"):
        analyze_experiment(output)
