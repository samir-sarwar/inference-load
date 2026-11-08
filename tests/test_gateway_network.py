import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import uvicorn

from inference_load_lab.gateway import GatewayConfig, create_app


@pytest.mark.parametrize(
    "mode_args",
    [
        ["--concurrency", "4"],
        ["--mode", "rate", "--request-rate", "100"],
        ["--mode", "burst", "--burst-size", "4"],
    ],
)
def test_benchmark_through_real_http_gateway_preserves_limits_and_queue_metrics(
    tmp_path,
    mode_args,
):
    active = peak = 0
    keys = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            nonlocal active, peak
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                keys.append(self.headers["Authorization"])
                active += 1
                peak = max(peak, active)
            first = b'data: {"choices":[{"delta":{"content":"network answer"}}]}\n\n'
            last = b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(first) + len(last)))
            self.end_headers()
            try:
                self.wfile.write(first)
                self.wfile.flush()
                time.sleep(0.03)
            finally:
                with lock:
                    active -= 1
            self.wfile.write(last)
            self.wfile.flush()

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    app = create_app(
        GatewayConfig(
            f"http://127.0.0.1:{upstream.server_port}", concurrency=1, max_queue=8
        ),
        "upstream-network-key",
        "frontend-network-key",
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="warning",
            timeout_graceful_shutdown=2,
        )
    )
    gateway_thread = threading.Thread(
        target=lambda: server.run(sockets=[listener]), daemon=True
    )
    gateway_thread.start()
    output = tmp_path / "queued-run"
    env = os.environ.copy()
    env["VLLM_API_KEY"] = "frontend-network-key"
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from inference_load_lab import main; main()",
                "bench",
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--requests",
                "4",
                *mode_args,
                "--warmup",
                "0",
                "--output-dir",
                str(output),
            ],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((output / "summary.json").read_text())["runs"][0]
        assert summary["successful_requests"] == 4
        assert summary["total_output_tokens"] == 8
        assert summary["queue_wait_ms"]["samples"] == 4
        assert summary["queue_wait_ms"]["p95"] > 1
        assert summary["request_manager_configs"][0]["concurrency_limit"] == 1
        assert peak == 1
        assert keys == ["Bearer upstream-network-key"] * 4
        for path in output.iterdir():
            assert "frontend-network-key" not in path.read_text()
            assert "upstream-network-key" not in path.read_text()
    finally:
        server.should_exit = True
        gateway_thread.join(timeout=5)
        listener.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)
    assert not gateway_thread.is_alive()
