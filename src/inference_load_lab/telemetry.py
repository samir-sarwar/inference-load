"""Sample timestamped GPU telemetry outside the inference connection pool."""

import asyncio
import json
import math
import os
import statistics
import tomllib
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

ENDPOINT = "https://api.runpod.io/graphql"
QUERY = """query($podId: String!) {
  pod(input: {podId: $podId}) {
    id runtime { uptimeInSeconds }
    latestTelemetry { time individualGpuMetrics { id percentUtilization memoryUtilization } }
  }
}"""


def load_runpod_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        path = Path.home() / ".runpod" / "config.toml"
        if path.is_file():
            key = tomllib.loads(path.read_text()).get("apikey")
    if not isinstance(key, str) or not key:
        raise ValueError(
            "GPU sampling requires RUNPOD_API_KEY or runpodctl's ~/.runpod/config.toml."
        )
    return key


def percentage(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 100
    ):
        raise ValueError("Invalid GPU utilization percentage.")
    return value


class Observation:
    def __init__(self, monitor, concurrency):
        self.monitor = monitor
        self.concurrency = concurrency
        self.started = perf_counter()
        self.started_wall = datetime.now(UTC)
        self.duration = None
        self.records = []

    async def sample(self):
        row = {
            "pod_id": self.monitor.pod_id,
            "concurrency": self.concurrency,
            "phase": "measured",
            "sample_started_offset_s": perf_counter() - self.started,
            "observed_at": datetime.now(UTC).isoformat(),
        }
        try:
            response = await self.monitor.client.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {self.monitor.api_key}"},
                json={"query": QUERY, "variables": {"podId": self.monitor.pod_id}},
                timeout=10,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("errors"):
                raise ValueError("Runpod rejected the telemetry query.")
            pod = body["data"]["pod"]
            telemetry = pod["latestTelemetry"]
            if (
                pod["id"] != self.monitor.pod_id
                or not telemetry
                or not telemetry["individualGpuMetrics"]
            ):
                raise ValueError("No GPU telemetry is available for this pod.")
            provider_time = datetime.fromisoformat(telemetry["time"])
            if provider_time.tzinfo is None:
                raise ValueError(
                    "Provider telemetry timestamp must include a timezone."
                )
            provider_time = provider_time.astimezone(UTC)
            row.update(
                status="ok",
                provider_uptime_s=(pod.get("runtime") or {}).get("uptimeInSeconds"),
                provider_sample_at=provider_time.isoformat(),
                provider_sample_offset_s=(
                    provider_time - self.started_wall
                ).total_seconds(),
                provider_sample_age_s=(
                    datetime.now(UTC) - provider_time
                ).total_seconds(),
                gpus=[
                    {
                        "id": gpu["id"],
                        "gpu_util_percent": percentage(gpu["percentUtilization"]),
                        "memory_util_percent": percentage(gpu["memoryUtilization"]),
                    }
                    for gpu in telemetry["individualGpuMetrics"]
                ],
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            # Do not save API response bodies, URLs with credentials, or account data.
            row.update(status="error", error_type=type(exc).__name__)
        row["sample_finished_offset_s"] = perf_counter() - self.started
        self.records.append(row)
        with self.monitor.path.open("a") as output:
            output.write(json.dumps(row) + "\n")

    async def collect(self, stop):
        deadline = self.started
        while not stop.is_set():
            await self.sample()
            deadline += self.monitor.interval
            if deadline < perf_counter():
                deadline = perf_counter() + self.monitor.interval
            delay = max(0, deadline - perf_counter())
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    def summary(self):
        successful = [r for r in self.records if r["status"] == "ok"]
        inside = [
            r for r in successful if 0 <= r["provider_sample_offset_s"] <= self.duration
        ]
        good = list({r["provider_sample_at"]: r for r in inside}.values())
        ids = sorted({gpu["id"] for r in good for gpu in r["gpus"]})
        errors = sum(r["status"] == "error" for r in self.records)
        return {
            "source": "runpod_graphql_latest_telemetry",
            "status": "unavailable" if not good else "partial" if errors else "ok",
            "interval_s": self.monitor.interval,
            "window_duration_s": self.duration,
            "window_started_at": self.started_wall.isoformat(),
            "successful_polls": len(successful),
            "unique_samples": len(good),
            "failed_polls": errors,
            "outside_window_polls_excluded": len(successful) - len(inside),
            "duplicate_polls_excluded": len(inside) - len(good),
            "gpus": [
                {
                    "id": gpu_id,
                    "samples": len(
                        values := [
                            g["gpu_util_percent"]
                            for r in good
                            for g in r["gpus"]
                            if g["id"] == gpu_id
                        ]
                    ),
                    "mean_util_percent": statistics.mean(values),
                    "max_util_percent": max(values),
                }
                for gpu_id in ids
            ],
        }


class RunpodMonitor:
    def __init__(self, client, api_key, pod_id, path, interval=2.0):
        if not api_key or not pod_id or not math.isfinite(interval) or interval <= 0:
            raise ValueError(
                "GPU pod ID and finite positive sampling interval required."
            )
        self.client = client
        self.api_key = api_key
        self.pod_id = pod_id
        self.path = Path(path)
        self.interval = interval

    def config(self):
        return {
            "source": "runpod_graphql_latest_telemetry",
            "pod_id": self.pod_id,
            "interval_s": self.interval,
            "file": self.path.name,
        }

    @asynccontextmanager
    async def measure(self, concurrency):
        observation = Observation(self, concurrency)
        stop = asyncio.Event()
        task = asyncio.create_task(observation.collect(stop))
        try:
            yield observation
        finally:
            observation.duration = perf_counter() - observation.started
            stop.set()
            try:
                await task
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
