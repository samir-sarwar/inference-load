"""Benchmark a streaming vLLM server with concurrency, rate, or burst load."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

from inference_load_lab.arrivals import arrival_offsets, dispatch_arrivals
from inference_load_lab.client import generate
from inference_load_lab.telemetry import RunpodMonitor, load_runpod_key

DEFAULT_PROMPT = "Explain KV caching in two sentences."
DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class BenchmarkConfig:
    base_url: str
    model: str = "qwen3-4b"
    requests: int = 100
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES
    warmup: int = 5
    max_tokens: int = 128
    prompts: tuple[str, ...] = (DEFAULT_PROMPT,)
    seed: int = 0
    timeout: float = 120.0
    mode: str = "concurrency"
    request_rate: float | None = None
    burst_size: int = 16
    burst_interval: float = 1.0
    max_in_flight: int = 128
    ignore_eos: bool = False
    prompt_classes: tuple[str, ...] = ()
    prompt_max_tokens: tuple[int | None, ...] = ()

    def __post_init__(self):
        if self.requests < 1 or self.max_tokens < 1:
            raise ValueError("--requests and --max-tokens must be positive.")
        if self.mode not in {"concurrency", "rate", "burst"}:
            raise ValueError("Unknown load mode.")
        if self.mode == "concurrency":
            if not self.concurrencies or any(c < 1 for c in self.concurrencies):
                raise ValueError("Concurrency levels must be positive.")
            if len(set(self.concurrencies)) != len(self.concurrencies):
                raise ValueError("Concurrency levels must be unique.")
        else:
            arrival_offsets(
                self.mode,
                self.requests,
                request_rate=self.request_rate,
                burst_size=self.burst_size,
                burst_interval=self.burst_interval,
            )
        if self.mode != "rate" and self.request_rate is not None:
            raise ValueError("--request-rate applies only to rate mode.")
        if self.max_in_flight < 1:
            raise ValueError("--max-in-flight must be positive.")
        if self.warmup < 0:
            raise ValueError("--warmup cannot be negative.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("--timeout must be a finite positive number.")
        if not self.prompts or any(not p.strip() for p in self.prompts):
            raise ValueError("Provide at least one nonempty prompt.")
        if self.prompt_classes and (
            len(self.prompt_classes) != len(self.prompts)
            or any(
                not isinstance(label, str)
                or not label
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label)
                for label in self.prompt_classes
            )
        ):
            raise ValueError("Provide one lowercase class label per prompt.")
        if self.prompt_max_tokens and (
            len(self.prompt_max_tokens) != len(self.prompts)
            or any(
                value is not None and (type(value) is not int or value < 1)
                for value in self.prompt_max_tokens
            )
        ):
            raise ValueError("Provide one positive output budget (or null) per prompt.")
        url = httpx.URL(self.base_url)
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("--base-url must be an absolute HTTP or HTTPS URL.")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError(
                "--base-url cannot contain credentials, a query, or a fragment."
            )

    def output_budget(self, index):
        return (
            self.prompt_max_tokens[index] if self.prompt_max_tokens else None
        ) or self.max_tokens

    def workload_class(self, index):
        return self.prompt_classes[index] if self.prompt_classes else "default"


def percentile(values: list[float], fraction: float) -> float | None:
    """Use linear interpolation at index (n - 1) * fraction."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(records: list[dict], duration: float, *, include_classes=True) -> dict:
    successful = [r for r in records if r["status"] == "ok"]
    latencies = [r["e2e_ms"] for r in successful]
    ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
    tpots = [r["tpot_ms"] for r in successful if r.get("tpot_ms") is not None]
    queue_waits = [
        r["queue_wait_ms"] for r in successful if r.get("queue_wait_ms") is not None
    ]
    missing_usage = sum(r["output_tokens"] is None for r in successful)
    attempted = [r for r in records if r["status"] != "dropped"]
    dispatch_lags = [
        r["dispatch_lag_ms"] for r in attempted if r.get("dispatch_lag_ms") is not None
    ]
    arrival_latencies = [
        r["arrival_e2e_ms"] for r in successful if r.get("arrival_e2e_ms") is not None
    ]
    managers = {
        json.dumps(r["request_manager"], sort_keys=True)
        for r in successful
        if r.get("request_manager") is not None
    }
    tokens = (
        sum(r["output_tokens"] for r in successful)
        if successful and not missing_usage
        else None
    )
    result = {
        "recorded_requests": len(records),
        "completed_requests": len(attempted),
        "successful_requests": len(successful),
        "failed_requests": sum(r["status"] == "error" for r in records),
        "dropped_requests": sum(r["status"] == "dropped" for r in records),
        "http_status_counts": dict(
            Counter(
                str(r["http_status"])
                for r in attempted
                if r.get("http_status") is not None
            )
        ),
        "error_types": dict(
            Counter(
                r["error"].get("server_type") or r["error"]["type"]
                for r in records
                if r.get("error")
            )
        ),
        "duration_s": duration,
        "e2e_ms": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "ttft_ms": {
            "p50": percentile(ttfts, 0.5),
            "p95": percentile(ttfts, 0.95),
            "samples": len(ttfts),
        },
        "tpot_ms": {
            "p50": percentile(tpots, 0.5),
            "p95": percentile(tpots, 0.95),
            "samples": len(tpots),
        },
        "missing_token_usage": missing_usage,
        "dispatch_lag_ms": {
            "p50": percentile(dispatch_lags, 0.5),
            "p95": percentile(dispatch_lags, 0.95),
            "max": max(dispatch_lags) if dispatch_lags else None,
            "samples": len(dispatch_lags),
        },
        "arrival_e2e_ms": {
            "p50": percentile(arrival_latencies, 0.5),
            "p95": percentile(arrival_latencies, 0.95),
            "samples": len(arrival_latencies),
        },
        "request_manager_configs": [json.loads(value) for value in sorted(managers)],
        "queue_wait_ms": {
            "p50": percentile(queue_waits, 0.5),
            "p95": percentile(queue_waits, 0.95),
            "samples": len(queue_waits),
            "max": max(queue_waits) if queue_waits else None,
        },
        "total_output_tokens": tokens,
        "output_tokens_per_second": tokens / duration if tokens is not None else None,
        "successful_requests_per_second": len(successful) / duration,
        "aged_promotions": sum(r.get("aged_promotion") is True for r in successful),
    }
    if include_classes:
        result["by_class"] = {
            label: summarize(
                [r for r in records if r.get("workload_class", "default") == label],
                duration,
                include_classes=False,
            )
            for label in sorted({r.get("workload_class", "default") for r in records})
        }
    return result


def request_record(
    concurrency,
    phase,
    request_id,
    prompt_index,
    scheduled_offset=None,
    workload_class="default",
    requested_max_tokens=None,
):
    return {
        "concurrency": concurrency,
        "phase": phase,
        "request_id": request_id,
        "prompt_index": prompt_index,
        "workload_class": workload_class,
        "requested_max_tokens": requested_max_tokens,
        "scheduled_offset_s": scheduled_offset,
        "started_offset_s": None,
        "status": "error",
        "http_status": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "queue_wait_ms": None,
        "request_manager": None,
        "output_tokens": None,
        "finish_reason": None,
        "error": None,
        "e2e_ms": None,
        "dispatch_lag_ms": None,
        "arrival_e2e_ms": None,
        "arrival_order": None,
        "admission_order": None,
        "aged_promotion": None,
        "estimated_cost": None,
    }


async def perform_request(
    client,
    config,
    api_key,
    concurrency,
    phase,
    records,
    request_id,
    prompt_index,
    started,
    scheduled_offset=None,
):
    request_started = perf_counter()
    record = request_record(
        concurrency,
        phase,
        request_id,
        prompt_index,
        scheduled_offset,
        config.workload_class(prompt_index),
        config.output_budget(prompt_index),
    )
    record["started_offset_s"] = request_started - started
    if scheduled_offset is not None:
        record["dispatch_lag_ms"] = max(
            0.0, (request_started - started - scheduled_offset) * 1000
        )
    try:
        async with asyncio.timeout(config.timeout):
            result = await generate(
                client,
                base_url=config.base_url,
                model=config.model,
                prompt=config.prompts[prompt_index],
                api_key=api_key,
                max_tokens=config.output_budget(prompt_index),
                ignore_eos=config.ignore_eos,
            )
        record.update(
            status="ok",
            http_status=200,
            tpot_ms=result.get("tpot_ms"),
            **{
                field: result.get(field)
                for field in [
                    "arrival_order",
                    "admission_order",
                    "aged_promotion",
                    "estimated_cost",
                ]
            },
            queue_wait_ms=result.get("queue_wait_ms"),
            request_manager=result.get("request_manager"),
            **{
                k: result[k]
                for k in ("ttft_ms", "e2e_ms", "output_tokens", "finish_reason")
            },
        )
    except (httpx.HTTPError, RuntimeError, ValueError, TimeoutError) as exc:
        message = str(exc) or "Request exceeded its total timeout."
        record["http_status"] = getattr(exc, "status_code", None)
        record["error"] = {
            "type": type(exc).__name__,
            "server_type": getattr(exc, "server_type", None),
            "message": message.replace(api_key, "[redacted]") if api_key else message,
        }
        record["e2e_ms"] = (perf_counter() - request_started) * 1000
    record["finished_offset_s"] = perf_counter() - started
    if scheduled_offset is not None:
        record["arrival_e2e_ms"] = (
            record["finished_offset_s"] - scheduled_offset
        ) * 1000
    records.append(record)


async def run_batch(
    client: httpx.AsyncClient,
    config: BenchmarkConfig,
    api_key: str,
    concurrency: int,
    schedule: list[int],
    phase: str,
    records: list[dict],
) -> float:
    """Use bounded workers; append completed results for interruption recovery."""
    work = iter(enumerate(schedule))
    started = perf_counter()

    async def worker():
        for request_id, prompt_index in work:
            await perform_request(
                client,
                config,
                api_key,
                concurrency,
                phase,
                records,
                request_id,
                prompt_index,
                started,
            )

    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(schedule))):
            group.create_task(worker())
    return perf_counter() - started


async def run_arrival_batch(client, config, api_key, schedule, records):
    offsets = arrival_offsets(
        config.mode,
        config.requests,
        request_rate=config.request_rate,
        burst_size=config.burst_size,
        burst_interval=config.burst_interval,
    )

    async def send(request_id, offset, started):
        await perform_request(
            client,
            config,
            api_key,
            None,
            "measured",
            records,
            request_id,
            schedule[request_id],
            started,
            offset,
        )

    def drop(request_id, offset, started):
        record = request_record(
            None,
            "measured",
            request_id,
            schedule[request_id],
            offset,
            config.workload_class(schedule[request_id]),
            config.output_budget(schedule[request_id]),
        )
        record.update(
            status="dropped",
            finished_offset_s=perf_counter() - started,
            error={
                "type": "ClientCapacityExceeded",
                "server_type": None,
                "message": "Driver in-flight limit reached; this arrival was not sent.",
            },
        )
        records.append(record)

    return await dispatch_arrivals(offsets, config.max_in_flight, send, drop)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_state() -> dict:
    def read(*args):
        try:
            result = subprocess.run(
                ["git", *args], check=False, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = read("status", "--porcelain")
    return {
        "commit": read("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def redact_key(value, api_key: str):
    if isinstance(value, str):
        return value.replace(api_key, "[redacted]") if api_key else value
    if isinstance(value, dict):
        return {k: redact_key(v, api_key) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact_key(v, api_key) for v in value]
    return value


async def run_suite(
    client: httpx.AsyncClient,
    config: BenchmarkConfig,
    api_key: str,
    output_dir: Path,
    *,
    server_config: dict | None = None,
    progress=None,
    gpu_monitor: RunpodMonitor | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = asdict(config)
    settings.update(
        started_at=utc_now(),
        load_model="closed_loop" if config.mode == "concurrency" else "open_loop",
        arrival_pattern=config.mode if config.mode != "concurrency" else None,
        temperature=0,
        percentile_method="linear interpolation at (n - 1) * quantile",
        python_version=platform.python_version(),
        httpx_version=httpx.__version__,
        git=git_state(),
        declared_server_config=server_config,
        tpot_definition="(client e2e_ms - client ttft_ms) / (server output_tokens - 1)",
        gpu_telemetry=gpu_monitor.config() if gpu_monitor else None,
        prompt_sha256=hashlib.sha256(
            json.dumps(config.prompts, ensure_ascii=False).encode()
        ).hexdigest(),
    )
    (output_dir / "config.json").write_text(
        json.dumps(redact_key(settings, api_key), indent=2) + "\n"
    )
    schedule = [i % len(config.prompts) for i in range(config.requests)]
    random.Random(config.seed).shuffle(schedule)
    warmup_schedule = [i % len(config.prompts) for i in range(config.warmup)]
    report = {"status": "running", "runs": []}
    pending = []

    def save_report():
        (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    with (output_dir / "requests.jsonl").open("x") as output:

        def save_requests():
            for record in pending:
                output.write(json.dumps(record) + "\n")
            output.flush()
            pending.clear()

        save_report()
        try:
            levels = config.concurrencies if config.mode == "concurrency" else (None,)
            for concurrency in levels:
                if progress:
                    progress("start", {"concurrency": concurrency, "mode": config.mode})
                warmup_concurrency = (
                    concurrency
                    if concurrency is not None
                    else min(16, config.max_in_flight)
                )
                await run_batch(
                    client,
                    config,
                    api_key,
                    warmup_concurrency,
                    warmup_schedule,
                    "warmup",
                    pending,
                )
                warmup_failures = sum(r["status"] == "error" for r in pending)
                warmup_error = next((r["error"] for r in pending if r["error"]), None)
                save_requests()
                if warmup_failures:
                    report["runs"].append(
                        {
                            "concurrency": concurrency,
                            "status": "warmup_failed",
                            "planned_requests": config.requests,
                            "completed_requests": 0,
                            "warmup_failed_requests": warmup_failures,
                            "error": warmup_error,
                        }
                    )
                    break

                arrival_details = {}
                measurement = (
                    gpu_monitor.measure(concurrency) if gpu_monitor else nullcontext()
                )
                async with measurement as observation:
                    if config.mode == "concurrency":
                        duration = await run_batch(
                            client,
                            config,
                            api_key,
                            concurrency,
                            schedule,
                            "measured",
                            pending,
                        )
                    else:
                        arrival_details = await run_arrival_batch(
                            client, config, api_key, schedule, pending
                        )
                        duration = arrival_details.pop("duration_s")
                summary = {
                    "concurrency": concurrency,
                    "mode": config.mode,
                    "planned_requests": config.requests,
                    "warmup_requests": config.warmup,
                    "warmup_concurrency": warmup_concurrency,
                    **arrival_details,
                    **summarize(pending, duration),
                }
                if observation is not None:
                    summary["gpu"] = observation.summary()
                summary["status"] = (
                    "ok"
                    if summary["failed_requests"] == 0
                    and summary["dropped_requests"] == 0
                    else "request_failures"
                )
                report["runs"].append(summary)
                save_requests()
                save_report()
                if progress:
                    progress("complete", summary)
            report["status"] = (
                "ok" if all(r["status"] == "ok" for r in report["runs"]) else "failed"
            )
        except asyncio.CancelledError:
            report["status"] = "interrupted"
            raise
        except Exception:
            report["status"] = "error"
            raise
        finally:
            save_requests()
            report["ended_at"] = utc_now()
            save_report()
    return report


def load_workload(path: Path) -> dict:
    prompts = []
    classes = []
    budgets = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON.") from exc
        prompt = value.get("prompt") if isinstance(value, dict) else value
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f'{path}:{number}: expected a nonempty string or {{"prompt": "..."}}.'
            )
        prompts.append(prompt)
        label = value.get("class", "default") if isinstance(value, dict) else "default"
        budget = value.get("max_tokens") if isinstance(value, dict) else None
        if (
            not isinstance(label, str)
            or not label
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label)
        ):
            raise ValueError(f"{path}:{number}: invalid workload class.")
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise ValueError(f"{path}:{number}: output budget must be positive.")
        classes.append(label)
        budgets.append(budget)
    if not prompts:
        raise ValueError(f"{path}: no prompts found.")
    return {
        "prompts": tuple(prompts),
        "prompt_classes": tuple(classes),
        "prompt_max_tokens": tuple(budgets),
    }


def load_prompts(path: Path) -> tuple[str, ...]:
    return load_workload(path)["prompts"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="inference-load-lab bench", description=__doc__
    )
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3-4b"))
    parser.add_argument(
        "--mode", choices=["concurrency", "rate", "burst"], default="concurrency"
    )
    parser.add_argument(
        "--request-rate", type=float, help="Scheduled requests per second in rate mode."
    )
    parser.add_argument(
        "--burst-size", type=int, help="Requests per burst (default 16)."
    )
    parser.add_argument(
        "--burst-interval", type=float, help="Seconds between bursts (default 1)."
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        help="Driver cap for arrival modes (default 128); excess arrivals are dropped, not queued.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Measured requests per concurrency level or arrival run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        help="Concurrency-mode levels (default 1 2 4 8 16).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Unmeasured requests per level; 0 disables warmup.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="vLLM extension: continue to the token limit for a fixed-length decode workload.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Total timeout per request in seconds.",
    )
    parser.add_argument("--seed", type=int, default=0)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument(
        "--prompt", action="append", help="Repeat to supply multiple prompts."
    )
    prompts.add_argument(
        "--prompts-file",
        type=Path,
        help="JSONL strings or objects with a prompt field.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; existing directories are never overwritten.",
    )
    parser.add_argument(
        "--server-config",
        type=Path,
        help="Optional declared server settings to record with the run.",
    )
    parser.add_argument(
        "--gpu-pod-id", help="Opt in to sampling this Runpod pod's GPU utilization."
    )
    parser.add_argument("--gpu-interval", type=float, default=2.0)
    args = parser.parse_args(argv)
    if not args.base_url:
        parser.error("Set VLLM_BASE_URL or pass --base-url.")
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        parser.error("Set VLLM_API_KEY in your environment or .env.local.")
    try:
        if not math.isfinite(args.gpu_interval) or args.gpu_interval < 1:
            raise ValueError("--gpu-interval must be finite and at least one second.")
        gpu_key = load_runpod_key() if args.gpu_pod_id else None
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.mode != "concurrency" and args.concurrency is not None:
        parser.error(
            "Use --max-in-flight for arrival modes; --concurrency is only for concurrency mode."
        )
    if args.mode == "concurrency" and any(
        v is not None
        for v in [
            args.request_rate,
            args.burst_size,
            args.burst_interval,
            args.max_in_flight,
        ]
    ):
        parser.error("Arrival options require --mode rate or --mode burst.")
    if args.mode != "burst" and (
        args.burst_size is not None or args.burst_interval is not None
    ):
        parser.error("Burst options require --mode burst.")
    try:
        prompt_settings = (
            load_workload(args.prompts_file)
            if args.prompts_file
            else {"prompts": tuple(args.prompt or [DEFAULT_PROMPT])}
        )
        config = BenchmarkConfig(
            base_url=args.base_url.rstrip("/"),
            model=args.model,
            requests=args.requests,
            concurrencies=tuple(args.concurrency or DEFAULT_CONCURRENCIES)
            if args.mode == "concurrency"
            else (),
            warmup=args.warmup,
            max_tokens=args.max_tokens,
            **prompt_settings,
            seed=args.seed,
            timeout=args.timeout,
            mode=args.mode,
            request_rate=args.request_rate,
            burst_size=args.burst_size if args.burst_size is not None else 16,
            burst_interval=args.burst_interval
            if args.burst_interval is not None
            else 1.0,
            max_in_flight=args.max_in_flight if args.max_in_flight is not None else 128,
            ignore_eos=args.ignore_eos,
        )
        server_config = (
            json.loads(args.server_config.read_text()) if args.server_config else None
        )
    except (OSError, ValueError, httpx.InvalidURL) as exc:
        parser.error(str(exc))
    output_dir = args.output_dir or Path("results") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )

    def display(event, summary):
        if event == "start":
            if config.mode == "concurrency":
                label = f"Concurrency {summary['concurrency']}"
            elif config.mode == "rate":
                label = f"Rate {config.request_rate:g} requests/s (driver cap {config.max_in_flight})"
            else:
                label = f"Bursts of {config.burst_size} every {config.burst_interval:g}s (driver cap {config.max_in_flight})"
            print(
                f"{label}: {config.warmup} warmup + {config.requests} measured requests",
                flush=True,
            )
            return

        def number(value):
            return f"{value:.2f}" if value is not None else "n/a"

        print(
            f"  OK={summary['successful_requests']} failed={summary['failed_requests']} dropped={summary['dropped_requests']} | "
            f"TTFT p50/p95={number(summary['ttft_ms']['p50'])}/{number(summary['ttft_ms']['p95'])} ms | "
            f"TPOT p50/p95={number(summary['tpot_ms']['p50'])}/{number(summary['tpot_ms']['p95'])} ms | "
            f"E2E p50/p95={number(summary['e2e_ms']['p50'])}/{number(summary['e2e_ms']['p95'])} ms | "
            f"queue p50/p95={number(summary['queue_wait_ms']['p50'])}/{number(summary['queue_wait_ms']['p95'])} ms | "
            f"output tokens/s={number(summary['output_tokens_per_second'])}",
            flush=True,
        )
        if config.mode != "concurrency":
            print(
                f"  dispatch lag p95={number(summary['dispatch_lag_ms']['p95'])} ms | "
                f"arrival-to-completion p95={number(summary['arrival_e2e_ms']['p95'])} ms | "
                f"peak in-flight={summary['peak_in_flight']}",
                flush=True,
            )
        if "gpu" in summary:
            gpu = summary["gpu"]
            print(
                f"  GPU telemetry={gpu['status']} | unique samples={gpu['unique_samples']} | "
                f"failed polls={gpu['failed_polls']}",
                flush=True,
            )

    async def run():
        maximum = (
            max(config.concurrencies)
            if config.mode == "concurrency"
            else config.max_in_flight
        )
        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(
                    config.timeout, connect=min(10.0, config.timeout)
                ),
                limits=httpx.Limits(
                    max_connections=maximum,
                    max_keepalive_connections=maximum,
                    keepalive_expiry=60.0,
                ),
            ) as client,
            httpx.AsyncClient(timeout=10) as gpu_client,
        ):
            return await run_suite(
                client,
                config,
                api_key,
                output_dir,
                server_config=server_config,
                progress=display,
                gpu_monitor=RunpodMonitor(
                    gpu_client,
                    gpu_key,
                    args.gpu_pod_id,
                    output_dir / "gpu.jsonl",
                    args.gpu_interval,
                )
                if args.gpu_pod_id
                else None,
            )

    try:
        report = asyncio.run(run())
    except FileExistsError:
        parser.exit(
            1,
            f"Output directory already exists: {output_dir}. Choose a new --output-dir.\n",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Benchmark failed: {str(exc).replace(api_key, '[redacted]')}\n")
    except KeyboardInterrupt:
        parser.exit(
            130, f"\nInterrupted. Saved completed requests to {output_dir.resolve()}\n"
        )
    print(f"Results: {output_dir.resolve()}")
    if report["status"] != "ok":
        parser.exit(
            1,
            "Some requests failed or were dropped. Inspect summary.json and requests.jsonl.\n",
        )


if __name__ == "__main__":
    main()
