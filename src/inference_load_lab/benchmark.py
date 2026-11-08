"""Run a closed-loop concurrency sweep against a streaming vLLM server."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

from inference_load_lab.client import generate

DEFAULT_PROMPT = "Explain KV caching in two sentences."


@dataclass(frozen=True)
class BenchmarkConfig:
    base_url: str
    model: str = "qwen3-4b"
    requests: int = 100
    concurrencies: tuple[int, ...] = (1, 2, 4, 8, 16)
    warmup: int = 5
    max_tokens: int = 128
    prompts: tuple[str, ...] = (DEFAULT_PROMPT,)
    seed: int = 0
    timeout: float = 120.0

    def __post_init__(self):
        if self.requests < 1 or self.max_tokens < 1:
            raise ValueError("--requests and --max-tokens must be positive.")
        if not self.concurrencies or any(c < 1 for c in self.concurrencies):
            raise ValueError("Concurrency levels must be positive.")
        if len(set(self.concurrencies)) != len(self.concurrencies):
            raise ValueError("Concurrency levels must be unique.")
        if self.warmup < 0:
            raise ValueError("--warmup cannot be negative.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("--timeout must be a finite positive number.")
        if not self.prompts or any(not p.strip() for p in self.prompts):
            raise ValueError("Provide at least one nonempty prompt.")
        url = httpx.URL(self.base_url)
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("--base-url must be an absolute HTTP or HTTPS URL.")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError(
                "--base-url cannot contain credentials, a query, or a fragment."
            )


def percentile(values: list[float], fraction: float) -> float | None:
    """Use linear interpolation at index (n - 1) * fraction."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(records: list[dict], duration: float) -> dict:
    successful = [r for r in records if r["status"] == "ok"]
    latencies = [r["e2e_ms"] for r in successful]
    ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
    queue_waits = [
        r["queue_wait_ms"] for r in successful if r.get("queue_wait_ms") is not None
    ]
    missing_usage = sum(r["output_tokens"] is None for r in successful)
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
    return {
        "completed_requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
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
        "missing_token_usage": missing_usage,
        "request_manager_configs": [json.loads(value) for value in sorted(managers)],
        "queue_wait_ms": {
            "p50": percentile(queue_waits, 0.5),
            "p95": percentile(queue_waits, 0.95),
            "samples": len(queue_waits),
        },
        "total_output_tokens": tokens,
        "output_tokens_per_second": tokens / duration if tokens is not None else None,
        "successful_requests_per_second": len(successful) / duration,
    }


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
            request_started = perf_counter()
            record = {
                "concurrency": concurrency,
                "phase": phase,
                "request_id": request_id,
                "prompt_index": prompt_index,
                "started_offset_s": request_started - started,
                "status": "error",
                "ttft_ms": None,
                "queue_wait_ms": None,
                "request_manager": None,
                "output_tokens": None,
                "finish_reason": None,
                "error": None,
            }
            try:
                async with asyncio.timeout(config.timeout):
                    result = await generate(
                        client,
                        base_url=config.base_url,
                        model=config.model,
                        prompt=config.prompts[prompt_index],
                        api_key=api_key,
                        max_tokens=config.max_tokens,
                    )
                record.update(
                    status="ok",
                    queue_wait_ms=result.get("queue_wait_ms"),
                    request_manager=result.get("request_manager"),
                    **{
                        k: result[k]
                        for k in ("ttft_ms", "e2e_ms", "output_tokens", "finish_reason")
                    },
                )
            except (httpx.HTTPError, RuntimeError, ValueError, TimeoutError) as exc:
                message = str(exc) or "Request exceeded its total timeout."
                record["error"] = {
                    "type": type(exc).__name__,
                    "message": message.replace(api_key, "[redacted]")
                    if api_key
                    else message,
                }
                record["e2e_ms"] = (perf_counter() - request_started) * 1000
            record["finished_offset_s"] = perf_counter() - started
            records.append(record)

    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(schedule))):
            group.create_task(worker())
    return perf_counter() - started


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
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = asdict(config)
    settings.update(
        started_at=utc_now(),
        load_model="closed_loop",
        temperature=0,
        percentile_method="linear interpolation at (n - 1) * quantile",
        python_version=platform.python_version(),
        httpx_version=httpx.__version__,
        git=git_state(),
        declared_server_config=server_config,
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
            for concurrency in config.concurrencies:
                if progress:
                    progress("start", {"concurrency": concurrency})
                await run_batch(
                    client,
                    config,
                    api_key,
                    concurrency,
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

                duration = await run_batch(
                    client, config, api_key, concurrency, schedule, "measured", pending
                )
                summary = {
                    "concurrency": concurrency,
                    "planned_requests": config.requests,
                    "warmup_requests": config.warmup,
                    **summarize(pending, duration),
                }
                summary["status"] = (
                    "ok" if summary["failed_requests"] == 0 else "request_failures"
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


def load_prompts(path: Path) -> tuple[str, ...]:
    prompts = []
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
    if not prompts:
        raise ValueError(f"{path}: no prompts found.")
    return tuple(prompts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="inference-load-lab bench", description=__doc__
    )
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3-4b"))
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Measured requests per concurrency level.",
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Unmeasured requests per level; 0 disables warmup.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
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
    args = parser.parse_args(argv)
    if not args.base_url:
        parser.error("Set VLLM_BASE_URL or pass --base-url.")
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        parser.error("Set VLLM_API_KEY in your environment or .env.local.")
    try:
        config = BenchmarkConfig(
            base_url=args.base_url.rstrip("/"),
            model=args.model,
            requests=args.requests,
            concurrencies=tuple(args.concurrency),
            warmup=args.warmup,
            max_tokens=args.max_tokens,
            prompts=load_prompts(args.prompts_file)
            if args.prompts_file
            else tuple(args.prompt or [DEFAULT_PROMPT]),
            seed=args.seed,
            timeout=args.timeout,
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
            print(
                f"Concurrency {summary['concurrency']}: {config.warmup} warmup + {config.requests} measured requests",
                flush=True,
            )
            return

        def number(value):
            return f"{value:.2f}" if value is not None else "n/a"

        print(
            f"  OK={summary['successful_requests']} failed={summary['failed_requests']} | "
            f"TTFT p50/p95={number(summary['ttft_ms']['p50'])}/{number(summary['ttft_ms']['p95'])} ms | "
            f"E2E p50/p95={number(summary['e2e_ms']['p50'])}/{number(summary['e2e_ms']['p95'])} ms | "
            f"queue p50/p95={number(summary['queue_wait_ms']['p50'])}/{number(summary['queue_wait_ms']['p95'])} ms | "
            f"output tokens/s={number(summary['output_tokens_per_second'])}",
            flush=True,
        )

    async def run():
        maximum = max(config.concurrencies)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=min(10.0, config.timeout)),
            limits=httpx.Limits(
                max_connections=maximum,
                max_keepalive_connections=maximum,
                keepalive_expiry=60.0,
            ),
        ) as client:
            return await run_suite(
                client,
                config,
                api_key,
                output_dir,
                server_config=server_config,
                progress=display,
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
            1, "Some requests failed. Inspect summary.json and requests.jsonl.\n"
        )


if __name__ == "__main__":
    main()
