"""Repeat a controlled comparison of direct vLLM and FIFO admission settings."""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import httpx

from inference_load_lab.benchmark import (
    BenchmarkConfig,
    git_state,
    load_prompts,
    redact_key,
    run_suite,
    utc_now,
)
from inference_load_lab.gateway import GatewayConfig
from inference_load_lab.telemetry import RunpodMonitor, load_runpod_key


def trial_order(names, trial):
    """Rotate positions; reverse direction on alternating complete cycles."""
    cycle, offset = divmod(trial, len(names))
    order = list(reversed(names)) if cycle % 2 else list(names)
    return order[offset:] + order[:offset]


def validate_spec(spec, base_url):
    if type(spec["trials"]) is not int or spec["trials"] < 3:
        raise ValueError("Use at least three trials for the controlled comparison.")
    variants = spec["variants"]
    names = [v["name"] for v in variants]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Provide at least two uniquely named variants.")
    for variant in variants:
        name = variant["name"]
        if not name or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name
        ):
            raise ValueError(
                "Variant names must contain lowercase letters, digits or hyphens."
            )
        if variant.get("concurrency") is not None:
            GatewayConfig(
                base_url,
                concurrency=variant["concurrency"],
                max_queue=spec["queue_size"],
                queue_timeout=spec["queue_timeout"],
                upstream_timeout=spec["workload"]["timeout"],
            )
    if spec["baseline"] not in names or spec["candidate"] not in names:
        raise ValueError("Baseline and candidate must name configured variants.")
    if spec["baseline"] == spec["candidate"]:
        raise ValueError("Baseline and candidate must differ.")


@asynccontextmanager
async def route(variant, spec, upstream, upstream_key, frontend_key, log_path):
    if variant.get("concurrency") is None:
        yield upstream
        return
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment.update(
        VLLM_BASE_URL=upstream, VLLM_API_KEY=upstream_key, GATEWAY_API_KEY=frontend_key
    )
    command = [
        sys.executable,
        "-c",
        "from inference_load_lab import main; main()",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--concurrency",
        str(variant["concurrency"]),
        "--queue-size",
        str(spec["queue_size"]),
        "--queue-timeout",
        str(spec["queue_timeout"]),
        "--upstream-timeout",
        str(spec["workload"]["timeout"]),
    ]
    with log_path.open("w") as log:
        process = await asyncio.create_subprocess_exec(
            *command, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                deadline = monotonic() + 15
                while monotonic() < deadline:
                    if process.returncode is not None:
                        raise RuntimeError(
                            "Temporary gateway exited; see its console log."
                        )
                    try:
                        if (await client.get(url + "/health")).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.05)
                else:
                    raise RuntimeError("Temporary gateway did not become ready.")
            yield url
        finally:
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=15)
            except TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=5)


async def run_experiment(
    spec,
    base_url,
    api_key,
    output,
    *,
    gpu_pod_id=None,
    gpu_key=None,
    gpu_interval=2,
    gateway_key=None,
    deployment_info=None,
):
    validate_spec(spec, base_url)
    if spec.get("require_gpu_telemetry") and not gpu_pod_id:
        raise ValueError("This experiment requires --gpu-pod-id for GPU measurements.")
    prompts = load_prompts(Path(spec["prompts_file"]))
    workload = dict(spec["workload"])
    workload["concurrencies"] = tuple(workload["concurrencies"])
    template = BenchmarkConfig(base_url, prompts=prompts, **workload)
    if template.mode != "concurrency" or len(template.concurrencies) != 1:
        raise ValueError(
            "Controlled experiments currently require one closed-loop concurrency level."
        )
    server_config = json.loads(Path(spec["server_config"]).read_text())
    deployment = None
    if deployment_info:
        deployment = json.loads(Path(deployment_info).read_text())
        if deployment.get("base_url") != base_url:
            raise ValueError(
                "Deployment metadata URL does not match the experiment upstream."
            )
    frontend_key = gateway_key or api_key
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    keys = (api_key, frontend_key, gpu_key)

    def save(path, data):
        for key in keys:
            if key:
                data = redact_key(data, key)
        path.write_text(json.dumps(data, indent=2) + "\n")

    metadata = {
        "status": "running",
        "started_at": utc_now(),
        "spec": spec,
        "git": git_state(),
        "server_config": server_config,
        "gpu_pod_id": gpu_pod_id,
        "runs": [],
    }
    save(output / "experiment.json", metadata)
    sources = [
        *Path("src").rglob("*.py"),
        *Path("tests").rglob("*.py"),
        *Path("scripts").rglob("*.py"),
        *Path("configs").rglob("*.json"),
        *Path(".github").rglob("*.yml"),
        Path(".gitignore"),
        Path("README.md"),
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path(spec["server_config"]),
        Path(spec["prompts_file"]),
    ]
    save(
        output / "source-manifest.json",
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(sources))
            if p.is_file()
        },
    )
    if deployment is not None:
        # This allowlist excludes pod environment variables and credentials.
        save(
            output / "deployment.json",
            {
                k: deployment[k]
                for k in ["image", "model", "revision", "server_args", "base_url"]
                if k in deployment
            },
        )
        save(
            output / "pod.json",
            {
                k: v
                for k, v in deployment.get("pod", {}).items()
                if k
                in [
                    "id",
                    "name",
                    "gpuId",
                    "gpuCount",
                    "costPerHr",
                    "imageName",
                    "containerDiskInGb",
                    "volumeInGb",
                ]
            },
        )

    async def snapshot(client, name):
        response = await client.get(
            base_url + "/metrics", headers={"Authorization": "Bearer " + api_key}
        )
        status = {"http_status": response.status_code, "observed_at": utc_now()}
        if response.status_code == 200:
            text = response.text
            for key in keys:
                if key:
                    text = text.replace(key, "[redacted]")
            path = output / (name + ".prom")
            path.write_text(text)
            status["file"] = path.name
        return status

    try:
        async with httpx.AsyncClient(timeout=15) as control:
            models = await control.get(
                base_url + "/v1/models", headers={"Authorization": "Bearer " + api_key}
            )
            models.raise_for_status()
            save(output / "models.json", models.json())
            version = await control.get(
                base_url + "/version", headers={"Authorization": "Bearer " + api_key}
            )
            save(
                output / "version.json",
                {"http_status": version.status_code, "body": version.text},
            )
            names = [v["name"] for v in spec["variants"]]
            by_name = {v["name"]: v for v in spec["variants"]}
            for trial in range(spec["trials"]):
                order = trial_order(names, trial)
                for position, name in enumerate(order):
                    variant = by_name[name]
                    label = f"trial-{trial + 1:02}-{name}"
                    directory = output / "runs" / f"trial-{trial + 1:02}" / name
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    entry = {
                        "trial": trial + 1,
                        "position": position + 1,
                        "variant": name,
                        "started_at": utc_now(),
                        "output_dir": str(directory.relative_to(output)),
                        "metrics_before": await snapshot(control, label + "-before"),
                    }
                    metadata["runs"].append(entry)
                    save(output / "experiment.json", metadata)
                    print(
                        f"Trial {trial + 1}/{spec['trials']} {name}: starting.",
                        flush=True,
                    )
                    async with route(
                        variant,
                        spec,
                        base_url,
                        api_key,
                        frontend_key,
                        output / (label + "-gateway.log"),
                    ) as url:
                        settings = asdict(template)
                        settings.update(base_url=url)
                        config = BenchmarkConfig(**settings)
                        key = (
                            api_key
                            if variant.get("concurrency") is None
                            else frontend_key
                        )
                        maximum = max(config.concurrencies)
                        async with (
                            httpx.AsyncClient(
                                timeout=httpx.Timeout(
                                    config.timeout, connect=min(10, config.timeout)
                                ),
                                limits=httpx.Limits(
                                    max_connections=maximum,
                                    max_keepalive_connections=maximum,
                                    keepalive_expiry=60,
                                ),
                            ) as inference,
                            httpx.AsyncClient(timeout=10) as telemetry,
                        ):
                            monitor = (
                                RunpodMonitor(
                                    telemetry,
                                    gpu_key,
                                    gpu_pod_id,
                                    directory / "gpu.jsonl",
                                    gpu_interval,
                                )
                                if gpu_pod_id
                                else None
                            )
                            report = await run_suite(
                                inference,
                                config,
                                key,
                                directory,
                                server_config=server_config,
                                gpu_monitor=monitor,
                            )
                        if variant.get("concurrency") is not None:
                            stats = await control.get(
                                url + "/stats",
                                headers={"Authorization": "Bearer " + frontend_key},
                            )
                            stats.raise_for_status()
                            save(output / (label + "-gateway-stats.json"), stats.json())
                    entry.update(
                        ended_at=utc_now(),
                        status=report["status"],
                        metrics_after=await snapshot(control, label + "-after"),
                    )
                    save(output / "experiment.json", metadata)
                    summary = report["runs"][0]
                    print(
                        f"  {report['status']}: E2E p95={summary.get('e2e_ms', {}).get('p95')} ms; "
                        f"tokens/s={summary.get('output_tokens_per_second')}",
                        flush=True,
                    )
                    if report["status"] != "ok":
                        raise RuntimeError(
                            "A benchmark run failed; partial results were preserved."
                        )
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata["status"] = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            else "failed"
        )
        metadata["error_type"] = type(exc).__name__
        raise
    finally:
        metadata["ended_at"] = utc_now()
        save(output / "experiment.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="inference-load-lab experiment", description=__doc__
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/admission-experiment.json")
    )
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-pod-id")
    parser.add_argument("--gpu-interval", type=float, default=2)
    parser.add_argument("--deployment-info", type=Path)
    args = parser.parse_args(argv)
    api_key = os.environ.get("VLLM_API_KEY")
    if not args.base_url or not api_key:
        parser.error("Set VLLM_BASE_URL and VLLM_API_KEY, or pass --base-url.")
    try:
        spec = json.loads(args.config.read_text())
        gpu_key = load_runpod_key() if args.gpu_pod_id else None
        if args.gpu_pod_id:
            RunpodMonitor(
                None, gpu_key, args.gpu_pod_id, Path("unused"), args.gpu_interval
            )
        output = args.output_dir or Path("results") / (
            "experiment-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        )
        asyncio.run(
            run_experiment(
                spec,
                args.base_url.rstrip("/"),
                api_key,
                output,
                gpu_pod_id=args.gpu_pod_id,
                gpu_key=gpu_key,
                gpu_interval=args.gpu_interval,
                gateway_key=os.environ.get("GATEWAY_API_KEY"),
                deployment_info=args.deployment_info,
            )
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        httpx.HTTPError,
        RuntimeError,
    ) as exc:
        message = str(exc).replace(api_key, "[redacted]")
        parser.exit(1, f"Experiment failed: {message}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Experiment interrupted; partial results preserved.\n")
    print(f"Experiment saved: {output.resolve()}")
