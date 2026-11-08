"""Validate controlled comparisons and export numeric results and figures."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRICS = {
    "e2e_p95_ms": ("e2e_ms", "p95"),
    "ttft_p95_ms": ("ttft_ms", "p95"),
    "tpot_p50_ms": ("tpot_ms", "p50"),
    "queue_p95_ms": ("queue_wait_ms", "p95"),
    "tokens_per_second": ("output_tokens_per_second",),
}
MATCH_FIELDS = [
    "model",
    "requests",
    "concurrencies",
    "warmup",
    "max_tokens",
    "prompts",
    "prompt_sha256",
    "seed",
    "timeout",
    "mode",
    "temperature",
    "declared_server_config",
    "tpot_definition",
    "ignore_eos",
    "gpu_telemetry",
    "python_version",
    "httpx_version",
]


def read_json(path):
    return json.loads(path.read_text())


def metric(row, keys):
    value = row
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Nonfinite or nonnumeric benchmark metric.")
    return value


def run_directory(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Run path escapes the experiment directory.")
    return path


def analyze_experiment(root):
    root = Path(root)
    experiment = read_json(root / "experiment.json")
    if experiment["status"] != "complete":
        raise ValueError(
            "Only completed experiments can produce a controlled comparison."
        )
    spec = experiment["spec"]
    variants = {v["name"]: v for v in spec["variants"]}
    expected = {
        (trial, name) for trial in range(1, spec["trials"] + 1) for name in variants
    }
    actual = [(entry["trial"], entry["variant"]) for entry in experiment["runs"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Missing or duplicate trial/variant runs.")
    rows, issues = [], []
    reference_config = reference_tokens = reference_gpus = None
    for entry in experiment["runs"]:
        directory = run_directory(root, entry["output_dir"])
        config = read_json(directory / "config.json")
        report = read_json(directory / "summary.json")
        if len(report["runs"]) != 1:
            raise ValueError("Expected one workload level per controlled run.")
        summary = report["runs"][0]
        label = f"trial {entry['trial']} {entry['variant']}"
        matched = {k: config.get(k) for k in MATCH_FIELDS}
        if reference_config is None:
            reference_config = matched
        elif matched != reference_config:
            raise ValueError(
                "Workload, server, telemetry, or metric definitions changed between runs."
            )
        if report["status"] != "ok" or any(
            [
                summary.get("successful_requests") != config["requests"],
                summary.get("failed_requests") != 0,
                summary.get("dropped_requests") != 0,
                summary.get("missing_token_usage") != 0,
            ]
        ):
            issues.append(
                label
                + ": incomplete requests, failures, drops, or missing token counts"
            )
        records = [
            json.loads(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        tokens = sorted(
            (r["request_id"], r["prompt_index"], r["output_tokens"])
            for r in records
            if r["phase"] == "measured" and r["status"] == "ok"
        )
        if len(tokens) != config["requests"] or {r[0] for r in tokens} != set(
            range(config["requests"])
        ):
            issues.append(
                label + ": raw successful request records are missing or duplicated"
            )
        if reference_tokens is None:
            reference_tokens = tokens
        elif tokens != reference_tokens:
            issues.append(
                label + ": generated token counts or request/prompt mapping changed"
            )
        limit = variants[entry["variant"]].get("concurrency")
        managers = summary.get("request_manager_configs", [])
        if limit is None:
            if managers:
                issues.append(
                    label + ": direct route unexpectedly reported admission settings"
                )
        elif managers != [
            {
                "policy": "fifo",
                "concurrency_limit": limit,
                "queue_capacity": spec["queue_size"],
                "queue_timeout_s": spec["queue_timeout"],
                "upstream_timeout_s": spec["workload"]["timeout"],
            }
        ]:
            issues.append(
                label + ": observed gateway limit does not match the planned variant"
            )
        gpu = summary.get("gpu", {})
        gpu_ids = sorted(g["id"] for g in gpu.get("gpus", []))
        if spec.get("require_gpu_telemetry"):
            if gpu.get("status") != "ok" or gpu.get("unique_samples", 0) < 1:
                issues.append(
                    label + ": no fresh unique GPU sample in the measured window"
                )
            if reference_gpus is None:
                reference_gpus = gpu_ids
            elif gpu_ids != reference_gpus:
                issues.append(label + ": GPU identity changed")
        row = {
            "trial": entry["trial"],
            "position": entry["position"],
            "variant": entry["variant"],
            "successful_requests": summary.get("successful_requests"),
            "failed_requests": summary.get("failed_requests"),
            "dropped_requests": summary.get("dropped_requests"),
            "total_output_tokens": summary.get("total_output_tokens"),
            "duration_s": summary.get("duration_s"),
            "gpu_successful_polls": gpu.get("successful_polls", 0),
            "gpu_unique_samples": gpu.get("unique_samples", 0),
            "gpu_mean_util_percent": statistics.mean(
                g["mean_util_percent"] for g in gpu["gpus"]
            )
            if gpu.get("gpus")
            else None,
            "output_dir": entry["output_dir"],
        }
        row.update({name: metric(summary, keys) for name, keys in METRICS.items()})
        if any(
            row[k] is None or row[k] <= 0
            for k in ["e2e_p95_ms", "tokens_per_second", "tpot_p50_ms"]
        ):
            issues.append(
                label + ": required positive timing or throughput metrics are missing"
            )
        rows.append(row)
    aggregates = {}
    for name in variants:
        runs = [r for r in rows if r["variant"] == name]
        aggregates[name] = {"trials": len(runs)}
        for field in [*METRICS, "gpu_mean_util_percent"]:
            values = [r[field] for r in runs if r[field] is not None]
            aggregates[name][field] = (
                {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
                if values
                else None
            )
    pairs = []
    if not issues:
        by_run = {(r["trial"], r["variant"]): r for r in rows}
        for trial in range(1, spec["trials"] + 1):
            base = by_run[trial, spec["baseline"]]
            candidate = by_run[trial, spec["candidate"]]
            pairs.append(
                {
                    "trial": trial,
                    "e2e_p95_reduction_percent": 100
                    * (base["e2e_p95_ms"] - candidate["e2e_p95_ms"])
                    / base["e2e_p95_ms"],
                    "throughput_increase_percent": 100
                    * (candidate["tokens_per_second"] - base["tokens_per_second"])
                    / base["tokens_per_second"],
                }
            )
    return {
        "eligible_comparison": not issues,
        "issues": issues,
        "baseline": spec["baseline"],
        "candidate": spec["candidate"],
        "primary_metric": "per-run p95 client end-to-end latency",
        "aggregation": "median of trial metrics; each paired change compares the same trial",
        "consistent_improvement_observed": bool(pairs)
        and all(
            p["e2e_p95_reduction_percent"] > 0 and p["throughput_increase_percent"] >= 0
            for p in pairs
        ),
        "paired_changes": pairs,
        "median_paired_e2e_p95_reduction_percent": statistics.median(
            p["e2e_p95_reduction_percent"] for p in pairs
        )
        if pairs
        else None,
        "median_paired_throughput_increase_percent": statistics.median(
            p["throughput_increase_percent"] for p in pairs
        )
        if pairs
        else None,
        "variants": aggregates,
        "runs": rows,
    }


def plot_results(root, output, report):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError(
            "Plots require the analysis extra: uv sync --extra analysis"
        ) from exc
    names = list(report["variants"])
    colors = ["#2563eb", "#ea580c", "#059669"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, field, label in zip(
        axes,
        ["e2e_p95_ms", "tokens_per_second"],
        ["p95 end-to-end latency (ms)", "Output tokens / second"],
    ):
        for index, name in enumerate(names):
            rows = sorted(
                (r for r in report["runs"] if r["variant"] == name),
                key=lambda r: r["trial"],
            )
            for j, row in enumerate(rows):
                axis.scatter(
                    index + (j - (len(rows) - 1) / 2) * 0.08,
                    row[field],
                    color=colors[j % len(colors)],
                    s=45,
                    label=f"Trial {j + 1}" if index == 0 else None,
                )
            median = report["variants"][name][field]["median"]
            axis.hlines(
                median, index - 0.24, index + 0.24, color="#111827", linewidth=2
            )
        axis.set_xticks(range(len(names)), names)
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Controlled admission comparison · dots are trials, bars are medians")
    fig.savefig(output / "latency-throughput.png", dpi=180)
    plt.close(fig)
    if not any(r["gpu_unique_samples"] for r in report["runs"]):
        return
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(4 * len(names), 3.5),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for axis, name in zip(axes[0], names):
        for row in report["runs"]:
            if row["variant"] != name:
                continue
            directory = run_directory(Path(root), row["output_dir"])
            path = directory / "gpu.jsonl"
            if not path.is_file():
                continue
            summary = read_json(directory / "summary.json")["runs"][0]
            duration = summary["gpu"]["window_duration_s"]
            samples = [json.loads(line) for line in path.read_text().splitlines()]
            samples = list(
                {
                    s["provider_sample_at"]: s
                    for s in samples
                    if s["status"] == "ok"
                    and 0 <= s["provider_sample_offset_s"] <= duration
                }.values()
            )
            axis.plot(
                [s["provider_sample_offset_s"] for s in samples],
                [
                    statistics.mean(g["gpu_util_percent"] for g in s["gpus"])
                    for s in samples
                ],
                marker=".",
                color=colors[(row["trial"] - 1) % len(colors)],
                label=f"Trial {row['trial']}",
            )
        axis.set_title(name)
        axis.set_xlabel("Seconds since measured run start")
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Runpod-reported GPU utilization (%)")
    axes[0][-1].legend(frameon=False)
    fig.savefig(output / "gpu-utilization.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="inference-load-lab analyze", description=__doc__
    )
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = analyze_experiment(args.experiment)
        output = args.output_dir or args.experiment / "analysis"
        output.mkdir(parents=True, exist_ok=False)
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        with (output / "trials.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(report["runs"][0]))
            writer.writeheader()
            writer.writerows(report["runs"])
        if not args.no_plots and report["eligible_comparison"]:
            plot_results(args.experiment, output, report)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(
        f"Eligible comparison: {report['eligible_comparison']}; consistent improvement observed: {report['consistent_improvement_observed']}"
    )
    print(f"Analysis saved: {output.resolve()}")
    if not report["eligible_comparison"]:
        parser.exit(
            1,
            "Comparison checks failed; inspect comparison.json before using the figures.\n",
        )
