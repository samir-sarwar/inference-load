# Saved benchmark artifacts

These captures retain request-level data, workload settings, validation logs,
and environment details for later analysis and a portfolio write-up. Directory
timestamps use UTC.

| Capture | Contents |
| --- | --- |
| [Length-aware scheduling](scheduling-validation-20261108T231211.537306Z/capture.json) | 100 automated tests and four paired FIFO/length-aware trials on mixed short/long bursts. |
| [Final scope validation](final-scope-20261108T221323.184479Z/capture.json) | 84 automated tests, a package build, and the completed nine-run admission experiment. |
| [GPU telemetry diagnostic](final-scope-20261108T220557.471002Z/capture.json) | Intentionally interrupted initial experiment; cached runtime readings led to adding provider timestamps and stale/duplicate sample handling. |
| [Arrival workloads](arrivals-validation-20261108T185225.795608Z/capture.json) | 58 automated tests and five live rate, burst, queue-rejection, and driver-cap cases. |
| [FIFO validation](validation-20261108T165657.214596Z/capture.json) | Earlier gateway tests and repeated direct/FIFO benchmark captures. |

The completed uniform-workload admission comparison uses Qwen3-4B-Instruct-2507 on one A40,
the pinned vLLM 0.11.2 image and model revision, and the checked-in
[experiment plan](../configs/admission-experiment.json). Each of three variants
runs three times at client concurrency 32, with 300 measured requests and
32 excluded warmups per run. Every request uses a fixed 128-token decode budget.
The complete dataset contains 2,700 measured requests and 288 warmups.

The prespecified comparison is FIFO admission concurrency 4 versus concurrency
16, with direct vLLM included as a reference. Figures show individual trials;
aggregate values in the numeric export are medians of trial metrics. The
comparison's checks and paired changes are recorded in its JSON export.

- [Comparison checks and paired changes](final-scope-20261108T221323.184479Z/experiment/analysis/comparison.json)
- [Per-trial metrics CSV](final-scope-20261108T221323.184479Z/experiment/analysis/trials.csv)
- [Latency and throughput figure](final-scope-20261108T221323.184479Z/experiment/analysis/latency-throughput.png)
- [GPU sample figure](final-scope-20261108T221323.184479Z/experiment/analysis/gpu-utilization.png)
- [Run order and settings](final-scope-20261108T221323.184479Z/experiment/experiment.json)
- [Source hashes](final-scope-20261108T221323.184479Z/experiment/source-manifest.json)
- [Automated test log](final-scope-20261108T221323.184479Z/validation/pytest.log)
- [JUnit results](final-scope-20261108T221323.184479Z/validation/junit.xml)
- [GPU shutdown read-back](final-scope-20261108T221323.184479Z/pod-after-stop.json)

Each run directory retains its config, request JSONL, summary, and raw GPU polls.
The experiment directory also includes vLLM metrics snapshots, server version,
model metadata, gateway logs, and final gateway counts. Provider GPU samples
have coarse refresh timing. Summaries deduplicate their timestamps and include
only samples taken inside the measured window; coverage remains visible in
each run's summary and in the CSV.

The GPU was stopped after final validation. Reproduction commands and metric
definitions are in the project [README](../README.md).

The later scheduling experiment holds concurrency at 16 and changes only the
pending-request policy. It completed 2,400 measured requests and 256 warmups,
with zero measured failures or drops. Its prespecified metric is short-class
p95 end-to-end latency; long-class latency and total throughput remain visible
as trade-offs. The experiment uses a different workload from the uniform test.

- [Scheduling comparison JSON](scheduling-validation-20261108T231211.537306Z/experiment/analysis/comparison.json)
- [Scheduling trial CSV](scheduling-validation-20261108T231211.537306Z/experiment/analysis/trials.csv)
- [Short/long latency figure](scheduling-validation-20261108T231211.537306Z/experiment/analysis/class-latencies.png)
- [Actual arrival/admission rank figure](scheduling-validation-20261108T231211.537306Z/experiment/analysis/admission-order.png)
- [Scheduling source hashes](scheduling-validation-20261108T231211.537306Z/experiment/source-manifest.json)
- [Scheduling test log](scheduling-validation-20261108T231211.537306Z/validation/pytest.log)
- [Scheduling GPU shutdown read-back](scheduling-validation-20261108T231211.537306Z/pod-after-stop.json)
