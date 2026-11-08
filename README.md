# Inference Load Lab

A small LLM serving and benchmarking project built around vLLM.

Send streaming requests to a Qwen model on a Runpod GPU Pod, then run a controlled
concurrency sweep to establish a direct-server baseline before adding a FIFO queue.

## Send a request

Install the locked Python environment:

```bash
uv sync --locked
```

Deployment setup stores the server URL, model alias, and generated API key in
`.env.local`. That file is private and ignored by Git. On another machine, copy
`.env.example` to `.env.local` and fill in the connection details.

```bash
uv run --env-file .env.local inference-load-lab
```

Choose a prompt and output limit:

```bash
uv run --env-file .env.local inference-load-lab \
  --prompt "Explain continuous batching in two sentences." \
  --max-tokens 128
```

The client prints the generated text as it arrives, followed by a JSON summary:

- `ttft_ms`: time from request start to the first nonempty generated content.
- `e2e_ms`: time from request start to the stream's `[DONE]` event.
- `output_tokens`: the server's reported completion-token count.
- `finish_reason`: why generation stopped, such as `stop` or `length`.

These are client-observed timings. They include the network and Runpod proxy.
The first visible content is a practical TTFT proxy, and a streaming chunk can
contain multiple tokens. The client uses the server's token count rather than
counting chunks. An interrupted stream is reported as a failure.

## Run a load test

Start with one concurrency level:

```bash
uv run --env-file .env.local inference-load-lab bench \
  --requests 100 --concurrency 4 --warmup 5 --max-tokens 128 \
  --server-config configs/vllm.json
```

Run the baseline sweep with the included prompt set:

```bash
uv run --env-file .env.local inference-load-lab bench \
  --requests 100 --concurrency 1 2 4 8 16 --warmup 16 \
  --prompts-file data/prompts.jsonl --max-tokens 128 \
  --server-config configs/vllm.json
```

`--requests` is the number of measured requests **at each concurrency level**.
The sweep above sends 500 measured requests and 80 warmup requests. Levels run
sequentially. A fixed number of workers continuously send requests as previous
ones finish, reusing one HTTP client and its connection pool. This is a
closed-loop concurrency test; the offered request rate depends on response time.

Each level replays the same balanced, seeded prompt sequence. Supply repeatable
`--prompt` flags or a JSONL file containing strings or `{"prompt": "..."}` objects.
Use `--seed` to change the measured prompt order, and `--timeout` to set the total
deadline for each request, including its streamed response.

Warmups finish before measurement starts and are excluded from every reported
latency and throughput metric. Choose enough warmup requests to cover your prompt
set and concurrency. Warmups and repeated prompts can populate vLLM's prefix
cache: the default repeated-prompt test is a warm-prefix workload. Keep the
prompts, generation limits, cache policy, hardware, and server settings consistent
when comparing runs. The supplied server config is recorded as declared settings,
not queried from the live server.

The terminal reports successful and failed requests, p50/p95 TTFT and end-to-end
latency, and aggregate output tokens per second. Latency percentiles use successful
requests only, with linear interpolation at `(n - 1) * quantile`. Throughput is
the sum of successful requests' output tokens divided by the measured wall time
from launching the workers until they all finish, including failed requests' time.
It is not the average of individual requests' token rates. If any successful
request lacks a token count, output-token throughput is reported as `n/a`.

Every invocation creates a new, Git-ignored `results/<UTC timestamp>/` directory:

- `config.json`: workload, prompts and their hash, seed, versions, Git state,
  and optional declared server settings. The API key is excluded.
- `requests.jsonl`: one record per completed request, including its phase
  (`warmup` or `measured`), concurrency, request ID, prompt index, timings, token
  count, finish reason, and error details. Generated text is not printed or saved.
- `summary.json`: metrics for each level and the overall completion status.

Use `--output-dir results/my-run` to name a new directory. Existing directories
are refused to protect earlier results. Failed warmups stop the sweep before
measured load; measured request failures are recorded and the command exits
nonzero after the sweep. Ctrl+C preserves completed requests and marks the run as
interrupted. Requests are not automatically retried.

## Server configuration

- Model: `Qwen/Qwen3-4B-Instruct-2507` (public, BF16 weights).
- Model revision: `cdbee75f17c01a7cc42f958dc650907174af0554`.
- vLLM image: `vllm/vllm-openai:v0.11.2`, pinned by digest at deployment.
- API model alias: `qwen3-4b`.
- Context limit: 4,096 tokens, shared by the prompt and output.
- Maximum scheduled sequences: 16.
- GPU memory utilization setting: 0.90.
- The HTTP API listens on `0.0.0.0:8000` and requires a generated bearer key.

`configs/vllm.json` records the exact image digest and launch arguments. Keep
these fixed while comparing the direct-server and FIFO-queue experiments.

vLLM downloads the public model directly from Hugging Face when it starts.
Hugging Face MCP is useful for inspecting the Hub; downloading this public model
does not require a Hugging Face login.

The local `.local/deployment.json` records the Pod ID, image digest, model
revision, and launch arguments after a successful create. No network volume is
created by this setup; cached weights on container disk must be downloaded again
after stopping the Pod.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The tests cover streamed content and metadata, accurate use of the server's
token count, HTTP errors, in-stream errors, and truncated responses. They do not
need a GPU.
Benchmark tests also check bounded concurrency, warmup exclusion, repeatable
workloads, timeout and error reporting, percentile and throughput calculations,
and a full CLI run against a local streaming HTTP server.
