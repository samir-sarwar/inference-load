# Inference Load Lab

A small LLM serving and benchmarking project built around vLLM.

The first milestone is one streaming request to a Qwen model on a Runpod GPU Pod.
The FIFO queue and load generator will build on this measured baseline.

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
