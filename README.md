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
