"""Send one streaming request to a vLLM server."""

import argparse
import asyncio
import json
import math
import os
from collections.abc import Callable
from time import perf_counter

import httpx


class InferenceHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.server_type = None
        try:
            value = json.loads(body)
        except ValueError:
            return
        if isinstance(value, dict) and isinstance(value.get("error"), dict):
            server_type = value["error"].get("type")
            if isinstance(server_type, str):
                self.server_type = server_type


async def generate(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int = 128,
    ignore_eos: bool = False,
    on_text: Callable[[str], None] | None = None,
) -> dict:
    """Return client-observed timings and the server's output-token count."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    started = perf_counter()
    first_content_at = None
    parts = []
    usage = {}
    finish_reason = None
    completed = False

    async with client.stream(
        "POST",
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    ) as response:
        if response.status_code != 200:
            await response.aread()
            raise InferenceHTTPError(response.status_code, response.text)

        raw_queue_wait = response.headers.get("x-queue-wait-ms")
        queue_wait_ms = float(raw_queue_wait) if raw_queue_wait is not None else None
        if queue_wait_ms is not None and (
            not math.isfinite(queue_wait_ms) or queue_wait_ms < 0
        ):
            raise RuntimeError("Invalid gateway queue-wait measurement.")
        raw_manager = response.headers.get("x-request-manager")
        request_manager = json.loads(raw_manager) if raw_manager is not None else None
        if request_manager is not None and not isinstance(request_manager, dict):
            raise RuntimeError("Invalid request-manager configuration header.")

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                finished = perf_counter()
                completed = True
                break

            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"vLLM stream error: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]

            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                text = choice.get("delta", {}).get("content")
                if text:
                    if first_content_at is None:
                        first_content_at = perf_counter()
                    parts.append(text)
                    if on_text is not None:
                        on_text(text)

    if not completed:
        raise RuntimeError(
            "The stream ended before [DONE]; the request was incomplete."
        )

    output_tokens = usage.get("completion_tokens")
    if output_tokens is not None and (
        type(output_tokens) is not int or output_tokens < 0
    ):
        raise RuntimeError("Invalid server completion-token count.")

    return {
        "model": model,
        "text": "".join(parts),
        "ttft_ms": (
            (first_content_at - started) * 1000
            if first_content_at is not None
            else None
        ),
        "e2e_ms": (finished - started) * 1000,
        "tpot_ms": (
            (finished - first_content_at) * 1000 / (output_tokens - 1)
            if first_content_at is not None
            and output_tokens is not None
            and output_tokens > 1
            else None
        ),
        "output_tokens": output_tokens,
        "queue_wait_ms": queue_wait_ms,
        "request_manager": request_manager,
        "finish_reason": finish_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Other commands: inference-load-lab {bench,serve,experiment,analyze} --help",
    )
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3-4b"))
    parser.add_argument("--prompt", default="Explain KV caching in two sentences.")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Set VLLM_BASE_URL or pass --base-url.")
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        parser.error("Set VLLM_API_KEY in your environment or .env.local.")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive.")

    async def run() -> dict:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0)
        ) as client:
            return await generate(
                client,
                base_url=args.base_url,
                model=args.model,
                prompt=args.prompt,
                api_key=api_key,
                max_tokens=args.max_tokens,
                on_text=lambda text: print(text, end="", flush=True),
            )

    try:
        result = asyncio.run(run())
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"\nRequest failed: {exc}\n")
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "text"}, indent=2))


if __name__ == "__main__":
    main()
