import json

import httpx
import pytest

from inference_load_lab.client import generate


def event(data):
    return f"data: {json.dumps(data)}\n\n"


@pytest.mark.asyncio
async def test_stream_uses_content_for_ttft_and_server_usage_for_tokens(monkeypatch):
    clock = iter([10.0, 12.0, 15.0])
    monkeypatch.setattr("inference_load_lab.client.perf_counter", lambda: next(clock))
    body = (
        event({"choices": [{"delta": {"role": "assistant"}}]})
        + event({"choices": [{"delta": {"content": "Hello "}}]})
        + event({"choices": [{"delta": {"content": "world"}}]})
        + event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + event({"choices": [], "usage": {"completion_tokens": 7}})
        + "data: [DONE]\n\n"
    )

    def respond(request):
        assert request.url == "https://example.test/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True
        return httpx.Response(200, text=body)

    chunks = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await generate(
            client,
            base_url="https://example.test/",
            model="qwen3-4b",
            prompt="Hello",
            api_key="test-key",
            on_text=chunks.append,
        )

    assert result["text"] == "Hello world"
    assert chunks == ["Hello ", "world"]
    assert result["ttft_ms"] == 2000
    assert result["e2e_ms"] == 5000
    assert result["output_tokens"] == 7
    assert result["finish_reason"] == "stop"
    assert result["queue_wait_ms"] is None


@pytest.mark.asyncio
async def test_reads_gateway_queue_wait_header():
    body = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, text=body, headers={"X-Queue-Wait-Ms": "12.5"}
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        result = await generate(
            client,
            base_url="https://example.test",
            model="qwen3-4b",
            prompt="hello",
            api_key="test-key",
        )
    assert result["queue_wait_ms"] == 12.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (401, "Invalid API key", "HTTP 401: Invalid API key"),
        (200, event({"error": {"message": "Engine failed"}}), "Engine failed"),
        (
            200,
            event({"choices": [{"delta": {"content": "partial"}}]}),
            "stream ended before",
        ),
    ],
)
async def test_failed_or_incomplete_requests_are_not_reported_as_success(
    status, body, message
):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=body))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match=message):
            await generate(
                client,
                base_url="https://example.test",
                model="qwen3-4b",
                prompt="Hello",
                api_key="test-key",
            )
