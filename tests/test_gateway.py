import asyncio
import gzip
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from starlette.requests import ClientDisconnect

from inference_load_lab.client import generate
from inference_load_lab.gateway import GatewayConfig, create_app

BODY = {
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": True,
}
SSE = b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n'


class HeldStream(httpx.AsyncByteStream):
    def __init__(self, release):
        self.release = release
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"start"}}]}\n\n'
        await self.release.wait()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def gateway(respond, **options):
    config = GatewayConfig("https://upstream.test", **options)
    app = create_app(
        config, "upstream-key", "frontend-key", transport=httpx.MockTransport(respond)
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
            headers={"Authorization": "Bearer frontend-key"},
        ) as client,
    ):
        yield app, client


async def wait_queued(app, number):
    async def wait():
        while (await app.state.gate.snapshot())["queued_requests"] != number:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


def raw_scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "server": ("gateway.test", 80),
        "client": ("127.0.0.1", 12345),
        "headers": [
            (b"authorization", b"Bearer frontend-key"),
            (b"content-type", b"application/json"),
        ],
    }


@pytest.mark.asyncio
async def test_streaming_client_works_and_upstream_uses_its_own_key():
    def respond(request):
        assert request.headers["authorization"] == "Bearer upstream-key"
        assert request.url == "https://upstream.test/v1/chat/completions"
        assert json.loads(request.content)["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200, content=SSE, headers={"Content-Type": "text/event-stream"}
        )

    async with gateway(respond) as (app, client):
        result = await generate(
            client,
            base_url="http://gateway.test",
            model="qwen3-4b",
            prompt="hello",
            api_key="frontend-key",
        )
        assert result["text"] == "answer"
        assert result["output_tokens"] == 2
        assert result["queue_wait_ms"] >= 0
        assert result["request_manager"]["concurrency_limit"] == 4
        assert (await app.state.gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_slot_is_held_for_entire_stream_and_overflow_is_rejected():
    release = asyncio.Event()
    entered = asyncio.Event()
    calls = []
    streams = []

    def respond(request):
        calls.append(json.loads(request.content)["messages"][0]["content"])
        body = HeldStream(release)
        streams.append(body)
        entered.set()
        return httpx.Response(
            200, stream=body, headers={"Content-Type": "text/event-stream"}
        )

    async with gateway(respond, concurrency=1, max_queue=1) as (app, client):
        first = asyncio.create_task(client.post("/v1/chat/completions", json=BODY))
        await asyncio.wait_for(entered.wait(), timeout=2)
        second_body = {**BODY, "messages": [{"role": "user", "content": "second"}]}
        second = asyncio.create_task(
            client.post("/v1/chat/completions", json=second_body)
        )
        try:
            await wait_queued(app, 1)
            overflow = await client.post("/v1/chat/completions", json=BODY)
            assert overflow.status_code == 429
            assert overflow.json()["error"]["type"] == "queue_full"
            assert overflow.headers["retry-after"] == "1"
            assert calls == ["hello"]
            assert (await app.state.gate.snapshot())["active_requests"] == 1
        finally:
            release.set()
        responses = await asyncio.gather(first, second)
        assert [r.status_code for r in responses] == [200, 200]
        assert calls == ["hello", "second"]
        assert float(responses[1].headers["x-queue-wait-ms"]) > 0
        assert all(stream.closed for stream in streams)
        assert (await app.state.gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_queue_timeout_does_not_send_request_upstream():
    release = asyncio.Event()
    entered = asyncio.Event()
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        entered.set()
        return httpx.Response(200, stream=HeldStream(release))

    async with gateway(respond, concurrency=1, queue_timeout=0.02) as (app, client):
        first = asyncio.create_task(client.post("/v1/chat/completions", json=BODY))
        await asyncio.wait_for(entered.wait(), timeout=2)
        try:
            response = await client.post("/v1/chat/completions", json=BODY)
            assert response.status_code == 504
            assert response.json()["error"]["type"] == "queue_timeout"
            assert calls == 1
            assert (await app.state.gate.snapshot())["queued_requests"] == 0
        finally:
            release.set()
            await first


@pytest.mark.asyncio
async def test_disconnected_waiter_is_removed_before_admission():
    release = asyncio.Event()
    entered = asyncio.Event()
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        entered.set()
        return httpx.Response(200, stream=HeldStream(release))

    async with gateway(respond, concurrency=1) as (app, client):
        active = asyncio.create_task(client.post("/v1/chat/completions", json=BODY))
        await asyncio.wait_for(entered.wait(), timeout=2)
        incoming = asyncio.Queue()
        incoming.put_nowait(
            {
                "type": "http.request",
                "body": json.dumps(BODY).encode(),
                "more_body": False,
            }
        )
        sent = []

        async def send(message):
            sent.append(message)

        queued = asyncio.create_task(app(raw_scope(), incoming.get, send))
        try:
            await wait_queued(app, 1)
            incoming.put_nowait({"type": "http.disconnect"})
            await asyncio.wait_for(queued, timeout=2)
            assert (await app.state.gate.snapshot())["queued_requests"] == 0
            assert calls == 1
            assert sent[0]["status"] == 499
        finally:
            release.set()
            await active


@pytest.mark.asyncio
async def test_disconnected_stream_closes_upstream_and_releases_slot():
    body = HeldStream(asyncio.Event())
    async with gateway(
        lambda request: httpx.Response(200, stream=body), concurrency=1
    ) as (app, _client):
        incoming = asyncio.Queue()
        incoming.put_nowait(
            {
                "type": "http.request",
                "body": json.dumps(BODY).encode(),
                "more_body": False,
            }
        )

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                raise OSError("client closed the socket")

        with pytest.raises(ClientDisconnect):
            await app(raw_scope(), incoming.get, send)
        assert body.closed
        assert (await app.state.gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_cancel_during_upstream_headers_releases_granted_slot():
    entered = asyncio.Event()

    async def respond(request):
        entered.set()
        await asyncio.Event().wait()

    async with gateway(respond, concurrency=1) as (app, client):
        task = asyncio.create_task(client.post("/v1/chat/completions", json=BODY))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await app.state.gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_upstream_header_timeout_releases_slot():
    async def respond(request):
        await asyncio.sleep(1)

    async with gateway(respond, upstream_timeout=0.01) as (app, client):
        response = await client.post("/v1/chat/completions", json=BODY)
        assert response.status_code == 504
        assert response.json()["error"]["type"] == "upstream_timeout"
        assert (await app.state.gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_stream_timeout_is_an_error_event_and_not_a_successful_completion():
    body = HeldStream(asyncio.Event())
    async with gateway(
        lambda request: httpx.Response(200, stream=body), upstream_timeout=0.02
    ) as (app, client):
        with pytest.raises(RuntimeError, match="Upstream stream timed out"):
            await generate(
                client,
                base_url="http://gateway.test",
                model="qwen3-4b",
                prompt="hello",
                api_key="frontend-key",
            )
        assert body.closed
        assert (await app.state.gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_upstream_errors_do_not_leak_admission_slots():
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("offline")
        return httpx.Response(503, json={"error": "unavailable"})

    async with gateway(respond, concurrency=1) as (app, client):
        assert (await client.post("/v1/chat/completions", json=BODY)).status_code == 502
        assert (await client.post("/v1/chat/completions", json=BODY)).status_code == 503
        assert (await app.state.gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_nonstreaming_json_and_encoded_upstream_responses_are_forwarded():
    payload = {"choices": [{"message": {"content": "ordinary response"}}]}

    def respond(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            content=gzip.compress(json.dumps(payload).encode()),
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
                "x-request-id": "request-123",
            },
        )

    async with gateway(respond) as (app, client):
        response = await client.post(
            "/v1/chat/completions", json={**BODY, "stream": False}
        )
        assert response.json() == payload
        assert "content-encoding" not in response.headers
        assert response.headers["x-request-id"] == "request-123"
        assert (await app.state.gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_midstream_network_error_is_reported_without_exposing_upstream_key():
    class FailedStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("connection lost: upstream-key")

        async def aclose(self):
            self.closed = True

    body = FailedStream()
    async with gateway(lambda request: httpx.Response(200, stream=body)) as (
        app,
        client,
    ):
        response = await client.post("/v1/chat/completions", json=BODY)
        assert '"type":"upstream_error"' in response.text
        assert "upstream-key" not in response.text
        assert body.closed
        assert (await app.state.gate.snapshot())["active_requests"] == 0


@pytest.mark.asyncio
async def test_auth_validation_and_model_list_do_not_enter_inference_queue():
    calls = []

    def respond(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"data": [{"id": "qwen3-4b"}]})

    async with gateway(respond, max_request_bytes=512) as (_app, client):
        invalid = await client.post(
            "/v1/chat/completions",
            json=BODY,
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert invalid.status_code == 401
        assert (
            await client.post("/v1/chat/completions", content=b"not-json")
        ).status_code == 400
        assert (
            await client.post("/v1/chat/completions", content=b"x" * 513)
        ).status_code == 413
        assert (
            await client.post("/v1/chat/completions", json={"stream": "true"})
        ).status_code == 400
        assert (await client.get("/v1/models")).json()["data"][0]["id"] == "qwen3-4b"
        stats = (await client.get("/stats")).json()
        assert stats["active_requests"] == stats["queued_requests"] == 0
        assert calls == ["/v1/models"]
