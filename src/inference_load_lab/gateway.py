"""Serve a FIFO admission gateway in front of one vLLM server."""

import argparse
import asyncio
import json
import math
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import anyio
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from inference_load_lab.admission import (
    AdmissionClosed,
    FifoAdmission,
    Lease,
    QueueFull,
    QueueTimedOut,
)


@dataclass(frozen=True)
class GatewayConfig:
    upstream_base_url: str
    concurrency: int = 4
    max_queue: int = 64
    queue_timeout: float = 30.0
    upstream_timeout: float = 120.0
    max_request_bytes: int = 1024 * 1024

    def __post_init__(self):
        url = httpx.URL(self.upstream_base_url)
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("The upstream must be an absolute HTTP or HTTPS URL.")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError(
                "The upstream URL cannot contain credentials, a query, or a fragment."
            )
        if self.concurrency < 1 or self.max_queue < 0 or self.max_request_bytes < 1:
            raise ValueError(
                "Concurrency and body limit must be positive; queue size may be zero."
            )
        if any(
            not math.isfinite(t) or t <= 0
            for t in (self.queue_timeout, self.upstream_timeout)
        ):
            raise ValueError(
                "Queue and upstream timeouts must be finite positive numbers."
            )


@dataclass
class Resources:
    lease: Lease | None = None
    response: httpx.Response | None = None
    helpers: list[asyncio.Task] = field(default_factory=list)

    async def close(self):
        try:
            for task in self.helpers:
                task.cancel()
            if self.helpers:
                await asyncio.gather(*self.helpers, return_exceptions=True)
            if self.response is not None:
                await self.response.aclose()
        finally:
            if self.lease is not None:
                await self.lease.release()


class ResourceCleanupMiddleware:
    """Clean up even when response headers fail or a streaming client disconnects."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        resources = Resources()
        scope.setdefault("state", {})["gateway_resources"] = resources
        try:
            await self.app(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await resources.close()


def error(status: int, message: str, kind: str, *, retry: bool = False) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind}},
        status_code=status,
        headers={"Retry-After": "1"} if retry else None,
    )


def forward_headers(response: httpx.Response) -> dict:
    # aiter_bytes()/aread() decode content encodings; hop-by-hop headers and the
    # original content length must not accompany the decoded downstream body.
    return {
        k: v
        for k, v in response.headers.items()
        if k in {"content-type", "cache-control", "retry-after", "x-request-id"}
    }


async def wait_for_disconnect(request: Request):
    # Called only after the entire request body has been consumed.
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


def create_app(
    config: GatewayConfig,
    upstream_api_key: str,
    gateway_api_key: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    if not upstream_api_key or not gateway_api_key:
        raise ValueError("Both upstream and gateway API keys are required.")
    gate = FifoAdmission(config.concurrency, config.max_queue)
    public_config = {
        "policy": "fifo",
        "concurrency_limit": config.concurrency,
        "queue_capacity": config.max_queue,
        "queue_timeout_s": config.queue_timeout,
        "upstream_timeout_s": config.upstream_timeout,
    }

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(
                config.upstream_timeout, connect=min(10.0, config.upstream_timeout)
            ),
            limits=httpx.Limits(
                max_connections=config.concurrency + 1,
                max_keepalive_connections=config.concurrency + 1,
            ),
        ) as client:
            app.state.upstream = client
            try:
                yield
            finally:
                await gate.close()

    def authorized(request):
        supplied = request.headers.get("authorization", "")
        return secrets.compare_digest(
            supplied.encode(), f"Bearer {gateway_api_key}".encode()
        )

    async def health(request):
        return JSONResponse({"status": "ok"})

    async def stats(request):
        if not authorized(request):
            return error(401, "Invalid API key.", "authentication_error")
        return JSONResponse({**public_config, **await gate.snapshot()})

    async def models(request):
        if not authorized(request):
            return error(401, "Invalid API key.", "authentication_error")
        try:
            async with asyncio.timeout(config.upstream_timeout):
                response = await request.app.state.upstream.get(
                    config.upstream_base_url.rstrip("/") + "/v1/models",
                    headers={"Authorization": f"Bearer {upstream_api_key}"},
                )
            return Response(
                response.content,
                status_code=response.status_code,
                headers=forward_headers(response),
            )
        except (TimeoutError, httpx.TimeoutException):
            return error(
                504, "The upstream model-list request timed out.", "upstream_timeout"
            )
        except httpx.HTTPError:
            return error(502, "Could not reach the upstream server.", "upstream_error")

    async def completion(request):
        if not authorized(request):
            return error(401, "Invalid API key.", "authentication_error")
        body = bytearray()
        try:
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > config.max_request_bytes:
                    return error(413, "Request body is too large.", "invalid_request")
            payload = json.loads(body)
        except ClientDisconnect:
            return Response(status_code=499)
        except (ValueError, UnicodeError):
            return error(400, "Request body must be a JSON object.", "invalid_request")
        if not isinstance(payload, dict) or not isinstance(
            payload.get("stream", False), bool
        ):
            return error(
                400,
                "Expected a JSON object with a boolean stream option.",
                "invalid_request",
            )

        resources = request.state.gateway_resources
        deadline = None

        async def open_upstream():
            nonlocal deadline
            resources.lease = await gate.acquire(config.queue_timeout)
            deadline = asyncio.get_running_loop().time() + config.upstream_timeout
            upstream_request = request.app.state.upstream.build_request(
                "POST",
                config.upstream_base_url.rstrip("/") + "/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {upstream_api_key}",
                    "Accept-Encoding": "identity",
                },
                json=payload,
            )
            async with asyncio.timeout_at(deadline):
                resources.response = await request.app.state.upstream.send(
                    upstream_request, stream=True
                )

        operation = asyncio.create_task(open_upstream())
        disconnect = asyncio.create_task(wait_for_disconnect(request))
        resources.helpers.extend([operation, disconnect])
        try:
            finished, _ = await asyncio.wait(
                [operation, disconnect], return_when=asyncio.FIRST_COMPLETED
            )
            if disconnect in finished:
                return Response(status_code=499)
            await operation
        except QueueFull:
            return error(429, "The request queue is full.", "queue_full", retry=True)
        except QueueTimedOut:
            return error(
                504, "Timed out waiting in the request queue.", "queue_timeout"
            )
        except AdmissionClosed:
            return error(
                503, "The request manager is shutting down.", "unavailable", retry=True
            )
        except (TimeoutError, httpx.TimeoutException):
            return error(504, "The upstream request timed out.", "upstream_timeout")
        except httpx.HTTPError:
            return error(502, "Could not reach the upstream server.", "upstream_error")
        finally:
            disconnect.cancel()
            await asyncio.gather(disconnect, return_exceptions=True)

        response = resources.response
        headers = forward_headers(response)
        headers.update(
            {
                "X-Queue-Wait-Ms": f"{resources.lease.queue_wait_ms:.6f}",
                "X-Admission-Policy": "fifo",
                "X-Request-Manager": json.dumps(public_config, separators=(",", ":")),
            }
        )
        if not payload.get("stream") or response.status_code != 200:
            try:
                async with asyncio.timeout_at(deadline):
                    content = await response.aread()
                return Response(
                    content, status_code=response.status_code, headers=headers
                )
            except (TimeoutError, httpx.TimeoutException):
                return error(
                    504, "The upstream response timed out.", "upstream_timeout"
                )
            except httpx.HTTPError:
                return error(502, "The upstream response failed.", "upstream_error")

        async def stream_body():
            iterator = response.aiter_bytes().__aiter__()
            try:
                while True:
                    try:
                        async with asyncio.timeout_at(deadline):
                            chunk = await anext(iterator)
                    except StopAsyncIteration:
                        return
                    yield chunk
            except (TimeoutError, httpx.TimeoutException):
                yield b'data: {"error":{"message":"Upstream stream timed out.","type":"upstream_timeout"}}\n\n'
            except httpx.HTTPError:
                yield b'data: {"error":{"message":"Upstream stream failed.","type":"upstream_error"}}\n\n'

        return StreamingResponse(
            stream_body(), status_code=response.status_code, headers=headers
        )

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/stats", stats),
            Route("/v1/models", models),
            Route("/v1/chat/completions", completion, methods=["POST"]),
        ],
        middleware=[Middleware(ResourceCleanupMiddleware)],
        lifespan=lifespan,
    )
    app.state.gate = gate
    app.state.gateway_config = config
    return app


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="inference-load-lab serve", description=__doc__
    )
    parser.add_argument("--upstream-base-url", default=os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--queue-timeout", type=float, default=30.0)
    parser.add_argument("--upstream-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    upstream_key = os.environ.get("VLLM_API_KEY", "")
    gateway_key = os.environ.get("GATEWAY_API_KEY", upstream_key)
    if not args.upstream_base_url or not upstream_key or not gateway_key:
        parser.error(
            "Set VLLM_BASE_URL and VLLM_API_KEY; GATEWAY_API_KEY optionally selects a separate client key."
        )
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    try:
        config = GatewayConfig(
            args.upstream_base_url.rstrip("/"),
            args.concurrency,
            args.queue_size,
            args.queue_timeout,
            args.upstream_timeout,
        )
        url = httpx.URL(config.upstream_base_url)
        upstream_port = url.port or (443 if url.scheme == "https" else 80)
        if url.host in {"127.0.0.1", "localhost", "::1"} and upstream_port == args.port:
            parser.error("The upstream URL points to this gateway's own port.")
        app = create_app(config, upstream_key, gateway_key)
    except (ValueError, httpx.InvalidURL) as exc:
        parser.error(str(exc))
    uvicorn.run(
        app, host=args.host, port=args.port, workers=1, timeout_graceful_shutdown=10
    )


if __name__ == "__main__":
    main()
