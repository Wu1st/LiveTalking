"""Expose the OCR/translation document service through LiveTalking.

The OCR runtime remains an independent service because it owns large models,
GPU scheduling, checkpoints, and persistent job state.  LiveTalking presents a
single public HTTP surface and streams request/response bodies to that service.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import aiohttp
from aiohttp import web


CLIENT_KEY = web.AppKey("document_proxy_client", aiohttp.ClientSession)
BASE_URL_KEY = web.AppKey("document_proxy_base_url", str)

_REQUEST_HEADERS = {"accept", "content-type", "idempotency-key"}
_RESPONSE_HEADERS = {
    "cache-control",
    "content-disposition",
    "content-length",
    "content-type",
    "etag",
    "last-modified",
}


async def document_proxy_context(app: web.Application):
    timeout = aiohttp.ClientTimeout(
        total=float(os.getenv("OCR_DOCUMENT_PROXY_TIMEOUT_SECONDS", "3600")),
        connect=float(os.getenv("OCR_DOCUMENT_PROXY_CONNECT_TIMEOUT_SECONDS", "10")),
    )
    connector = aiohttp.TCPConnector(limit=32, ttl_dns_cache=300)
    app[CLIENT_KEY] = aiohttp.ClientSession(timeout=timeout, connector=connector)
    try:
        yield
    finally:
        await app[CLIENT_KEY].close()


def _forward_request_headers(request: web.Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() in _REQUEST_HEADERS or key.lower().startswith("x-")
    }


def _forward_response_headers(response: aiohttp.ClientResponse) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() in _RESPONSE_HEADERS or key.lower().startswith("x-")
    }


async def _request_body(request: web.Request) -> AsyncIterator[bytes]:
    async for chunk in request.content.iter_chunked(1024 * 1024):
        yield chunk


async def _proxy(request: web.Request, upstream_path: str) -> web.StreamResponse:
    base_url = request.app[BASE_URL_KEY]
    kwargs: dict[str, object] = {
        "params": request.query,
        "headers": _forward_request_headers(request),
        "allow_redirects": False,
    }
    if request.can_read_body:
        kwargs["data"] = _request_body(request)

    try:
        async with request.app[CLIENT_KEY].request(
            request.method,
            f"{base_url}{upstream_path}",
            **kwargs,
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=_forward_response_headers(upstream),
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(1024 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response
    except (aiohttp.ClientError, TimeoutError) as error:
        return web.json_response(
            {
                "code": -1,
                "msg": "OCR/translation service unavailable",
                "detail": str(error),
            },
            status=503,
        )


def _handler(upstream_path: str):
    async def proxy_handler(request: web.Request) -> web.StreamResponse:
        path = upstream_path.format(**request.match_info)
        return await _proxy(request, path)

    return proxy_handler


def setup_document_proxy_routes(
    app: web.Application,
    *,
    base_url: str | None = None,
) -> None:
    """Register integrated and source-compatible document API routes."""

    app[BASE_URL_KEY] = (
        base_url
        or os.getenv("OCR_DOCUMENT_API_URL", "http://127.0.0.1:8000")
    ).rstrip("/")
    app.cleanup_ctx.append(document_proxy_context)

    routes = [
        # LiveTalking-style integrated surface.
        ("GET", "/api/document/health", "/health"),
        ("POST", "/api/document/image", "/v1/ocr/image"),
        ("POST", "/api/document/jobs", "/v1/document/jobs"),
        ("GET", "/api/document/jobs/{job_id}", "/v1/document/jobs/{job_id}"),
        ("GET", "/api/document/jobs/{job_id}/result", "/v1/document/jobs/{job_id}/result"),
        ("GET", "/api/document/jobs/{job_id}/download", "/v1/document/jobs/{job_id}/download"),
        ("POST", "/api/document/jobs/{job_id}/cancel", "/v1/document/jobs/{job_id}/cancel"),
        ("POST", "/api/document/jobs/{job_id}/resume", "/v1/document/jobs/{job_id}/resume"),
        ("POST", "/api/document/jobs/{job_id}/livetalking", "/v1/document/jobs/{job_id}/livetalking"),
        ("POST", "/api/document/translation-jobs", "/v1/translation/document-jobs"),
        ("GET", "/api/document/translation-jobs/{job_id}", "/v1/translation/document-jobs/{job_id}"),
        ("GET", "/api/document/translation-jobs/{job_id}/result", "/v1/translation/document-jobs/{job_id}/result"),
        ("GET", "/api/document/translation-jobs/{job_id}/download", "/v1/translation/document-jobs/{job_id}/download"),
        ("POST", "/api/document/translation-jobs/{job_id}/cancel", "/v1/translation/document-jobs/{job_id}/cancel"),
        ("POST", "/api/document/translation-jobs/{job_id}/resume", "/v1/translation/document-jobs/{job_id}/resume"),
        # Transparent aliases let existing OCR clients point at LiveTalking:8010.
        ("POST", "/v1/ocr/image", "/v1/ocr/image"),
        ("POST", "/v1/document/jobs", "/v1/document/jobs"),
        ("GET", "/v1/document/jobs/{job_id}", "/v1/document/jobs/{job_id}"),
        ("GET", "/v1/document/jobs/{job_id}/result", "/v1/document/jobs/{job_id}/result"),
        ("GET", "/v1/document/jobs/{job_id}/download", "/v1/document/jobs/{job_id}/download"),
        ("POST", "/v1/document/jobs/{job_id}/cancel", "/v1/document/jobs/{job_id}/cancel"),
        ("POST", "/v1/document/jobs/{job_id}/resume", "/v1/document/jobs/{job_id}/resume"),
        ("POST", "/v1/document/jobs/{job_id}/livetalking", "/v1/document/jobs/{job_id}/livetalking"),
        ("POST", "/v1/translation/document-jobs", "/v1/translation/document-jobs"),
        ("GET", "/v1/translation/document-jobs/{job_id}", "/v1/translation/document-jobs/{job_id}"),
        ("GET", "/v1/translation/document-jobs/{job_id}/result", "/v1/translation/document-jobs/{job_id}/result"),
        ("GET", "/v1/translation/document-jobs/{job_id}/download", "/v1/translation/document-jobs/{job_id}/download"),
        ("POST", "/v1/translation/document-jobs/{job_id}/cancel", "/v1/translation/document-jobs/{job_id}/cancel"),
        ("POST", "/v1/translation/document-jobs/{job_id}/resume", "/v1/translation/document-jobs/{job_id}/resume"),
    ]
    for method, public_path, upstream_path in routes:
        app.router.add_route(method, public_path, _handler(upstream_path))
