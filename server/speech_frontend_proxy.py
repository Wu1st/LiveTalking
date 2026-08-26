"""Proxy LiveTalking requests to the local lightweight speech-frontend sidecar."""

from __future__ import annotations

import os

import aiohttp
from aiohttp import web


CLIENT_KEY = web.AppKey("speech_frontend_proxy_client", aiohttp.ClientSession)
BASE_URL = os.getenv("SPEECH_FRONTEND_URL", "http://127.0.0.1:8092").rstrip("/")


async def speech_frontend_proxy_context(app: web.Application):
    timeout = aiohttp.ClientTimeout(total=120)
    app[CLIENT_KEY] = aiohttp.ClientSession(timeout=timeout)
    try:
        yield
    finally:
        await app[CLIENT_KEY].close()


def _response_headers(response: aiohttp.ClientResponse) -> dict[str, str]:
    headers = {}
    for key, value in response.headers.items():
        if key.lower() == "content-type" or key.lower().startswith("x-"):
            headers[key] = value
    return headers


async def _proxy_upload(request: web.Request, target_path: str) -> web.Response:
    source = await request.post()
    target = aiohttp.FormData()
    for key, value in source.items():
        if hasattr(value, "file"):
            target.add_field(
                key,
                value.file.read(),
                filename=getattr(value, "filename", "audio") or "audio",
                content_type=getattr(value, "content_type", None) or "application/octet-stream",
            )
        else:
            target.add_field(key, str(value))
    async with request.app[CLIENT_KEY].post(
        f"{BASE_URL}{target_path}",
        params=request.query,
        data=target,
    ) as upstream:
        return web.Response(
            status=upstream.status,
            body=await upstream.read(),
            headers=_response_headers(upstream),
        )


async def health(request: web.Request) -> web.Response:
    try:
        async with request.app[CLIENT_KEY].get(f"{BASE_URL}/health") as upstream:
            return web.Response(
                status=upstream.status,
                body=await upstream.read(),
                headers=_response_headers(upstream),
            )
    except Exception as error:
        return web.json_response({"code": -1, "msg": f"speech frontend unavailable: {error}"})


async def analyze(request: web.Request) -> web.Response:
    return await _proxy_upload(request, "/api/v1/speech_frontend/analyze")


async def prepare(request: web.Request) -> web.Response:
    return await _proxy_upload(request, "/api/v1/speech_frontend/prepare")


def setup_speech_frontend_proxy_routes(app: web.Application) -> None:
    app.cleanup_ctx.append(speech_frontend_proxy_context)
    app.router.add_get("/api/speech_frontend/health", health)
    app.router.add_post("/api/speech_frontend/analyze", analyze)
    app.router.add_post("/api/speech_frontend/prepare", prepare)
