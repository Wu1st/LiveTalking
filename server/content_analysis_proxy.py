"""Same-origin proxy for the private conversation-analysis service.

The browser never receives the upstream API key.  LiveTalking accepts the
uploaded audio, applies a strict size limit, and forwards only the supported
form fields to the loopback-only content-analysis API.
"""

import asyncio
import json
import os
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, FormData, web


MAX_AUDIO_BYTES = 80 * 1024 * 1024
MAX_FIELD_CHARS = 10_000
FORWARDED_FIELDS = {
    "analysis_mode",
    "question",
    "tasks",
    "language",
    "content_type_hint",
    "background_context",
    "speaker_aliases",
}


def _upstream_url() -> str:
    base_url = os.getenv(
        "LIVETALKING_CONTENT_ANALYSIS_URL", "http://127.0.0.1:18081"
    ).rstrip("/")
    return f"{base_url}/v1/audio/analyze"


def _api_key() -> str:
    configured = os.getenv("LIVETALKING_CONTENT_ANALYSIS_API_KEY", "").strip()
    if configured:
        return configured
    key_file = Path(
        os.getenv(
            "LIVETALKING_CONTENT_ANALYSIS_API_KEY_FILE",
            str(Path.home() / ".config" / "content-analysis-api.key"),
        )
    )
    try:
        return key_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


async def _read_request(request: web.Request):
    if not request.content_type.startswith("multipart/"):
        raise web.HTTPUnsupportedMediaType(text="请使用 multipart/form-data 上传音频")
    if request.content_length and request.content_length > MAX_AUDIO_BYTES + 64 * 1024:
        raise web.HTTPRequestEntityTooLarge(
            max_size=MAX_AUDIO_BYTES, actual_size=request.content_length
        )

    reader = await request.multipart()
    audio = None
    filename = "conversation.audio"
    content_type = "application/octet-stream"
    fields = {}

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            if audio is not None:
                raise web.HTTPBadRequest(text="一次只能上传一个音频文件")
            filename = Path(part.filename or filename).name[:255]
            content_type = part.headers.get("Content-Type", content_type)
            chunks = bytearray()
            while True:
                chunk = await part.read_chunk()
                if not chunk:
                    break
                chunks.extend(chunk)
                if len(chunks) > MAX_AUDIO_BYTES:
                    raise web.HTTPRequestEntityTooLarge(
                        max_size=MAX_AUDIO_BYTES, actual_size=len(chunks)
                    )
            audio = bytes(chunks)
        elif part.name in FORWARDED_FIELDS:
            value = (await part.text()).strip()
            if len(value) > MAX_FIELD_CHARS:
                raise web.HTTPBadRequest(text=f"字段 {part.name} 过长")
            if value:
                fields[part.name] = value

    if not audio:
        raise web.HTTPBadRequest(text="没有收到音频文件")
    return audio, filename, content_type, fields


async def analyze_conversation_audio(request: web.Request):
    """Forward an uploaded audio file to the private multimodal analyzer."""
    key = _api_key()
    if not key:
        return web.json_response(
            {"detail": "对话分析服务尚未配置，请联系管理员"}, status=503
        )

    try:
        audio, filename, content_type, fields = await _read_request(request)
    except web.HTTPException as exc:
        return web.json_response({"detail": exc.text}, status=exc.status)

    form = FormData()
    form.add_field("file", audio, filename=filename, content_type=content_type)
    for name, value in fields.items():
        form.add_field(name, value)

    timeout_seconds = float(
        os.getenv("LIVETALKING_CONTENT_ANALYSIS_TIMEOUT", "600")
    )
    timeout = ClientTimeout(total=max(30.0, timeout_seconds), connect=15.0)
    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.post(
                _upstream_url(), data=form, headers={"X-API-Key": key}
            ) as response:
                body = await response.read()
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return web.json_response(
                        {"detail": "对话分析服务返回了无法解析的响应"}, status=502
                    )
                return web.json_response(payload, status=response.status)
    except asyncio.TimeoutError:
        return web.json_response({"detail": "对话分析超时，请稍后重试"}, status=504)
    except ClientError:
        return web.json_response({"detail": "暂时无法连接对话分析服务"}, status=502)
