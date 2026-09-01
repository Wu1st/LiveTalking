from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any

from aiohttp import web

from .config import DEFAULT_MODEL, DEFAULT_PROFILES, sha256_file
from .media import decode_audio_bytes, encode_pcm16_wav, extract_speech
from .pipeline import PipelineResult, SpeechFrontendEngine


MAX_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_AUDIO_SECONDS = 30 * 60
ENGINE_KEY: web.AppKey[SpeechFrontendEngine] = web.AppKey("engine", SpeechFrontendEngine)
SEMAPHORE_KEY: web.AppKey[asyncio.Semaphore] = web.AppKey("semaphore", asyncio.Semaphore)
logger = logging.getLogger(__name__)


def json_ok(data: dict | None = None) -> web.Response:
    body: dict[str, Any] = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.json_response(body, dumps=lambda value: json.dumps(value, ensure_ascii=False))


def json_error(message: str, code: int = -1, status: int = 422) -> web.Response:
    return web.json_response(
        {"code": code, "msg": str(message)},
        status=status,
        dumps=lambda value: json.dumps(value, ensure_ascii=False),
    )


@web.middleware
async def error_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        try:
            response = await handler(request)
        except web.HTTPException:
            raise
        except ValueError as error:
            response = json_error(str(error), status=422)
        except Exception as error:
            logger.exception("Speech frontend request failed")
            response = json_error("语音前端处理失败", status=500)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


async def _read_upload(request: web.Request) -> tuple[bytes, dict[str, str]]:
    if request.content_length is not None and request.content_length > MAX_UPLOAD_BYTES:
        raise ValueError("音频文件不能超过 80 MB")
    form = await request.post()
    file_field = form.get("file")
    if file_field is None or not hasattr(file_field, "file"):
        raise ValueError("缺少 multipart 音频字段 file")
    audio_bytes = file_field.file.read(MAX_UPLOAD_BYTES + 1)
    if not audio_bytes:
        raise ValueError("上传的音频为空")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("音频文件不能超过 80 MB")
    values = {
        "sessionid": str(form.get("sessionid", request.query.get("sessionid", "0"))),
        "trace_id": str(
            form.get("client_trace_id", request.query.get("client_trace_id", ""))
            or uuid.uuid4().hex
        ),
        "filename": str(getattr(file_field, "filename", "audio") or "audio"),
        "output": str(form.get("output", request.query.get("output", "speech"))).lower(),
        "include_probabilities": str(
            form.get("include_probabilities", request.query.get("include_probabilities", "0"))
        ).lower(),
        "include_audit": str(form.get("include_audit", request.query.get("include_audit", "0"))).lower(),
        "gap_ms": str(form.get("gap_ms", request.query.get("gap_ms", "120"))),
    }
    return audio_bytes, values


async def _run(request: web.Request) -> tuple[PipelineResult, dict[str, str]]:
    audio_bytes, values = await _read_upload(request)
    loop = asyncio.get_running_loop()
    audio = await loop.run_in_executor(
        None,
        decode_audio_bytes,
        audio_bytes,
        values["filename"],
    )
    if len(audio) / 16000.0 > MAX_AUDIO_SECONDS:
        raise ValueError("音频时长不能超过 30 分钟")
    async with request.app[SEMAPHORE_KEY]:
        result = await loop.run_in_executor(None, request.app[ENGINE_KEY].process, audio)
    return result, values


def _result_data(result: PipelineResult, values: dict[str, str]) -> dict:
    data = {
        "schema_version": 1,
        "sessionid": values["sessionid"],
        "trace_id": values["trace_id"],
        "filename": values["filename"],
        "pipeline": "patent_scheme_d",
        "sample_rate": 16000,
        "sample_count": int(len(result.original)),
        "duration_seconds": result.duration_seconds,
        "speech_interval_count": len(result.intervals),
        "speech_duration_seconds": result.speech_duration_seconds,
        "speech_ratio": result.speech_duration_seconds / max(result.duration_seconds, 1e-12),
        "intervals": result.intervals,
        "timing": {"processing_ms": result.processing_ms, "core_rtf": result.core_rtf},
    }
    if values["include_probabilities"] in {"1", "true", "yes"}:
        data["probabilities"] = {
            "frame_samples": 512,
            "raw": result.raw_probabilities.astype(float).tolist(),
            "wiener": result.wiener_probabilities.astype(float).tolist(),
        }
    if values["include_audit"] in {"1", "true", "yes"}:
        data["patent_audit"] = result.patent_audit
    return data


async def health(request: web.Request) -> web.Response:
    engine = request.app[ENGINE_KEY]
    return json_ok(
        {
            "service": "speech_frontend",
            "status": "ready",
            "pipeline": "patent_scheme_d",
            "sample_rate": 16000,
            "frame_samples": 512,
            "model_sha256": sha256_file(DEFAULT_MODEL),
            "profile_sha256": sha256_file(DEFAULT_PROFILES),
            "product": engine.config["product"],
        }
    )


async def analyze(request: web.Request) -> web.Response:
    result, values = await _run(request)
    return json_ok(_result_data(result, values))


async def prepare(request: web.Request) -> web.Response:
    result, values = await _run(request)
    output = values["output"]
    try:
        gap_ms = min(1000, max(0, int(values["gap_ms"])))
    except ValueError as error:
        raise ValueError("gap_ms 必须是 0–1000 的整数") from error
    if output == "speech":
        selected = extract_speech(result.enhanced, result.intervals, gap_ms=gap_ms)
        if not len(selected):
            response = web.Response(status=204)
        else:
            response = web.Response(body=encode_pcm16_wav(selected), content_type="audio/wav")
    elif output in {"enhanced", "wiener"}:
        selected = result.enhanced
        response = web.Response(body=encode_pcm16_wav(selected), content_type="audio/wav")
    elif output in {"original", "raw"}:
        selected = result.original
        response = web.Response(body=encode_pcm16_wav(selected), content_type="audio/wav")
    else:
        raise ValueError("output 仅支持 speech、enhanced 或 original")
    response.headers.update(
        {
            "X-Session-ID": values["sessionid"],
            "X-Trace-ID": values["trace_id"],
            "X-VAD-Pipeline": "patent_scheme_d",
            "X-VAD-Interval-Count": str(len(result.intervals)),
            "X-VAD-Speech-Seconds": f"{result.speech_duration_seconds:.6f}",
            "X-VAD-Processing-Ms": f"{result.processing_ms:.3f}",
            "X-VAD-Core-RTF": f"{result.core_rtf:.6f}",
            "X-Audio-Output": output,
        }
    )
    return response


async def options(request: web.Request) -> web.Response:
    return web.Response(status=204)


def create_app(
    engine: SpeechFrontendEngine | None = None,
    max_concurrency: int = 2,
) -> web.Application:
    app = web.Application(middlewares=[error_middleware], client_max_size=MAX_UPLOAD_BYTES)
    app[ENGINE_KEY] = engine or SpeechFrontendEngine()
    app[SEMAPHORE_KEY] = asyncio.Semaphore(max(1, max_concurrency))
    app.router.add_get("/health", health)
    app.router.add_get("/api/v1/speech_frontend/health", health)
    app.router.add_post("/api/v1/speech_frontend/analyze", analyze)
    app.router.add_post("/api/v1/speech_frontend/prepare", prepare)
    app.router.add_post("/speech_frontend", analyze)
    app.router.add_post("/humanaudio_vad", analyze)
    app.router.add_route("OPTIONS", "/{tail:.*}", options)
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiveTalking-compatible speech frontend API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--max-concurrency", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    web.run_app(
        create_app(max_concurrency=arguments.max_concurrency),
        host=arguments.host,
        port=arguments.port,
        print=lambda message: print(message.replace("======== Running on", "Speech frontend API:")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
