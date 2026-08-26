###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import time
import json
import numpy as np
import asyncio
import subprocess
from pathlib import Path
from aiohttp import web

from utils.logger import logger
from utils.latency import (
    attach_trace,
    emit_latency,
    new_trace,
    optional_float,
    trace_from_id,
)
from server.beats_client import BEATS_CLIENT_KEY, beats_client_context

import tempfile
import os
import qwen3asr_service as asr_svc


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


MAX_TRANSCRIBE_BYTES = 80 * 1024 * 1024
MAX_TRANSLATION_CHARS = 20_000
MAX_ACOUSTIC_BYTES = 25 * 1024 * 1024


def _uploaded_audio_suffix(filename: str) -> str:
    """只保留安全的扩展名；实际格式仍由 ffmpeg 根据文件内容识别。"""
    suffix = Path(filename or "").suffix.lower()
    if suffix and len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".audio"


def _convert_audio_to_wav(source_path: str, wav_path: str):
    """调用 ffmpeg，将常见音频容器统一为 16 kHz 单声道 WAV。"""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", source_path,
            "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", wav_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("Audio conversion failed: %s", result.stderr[-1200:])
        raise ValueError("无法解析该音频文件，请确认文件格式正确且未损坏")


def _probe_audio_duration(wav_path: str):
    """读取转换后音频时长，失败时返回 None，不影响转写。"""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", wav_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if streams:
            return round(float(streams[0].get("duration", 0)), 2)
    except (subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning("Unable to probe audio duration: %s", wav_path)
    return None


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    request_started = time.perf_counter()
    trace = None
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        trace = new_trace(
            sessionid,
            f"text_{params.get('type', 'unknown')}",
            started_monotonic=request_started,
            text_chars=len(str(params.get('text', ''))),
            interrupt=bool(params.get('interrupt')),
        )
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            emit_latency("human_error", trace, error="session not found")
            return json_error("session not found")

        if params.get('interrupt'):
            emit_latency("interrupt_requested", trace, trigger="human")
            avatar_session.flush_talk(trace=trace, trigger="human")

        datainfo = attach_trace({}, trace)
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            llm_response = request.app.get("llm_response")
            if llm_response:
                datainfo["_llm_dispatched_monotonic"] = time.perf_counter()
                emit_latency("llm_dispatched", trace)
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        emit_latency("human_accepted", trace)
        return json_ok({"trace_id": trace["trace_id"]})
    except Exception as e:
        emit_latency("human_error", trace, error=repr(e))
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    request_started = time.perf_counter()
    trace = None
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        trace = new_trace(
            sessionid,
            "manual_interrupt",
            started_monotonic=request_started,
        )
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            emit_latency("interrupt_error", trace, error="session not found")
            return json_error("session not found")
        emit_latency("interrupt_requested", trace, trigger="button")
        avatar_session.flush_talk(trace=trace, trigger="button")
        emit_latency("interrupt_returned", trace)
        return json_ok({"trace_id": trace["trace_id"]})
    except Exception as e:
        emit_latency("interrupt_error", trace, error=repr(e))
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))

"""
async def humanaudio(request):
    # 上传音频文件
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))
"""

async def humanaudio(request):
    """上传音频 → ASR识别 → LLM对话"""
    request_started = time.perf_counter()
    trace = None
    try:
        form = await request.post()
        form_received = time.perf_counter()

        sessionid = str(form.get('sessionid', ''))
        client_silence_wait_ms = optional_float(form.get("client_silence_wait_ms"))
        client_speech_end_to_upload_ms = optional_float(
            form.get("client_speech_end_to_upload_ms")
        )
        trace = new_trace(
            sessionid,
            "audio_chat",
            started_monotonic=request_started,
            trace_id=str(form.get("client_trace_id") or "") or None,
            client_mode=str(form.get("client_mode", "unknown")),
            client_silence_wait_ms=client_silence_wait_ms,
            client_speech_end_to_upload_ms=client_speech_end_to_upload_ms,
        )
        fileobj = form["file"]
        filebytes = fileobj.file.read()
        file_read_end = time.perf_counter()
        # 判断文件格式
        logger.info(f"audio content_type={fileobj.content_type}, filename={fileobj.filename}, size={len(filebytes)}")
        emit_latency(
            "audio_upload_received",
            trace,
            form_parse_ms=(form_received - request_started) * 1000,
            file_read_ms=(file_read_end - form_received) * 1000,
            bytes=len(filebytes),
            content_type=fileobj.content_type,
        )

        datainfo = attach_trace({}, trace)

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            emit_latency("audio_chat_error", trace, error="session not found")
            return json_error("session not found")

        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(filebytes)
            raw_path = f.name

        tmp_path = raw_path.replace('.webm', '.wav')
        try:
            t0 = time.perf_counter()
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
                check=True, capture_output=True
            )
            t1 = time.perf_counter()

            loop = asyncio.get_running_loop()
            recognized_text = await loop.run_in_executor(
                None, asr_svc.transcribe, tmp_path
            )
            t2 = time.perf_counter()

            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_streams', tmp_path],
                capture_output=True, text=True
            )
            dur = float(json.loads(result.stdout)['streams'][0]['duration'])

        finally:
            os.unlink(raw_path)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        logger.info(f"ASR recognized: {recognized_text}")
        logger.info(f"LATENCY ffmpeg={t1-t0:.3f}s ASR={t2-t1:.3f}s total_asr={t2-t0:.3f}s audio_dur={dur:.1f}s RTF={((t2-t1)/dur):.2f}")
        emit_latency(
            "asr_final",
            trace,
            ffmpeg_ms=(t1 - t0) * 1000,
            asr_ms=(t2 - t1) * 1000,
            total_asr_ms=(t2 - t0) * 1000,
            audio_duration_s=dur,
            rtf=((t2 - t1) / dur) if dur > 0 else None,
            recognized_chars=len(recognized_text or ""),
        )

        if not recognized_text:
            emit_latency("audio_chat_error", trace, error="empty ASR result")
            return json_error("ASR 识别结果为空")

        # 走 LLM 流程（与 human 路由的 chat 模式完全一致）
        llm_response = request.app.get("llm_response")
        if llm_response:
            datainfo["_llm_dispatched_monotonic"] = time.perf_counter()
            emit_latency("llm_dispatched", trace)
            loop.run_in_executor(
                None, llm_response, recognized_text, avatar_session, datainfo
            )

        emit_latency("audio_chat_accepted", trace)
        return json_ok({"recognized": recognized_text, "trace_id": trace["trace_id"]})

    except Exception as e:
        emit_latency("audio_chat_error", trace, error=repr(e))
        logger.exception('humanaudio exception:')
        return json_error(str(e))

async def humanaudio_monitor(request):
    """监听模式：只做 ASR 转写，不触发 LLM"""
    request_started = time.perf_counter()
    trace = None
    try:
        form = await request.post()
        form_received = time.perf_counter()
        sessionid = str(form.get("sessionid", "0"))
        trace = new_trace(
            sessionid,
            "audio_monitor",
            started_monotonic=request_started,
            trace_id=str(form.get("client_trace_id") or "") or None,
            client_silence_wait_ms=optional_float(form.get("client_silence_wait_ms")),
        )
        fileobj = form["file"]
        filebytes = fileobj.file.read()
        emit_latency(
            "audio_upload_received",
            trace,
            form_parse_ms=(form_received - request_started) * 1000,
            bytes=len(filebytes),
            content_type=fileobj.content_type,
        )

        import tempfile, os, subprocess
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(filebytes)
            raw_path = f.name

        tmp_path = raw_path.replace('.webm', '.wav')
        try:
            convert_started = time.perf_counter()
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path,
                 "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
                check=True, capture_output=True
            )
            converted_at = time.perf_counter()
        finally:
            os.unlink(raw_path)

        try:
            loop = asyncio.get_event_loop()
            recognized_text = await loop.run_in_executor(
                None, asr_svc.transcribe, tmp_path
            )
            asr_finished = time.perf_counter()
            duration = _probe_audio_duration(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        logger.info(f"Monitor ASR: {recognized_text}")
        emit_latency(
            "asr_final",
            trace,
            ffmpeg_ms=(converted_at - convert_started) * 1000,
            asr_ms=(asr_finished - converted_at) * 1000,
            total_asr_ms=(asr_finished - convert_started) * 1000,
            audio_duration_s=duration,
            rtf=((asr_finished - converted_at) / duration) if duration else None,
            recognized_chars=len(recognized_text or ""),
        )
        return json_ok({"recognized": recognized_text, "trace_id": trace["trace_id"]})

    except Exception as e:
        emit_latency("audio_monitor_error", trace, error=repr(e))
        logger.exception('humanaudio_monitor exception:')
        return json_error(str(e))


async def beats_health(request):
    """Expose the local BEATs health state through the LiveTalking server."""
    client = request.app.get(BEATS_CLIENT_KEY)
    if client is None:
        return json_error("BEATs客户端未初始化")
    try:
        return json_ok(await client.health())
    except Exception as exc:
        logger.warning("BEATs health check failed: %s", exc)
        return json_error("BEATs服务暂不可用")


async def analyze_environment(request):
    """Analyze a continuous environment-audio window without invoking ASR/LLM/TTS."""
    request_started = time.perf_counter()
    trace = None
    try:
        form = await request.post()
        fileobj = form.get("file")
        if fileobj is None or not hasattr(fileobj, "file"):
            return json_error("缺少环境音频文件")

        filebytes = fileobj.file.read()
        if not filebytes:
            return json_error("上传的环境音频为空")
        if len(filebytes) > MAX_ACOUSTIC_BYTES:
            return json_error("环境音频文件不能超过 25 MB")

        sessionid = str(form.get("sessionid", "0"))
        trace = new_trace(
            sessionid,
            "beats_environment",
            started_monotonic=request_started,
        )
        client = request.app.get(BEATS_CLIENT_KEY)
        if client is None:
            return json_error("BEATs客户端未初始化")

        filename = Path(getattr(fileobj, "filename", "environment.webm") or "environment.webm").name
        content_type = getattr(fileobj, "content_type", None) or "application/octet-stream"
        result = await client.analyze(
            filebytes,
            filename=filename,
            content_type=content_type,
        )
        elapsed_ms = (time.perf_counter() - request_started) * 1000
        logger.info(
            "BEATs environment | session=%s | text=%s | decode_ms=%s | inference_ms=%s | total_ms=%.2f",
            sessionid,
            result.get("text"),
            result.get("decode_ms"),
            result.get("inference_ms"),
            elapsed_ms,
        )
        emit_latency(
            "beats_final",
            trace,
            bytes=len(filebytes),
            audio_duration_s=result.get("audio_duration_seconds"),
            decode_ms=result.get("decode_ms"),
            inference_ms=result.get("inference_ms"),
            total_ms=elapsed_ms,
            event_count=len(result.get("events") or []),
        )
        return json_ok(result)
    except Exception as exc:
        emit_latency("beats_error", trace, error=repr(exc))
        logger.exception("BEATs environment analysis failed")
        return json_error("环境声音分析暂不可用")


async def transcribe_audio(request):
    """音频转写工作台：接收上传文件或浏览器录音，仅返回可编辑的 ASR 文字。"""
    raw_path = None
    wav_path = None
    request_started = time.perf_counter()
    trace = None
    try:
        form = await request.post()
        form_received = time.perf_counter()
        trace = new_trace(
            "transcribe",
            "audio_transcribe",
            started_monotonic=request_started,
            trace_id=str(form.get("client_trace_id") or "") or None,
            client_silence_wait_ms=optional_float(form.get("client_silence_wait_ms")),
        )
        fileobj = form.get("file")
        if fileobj is None or not hasattr(fileobj, "file"):
            return json_error("请选择需要转写的音频文件")

        filebytes = fileobj.file.read()
        emit_latency(
            "audio_upload_received",
            trace,
            form_parse_ms=(form_received - request_started) * 1000,
            bytes=len(filebytes),
            content_type=getattr(fileobj, "content_type", None),
        )
        if not filebytes:
            return json_error("上传的音频文件为空")
        if len(filebytes) > MAX_TRANSCRIBE_BYTES:
            return json_error("音频文件不能超过 80 MB")

        filename = Path(getattr(fileobj, "filename", "audio") or "audio").name
        suffix = _uploaded_audio_suffix(filename)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as raw_file:
            raw_file.write(filebytes)
            raw_path = raw_file.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        started_at = time.perf_counter()
        loop = asyncio.get_running_loop()
        _convert_audio_to_wav(raw_path, wav_path)
        converted_at = time.perf_counter()
        duration = _probe_audio_duration(wav_path)
        recognized_text = await loop.run_in_executor(None, asr_svc.transcribe, wav_path)
        finished_at = time.perf_counter()

        logger.info(
            "Audio transcription filename=%s size=%d duration=%s ffmpeg=%.3fs ASR=%.3fs",
            filename,
            len(filebytes),
            duration,
            converted_at - started_at,
            finished_at - converted_at,
        )
        emit_latency(
            "asr_final",
            trace,
            ffmpeg_ms=(converted_at - started_at) * 1000,
            asr_ms=(finished_at - converted_at) * 1000,
            total_asr_ms=(finished_at - started_at) * 1000,
            audio_duration_s=duration,
            rtf=((finished_at - converted_at) / duration) if duration else None,
            recognized_chars=len(recognized_text or ""),
        )

        if not recognized_text:
            return json_error("ASR 识别结果为空")

        return json_ok({
            "recognized": recognized_text,
            "filename": filename,
            "duration": duration,
            "elapsed": round(finished_at - started_at, 2),
            "trace_id": trace["trace_id"],
        })
    except ValueError as e:
        emit_latency("audio_transcribe_error", trace, error=repr(e))
        return json_error(str(e))
    except Exception as e:
        emit_latency("audio_transcribe_error", trace, error=repr(e))
        logger.exception('transcribe_audio exception:')
        return json_error(str(e))
    finally:
        for path in (raw_path, wav_path):
            if path and os.path.exists(path):
                os.unlink(path)


async def translate_text(request):
    """文本翻译：单次调用 TranslateGemma，不保留对话历史。"""
    request_started = time.perf_counter()
    trace = None
    try:
        params = await request.json()
        text = str(params.get("text", "")).strip()
        source_language = str(params.get("source_language", "")).strip()
        target_language = str(params.get("target_language", "")).strip()
        trace = new_trace(
            "translation",
            "translation",
            started_monotonic=request_started,
            text_chars=len(text),
            source_language=source_language,
            target_language=target_language,
        )

        if not text:
            return json_error("请输入需要翻译的文字")
        if len(text) > MAX_TRANSLATION_CHARS:
            return json_error(f"单次翻译不能超过 {MAX_TRANSLATION_CHARS} 个字符")

        translation_service = request.app.get("translation_service")
        if translation_service is None:
            return json_error("翻译服务未配置")

        loop = asyncio.get_running_loop()
        inference_started = time.perf_counter()
        result = await loop.run_in_executor(
            None,
            translation_service.translate,
            text,
            source_language,
            target_language,
        )
        emit_latency(
            "translation_finished",
            trace,
            duration_ms=(time.perf_counter() - inference_started) * 1000,
            output_chars=len(str(result.get("translated", ""))) if isinstance(result, dict) else None,
        )
        if isinstance(result, dict):
            result = dict(result)
            result["trace_id"] = trace["trace_id"]
        return json_ok(result)
    except (ValueError, TypeError) as e:
        emit_latency("translation_error", trace, error=repr(e))
        return json_error(str(e))
    except Exception as e:
        emit_latency("translation_error", trace, error=repr(e))
        logger.exception('translate_text exception:')
        return json_error(f"翻译失败：{e}")


async def client_metrics(request):
    """接收浏览器端 fetch 与 WebRTC 指标，仅写入结构化日志。"""
    try:
        payload = await request.json()
        stage = str(payload.get("stage", ""))
        allowed_stages = {
            "client_fetch_complete",
            "client_webrtc_stats",
            "client_connection_state",
        }
        if stage not in allowed_stages:
            return json_error("unsupported metrics stage")

        sessionid = str(payload.get("sessionid", "0"))
        trace_id = str(payload.get("trace_id") or f"webrtc-{sessionid}")
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}

        # 限制字段数量，避免遥测接口被用来写入超大日志。
        metrics = {str(key): value for key, value in list(metrics.items())[:50]}
        trace = trace_from_id(sessionid, trace_id, "browser")
        emit_latency(stage, trace, **metrics)
        return json_ok()
    except Exception as e:
        logger.debug("client_metrics ignored invalid payload: %r", e)
        return json_error("invalid metrics payload")


async def save_monitor_clip(request):
    """接收前端发来的片段文字，写入服务器本地文件"""
    try:
        params = await request.json()
        save_dir = params.get('save_dir', './monitor_clips').strip() or './monitor_clips'
        filename = params.get('filename', 'clip.txt')
        content  = params.get('content', '')

        import os
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Monitor clip saved: {filepath}")
        return json_ok({"saved_to": filepath})
    except Exception as e:
        logger.exception('save_monitor_clip exception:')
        return json_error(str(e))
    
async def append_monitor_log(request):
    """追加一条转写记录到滚动日志文件，超出最大行数时从头部删除"""
    try:
        params   = await request.json()
        save_dir = params.get('save_dir', './monitor_clips').strip() or './monitor_clips'
        text     = params.get('text', '').strip()
        if not text:
            return json_ok()

        import os
        MAX_LINES = 5000   # 最大行数，超过此行数删除最早的 500 行

        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, 'monitor_log.txt')

        # 追加
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

        # 检查行数，超出则裁剪
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) > MAX_LINES:
            lines = lines[500:]   # 删除最早 500 行
            with open(log_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            logger.info(f"Monitor log trimmed to {len(lines)} lines")

        return json_ok()
    except Exception as e:
        logger.exception('append_monitor_log exception:')
        return json_error(str(e))

async def get_reply(request):
    """前端轮询接口：取该 session 最新的 LLM 完整回复，取完即清空"""
    try:
        params = await request.json()
        sessionid = str(params.get('sessionid', '0'))
        from llm import get_pending_reply
        text = get_pending_reply(sessionid)
        return json_ok({"reply": text})
    except Exception as e:
        logger.exception('get_reply exception:')
        return json_error(str(e))

async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


# ─── 路由注册 ──────────────────────────────────────────────────────────────

def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.cleanup_ctx.append(beats_client_context)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/humanaudio_monitor", humanaudio_monitor)
    app.router.add_get("/api/beats/health", beats_health)
    app.router.add_post("/analyze_environment", analyze_environment)
    app.router.add_post("/transcribe_audio", transcribe_audio)
    app.router.add_post("/translate_text", translate_text)
    app.router.add_post("/api/metrics/client", client_metrics)
    app.router.add_post("/save_monitor_clip", save_monitor_clip)
    app.router.add_post("/append_monitor_log", append_monitor_log)
    app.router.add_post("/get_reply", get_reply)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_static('/', path='web')
