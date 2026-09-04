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
from server.acoustic_context import (
    ACOUSTIC_CONTEXT_KEY,
    ACOUSTIC_REPORT_KEY,
    AcousticContextManager,
    AcousticReportService,
)

import tempfile
import os
from server import asr_service as asr_svc
from server import diarization_service as diarization_svc
from server.content_analysis_proxy import analyze_conversation_audio


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
MAX_ACOUSTIC_BYTES = MAX_TRANSCRIBE_BYTES
MAX_DIARIZATION_BYTES = diarization_svc.MAX_UPLOAD_BYTES


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
    speaker_diarization = None
    speaker_diarization_error = None
    diarization_task = None
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
            diarization_task = _start_diarization(form, tmp_path)
            try:
                # Both requests use the same normalized WAV. Waiting for the
                # optional task before cleanup guarantees the worker never
                # observes a deleted temporary file.
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
                speaker_diarization, speaker_diarization_error = await _await_diarization(
                    diarization_task,
                    trace,
                    "humanaudio",
                )
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

        if speaker_diarization is not None:
            datainfo["speaker_diarization"] = speaker_diarization

        # 走 LLM 流程（与 human 路由的 chat 模式完全一致）
        llm_response = request.app.get("llm_response")
        if llm_response:
            datainfo["_llm_dispatched_monotonic"] = time.perf_counter()
            emit_latency("llm_dispatched", trace)
            loop.run_in_executor(
                None, llm_response, recognized_text, avatar_session, datainfo
            )

        emit_latency("audio_chat_accepted", trace)
        response_data = {
            "recognized": recognized_text,
            "trace_id": trace["trace_id"],
        }
        if speaker_diarization is not None:
            response_data["speaker_diarization"] = speaker_diarization
        if speaker_diarization_error:
            response_data["speaker_diarization_error"] = speaker_diarization_error
        return json_ok(response_data)

    except Exception as e:
        emit_latency("audio_chat_error", trace, error=repr(e))
        logger.exception('humanaudio exception:')
        return json_error(str(e))

async def humanaudio_monitor(request):
    """监听模式：只做 ASR 转写，不触发 LLM"""
    request_started = time.perf_counter()
    trace = None
    speaker_diarization = None
    speaker_diarization_error = None
    diarization_task = None
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
            diarization_task = _start_diarization(form, tmp_path)
            try:
                recognized_text = await loop.run_in_executor(
                    None, asr_svc.transcribe, tmp_path
                )
                asr_finished = time.perf_counter()
                duration = _probe_audio_duration(tmp_path)
            finally:
                speaker_diarization, speaker_diarization_error = await _await_diarization(
                    diarization_task,
                    trace,
                    "humanaudio_monitor",
                )
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
        response_data = {
            "recognized": recognized_text,
            "trace_id": trace["trace_id"],
        }
        if speaker_diarization is not None:
            response_data["speaker_diarization"] = speaker_diarization
        if speaker_diarization_error:
            response_data["speaker_diarization_error"] = speaker_diarization_error
        return json_ok(response_data)

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


async def asr_health(request):
    """Expose the selected ASR backend and optional WhisperX status."""
    try:
        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, asr_svc.health)
        return json_ok(status)
    except Exception as exc:
        logger.warning("ASR health check failed: %s", exc)
        return json_error("ASR服务暂不可用")


async def diarization_health(request):
    """Expose the in-process FunASR speaker-diarization health.

    Args:
        request: Current aiohttp request. It is used only to keep the route
            signature consistent with the other health handlers.

    Returns:
        A JSON response containing local model configuration and health state.
    """
    try:
        return json_ok(diarization_svc.health())
    except Exception as exc:
        logger.warning("Speaker diarization health check failed: %s", exc)
        return json_error("说话人日志化服务暂不可用")


def _start_diarization(form, audio_path: str, force: bool = False):
    """Start optional diarization while the primary ASR runs.

    Args:
        form: Parsed multipart form from an audio request.
        audio_path: Converted WAV path shared with the diarization worker.
        force: Start the task even when the request has no opt-in form field.

    Returns:
        An asyncio task, or None when diarization was not requested.
    """
    if not force and not diarization_svc.requested(form.get("speaker_diarization")):
        return None
    language = str(form.get("language") or diarization_svc.LANGUAGE).strip()
    return asyncio.create_task(
        asyncio.to_thread(
            diarization_svc.diarize,
            audio_path,
            language or "auto",
        )
    )


async def _await_diarization(task, trace, source: str):
    """Resolve optional diarization without failing the ASR request.

    Args:
        task: Task returned by `_start_diarization`.
        trace: Latency trace for the current request.
        source: Route name used in logs and metrics.

    Returns:
        A tuple `(result, error_message)`. Both are None when disabled.
    """
    if task is None:
        return None, None
    try:
        result = await task
        emit_latency(
            "speaker_diarization_final",
            trace,
            source=source,
            speaker_count=result.get("speaker_count", 0),
            segment_count=len(result.get("speaker_segments") or []),
        )
        logger.info(
            "Speaker diarization source=%s speakers=%s segments=%s",
            source,
            result.get("speakers") or [],
            result.get("speaker_segments") or [],
        )
        return result, None
    except Exception as exc:
        logger.warning("Speaker diarization failed source=%s: %s", source, exc)
        emit_latency(
            "speaker_diarization_error",
            trace,
            source=source,
            error=repr(exc),
        )
        return None, "说话人日志化暂不可用"


async def analyze_environment(request):
    """Analyze an environment window and optionally update transcription context."""
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
        analysis_session_id = str(
            form.get("analysis_session_id") or sessionid or "transcribe-default"
        )[:128]
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
        context_manager = request.app.get(ACOUSTIC_CONTEXT_KEY)
        if context_manager is not None:
            context_manager.add_window(analysis_session_id, result)
            transcript = str(form.get("transcript", "")).strip()
            if transcript:
                context_manager.set_transcript(analysis_session_id, transcript)
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
    """Audio transcription plus optional concurrent acoustic-context analysis."""
    raw_path = None
    wav_path = None
    beats_task = None
    diarization_task = None
    speaker_diarization = None
    speaker_diarization_error = None
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
            return json_error(
                f"音频文件不能超过 {MAX_TRANSCRIBE_BYTES // (1024 * 1024)} MB"
            )

        filename = Path(getattr(fileobj, "filename", "audio") or "audio").name
        analysis_session_id = str(
            form.get("analysis_session_id") or "transcribe-default"
        )[:128]
        analyze_acoustic = str(form.get("analyze_acoustic", "1")).lower() not in {
            "0", "false", "no",
        }
        beats_client = request.app.get(BEATS_CLIENT_KEY)
        if analyze_acoustic and beats_client is not None:
            beats_task = asyncio.create_task(
                beats_client.analyze(
                    filebytes,
                    filename=filename,
                    content_type=getattr(fileobj, "content_type", None)
                    or "application/octet-stream",
                )
            )
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
        diarization_task = _start_diarization(form, wav_path)
        try:
            # Start both jobs from the same normalized WAV. The outer
            # finally block only removes the file after this task is drained.
            recognized_text = await loop.run_in_executor(
                None,
                asr_svc.transcribe,
                wav_path,
            )
            finished_at = time.perf_counter()
        finally:
            speaker_diarization, speaker_diarization_error = await _await_diarization(
                diarization_task,
                trace,
                "transcribe_audio",
            )

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

        context_manager = request.app.get(ACOUSTIC_CONTEXT_KEY)
        acoustic_result = None
        acoustic_error = None
        if context_manager is not None:
            context_manager.add_transcript(analysis_session_id, recognized_text)
        if beats_task is not None:
            try:
                acoustic_result = await beats_task
                if context_manager is not None:
                    context_manager.add_window(analysis_session_id, acoustic_result)
            except Exception as exc:
                acoustic_error = "环境声音分析暂不可用"
                logger.warning("Transcription acoustic analysis failed: %s", exc)
        response_data = {
            "recognized": recognized_text,
            "filename": filename,
            "duration": duration,
            "elapsed": round(finished_at - started_at, 2),
            "trace_id": trace["trace_id"],
            "analysis_session_id": analysis_session_id,
            "acoustic": acoustic_result,
        }
        if acoustic_error:
            response_data["acoustic_error"] = acoustic_error
        if speaker_diarization is not None:
            response_data["speaker_diarization"] = speaker_diarization
        if speaker_diarization_error:
            response_data["speaker_diarization_error"] = speaker_diarization_error
        return json_ok(response_data)
    except ValueError as e:
        emit_latency("audio_transcribe_error", trace, error=repr(e))
        return json_error(str(e))
    except Exception as e:
        emit_latency("audio_transcribe_error", trace, error=repr(e))
        logger.exception('transcribe_audio exception:')
        return json_error(str(e))
    finally:
        if beats_task is not None and not beats_task.done():
            beats_task.cancel()
        if diarization_task is not None and not diarization_task.done():
            diarization_task.cancel()
        for path in (raw_path, wav_path):
            if path and os.path.exists(path):
                os.unlink(path)


async def diarize_audio(request):
    """Run standalone speaker diarization without invoking the primary ASR.

    Args:
        request: Multipart request containing `file`; optional `language`
            selects the SenseVoice language hint.

    Returns:
        A JSON response with anonymous speakers, sentence segments, and merged
        `speaker_segments` intervals.
    """
    raw_path = None
    wav_path = None
    request_started = time.perf_counter()
    trace = None
    try:
        form = await request.post()
        trace = new_trace(
            "diarize",
            "speaker_diarization",
            started_monotonic=request_started,
            trace_id=str(form.get("client_trace_id") or "") or None,
        )
        fileobj = form.get("file")
        if fileobj is None or not hasattr(fileobj, "file"):
            return json_error("请选择需要日志化的音频文件")

        filebytes = fileobj.file.read()
        if not filebytes:
            return json_error("上传的音频文件为空")
        if len(filebytes) > MAX_DIARIZATION_BYTES:
            return json_error(
                f"音频文件不能超过 {MAX_DIARIZATION_BYTES // (1024 * 1024)} MB"
            )

        filename = Path(getattr(fileobj, "filename", "audio") or "audio").name
        suffix = _uploaded_audio_suffix(filename)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as raw_file:
            raw_file.write(filebytes)
            raw_path = raw_file.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        _convert_audio_to_wav(raw_path, wav_path)
        task = _start_diarization(form, wav_path, force=True)
        result, error = await _await_diarization(task, trace, "diarize_audio")
        if error or result is None:
            return json_error(error or "说话人日志化暂不可用")
        result["filename"] = filename
        result["duration"] = _probe_audio_duration(wav_path)
        return json_ok(result)
    except ValueError as exc:
        emit_latency("speaker_diarization_error", trace, error=repr(exc))
        return json_error(str(exc))
    except Exception as exc:
        emit_latency("speaker_diarization_error", trace, error=repr(exc))
        logger.exception("Standalone speaker diarization failed")
        return json_error(str(exc))
    finally:
        for path in (raw_path, wav_path):
            if path and os.path.exists(path):
                os.unlink(path)


async def reset_acoustic_context(request):
    """Clear one browser's rolling acoustic windows and transcript context."""
    try:
        params = await request.json()
        analysis_session_id = str(
            params.get("analysis_session_id") or "transcribe-default"
        )[:128]
        manager = request.app.get(ACOUSTIC_CONTEXT_KEY)
        if manager is not None:
            manager.reset(analysis_session_id)
        return json_ok({"analysis_session_id": analysis_session_id})
    except Exception as exc:
        logger.warning("Unable to reset acoustic context: %s", exc)
        return json_error("清空环境分析上下文失败")


async def diarize_api(request):
    """Expose in-process speaker diarization as a direct remote API.

    Args:
        request: Multipart request containing ``file`` and optional
            ``language``. Authentication is checked with ``FUNASR_API_KEY``.

    Returns:
        A direct JSON response containing ``speaker_count``, anonymous
        speakers, transcript segments, and merged speaker intervals.
    """
    if not diarization_svc.authorize_request(request):
        return web.json_response(
            {"detail": "缺少或无效的 FunASR API Key"},
            status=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_path = None
    wav_path = None
    try:
        form = await request.post()
        fileobj = form.get("file")
        if fileobj is None or not hasattr(fileobj, "file"):
            return web.json_response({"detail": "缺少音频文件"}, status=400)

        filebytes = fileobj.file.read()
        if not filebytes:
            return web.json_response({"detail": "上传的音频文件为空"}, status=400)
        if len(filebytes) > MAX_DIARIZATION_BYTES:
            return web.json_response(
                {
                    "detail": (
                        "音频文件不能超过 "
                        f"{MAX_DIARIZATION_BYTES // (1024 * 1024)} MB"
                    )
                },
                status=413,
            )

        filename = Path(getattr(fileobj, "filename", "audio") or "audio").name
        suffix = _uploaded_audio_suffix(filename)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as raw_file:
            raw_file.write(filebytes)
            raw_path = raw_file.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        _convert_audio_to_wav(raw_path, wav_path)
        language = str(
            form.get("language") or diarization_svc.LANGUAGE
        ).strip() or "auto"
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            diarization_svc.diarize,
            wav_path,
            language,
        )
        result = dict(result)
        result["filename"] = filename
        result["duration"] = _probe_audio_duration(wav_path)
        return web.json_response(result)
    except ValueError as exc:
        return web.json_response({"detail": str(exc)}, status=422)
    except Exception as exc:
        logger.exception("Direct speaker diarization API failed")
        return web.json_response(
            {"detail": f"说话人日志化失败：{exc}"},
            status=500,
        )
    finally:
        for path in (raw_path, wav_path):
            if path and os.path.exists(path):
                os.unlink(path)


async def acoustic_report(request):
    """Generate the LLM report separately so ASR and BEATs responses stay fast."""
    try:
        params = await request.json()
        analysis_session_id = str(
            params.get("analysis_session_id") or "transcribe-default"
        )[:128]
        manager = request.app.get(ACOUSTIC_CONTEXT_KEY)
        service = request.app.get(ACOUSTIC_REPORT_KEY)
        if manager is None or service is None:
            return json_error("环境分析服务未初始化")
        if "transcript" in params:
            manager.set_transcript(
                analysis_session_id,
                str(params.get("transcript") or ""),
            )
        report = await manager.report(
            analysis_session_id,
            service,
            force=bool(params.get("force", False)),
        )
        return json_ok(report)
    except Exception as exc:
        logger.exception("Acoustic report generation failed")
        return json_error(f"环境智能分析失败：{exc}")


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
    app[ACOUSTIC_CONTEXT_KEY] = AcousticContextManager()
    app[ACOUSTIC_REPORT_KEY] = AcousticReportService()
    app.cleanup_ctx.append(beats_client_context)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/humanaudio_monitor", humanaudio_monitor)
    app.router.add_get("/api/beats/health", beats_health)
    app.router.add_get("/api/asr/health", asr_health)
    app.router.add_get("/api/diarization/health", diarization_health)
    app.router.add_get("/health", diarization_health)
    app.router.add_post("/analyze_environment", analyze_environment)
    app.router.add_post("/acoustic_report", acoustic_report)
    app.router.add_post("/reset_acoustic_context", reset_acoustic_context)
    app.router.add_post("/transcribe_audio", transcribe_audio)
    app.router.add_post("/api/content-analysis/audio", analyze_conversation_audio)
    app.router.add_post("/diarize_audio", diarize_audio)
    app.router.add_post("/v1/diarize", diarize_api)
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
