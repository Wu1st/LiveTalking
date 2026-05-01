###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import numpy as np
import asyncio
from aiohttp import web

from utils.logger import logger

import subprocess
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


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
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
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()
        # 判断文件格式
        logger.info(f"audio content_type={fileobj.content_type}, filename={fileobj.filename}, size={len(filebytes)}")

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(filebytes)
            raw_path = f.name

        tmp_path = raw_path.replace('.webm', '.wav')
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
                check=True, capture_output=True
            )
        finally:
            os.unlink(raw_path)

        try:
            # 在线程池执行，避免阻塞 aiohttp 事件循环
            loop = asyncio.get_event_loop()
            recognized_text = await loop.run_in_executor(
                None, asr_svc.transcribe, tmp_path
            )
        finally:
            os.unlink(tmp_path)

        logger.info(f"ASR recognized: {recognized_text}")

        if not recognized_text:
            return json_error("ASR 识别结果为空")

        # 走 LLM 流程（与 human 路由的 chat 模式完全一致）
        llm_response = request.app.get("llm_response")
        if llm_response:
            loop.run_in_executor(
                None, llm_response, recognized_text, avatar_session, datainfo
            )

        return json_ok({"recognized": recognized_text})

    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))

async def humanaudio_monitor(request):
    """监听模式：只做 ASR 转写，不触发 LLM"""
    try:
        form = await request.post()
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        import tempfile, os, subprocess
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(filebytes)
            raw_path = f.name

        tmp_path = raw_path.replace('.webm', '.wav')
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path,
                 "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
                check=True, capture_output=True
            )
        finally:
            os.unlink(raw_path)

        try:
            loop = asyncio.get_event_loop()
            recognized_text = await loop.run_in_executor(
                None, asr_svc.transcribe, tmp_path
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        logger.info(f"Monitor ASR: {recognized_text}")
        return json_ok({"recognized": recognized_text})

    except Exception as e:
        logger.exception('humanaudio_monitor exception:')
        return json_error(str(e))
    
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
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/humanaudio_monitor", humanaudio_monitor)
    app.router.add_post("/save_monitor_clip", save_monitor_clip)
    app.router.add_post("/append_monitor_log", append_monitor_log)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_static('/', path='web')
