###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from server.webrtc import HumanPlayer
from avatars.base_avatar import BaseAvatar
from llm import llm_response
from translator import TranslationService
import registry
from server.routes import setup_routes
from server.rtc_manager import RTCManager
from server.session_manager import session_manager

import argparse
import random
import shutil
import asyncio
import torch
from io import BytesIO
from typing import Dict
from utils.logger import logger
import copy
import gc

import qwen3asr_service as asr_svc

app = Flask(__name__)
#sockets = Sockets(app)
opt = None
model = None
global_avatars = {} # avatar_id: payload
        

#####webrtc###############################
# rtc_manager replaces the old pcs set and duplicate offer handlers.
rtc_manager = None

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = params.get('avatar',opt.avatar_id) 
    ref_audio = params.get('refaudio','') #音色
    ref_text = params.get('reftext','')
    if (avatar_id and avatar_id != opt.avatar_id):
        # Avoid reloading if already cached globally
        if avatar_id not in global_avatars:
            global_avatars[avatar_id] = load_avatar(avatar_id)
        avatar_this = global_avatars[avatar_id]
    else:
        # Default avatar loaded at startup
        avatar_this = global_avatars.get(opt.avatar_id)
    if ref_audio: #请求参数配置了参考音频
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
    custom_config=params.get('custom_config','') #动作编排配置
    if custom_config:
        opt_this.customopt = json.loads(custom_config)

    avatar_session = registry.create("avatar", opt.model, opt=opt_this, model=model, avatar=avatar_this)
    return avatar_session

async def offer(request):
    return await rtc_manager.handle_offer(request)

async def on_shutdown(app):
    stop_event = app.get("ollama_keeper_stop")
    if stop_event is not None:
        stop_event.set()

    await rtc_manager.shutdown()


def preload_cosyvoice_speaker(runtime_opt):
    """Register the configured zero-shot speaker before serving requests."""
    if runtime_opt.tts != 'cosyvoice' or not runtime_opt.REF_FILE:
        return

    import requests as _req

    try:
        with open(runtime_opt.REF_FILE, 'rb') as f:
            response = _req.post(
                f"{runtime_opt.TTS_SERVER}/register_zero_shot_spk",
                data={
                    'prompt_text': runtime_opt.REF_TEXT,
                },
                files={
                    'prompt_wav': (
                        'prompt_wav.wav',
                        f,
                        'audio/wav',
                    ),
                },
                timeout=(5, 60),
            )

        response.raise_for_status()
        result = response.json()
        logger.info(
            "CosyVoice speaker cache preloaded: spk_id=%s, cached=%s",
            result.get('spk_id'),
            result.get('cached'),
        )
    except Exception as e:
        # Preloading is an optimization. Keep LiveTalking available when the
        # CosyVoice service is temporarily unavailable or does not implement
        # the speaker-registration endpoint; the TTS adapter can still use its
        # normal fallback path when a reply is synthesized.
        logger.warning("CosyVoice preload failed: %s", e)

def start_ollama_model_keeper(
    base_url,
    models,
    interval_seconds=240,
    keep_alive="10m",
):
    """
    启动时预加载 Ollama 模型，并定期刷新模型存活时间。

    interval_seconds=240：每4分钟刷新一次
    keep_alive=10m：每次要求 Ollama至少保留10分钟
    """
    import requests
    import time

    base_url = base_url.rstrip("/")
    models = tuple(dict.fromkeys(model for model in models if model))
    stop_event = Event()

    def touch_models(session):
        for model_name in models:
            started_at = time.perf_counter()
            try:
                response = session.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": keep_alive,
                    },
                    timeout=(5, 180),
                )
                response.raise_for_status()
                result = response.json()

                elapsed = time.perf_counter() - started_at
                load_duration = result.get("load_duration", 0) / 1_000_000_000

                logger.info(
                    "Ollama model ready | model=%s | request=%.2fs | load=%.2fs",
                    model_name,
                    elapsed,
                    load_duration,
                )
            except Exception as exc:
                # 预热失败不阻止 LiveTalking启动
                logger.warning(
                    "Ollama model preload failed | model=%s | error=%s",
                    model_name,
                    exc,
                )

    # 第一次同步执行：让模型加载完成后再开放网页服务
    with requests.Session() as session:
        touch_models(session)

    # 后续后台保活
    def keeper_worker():
        with requests.Session() as session:
            while not stop_event.wait(interval_seconds):
                touch_models(session)

    Thread(
        target=keeper_worker,
        name="ollama-model-keeper",
        daemon=True,
    ).start()

    return stop_event

def main():
    global rtc_manager, opt, model,load_avatar
    # 解析命令行参数
    from config import parse_args
    opt = parse_args()

    # ─── 加载 avatar 插件（触发 @register 注册）──────────────────────
    _avatar_modules = {
        'musetalk':   'avatars.musetalk_avatar',
        'wav2lip':    'avatars.wav2lip_avatar',
        'ultralight': 'avatars.ultralight_avatar',
    }
    import importlib
    avatar_mod = importlib.import_module(_avatar_modules[opt.model])
    load_model = avatar_mod.load_model
    load_avatar = avatar_mod.load_avatar
    warm_up = avatar_mod.warm_up
    logger.info(opt)

    if opt.model == 'musetalk':
        model = load_model()
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        model = load_model("./models/wav2lip.pth")
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        model = load_model(opt)
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,global_avatars[opt.avatar_id],160)

    # init rtc manager
    session_manager.init_builder(build_avatar_session)
    rtc_manager = RTCManager(opt)
    # share avatar_sessions (RTCManager handles it but routes.py expects it)
    
    if opt.transport=='virtualcam' or opt.transport=='rtmp':
        thread_quit = Event()
        params = {}
        # session 0 for virtualcam
        session_manager.add_session('0', build_avatar_session('0', params))
        rendthrd = Thread(target=session_manager.get_session('0').render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync["llm_response"] = llm_response
    appasync["translation_service"] = TranslationService(
        base_url=opt.TRANSLATION_SERVER,
        model=opt.TRANSLATION_MODEL,
        timeout=opt.TRANSLATION_TIMEOUT,
    )

    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    
    # 注册 server/routes.py 中的通用 API 路由
    setup_routes(appasync) 

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='rtmpapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info("预加载 Qwen3-ASR 模型...")
    import qwen3asr_service as asr_svc
    asr_svc.get_model()
    logger.info("Qwen3-ASR 模型加载完成")
    # opt has already been initialized by parse_args() at this point.
    preload_cosyvoice_speaker(opt)
    appasync["ollama_keeper_stop"] = start_ollama_model_keeper(
        base_url="http://127.0.0.1:11434",
        models=(
            "qwen2.5:7b",
            opt.TRANSLATION_MODEL,
        ),
        interval_seconds=240,
        keep_alive="10m",
    )
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://localhost:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
    
    
    
