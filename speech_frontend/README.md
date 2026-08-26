# 语音前端服务

用途：在 LiveTalking 现有录音、文件转写和监听链路中执行语音增强与 VAD，并向后续 ASR 提供 16 kHz 单声道 WAV。

启动：

```bash
conda run --no-capture-output -n qwen3-asr bash speech_frontend/run_api.sh
```

健康检查：

```bash
curl http://127.0.0.1:8092/health
```

LiveTalking 默认通过 `http://127.0.0.1:8092` 调用；如需修改，可设置 `SPEECH_FRONTEND_URL`。
