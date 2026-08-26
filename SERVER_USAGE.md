# 服务器使用说明

## 1. 登录与进入项目

```bash
ssh -p 10022 hf@113.98.61.52
cd /home/hf/qwen3-asr/LiveTalking
conda activate qwen3-asr
```

密码不写入仓库，请向服务器管理员获取。

## 2. 启动服务

先启动语音前端：

```bash
cd /home/hf/qwen3-asr/LiveTalking
nohup setsid conda run --no-capture-output -n qwen3-asr \
  bash speech_frontend/run_api.sh \
  > speech_frontend/api.log 2>&1 < /dev/null &
```

再启动 LiveTalking：

```bash
nohup setsid conda run --no-capture-output -n qwen3-asr \
  python app.py \
  > livetalking.log 2>&1 < /dev/null &
```

不要重复启动。启动前可检查：

```bash
ss -ltnp | grep -E ':8010|:8092'
```

## 3. 访问页面

服务器本机：

```text
http://127.0.0.1:8010/dashboard.html
```

公网 `8010` 尚未开放时，在自己的电脑建立 SSH 隧道：

```bash
ssh -N -p 10022 -L 8010:127.0.0.1:8010 hf@113.98.61.52
```

然后浏览器访问：

```text
http://127.0.0.1:8010/dashboard.html
```

页面中的按住说话、音频文件转写和监测录音会自动经过语音前端；环境声音分类保持原流程。

## 4. 健康检查

```bash
curl http://127.0.0.1:8092/health
curl http://127.0.0.1:8010/api/speech_frontend/health
```

返回 `"code": 0` 和 `"status": "ready"` 表示正常。

## 5. 音频接口

分析音频：

```bash
curl -F file=@test.wav \
  -F sessionid=test \
  http://127.0.0.1:8010/api/speech_frontend/analyze
```

生成供 ASR 使用的 16 kHz 单声道语音 WAV：

```bash
curl -F file=@test.wav \
  -F sessionid=test \
  -F output=speech \
  -o speech.wav \
  http://127.0.0.1:8010/api/speech_frontend/prepare
```

支持浏览器可解码的 WAV、MP3、M4A/AAC、OGG/Opus、FLAC、WebM 等常见格式。

## 6. 日志与停止

查看日志：

```bash
tail -f livetalking.log
tail -f speech_frontend/api.log
```

查找进程：

```bash
pgrep -af 'python app.py'
pgrep -af 'speech_frontend.api'
```

停止时先确认 PID，再执行：

```bash
kill -TERM <PID>
```

## 7. 代码位置

```text
LiveTalking 主程序：/home/hf/qwen3-asr/LiveTalking
语音前端服务：    speech_frontend/
前端接入脚本：    web/speech_frontend_integration.js
代理接口：        server/speech_frontend_proxy.py
```

当前语音前端接入提交：`444a1b0`。
