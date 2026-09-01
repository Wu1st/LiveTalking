# WhisperX 部署

LiveTalking 支持把 ASR 转写请求转发给独立的 WhisperX sidecar。这样 WhisperX 与现有的 Qwen3-ASR 不会争抢同一个 Python 进程的模型显存。

## 1. 启动 WhisperX

要求：

- Linux 主机已安装 Docker Compose v2；
- NVIDIA Container Toolkit 可用；
- 主机有可用 NVIDIA GPU；
- Docker 能访问 PyPI 和 Hugging Face。

启动：

```bash
cd LiveTalking/services/whisperx_api
bash run.sh
```

首次启动会下载 `large-v3`、Silero VAD 以及对应的 Hugging Face 缓存，模型保存在 Docker volume `whisperx-models` 中。

检查：

```bash
curl http://127.0.0.1:8093/health
```

LiveTalking 启动后还可以检查统一 ASR 状态：

```bash
curl http://127.0.0.1:8010/api/asr/health
```

## 2. 让 LiveTalking 使用 WhisperX

在启动 LiveTalking 前设置：

```bash
export LIVETALKING_ASR_BACKEND=whisperx
export WHISPERX_URL=http://127.0.0.1:8093
```

然后按原方式启动 LiveTalking：

```bash
cd LiveTalking
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
```

默认 `WHISPERX_MODE=fast`，只执行转写，适合语音对话和实时监听。需要词级时间戳时：

```bash
export WHISPERX_MODE=full
export WHISPERX_ALIGN=1
```

说话人分离需要 Hugging Face token 和 pyannote 模型授权：

```bash
export HF_TOKEN=hf_xxx
export WHISPERX_DIARIZE=1
```

## 3. 回退到 Qwen3-ASR

不设置 `LIVETALKING_ASR_BACKEND` 时，LiveTalking 继续使用原有的 Qwen3-ASR。显式回退：

```bash
export LIVETALKING_ASR_BACKEND=qwen3
```

## 4. 常用参数

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `WHISPERX_MODEL` | `large-v3` | Whisper 模型名称 |
| `WHISPERX_COMPUTE_TYPE` | `float16` | GPU 推理精度 |
| `WHISPERX_BATCH_SIZE` | `4` | WhisperX batch size |
| `WHISPERX_LANGUAGE` | `auto` | 语言代码，如 `zh`、`en` |
| `WHISPERX_READ_TIMEOUT` | `600` | 单次转写读取超时，单位秒 |
| `WHISPERX_MAX_UPLOAD_MB` | `500` | sidecar 上传大小限制 |

WhisperX sidecar 提供 `GET /health` 和 `POST /v1/transcribe`。LiveTalking 的 `/transcribe_audio`、`/humanaudio_monitor` 和 `/humanaudio` 会共用同一套适配器。
