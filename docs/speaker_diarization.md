# 说话人日志化 API 部署

本功能复用现有 LiveTalking 容器和 `8010` 端口，不创建新的 Docker 容器。FunASR 模型在 LiveTalking 进程内按需加载，用于：

- SenseVoiceSmall 语音识别；
- FSMN-VAD 语音活动检测；
- CAM++ 说话人特征提取和聚类；
- CT-Punc 标点和句段切分。

其中 `Speaker_00`、`Speaker_01` 等标签由 CAM++ 产生，SenseVoiceSmall 本身只负责识别和音频语义能力。标签只在当前音频请求内有效，不代表持久化身份。

## 在现有容器中启用 API

要求：

- 现有 LiveTalking 容器；
- 容器内可用的 NVIDIA GPU；
- 容器能访问 GitHub、PyPI 和 ModelScope。

启动：

```bash
cd /home/hf/qwen3-asr/LiveTalking
pip install -r requirements.txt
export FUNASR_API_KEY="$(openssl rand -hex 32)"
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
```

服务继续监听原来的 `0.0.0.0:8010`，首次调用日志化 API 时会下载 SenseVoiceSmall、FSMN-VAD、CAM++ 和 CT-Punc 模型。模型缓存使用当前容器的 ModelScope/HuggingFace 缓存目录。

`FUNASR_API_KEY` 配置后，远程日志化接口必须携带以下任一请求头：

- `X-API-Key: <密钥>`
- `Authorization: Bearer <密钥>`

生产环境不要省略 `FUNASR_API_KEY`，并只在防火墙或反向代理允许的网络范围内开放端口。健康检查不要求 Key，便于 Docker 和负载均衡器探活。

检查主服务：

```bash
curl http://<服务器IP>:8010/health
```

## 远程调用接口

接口：`POST /v1/diarize`

请求类型：`multipart/form-data`

字段：

| 字段 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `file` | 是 | - | 音频文件，支持 FunASR/ffmpeg 可解析的格式 |
| `language` | 否 | `auto` | `zh`、`en`、`yue` 等语言提示，或 `auto` |

使用 API Key：

```bash
export FUNASR_API_URL="http://<服务器IP>:8010"
export FUNASR_API_KEY="部署服务时设置的密钥"

curl -X POST \
  -H "X-API-Key: ${FUNASR_API_KEY}" \
  -F "file=@/path/to/audio.wav" \
  -F "language=auto" \
  "${FUNASR_API_URL}/v1/diarize"
```

也可以使用 Bearer Token：

```bash
curl -X POST \
  -H "Authorization: Bearer ${FUNASR_API_KEY}" \
  -F "file=@/path/to/audio.wav" \
  "${FUNASR_API_URL}/v1/diarize"
```

Python 调用示例：

```python
import requests

with open("audio.wav", "rb") as audio:
    response = requests.post(
        "http://<服务器IP>:8010/v1/diarize",
        headers={"X-API-Key": "<API密钥>"},
        files={"file": ("audio.wav", audio, "audio/wav")},
        data={"language": "auto"},
        timeout=(10, 600),
    )
response.raise_for_status()
print(response.json())
```

成功响应：

```json
{
  "backend": "funasr-sensevoice-cam++",
  "speaker_count": 2,
  "speakers": ["Speaker_00", "Speaker_01"],
  "text": "完整转写文本",
  "segments": [
    {
      "speaker": "Speaker_00",
      "start": 0.12,
      "end": 2.48,
      "text": "第一段发言"
    }
  ],
  "speaker_segments": [
    {
      "speaker": "Speaker_00",
      "start": 0.12,
      "end": 2.48,
      "text": "第一段发言"
    }
  ]
}
```

`segments` 是模型句段级结果；`speaker_segments` 会把同一匿名说话人、间隔不超过 250 ms 的相邻句段合并为一个发言区间。时间单位为秒。`Speaker_00` 等标签只在当前音频请求内有效。

## 通过 LiveTalking 代理调用

LiveTalking 启动后，也可以通过其代理接口执行说话人日志化，不调用现有 Qwen3-ASR：

```bash
curl -X POST \
  -F "file=@/path/to/audio.wav" \
  -F "language=auto" \
  http://127.0.0.1:8010/diarize_audio
```

成功响应中的关键字段：

```json
{
  "speaker_count": 2,
  "speakers": ["Speaker_00", "Speaker_01"],
  "segments": [
    {
      "speaker": "Speaker_00",
      "start": 0.12,
      "end": 2.48,
      "text": "第一段发言"
    }
  ],
  "speaker_segments": [
    {
      "speaker": "Speaker_00",
      "start": 0.12,
      "end": 2.48,
      "text": "第一段发言"
    }
  ]
}
```

`segments` 是模型句段级结果；`speaker_segments` 会把同一匿名说话人、间隔不超过 250 ms 的相邻句段合并为一个发言区间。

## 接入现有 ASR

默认情况下，现有 Qwen3-ASR 不变。普通转写和监听请求可以通过 multipart 字段按请求开启：

```text
speaker_diarization=1
```

也可以全局开启：

```bash
export LIVETALKING_SPEAKER_DIARIZATION=1
export FUNASR_API_KEY="部署服务时设置的密钥"
export SENSEVOICE_LANGUAGE=auto
```

然后启动 LiveTalking：

```bash
cd LiveTalking
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
```

开启后，以下接口会在原有 `recognized` 字段之外附加 `speaker_diarization`：

- `POST /transcribe_audio`
- `POST /humanaudio_monitor`
- `POST /humanaudio`

健康检查：

```bash
curl http://127.0.0.1:8010/api/diarization/health
```

## 常用参数

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SENSEVOICE_MODEL` | `iic/SenseVoiceSmall` | SenseVoice 模型 |
| `SENSEVOICE_BATCH_SIZE_S` | `60` | FunASR 按秒累计的 batch 大小 |
| `SENSEVOICE_MERGE_LENGTH_S` | `15` | VAD 合并长度 |
| `SENSEVOICE_MAX_SINGLE_SEGMENT_MS` | `30000` | 单个 VAD 片段最大长度 |
| `SENSEVOICE_MAX_UPLOAD_MB` | `500` | 日志化 API 上传大小限制 |
| `SENSEVOICE_DEVICE` | `auto` | `auto`、`cpu` 或 CUDA 设备 |
| `SENSEVOICE_LANGUAGE` | `auto` | 默认语言提示 |
| `LIVETALKING_SPEAKER_DIARIZATION_PRELOAD` | `0` | 是否在主服务启动时预加载模型 |
| `FUNASR_API_KEY` | 空 | 远程 API Key；生产环境必须设置 |

`GET /health` 和 `GET /api/diarization/health` 返回模型状态；`/health` 不会触发模型加载。FunASR 不可用时，LiveTalking 会保留原有 ASR 结果，并返回 `speaker_diarization_error`，不会让主转写或语音对话失败。

当前主容器直接安装 FunASR 和 ModelScope，模型组合为 SenseVoiceSmall + FSMN-VAD + CAM++ + CT-Punc。升级 FunASR 时，请重新确认 `sentence_info` 和 CAM++ 输出格式。
