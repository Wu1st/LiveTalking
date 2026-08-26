# LiveTalking 延迟与传输观测

本项目保留原有 `livetalking.log`，并额外输出适合统计的
`latency_metrics.jsonl`。结构化日志只记录耗时、字符数、队列长度和媒体统计，
不记录完整对话文本。

## 日志文件

默认路径为项目运行目录下的：

```text
latency_metrics.jsonl
```

单文件达到 50 MB 后自动轮转，最多保留 5 份。可在启动前修改路径：

```bash
export LIVETALKING_METRICS_LOG=/path/to/latency_metrics.jsonl
```

每条事件包含：

- `timestamp`：带时区的事件时间；
- `trace_id`：一次输入到播放链路的关联编号；
- `session_id`：LiveTalking 会话编号；
- `source`：`audio_chat`、`text_chat`、`audio_monitor` 等入口；
- `stage`：ASR、LLM、TTS、播放或 WebRTC 阶段；
- `relative_ms`：相对本次服务端请求开始的毫秒数。

浏览器每 5 秒采集一次 WebRTC 指标，包括 RTT、jitter、丢包、视频丢帧、
freeze、NACK、PLI 和音频 concealment。浏览器不支持某项指标时，该字段为空，
不会影响页面功能。

## 生成统计报告

仅统计智能对话的音频链路：

```bash
python tools/analyze_latency.py latency_metrics.jsonl \
  --source audio_chat \
  --source browser \
  --output latency_report.txt
```

统计所有入口：

```bash
python tools/analyze_latency.py latency_metrics.jsonl \
  --output latency_report.txt
```

若要同时分析轮转文件，把文件按从旧到新的顺序全部传入：

```bash
python tools/analyze_latency.py \
  latency_metrics.jsonl.5 latency_metrics.jsonl.4 latency_metrics.jsonl.3 \
  latency_metrics.jsonl.2 latency_metrics.jsonl.1 latency_metrics.jsonl
```

报告包含：

- multipart 上传接收、ffmpeg、ASR 和 RTF；
- 0～3 秒、3～20 秒、20～60 秒和 60 秒以上的 ASR 分桶；
- LLM 排队、首 token、完整输出和首个 TTS 文本提交；
- TTS 文本排队、HTTP POST、首 HTTP 音频块和首 PCM；
- 请求开始到播放通知的服务端端到端延迟；
- 加上前端静音等待和录音封装等待后的体感估算；
- WebRTC RTT、音视频 jitter、累计丢包；
- 最近的结构化错误事件。

P95 在样本少于 20 条时只能用于观察。正式对比应使用固定测试语料，并至少采集
50 次热状态对话及 10 次打断场景。

服务器端完整操作流程、CosyVoice 内部指标、GPU 采样与组件基准测试见：

```text
/home/hf/qwen3-asr/observability/README.md
```
