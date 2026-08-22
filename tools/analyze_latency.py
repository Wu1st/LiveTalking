#!/usr/bin/env python3
"""Aggregate ``latency_metrics.jsonl`` into a data.txt-style report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: Iterable[Any]) -> dict[str, float | int] | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return {
        "n": len(numbers),
        "avg": statistics.fmean(numbers),
        "p50": percentile(numbers, 0.50),
        "p95": percentile(numbers, 0.95),
        "max": max(numbers),
    }


def load_events(paths: list[Path]) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    invalid = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and event.get("stage"):
                        events.append(event)
                    else:
                        invalid += 1
                except json.JSONDecodeError:
                    invalid += 1
    return events, invalid


def first_by_stage(trace_events: list[dict[str, Any]], stage: str):
    matches = [event for event in trace_events if event.get("stage") == stage]
    if not matches:
        return None
    return min(matches, key=lambda event: event.get("relative_ms", math.inf))


def duration_bucket(duration: float) -> str:
    if duration < 3:
        return "0～3秒"
    if duration < 20:
        return "3～20秒"
    if duration < 60:
        return "20～60秒"
    return "60秒以上"


def metric_row(name: str, stats: dict[str, float | int] | None, unit: str = "ms") -> str:
    if stats is None:
        return f"{name:<28} 无数据"
    return (
        f"{name:<28} n={stats['n']:>4}  avg={stats['avg']:>9.2f}{unit}  "
        f"P50={stats['p50']:>9.2f}{unit}  P95={stats['p95']:>9.2f}{unit}  "
        f"max={stats['max']:>9.2f}{unit}"
    )


def values_at(events: list[dict[str, Any]], stage: str, field: str) -> list[float]:
    return [
        float(event[field])
        for event in events
        if event.get("stage") == stage and isinstance(event.get(field), (int, float))
    ]


def build_report(events: list[dict[str, Any]], invalid: int) -> str:
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("trace_id"):
            traces[str(event["trace_id"])].append(event)

    lines = [
        "LiveTalking 延迟与传输统计",
        "=" * 78,
        f"有效事件: {len(events)}    trace数量: {len(traces)}    无效行: {invalid}",
        "数据来源: " + ", ".join(sorted({str(event.get('source', 'unknown')) for event in events})),
        "说明: P95少于20个样本时仅供观察，不应作为稳定SLO。",
        "",
        "一、ASR",
        "-" * 78,
        metric_row("multipart接收/解析", describe(values_at(events, "audio_upload_received", "form_parse_ms"))),
        metric_row("ffmpeg转换", describe(values_at(events, "asr_final", "ffmpeg_ms"))),
        metric_row("ASR模型推理", describe(values_at(events, "asr_final", "asr_ms"))),
        metric_row("服务端ASR总耗时", describe(values_at(events, "asr_final", "total_asr_ms"))),
    ]

    asr_events = [event for event in events if event.get("stage") == "asr_final"]
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in asr_events:
        duration = event.get("audio_duration_s")
        if isinstance(duration, (int, float)):
            bucketed[duration_bucket(float(duration))].append(event)
    lines.extend(["", "ASR按音频时长分桶:"])
    for bucket in ("0～3秒", "3～20秒", "20～60秒", "60秒以上"):
        group = bucketed.get(bucket, [])
        lines.append(
            metric_row(
                bucket,
                describe(event.get("total_asr_ms") for event in group),
            )
        )

    lines.extend([
        "",
        "二、LLM",
        "-" * 78,
        metric_row("LLM排队/调度", describe(values_at(events, "llm_started", "dispatch_wait_ms"))),
        metric_row("LLM首token", describe(values_at(events, "llm_first_token", "duration_ms"))),
        metric_row("LLM完整输出", describe(values_at(events, "llm_finished", "duration_ms"))),
        metric_row("首个TTS文本提交", describe(values_at(events, "llm_tts_first_enqueue", "duration_ms"))),
        "",
        "三、TTS",
        "-" * 78,
        metric_row("TTS文本队列等待", describe(values_at(events, "tts_dequeued", "queue_ms"))),
        metric_row("TTS HTTP POST", describe(values_at(events, "tts_http_posted", "duration_ms"))),
        metric_row("TTS请求到首HTTP块", describe(values_at(events, "tts_http_first_chunk", "duration_ms"))),
        metric_row("POST到首HTTP块", describe(values_at(events, "tts_http_first_chunk", "post_to_first_ms"))),
        metric_row("TTS出队到首PCM入队", describe(values_at(events, "tts_first_pcm_enqueued", "duration_ms"))),
        "",
        "四、端到端",
        "-" * 78,
    ])

    server_e2e: list[float] = []
    perceived_estimate: list[float] = []
    pcm_to_playback: list[float] = []
    for trace_events in traces.values():
        request = first_by_stage(trace_events, "request_started")
        playback = first_by_stage(trace_events, "avatar_playback_start")
        pcm = first_by_stage(trace_events, "tts_first_pcm_enqueued")
        if playback and isinstance(playback.get("relative_ms"), (int, float)):
            server_ms = float(playback["relative_ms"])
            server_e2e.append(server_ms)
            silence_ms = float((request or {}).get("client_silence_wait_ms") or 0)
            finalize_ms = float((request or {}).get("client_speech_end_to_upload_ms") or 0)
            perceived_estimate.append(server_ms + silence_ms + finalize_ms)
        if pcm and playback:
            if isinstance(pcm.get("relative_ms"), (int, float)) and isinstance(
                playback.get("relative_ms"), (int, float)
            ):
                pcm_to_playback.append(
                    float(playback["relative_ms"]) - float(pcm["relative_ms"])
                )

    lines.extend([
        metric_row("服务端请求开始→播放通知", describe(server_e2e)),
        metric_row("首PCM入队→播放通知", describe(pcm_to_playback)),
        metric_row("用户体感估算", describe(perceived_estimate)),
        "注: 用户体感估算=静音等待+录音封装等待+服务端请求开始到播放通知；",
        "    尚不含浏览器最终音频设备播放缓冲，客户端与服务端时钟未做同步。",
        "",
        "五、WebRTC客户端采样",
        "-" * 78,
        metric_row("当前RTT", describe(values_at(events, "client_webrtc_stats", "rtt_ms"))),
        metric_row("音频jitter", describe(values_at(events, "client_webrtc_stats", "audio_jitter_ms"))),
        metric_row("视频jitter", describe(values_at(events, "client_webrtc_stats", "video_jitter_ms"))),
        metric_row("音频累计丢包", describe(values_at(events, "client_webrtc_stats", "audio_packets_lost")), ""),
        metric_row("视频累计丢包", describe(values_at(events, "client_webrtc_stats", "video_packets_lost")), ""),
        "",
        "六、错误与异常",
        "-" * 78,
    ])

    errors = [event for event in events if str(event.get("stage", "")).endswith("_error")]
    lines.append(f"结构化错误事件: {len(errors)}")
    for event in errors[-20:]:
        lines.append(
            f"- {event.get('timestamp', '')} trace={event.get('trace_id', '-')} "
            f"stage={event.get('stage')} error={event.get('error', '')}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("latency_metrics.jsonl")],
        help="JSONL files, including rotated files when needed",
    )
    parser.add_argument("-o", "--output", type=Path, help="write report to this path")
    parser.add_argument(
        "--source",
        action="append",
        help="only include this source; may be supplied more than once (for example audio_chat)",
    )
    args = parser.parse_args()

    missing = [str(path) for path in args.paths if not path.exists()]
    if missing:
        parser.error("file not found: " + ", ".join(missing))

    events, invalid = load_events(args.paths)
    if args.source:
        selected = set(args.source)
        events = [event for event in events if event.get("source") in selected]
    report = build_report(events, invalid)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
