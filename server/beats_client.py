"""Reusable client and text formatting for the local BEATs service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiohttp

from utils.logger import logger


BEATS_CLIENT_KEY = "beats_client"

_LABELS_ZH = {
    "Speech": "人声",
    "Male speech, man speaking": "男性说话",
    "Female speech, woman speaking": "女性说话",
    "Child speech, kid speaking": "儿童说话",
    "Conversation": "多人交谈",
    "Narration, monologue": "独白或旁白",
    "Speech synthesizer": "合成语音",
    "Whispering": "低声耳语",
    "Shout": "喊叫",
    "Screaming": "尖叫",
    "Laughter": "笑声",
    "Crying, sobbing": "哭泣",
    "Cough": "咳嗽",
    "Sneeze": "喷嚏",
    "Breathing": "呼吸声",
    "Typing": "打字声",
    "Computer keyboard": "电脑键盘声",
    "Writing": "书写声",
    "Door": "门声",
    "Knock": "敲击声",
    "Walk, footsteps": "脚步声",
    "Music": "音乐",
    "Vehicle": "车辆声",
    "Car": "汽车声",
    "Motor vehicle (road)": "道路机动车声",
    "Engine": "发动机声",
    "Traffic noise, roadway noise": "道路交通声",
    "Rail transport": "轨道交通声",
    "Train": "火车声",
    "Aircraft": "飞行器声",
    "Siren": "警报器声",
    "Alarm": "警报声",
    "Rain": "雨声",
    "Wind": "风声",
    "Thunderstorm": "雷雨声",
    "Water": "水声",
    "Dog": "狗叫声",
    "Bark": "犬吠声",
    "Bird": "鸟鸣声",
    "Silence": "安静",
    "Inside, small room": "小型室内空间",
    "Inside, large room or hall": "大型室内空间或大厅",
    "Inside, public space": "室内公共空间",
    "Outside, urban or manmade": "城市室外环境",
    "Outside, rural or natural": "自然室外环境",
    "Reverberation": "明显混响",
    "Echo": "回声",
    "Noise": "噪声",
    "White noise": "白噪声",
    "Pink noise": "粉红噪声",
}


def _load_ontology() -> dict[str, str]:
    path = Path(__file__).with_name("audioset_ontology.json")
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return {
            str(entry["id"]): str(entry["name"])
            for entry in entries
            if entry.get("id") and entry.get("name")
        }
    except Exception as exc:
        logger.warning("Unable to load AudioSet ontology %s: %s", path, exc)
        return {}


_ONTOLOGY = _load_ontology()


def _format_summary(events: list[dict[str, Any]], limit: int = 5) -> str:
    if not events:
        return "未检测到置信度足够的声音事件"

    parts = []
    seen = set()
    for event in events:
        label = str(event.get("label") or event.get("label_id") or "未知声音")
        label_zh = str(event.get("label_zh") or label)
        if label_zh in seen:
            continue
        seen.add(label_zh)
        confidence = float(event.get("confidence") or 0.0)
        parts.append(f"{label_zh} {confidence * 100:.0f}%")
        if len(parts) >= limit:
            break

    if not parts:
        return "未检测到置信度足够的声音事件"
    return "检测到：" + "、".join(parts)


class BeatsClient:
    """One long-lived HTTP client shared by all LiveTalking requests."""

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "BEATS_SERVER",
            "http://127.0.0.1:50010",
        ).rstrip("/")
        self.timeout_seconds = float(os.getenv("BEATS_TIMEOUT", "15"))
        self.top_k = int(os.getenv("BEATS_TOP_K", "8"))
        self.threshold = float(os.getenv("BEATS_THRESHOLD", "0.10"))
        self.session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self.session is not None and not self.session.closed:
            return
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        connector = aiohttp.TCPConnector(limit=2, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        )

    async def close(self) -> None:
        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = None

    async def health(self) -> dict[str, Any]:
        await self.start()
        assert self.session is not None
        async with self.session.get(f"{self.base_url}/health") as response:
            payload = await response.json(content_type=None)
            if response.status != 200:
                raise RuntimeError(f"BEATs health HTTP {response.status}: {payload}")
            return payload

    async def analyze(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "environment.webm",
        content_type: str = "audio/webm",
    ) -> dict[str, Any]:
        await self.start()
        assert self.session is not None

        form = aiohttp.FormData()
        form.add_field(
            "file",
            audio_bytes,
            filename=filename,
            content_type=content_type or "application/octet-stream",
        )
        params = {
            "top_k": str(self.top_k),
            "threshold": str(self.threshold),
        }
        async with self.session.post(
            f"{self.base_url}/v1/acoustic/analyze",
            params=params,
            data=form,
        ) as response:
            payload = await response.json(content_type=None)
            if response.status != 200:
                detail = payload.get("detail") if isinstance(payload, dict) else payload
                raise RuntimeError(f"BEATs HTTP {response.status}: {detail}")

        events = []
        for raw_event in payload.get("events", []):
            event = dict(raw_event)
            label_id = str(event.get("label") or event.get("label_id") or "")
            label = _ONTOLOGY.get(label_id, label_id or "Unknown")
            event["label_id"] = label_id
            event["label"] = label
            event["label_zh"] = _LABELS_ZH.get(label, label)
            events.append(event)

        return {
            "text": _format_summary(events),
            "events": events,
            "model": payload.get("model"),
            "audio_duration_seconds": payload.get("audio_duration_seconds"),
            "decode_ms": payload.get("decode_ms"),
            "inference_ms": payload.get("inference_ms"),
        }


async def beats_client_context(app):
    """aiohttp cleanup context: start once and always close the session."""
    client = BeatsClient()
    app[BEATS_CLIENT_KEY] = client
    await client.start()
    try:
        try:
            health = await client.health()
            logger.info(
                "BEATs client ready | model=%s device=%s load=%.3fs",
                health.get("model"),
                health.get("device"),
                float(health.get("model_load_seconds") or 0.0),
            )
        except Exception as exc:
            logger.warning(
                "BEATs service is unavailable at startup; ASR/LLM/TTS remain usable: %s",
                exc,
            )
        yield
    finally:
        await client.close()
