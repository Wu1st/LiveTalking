from __future__ import annotations

import unittest

import numpy as np
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from speech_frontend.api import create_app
from speech_frontend.media import encode_pcm16_wav
from speech_frontend.pipeline import PipelineResult


class FakeEngine:
    config = {"product": "test"}

    def process(self, audio: np.ndarray) -> PipelineResult:
        has_speech = bool(np.max(np.abs(audio), initial=0.0) > 0.01)
        intervals = (
            [{"start_sample": 160, "end_sample": min(len(audio), 1600),
              "start_seconds": 0.01, "end_seconds": min(len(audio), 1600) / 16000.0}]
            if has_speech and len(audio) > 160 else []
        )
        frames = (len(audio) + 511) // 512
        return PipelineResult(
            original=audio,
            enhanced=audio * 0.5,
            raw_probabilities=np.zeros(frames, dtype=np.float32),
            wiener_probabilities=np.zeros(frames, dtype=np.float32),
            intervals=intervals,
            patent_audit={"mechanism": "test"},
            duration_seconds=len(audio) / 16000.0,
            speech_duration_seconds=sum(
                (item["end_sample"] - item["start_sample"]) / 16000.0 for item in intervals
            ),
            processing_ms=1.25,
            core_rtf=0.01,
        )


def upload(audio: np.ndarray, **fields: str) -> FormData:
    form = FormData()
    form.add_field("file", encode_pcm16_wav(audio), filename="test.wav", content_type="audio/wav")
    for key, value in fields.items():
        form.add_field(key, value)
    return form


class ApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = TestClient(TestServer(create_app(engine=FakeEngine())))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_health_uses_livetalking_envelope(self) -> None:
        response = await self.client.get("/health")
        payload = await response.json()
        self.assertEqual(payload["code"], 0)
        self.assertEqual(payload["msg"], "ok")
        self.assertEqual(payload["data"]["sample_rate"], 16000)

    async def test_analyze_accepts_file_and_sessionid(self) -> None:
        audio = np.full(3200, 0.25, dtype=np.float32)
        response = await self.client.post(
            "/api/v1/speech_frontend/analyze",
            data=upload(audio, sessionid="live-42", client_trace_id="trace-7"),
        )
        payload = await response.json()
        self.assertEqual(payload["code"], 0)
        self.assertEqual(payload["data"]["sessionid"], "live-42")
        self.assertEqual(payload["data"]["trace_id"], "trace-7")
        self.assertEqual(payload["data"]["speech_interval_count"], 1)

    async def test_prepare_returns_asr_ready_wav(self) -> None:
        audio = np.full(3200, 0.25, dtype=np.float32)
        response = await self.client.post(
            "/api/v1/speech_frontend/prepare",
            data=upload(audio, sessionid="live-42", output="speech"),
        )
        body = await response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(body[:4], b"RIFF")
        self.assertEqual(response.headers["X-Session-ID"], "live-42")
        self.assertEqual(response.headers["X-VAD-Interval-Count"], "1")

    async def test_prepare_no_speech_returns_204(self) -> None:
        response = await self.client.post(
            "/api/v1/speech_frontend/prepare",
            data=upload(np.zeros(3200, dtype=np.float32), output="speech"),
        )
        self.assertEqual(response.status, 204)
        self.assertEqual(response.headers["X-VAD-Interval-Count"], "0")

    async def test_livetalking_alias(self) -> None:
        response = await self.client.post(
            "/humanaudio_vad",
            data=upload(np.full(3200, 0.25, dtype=np.float32), sessionid="9"),
        )
        payload = await response.json()
        self.assertEqual(payload["code"], 0)
        self.assertEqual(payload["data"]["sessionid"], "9")


if __name__ == "__main__":
    unittest.main()

