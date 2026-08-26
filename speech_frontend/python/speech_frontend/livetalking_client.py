from __future__ import annotations

from dataclasses import dataclass

import aiohttp


@dataclass(frozen=True)
class PreparedAudio:
    audio_bytes: bytes
    has_speech: bool
    trace_id: str
    interval_count: int
    speech_seconds: float


class LiveTalkingSpeechFrontendClient:
    """Small client that can be copied into LiveTalking's server package."""

    def __init__(self, base_url: str = "http://127.0.0.1:8092", timeout_seconds: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def prepare_for_asr(
        self,
        audio_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        sessionid: str,
        trace_id: str = "",
    ) -> PreparedAudio:
        form = aiohttp.FormData()
        form.add_field("sessionid", str(sessionid))
        if trace_id:
            form.add_field("client_trace_id", trace_id)
        form.add_field("output", "speech")
        form.add_field(
            "file",
            audio_bytes,
            filename=filename or "audio.webm",
            content_type=content_type or "application/octet-stream",
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/api/v1/speech_frontend/prepare",
                data=form,
            ) as response:
                if response.status == 204:
                    return PreparedAudio(
                        b"",
                        False,
                        response.headers.get("X-Trace-ID", trace_id),
                        int(response.headers.get("X-VAD-Interval-Count", "0")),
                        float(response.headers.get("X-VAD-Speech-Seconds", "0")),
                    )
                if response.status != 200:
                    detail = await response.text()
                    raise RuntimeError(f"speech frontend HTTP {response.status}: {detail}")
                return PreparedAudio(
                    await response.read(),
                    True,
                    response.headers.get("X-Trace-ID", trace_id),
                    int(response.headers.get("X-VAD-Interval-Count", "0")),
                    float(response.headers.get("X-VAD-Speech-Seconds", "0")),
                )

