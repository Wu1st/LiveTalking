from __future__ import annotations

import io
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix and len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".audio"


def decode_audio_bytes(audio_bytes: bytes, filename: str = "audio") -> np.ndarray:
    """Decode a browser/LiveTalking audio upload to mono float32 PCM at 16 kHz."""
    input_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=_safe_suffix(filename),
            delete=False,
        ) as temporary:
            temporary.write(audio_bytes)
            input_path = temporary.name
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", input_path,
                "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace")[-600:]
        raise ValueError(f"无法解码上传的音频{': ' + detail if detail else ''}")
    audio = np.frombuffer(completed.stdout, dtype="<f4").copy()
    return np.ascontiguousarray(
        np.clip(np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0),
        dtype=np.float32,
    )


def encode_pcm16_wav(audio: np.ndarray) -> bytes:
    value = np.asarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)
    pcm = np.rint(value * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(pcm.tobytes())
    return output.getvalue()


def extract_speech(
    enhanced: np.ndarray,
    intervals: list[dict],
    gap_ms: int = 120,
) -> np.ndarray:
    if not intervals:
        return np.empty(0, dtype=np.float32)
    gap = np.zeros(max(0, round(16000 * gap_ms / 1000.0)), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for index, interval in enumerate(intervals):
        start = max(0, int(interval["start_sample"]))
        end = min(len(enhanced), int(interval["end_sample"]))
        if end <= start:
            continue
        if chunks and len(gap):
            chunks.append(gap)
        chunks.append(np.asarray(enhanced[start:end], dtype=np.float32))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
