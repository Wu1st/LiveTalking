"""ASR backend adapter used by LiveTalking audio routes.

The application can keep using the in-process Qwen3-ASR model or delegate
transcription to the WhisperX HTTP sidecar. Keeping that choice here prevents
the request handlers from depending on either model's implementation details.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests


BACKEND = os.getenv("LIVETALKING_ASR_BACKEND", "qwen3").strip().lower()
WHISPERX_URL = os.getenv(
    "WHISPERX_URL",
    "http://127.0.0.1:8093",
).rstrip("/")
WHISPERX_MODE = os.getenv("WHISPERX_MODE", "fast").strip().lower()
WHISPERX_LANGUAGE = os.getenv("WHISPERX_LANGUAGE", "auto").strip()
WHISPERX_ALIGN = os.getenv("WHISPERX_ALIGN", "0").lower() in {"1", "true", "yes"}
WHISPERX_DIARIZE = os.getenv("WHISPERX_DIARIZE", "0").lower() in {"1", "true", "yes"}
WHISPERX_TIMEOUT = (
    float(os.getenv("WHISPERX_CONNECT_TIMEOUT", "10")),
    float(os.getenv("WHISPERX_READ_TIMEOUT", "600")),
)


def _qwen3_transcribe(audio_path: str) -> str:
    """Use the existing in-process Qwen3-ASR implementation.

    Args:
        audio_path: Local audio path accepted by the Qwen3-ASR service.

    Returns:
        The normalized transcription text.
    """
    # Lazy import keeps WhisperX-only deployments from importing Qwen-specific
    # packages in the request process.
    import qwen3asr_service

    return qwen3asr_service.transcribe(audio_path).strip()


def _whisperx_transcribe(audio_path: str) -> str:
    """Upload one local audio file to the WhisperX sidecar.

    Args:
        audio_path: Local audio path readable by the LiveTalking process.

    Returns:
        The transcription text returned by WhisperX.

    Raises:
        RuntimeError: If the sidecar is unavailable or returns an invalid
            response. The original HTTP status and message are included to
            make deployment failures diagnosable from LiveTalking logs.
    """
    path = Path(audio_path)
    try:
        with path.open("rb") as audio:
            response = requests.post(
                f"{WHISPERX_URL}/v1/transcribe",
                files={
                    "file": (
                        path.name,
                        audio,
                        "audio/wav",
                    ),
                },
                data={
                    "mode": WHISPERX_MODE,
                    "language": WHISPERX_LANGUAGE or "auto",
                    "align": str(WHISPERX_ALIGN).lower(),
                    "diarize": str(WHISPERX_DIARIZE).lower(),
                },
                timeout=WHISPERX_TIMEOUT,
            )
    except requests.RequestException as error:
        raise RuntimeError(
            f"WhisperX 服务不可用（{WHISPERX_URL}）：{error}"
        ) from error

    try:
        payload: dict[str, Any] = response.json()
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"WhisperX 返回了非 JSON 响应（HTTP {response.status_code}）"
        ) from error

    if not response.ok:
        detail = payload.get("detail") or payload.get("msg") or str(payload)
        raise RuntimeError(f"WhisperX 请求失败（HTTP {response.status_code}）：{detail}")

    text = payload.get("text")
    if not isinstance(text, str):
        raise RuntimeError("WhisperX 响应缺少字符串字段 text")
    return text.strip()


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file using the configured backend.

    Args:
        audio_path: Local audio path. The caller owns temporary-file cleanup.

    Returns:
        A whitespace-trimmed transcription string.

    Raises:
        ValueError: If `LIVETALKING_ASR_BACKEND` is unsupported.
        RuntimeError: If the selected backend cannot complete transcription.
    """
    if BACKEND in {"qwen3", "qwen3-asr", "qwen"}:
        return _qwen3_transcribe(audio_path)
    if BACKEND in {"whisperx", "whisper-x"}:
        return _whisperx_transcribe(audio_path)
    raise ValueError(
        "LIVETALKING_ASR_BACKEND 仅支持 qwen3 或 whisperx，"
        f"当前值为 {BACKEND!r}"
    )


def preload() -> None:
    """Preload the selected in-process backend when startup warm-up is needed.

    WhisperX is loaded by its own sidecar container, so this function does
    nothing for that backend. This avoids loading two ASR models into the
    LiveTalking process.
    """
    if BACKEND in {"qwen3", "qwen3-asr", "qwen"}:
        import qwen3asr_service

        qwen3asr_service.get_model()


def health() -> dict[str, Any]:
    """Return local configuration and, for WhisperX, sidecar health.

    Returns:
        A JSON-serializable status dictionary for startup diagnostics.
    """
    result: dict[str, Any] = {
        "backend": BACKEND,
        "whisperx_url": WHISPERX_URL if BACKEND in {"whisperx", "whisper-x"} else None,
    }
    if BACKEND not in {"whisperx", "whisper-x"}:
        return result

    try:
        response = requests.get(
            f"{WHISPERX_URL}/health",
            timeout=WHISPERX_TIMEOUT[0],
        )
        result["sidecar_status_code"] = response.status_code
        result["sidecar"] = response.json()
    except (requests.RequestException, ValueError) as error:
        result["sidecar_error"] = str(error)
    return result
