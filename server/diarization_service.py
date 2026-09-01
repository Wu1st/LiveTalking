"""In-process FunASR speaker-diarization service for LiveTalking.

The model is hosted inside the existing LiveTalking process so the public API
stays on the original LiveTalking port and no second Docker container is
needed.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
from pathlib import Path
from typing import Any

from server.diarization_normalizer import normalize_result


LOGGER = logging.getLogger(__name__)

DEVICE_SETTING = os.getenv(
    "SENSEVOICE_DEVICE",
    os.getenv("FUNASR_DIARIZATION_DEVICE", "auto"),
).strip()
DEVICE_INDEX = int(os.getenv("SENSEVOICE_DEVICE_INDEX", "0"))
MODEL_NAME = os.getenv("SENSEVOICE_MODEL", "iic/SenseVoiceSmall")
VAD_MAX_SEGMENT_MS = int(os.getenv("SENSEVOICE_MAX_SINGLE_SEGMENT_MS", "30000"))
SPEECH_BATCH_SIZE_S = int(os.getenv("SENSEVOICE_BATCH_SIZE_S", "60"))
MERGE_LENGTH_S = int(os.getenv("SENSEVOICE_MERGE_LENGTH_S", "15"))
LANGUAGE = os.getenv("SENSEVOICE_LANGUAGE", "auto").strip() or "auto"
API_KEY = os.getenv("FUNASR_API_KEY", "").strip()
GLOBAL_ENABLED = os.getenv(
    "LIVETALKING_SPEAKER_DIARIZATION",
    "0",
).lower() in {"1", "true", "yes"}
PRELOAD_ENABLED = os.getenv(
    "LIVETALKING_SPEAKER_DIARIZATION_PRELOAD",
    "0",
).lower() in {"1", "true", "yes"}
MAX_UPLOAD_BYTES = int(
    os.getenv("SENSEVOICE_MAX_UPLOAD_MB", "500")
) * 1024 * 1024

MODEL: Any | None = None
MODEL_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()


def _resolve_device() -> str:
    """Resolve ``auto`` to a usable FunASR device string.

    Returns:
        The configured CPU or CUDA device used by the in-process model.
    """
    if DEVICE_SETTING.lower() not in {"", "auto"}:
        return DEVICE_SETTING

    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{DEVICE_INDEX}"
    except Exception as exc:
        LOGGER.warning("Unable to inspect CUDA for FunASR: %s", exc)
    return "cpu"


def _postprocess(text: str) -> str:
    """Remove SenseVoice control tokens from a transcript.

    Args:
        text: Raw SenseVoice transcript, which may contain emotion or event
            control tokens.

    Returns:
        Human-readable transcript text.
    """
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    return rich_transcription_postprocess(text)


def _load_model() -> Any:
    """Create the SenseVoice, VAD, punctuation, and CAM++ pipeline.

    Returns:
        A FunASR ``AutoModel`` configured for speaker diarization.

    Raises:
        RuntimeError: If FunASR is not installed in the existing container.
    """
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "当前 LiveTalking 容器未安装 FunASR，请在原容器内安装 "
            "funasr 和 modelscope"
        ) from exc

    return AutoModel(
        model=MODEL_NAME,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": VAD_MAX_SEGMENT_MS},
        spk_model="cam++",
        punc_model="ct-punc",
        device=_resolve_device(),
    )


def get_model() -> Any:
    """Load the in-process FunASR pipeline once and return it.

    Returns:
        The cached FunASR model pipeline.

    Notes:
        A process lock prevents concurrent first requests from downloading or
        constructing duplicate model instances.
    """
    global MODEL
    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:
        if MODEL is None:
            LOGGER.info(
                "Loading in-process FunASR diarization model=%s device=%s",
                MODEL_NAME,
                _resolve_device(),
            )
            MODEL = _load_model()
            LOGGER.info("FunASR diarization model is ready")
    return MODEL


def preload() -> None:
    """Optionally warm up the diarization model during LiveTalking startup.

    Lazy loading is the default so the existing Qwen3-ASR startup path keeps
    its current memory footprint. Set
    ``LIVETALKING_SPEAKER_DIARIZATION_PRELOAD=1`` to load it before serving.
    """
    if PRELOAD_ENABLED:
        get_model()


def shutdown() -> None:
    """Release the cached FunASR model before the process exits.

    This is best-effort because LiveTalking must still shut down cleanly when
    optional FunASR dependencies are unavailable.
    """
    global MODEL
    with MODEL_LOCK:
        MODEL = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def diarize(audio_path: str, language: str | None = None) -> dict[str, Any]:
    """Run local FunASR speaker diarization for one audio file.

    Args:
        audio_path: Local audio path. The caller owns temporary-file cleanup.
        language: SenseVoice language code such as ``zh`` or ``auto``.

    Returns:
        JSON-safe transcript and anonymous speaker intervals using labels such
        as ``Speaker_00`` and ``Speaker_01``.

    Raises:
        RuntimeError: If the model cannot load or inference fails.
        ValueError: If FunASR returns unusable speaker timestamps.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise RuntimeError(f"音频文件不存在：{audio_path}")

    # Serializing inference avoids concurrent GPU execution corrupting model
    # side caches while retaining concurrent HTTP request handling.
    with INFERENCE_LOCK:
        model = get_model()
        result = model.generate(
            input=str(path),
            cache={},
            language=(language or "auto").strip() or "auto",
            use_itn=True,
            batch_size_s=SPEECH_BATCH_SIZE_S,
            merge_vad=True,
            merge_length_s=MERGE_LENGTH_S,
        )

    normalized = normalize_result(result, postprocess_text=_postprocess)
    normalized.update(
        {
            "backend": "funasr-sensevoice-cam++",
            "model": MODEL_NAME,
            "device": _resolve_device(),
            "language": (language or "auto").strip() or "auto",
            "vad_model": "fsmn-vad",
            "speaker_model": "cam++",
            "punc_model": "ct-punc",
        }
    )
    return normalized


def requested(value: Any = None) -> bool:
    """Return whether diarization should run for a regular audio request.

    Args:
        value: Optional request-level override. Values such as ``1``, ``true``
            and ``on`` enable the feature.

    Returns:
        True when the request should start speaker diarization.
    """
    if value is None:
        return GLOBAL_ENABLED
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def authorize_request(request: Any) -> bool:
    """Check the API key for the public ``/v1/diarize`` endpoint.

    Args:
        request: aiohttp request containing HTTP headers.

    Returns:
        True when authentication succeeds or no API key is configured.
    """
    if not API_KEY:
        return True

    candidate = request.headers.get("X-API-Key", "").strip()
    if not candidate:
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            candidate = token.strip()
    return bool(candidate) and hmac.compare_digest(candidate, API_KEY)


def health() -> dict[str, Any]:
    """Return local model state and API configuration.

    Returns:
        A JSON-serializable health payload. This function never triggers model
        loading, so health probes do not cause a GPU allocation.
    """
    return {
        "status": "ready" if MODEL is not None else "loading",
        "backend": "funasr-sensevoice-cam++",
        "model": MODEL_NAME,
        "device": _resolve_device(),
        "model_loaded": MODEL is not None,
        "api_mode": "in-process",
        "authentication": {
            "enabled": bool(API_KEY),
            "accepted_headers": ["X-API-Key", "Authorization: Bearer"],
        },
        "models": {
            "vad": "fsmn-vad",
            "speaker": "cam++",
            "punctuation": "ct-punc",
        },
    }
