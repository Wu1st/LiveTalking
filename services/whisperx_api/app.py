import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import whisperx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from whisperx.diarize import DiarizationPipeline


DEVICE = os.getenv("WHISPERX_DEVICE", "cuda")
DEVICE_INDEX = int(os.getenv("WHISPERX_DEVICE_INDEX", "0"))
MODEL_NAME = os.getenv("WHISPERX_MODEL", "large-v3")
COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "4"))

MODEL_ROOT = os.getenv("WHISPERX_MODEL_ROOT", "/models/whisper")
ALIGN_ROOT = os.getenv("WHISPERX_ALIGN_ROOT", "/models/alignment")
ALIGN_DEVICE = os.getenv("WHISPERX_ALIGN_DEVICE", DEVICE)
DIARIZE_DEVICE = os.getenv("WHISPERX_DIARIZE_DEVICE", DEVICE)
HF_TOKEN = os.getenv("HF_TOKEN", "")

MAX_UPLOAD_BYTES = int(os.getenv("WHISPERX_MAX_UPLOAD_MB", "500")) * 1024 * 1024

ASR_MODEL = None
ALIGN_MODELS = {}
DIARIZER = None
GPU_LOCK = asyncio.Semaphore(1)


def _json_default(value):
    """Convert common NumPy/PyTorch scalar values into JSON-safe values."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _get_align_model(language: str):
    """Load and cache the language-specific forced-alignment model."""
    if language not in ALIGN_MODELS:
        ALIGN_MODELS[language] = whisperx.load_align_model(
            language_code=language,
            device=ALIGN_DEVICE,
            model_dir=ALIGN_ROOT,
        )
    return ALIGN_MODELS[language]


def _get_diarizer():
    """Load the optional pyannote diarization pipeline on first use."""
    global DIARIZER

    if DIARIZER is None:
        if not HF_TOKEN:
            raise RuntimeError("启用说话人分离时必须设置 HF_TOKEN")

        DIARIZER = DiarizationPipeline(
            token=HF_TOKEN,
            device=DIARIZE_DEVICE,
            cache_dir=os.getenv("HF_HOME", "/models/huggingface"),
        )

    return DIARIZER


def _run_pipeline(
    audio_path: str,
    mode: str,
    language: str | None,
    align: bool,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
):
    """Run WhisperX transcription and the optional alignment/diarization steps.

    Args:
        audio_path: Temporary local audio path.
        mode: `fast` for transcription only or `full` for extra processing.
        language: Forced language code, or None for language detection.
        align: Whether to produce word-level alignment.
        diarize: Whether to assign speaker labels.
        min_speakers: Optional lower bound for diarization.
        max_speakers: Optional upper bound for diarization.

    Returns:
        A JSON-serializable transcription result containing text, language,
        segments, and processing flags.
    """
    if mode == "fast":
        align = False
        diarize = False

    audio = whisperx.load_audio(audio_path)

    result = ASR_MODEL.transcribe(
        audio,
        batch_size=BATCH_SIZE,
        language=language,
    )

    if align:
        detected_language = result["language"]
        align_model, metadata = _get_align_model(detected_language)
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            ALIGN_DEVICE,
            return_char_alignments=False,
        )
        result["language"] = detected_language

    if diarize:
        diarize_segments = _get_diarizer()(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = whisperx.assign_word_speakers(
            diarize_segments,
            result,
        )

    result_language = result.get("language") or language or ""
    separator = "" if result_language in {"zh", "ja"} else " "
    text = separator.join(
        segment.get("text", "").strip()
        for segment in result.get("segments", [])
        if segment.get("text")
    ).strip()

    return {
        "text": text,
        "language": result_language,
        "segments": result.get("segments", []),
        "mode": mode,
        "aligned": align,
        "diarized": diarize,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the WhisperX model before accepting requests."""
    global ASR_MODEL

    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(ALIGN_ROOT, exist_ok=True)

    ASR_MODEL = whisperx.load_model(
        MODEL_NAME,
        DEVICE,
        device_index=DEVICE_INDEX,
        compute_type=COMPUTE_TYPE,
        download_root=MODEL_ROOT,
        vad_method="silero",
    )

    yield


app = FastAPI(
    title="LiveTalking WhisperX Service",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Return model loading state and CUDA availability."""
    return {
        "status": "ready" if ASR_MODEL is not None else "loading",
        "model": MODEL_NAME,
        "device": DEVICE,
        "device_index": DEVICE_INDEX,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    mode: str = Form("fast"),
    language: str = Form("zh"),
    align: bool = Form(False),
    diarize: bool = Form(False),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    """Accept one uploaded audio file and return its WhisperX transcription.

    Args:
        file: Multipart audio upload.
        mode: `fast` for text only or `full` for alignment/diarization options.
        language: Language code, or `auto` to enable detection.
        align: Enable word-level forced alignment in `full` mode.
        diarize: Enable speaker diarization in `full` mode.
        min_speakers: Optional minimum speaker count.
        max_speakers: Optional maximum speaker count.

    Returns:
        A JSON response with normalized text and WhisperX segments.
    """
    if mode not in {"fast", "full"}:
        raise HTTPException(400, "mode 只支持 fast 或 full")

    if language.lower() == "auto":
        language_value = None
    else:
        language_value = language.strip() or None

    suffix = Path(file.filename or "").suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".audio"

    fd, audio_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        total = 0
        with open(audio_path, "wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "上传音频超过大小限制")

                output.write(chunk)

        async with GPU_LOCK:
            result = await asyncio.to_thread(
                _run_pipeline,
                audio_path,
                mode,
                language_value,
                align,
                diarize,
                min_speakers,
                max_speakers,
            )

        result["filename"] = Path(file.filename or "audio").name

        serializable = json.loads(
            json.dumps(result, ensure_ascii=False, default=_json_default)
        )
        return JSONResponse(serializable)

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(500, f"WhisperX 转写失败：{error}") from error
    finally:
        try:
            os.unlink(audio_path)
        except FileNotFoundError:
            pass
