"""CPU-only emotion2vec HTTP service for the conversation analysis pipeline."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile


SERVICE_VERSION = "1.0.0"
MODEL_PATH = Path(
    os.getenv(
        "EMOTION_MODEL_PATH",
        "/home/hf/.cache/modelscope/hub/models/iic/emotion2vec_plus_base",
    )
).expanduser()
DEVICE = os.getenv("EMOTION_DEVICE", "cpu").strip()
FFMPEG = os.getenv(
    "EMOTION_FFMPEG",
    "/home/hf/.conda/envs/emotion-api/bin/ffmpeg",
).strip()
MAX_UPLOAD_BYTES = int(os.getenv("EMOTION_MAX_UPLOAD_MB", "100")) * 1024 * 1024
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("EMOTION_INFERENCE_TIMEOUT", "180"))


def _load_model():
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"emotion model directory does not exist: {MODEL_PATH}")
    from funasr import AutoModel

    return AutoModel(
        model=str(MODEL_PATH),
        device=DEVICE,
        disable_update=True,
    )


def _convert_to_wav(source: Path, target: Path) -> None:
    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("audio conversion timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace")[-500:]
        raise ValueError(f"unsupported or invalid audio: {detail}") from exc


def _parse_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        raise RuntimeError("emotion2vec returned an invalid result")
    row = raw[0]
    labels = row.get("labels")
    scores = row.get("scores")
    if not isinstance(labels, list) or not isinstance(scores, list) or not labels:
        raise RuntimeError("emotion2vec result is missing labels or scores")
    if len(labels) != len(scores):
        raise RuntimeError("emotion2vec labels and scores have different lengths")
    normalized_scores = {
        str(label): max(0.0, min(1.0, float(score)))
        for label, score in zip(labels, scores, strict=True)
    }
    label, confidence = max(normalized_scores.items(), key=lambda item: item[1])
    return {
        "emotion": label,
        "confidence": confidence,
        "scores": normalized_scores,
        "scope": "whole_audio",
    }


def _predict(model: Any, wav_path: Path) -> dict[str, Any]:
    raw = model.generate(
        input=str(wav_path),
        granularity="utterance",
        extract_embedding=False,
    )
    return _parse_result(raw)


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    app.state.model = await asyncio.to_thread(_load_model)
    app.state.model_load_seconds = round(time.perf_counter() - started, 3)
    app.state.inference_lock = asyncio.Lock()
    yield
    app.state.model = None


app = FastAPI(
    title="emotion2vec CPU API",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "emotion2vec",
        "version": SERVICE_VERSION,
        "model": MODEL_PATH.name,
        "device": DEVICE,
        "model_loaded": app.state.model is not None,
        "model_load_seconds": app.state.model_load_seconds,
    }


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(description="Audio file to classify")],
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=422, detail="audio file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio file is too large")

    suffix = Path(file.filename or "audio.bin").suffix[:16] or ".audio"
    with tempfile.TemporaryDirectory(prefix="emotion-api-") as temp_dir:
        source = Path(temp_dir) / f"source{suffix}"
        wav_path = Path(temp_dir) / "audio.wav"
        source.write_bytes(payload)
        try:
            await asyncio.to_thread(_convert_to_wav, source, wav_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            async with app.state.inference_lock:
                result = await asyncio.wait_for(
                    asyncio.to_thread(_predict, app.state.model, wav_path),
                    timeout=INFERENCE_TIMEOUT_SECONDS,
                )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="emotion inference timed out") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result

