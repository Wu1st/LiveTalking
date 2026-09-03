#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cj/qwen3-asr"
PROJECT_ROOT="${ROOT}/ocr-translation-api"
PYTHON="${ROOT}/runtimes/python312/bin/python"
OCR_SITE="${ROOT}/runtimes/ocr-final/lib/python3.12/site-packages"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python runtime is missing: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -d "${OCR_SITE}/paddle" ]]; then
  echo "Frozen OCR runtime is missing: ${OCR_SITE}" >&2
  exit 2
fi

OCR_CUDA_LIBS="$(find "${OCR_SITE}" -type d -path '*/nvidia/*/lib' -print | paste -sd:)"
export LD_LIBRARY_PATH="${OCR_SITE}/paddle/libs${OCR_CUDA_LIBS:+:${OCR_CUDA_LIBS}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${OCR_SITE}"
export CUDA_VISIBLE_DEVICES=0
export PADDLE_PDX_CACHE_HOME="${ROOT}/model_cache"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

export OCR_DEVICE=gpu:0
export OCR_CPU_THREADS=1
export OCR_DOCUMENT_RUNTIME=p1c2-primary
export OCR_DOCUMENT_WORKER_MODE=process
export OCR_DOCUMENT_PDF_RENDER_WORKERS=4
export OCR_DOCUMENT_RENDER_PNG_COMPRESS_LEVEL=1
export OCR_DOCUMENT_WORKERS=4
export OCR_DOCUMENT_MAX_IN_FLIGHT=8
export OCR_DOCUMENT_RETRIES=0
export OCR_DOCUMENT_CHECKPOINT_EVERY=8
export OCR_P1C2_CTC_MODEL_DIR="${ROOT}/models/ocr/PP-OCRv6_medium_rec_ctc_reduced"

export TRANSLATION_BACKEND=openai-compatible
export TRANSLATION_ENDPOINT="http://127.0.0.1:11434/v1"
export TRANSLATION_MODEL="translategemma:latest"
export TRANSLATION_REQUEST_CONCURRENCY=4
export TRANSLATION_USE_NATIVE_BATCH_ENDPOINT=0
export TRANSLATION_TEMPERATURE=0.0
export TRANSLATION_TOP_P=0.6
export TRANSLATION_TOP_K=20
export TRANSLATION_REPETITION_PENALTY=1.05
export TRANSLATION_GENERATION_SEED=20260823
export LIVETALKING_BASE_URL="http://127.0.0.1:8010"

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m uvicorn ocr_document.api:app --host 127.0.0.1 --port 8000
