FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache \
    NLTK_DATA=/models/nltk

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN useradd --create-home --uid 10001 whisperx \
    && mkdir -p /models/whisper /models/alignment /models/huggingface /models/nltk \
    && chown -R whisperx:whisperx /models /app

USER whisperx

EXPOSE 8093

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8093", "--workers", "1"]
