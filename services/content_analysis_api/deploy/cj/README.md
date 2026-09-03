# CJ deployment scope

This deployment is intentionally limited to FHR-owned functionality:

- OCR document API on `127.0.0.1:8000` using visible device 1 (24 GB MIG).
- Hy-MT2-1.8B translation on `127.0.0.1:8101` using visible device 0 (96 GB full GPU) with a bounded 45% vLLM reservation.
- Isolated Ollama on `127.0.0.1:11435` shares visible device 0 with Hy-MT2. Qwen3 8B and Qwen2.5 7B remain resident without consuming the OCR MIG.
- Dialogue understanding API on `127.0.0.1:18081`; Qwen3 8B is primary and Qwen2.5 7B is fallback.

No ASR, diarization, emotion, acoustic-scene, digital-human, or shared Ollama service is modified or started by these units.

The deployment files live under `/data/cj/qwen3-asr/run/deploy`. They are linked into the `cj` user systemd manager. Existing source-tree deployment files are retained unchanged for rollback and provenance.

The active OCR unit is `fhr-ocr-cj-api.service`. The earlier migration-only
`fhr-ocr-translation-api.service` name is not enabled; keeping a CJ-specific
unit name avoids colliding with copied deployment metadata.

The four consumed dialogue proxy samples are rerun with
`/opt/envs/ml-base/bin/python3 validate_dialogue.py`. On the 96 GB device the
September 3 validation completed in 6.846, 10.597, 10.879 and 9.843 seconds.
The samples are regression evidence, not an independent business acceptance
set.
