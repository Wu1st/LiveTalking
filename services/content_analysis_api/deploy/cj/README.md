# CJ deployment scope

This deployment is intentionally limited to FHR-owned functionality:

- OCR document API on `127.0.0.1:8000` using visible device 0 (96 GB full GPU); four persistent workers use roughly 7.0 GB after warm-up.
- Hy-MT2-1.8B translation configuration is retained for rollback, but its service is disabled and stopped; port `8101` is intentionally unavailable.
- OCR translation uses the existing root-owned `translategemma:latest` service at `127.0.0.1:11434`; this deployment does not modify or restart that shared Ollama process.
- Isolated Ollama on `127.0.0.1:11435` shares visible device 0 with OCR. Qwen3 8B is normally resident and Qwen2.5 7B remains available as an on-demand fallback.
- Dialogue understanding API on `127.0.0.1:18081`; Qwen3 8B is primary and Qwen2.5 7B is fallback.

No ASR, diarization, emotion, acoustic-scene, digital-human, or shared Ollama service is modified or started by these units.

The deployment files live under `/data/cj/qwen3-asr/run/deploy`. They are linked into the `cj` user systemd manager. Existing source-tree deployment files are retained unchanged for rollback and provenance.

The active OCR unit is `fhr-ocr-cj-api.service`. The earlier migration-only
`fhr-ocr-translation-api.service` name is not enabled; keeping a CJ-specific
unit name avoids colliding with copied deployment metadata.

OCR no longer declares a systemd dependency on Hy-MT2, so starting or
restarting OCR cannot automatically bring the old runtime back. OCR-only and
OCR-to-translation requests use independent GPU services.

The four consumed dialogue proxy samples are rerun with
`/opt/envs/ml-base/bin/python3 validate_dialogue.py`. On the 96 GB device the
September 3 validation completed in 6.846, 10.597, 10.879 and 9.843 seconds.
The samples are regression evidence, not an independent business acceptance
set.

On the same date, the first complete A196 run after moving OCR from the MIG to
the full GPU completed 34/34 pages with zero failures in 9.535 seconds
(213.95 PPM at the upload-to-result boundary). A simultaneous OCR, dialogue and
TranslateGemma stress request also completed all three operations. These are
deployment checks on proxy inputs, not business accuracy acceptance.
