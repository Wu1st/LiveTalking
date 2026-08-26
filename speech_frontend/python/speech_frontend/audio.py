from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _to_float32(value: np.ndarray) -> np.ndarray:
    if np.issubdtype(value.dtype, np.floating):
        return np.asarray(value, dtype=np.float32)
    if value.dtype == np.uint8:
        return ((value.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    limit = float(max(abs(np.iinfo(value.dtype).min), np.iinfo(value.dtype).max))
    return (value.astype(np.float32) / limit).astype(np.float32)


def read_wav_mono_16k(path: Path) -> tuple[np.ndarray, dict]:
    sample_rate, source = wavfile.read(path)
    source_float = _to_float32(source)
    channels = 1 if source_float.ndim == 1 else int(source_float.shape[1])
    mono = source_float if channels == 1 else np.mean(source_float, axis=1, dtype=np.float32)
    if sample_rate != 16000:
        from math import gcd
        divisor = gcd(int(sample_rate), 16000)
        mono = resample_poly(mono, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
        # The frozen C++/Web contract rounds the duration to the nearest
        # target sample. scipy.signal.resample_poly returns ceil(N*up/down),
        # so trim its identical FIR prefix to the shared duration contract.
        target_length = (len(source_float) * 16000 + int(sample_rate) // 2) // int(sample_rate)
        mono = mono[:target_length]
    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    mono = np.clip(mono, -1.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(mono), {
        "source_sample_rate": int(sample_rate),
        "source_channels": channels,
        "source_frames": int(len(source_float)),
        "prepared_sample_rate": 16000,
        "prepared_samples": int(len(mono)),
    }


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, 16000, np.asarray(np.clip(audio, -1.0, 1.0), dtype=np.float32))
