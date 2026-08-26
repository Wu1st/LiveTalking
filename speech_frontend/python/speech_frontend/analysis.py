from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .enhancement import low_percentile_noise_psd, stft


def _preview(values: np.ndarray, maximum: int = 2000) -> list[float]:
    flat = np.asarray(values).reshape(-1)
    if len(flat) > maximum:
        flat = flat[np.linspace(0, len(flat) - 1, maximum, dtype=np.int64)]
    return [float(value) for value in flat]


def analyze(audio: np.ndarray) -> dict:
    sample_rate = 16000
    frame_length, hop = 400, 160
    padded = np.pad(np.asarray(audio, dtype=np.float64), (0, max(0, frame_length - len(audio))))
    count = max(1, 1 + (len(padded) - frame_length) // hop)
    frames = np.stack([padded[index * hop:index * hop + frame_length] for index in range(count)])
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    fft_size = 1 << int(np.ceil(np.log2(max(1, len(audio)))))
    fft_size = min(fft_size, 65536)
    chunk = np.asarray(audio[:fft_size], dtype=np.float64)
    fft = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=fft_size)) if len(chunk) else np.zeros(1)
    spectrum = stft(audio)
    magnitude = np.abs(spectrum)
    power = magnitude * magnitude
    frequency = np.linspace(0.0, sample_rate / 2.0, power.shape[1])
    denominator = np.maximum(np.sum(power, axis=1), np.finfo(np.float64).tiny)
    centroid = np.sum(power * frequency[None, :], axis=1) / denominator
    flatness = np.exp(np.mean(np.log(np.maximum(power, np.finfo(np.float64).tiny)), axis=1)) / np.maximum(np.mean(power, axis=1), np.finfo(np.float64).tiny)
    noise = low_percentile_noise_psd(power)
    noise_score = 10.0 * np.log10((np.sum(power, axis=1) + 1e-30) / (np.sum(noise, axis=1) + 1e-30))
    welch_frequency, welch_psd = welch(audio, fs=sample_rate, nperseg=min(512, len(audio)))
    time_indices = np.linspace(0, max(0, len(magnitude) - 1), min(300, len(magnitude)), dtype=np.int64)
    freq_indices = np.linspace(0, magnitude.shape[1] - 1, min(129, magnitude.shape[1]), dtype=np.int64)
    spectrogram_db = 20.0 * np.log10(np.maximum(magnitude[np.ix_(time_indices, freq_indices)], 1e-15) / max(float(np.max(magnitude)), 1e-15))
    return {
        "sample_rate": sample_rate,
        "sample_count": int(len(audio)),
        "waveform": _preview(audio),
        "rms": _preview(rms),
        "zcr": _preview(zcr),
        "fft": {"frequency_hz": _preview(np.fft.rfftfreq(fft_size, 1 / sample_rate)), "magnitude": _preview(fft)},
        "stft": {"n_fft": 512, "win_length": 400, "hop_length": 160, "center": True, "frames": int(len(spectrum)), "bins": int(spectrum.shape[1])},
        "spectrogram_db": [[float(value) for value in row] for row in spectrogram_db],
        "welch_psd": {"frequency_hz": _preview(welch_frequency), "psd": _preview(welch_psd)},
        "spectral_features": {"centroid_hz": _preview(centroid), "flatness": _preview(flatness)},
        "noise_score_db": _preview(noise_score),
    }
