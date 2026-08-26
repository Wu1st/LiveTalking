from __future__ import annotations

from collections import deque

import numpy as np


def padded_hann(n_fft: int = 512, win_length: int = 400) -> np.ndarray:
    window = np.zeros(n_fft, dtype=np.float64)
    left = (n_fft - win_length) // 2
    window[left : left + win_length] = np.hanning(win_length)
    return window


def stft(audio: np.ndarray, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64)
    padded = np.pad(value, (n_fft // 2, n_fft // 2))
    if len(padded) < n_fft:
        padded = np.pad(padded, (0, n_fft - len(padded)))
    remainder = (len(padded) - n_fft) % hop_length
    if remainder:
        padded = np.pad(padded, (0, hop_length - remainder))
    count = 1 + (len(padded) - n_fft) // hop_length
    indices = np.arange(count)[:, None] * hop_length + np.arange(n_fft)[None, :]
    return np.fft.rfft(padded[indices] * padded_hann()[None, :], axis=1)


def istft(spectrum: np.ndarray, length: int, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
    window = padded_hann()
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1)
    output_length = (len(frames) - 1) * hop_length + n_fft
    output = np.zeros(output_length, dtype=np.float64)
    normalization = np.zeros(output_length, dtype=np.float64)
    for index, frame in enumerate(frames):
        left = index * hop_length
        output[left:left+n_fft] += frame * window
        normalization[left:left+n_fft] += window * window
    valid = normalization > np.finfo(np.float64).eps
    output[valid] /= normalization[valid]
    cropped = output[n_fft // 2 : n_fft // 2 + length]
    return np.pad(cropped, (0, max(0, length - len(cropped))))[:length]


def low_percentile_noise_psd(power: np.ndarray) -> np.ndarray:
    history: deque[np.ndarray] = deque(maxlen=100)
    estimate: np.ndarray | None = None
    result = np.empty_like(power, dtype=np.float64)
    for index, frame in enumerate(power):
        history.append(frame.copy())
        candidate = np.quantile(np.stack(tuple(history)), 0.2, axis=0)
        if estimate is None:
            estimate = candidate
        else:
            coefficient = np.where(candidate > estimate, 0.98, 0.70)
            estimate = coefficient * estimate + (1.0 - coefficient) * candidate
        result[index] = estimate
    return result


def wiener_decision_directed(audio: np.ndarray) -> np.ndarray:
    """Fixed low-percentile PSD + decision-directed Wiener VAD front end."""
    spectrum = stft(audio)
    power = np.square(np.abs(spectrum))
    noise = np.maximum(low_percentile_noise_psd(power), 1e-20)
    posterior = np.minimum(power / noise, np.finfo(np.float64).max / 4.0)
    instantaneous = np.maximum(posterior - 1.0, 0.0)
    prior = np.empty_like(posterior)
    if len(posterior):
        prior[0] = instantaneous[0]
        previous_gain = prior[0] / (1.0 + prior[0])
        previous_posterior = posterior[0]
        for frame in range(1, len(posterior)):
            carried = np.square(previous_gain) * previous_posterior
            prior[frame] = 0.98 * carried + 0.02 * instantaneous[frame]
            previous_gain = prior[frame] / (1.0 + prior[frame])
            previous_posterior = posterior[frame]
    gain = np.nan_to_num(prior / (1.0 + prior), nan=1.0, posinf=1.0, neginf=0.02)
    gain = np.clip(gain, 0.02, 1.0)
    return istft(spectrum * gain, len(audio)).astype(np.float32)
