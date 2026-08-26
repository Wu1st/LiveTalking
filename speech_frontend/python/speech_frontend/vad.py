from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort


FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class VadResult:
    probabilities: np.ndarray
    intervals: list[dict]
    frame_latency_ms: list[float]


class SileroV6:
    def __init__(self, model_path: Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])

    def infer(self, audio: np.ndarray, profile: dict) -> VadResult:
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        frame_count = (len(audio) + FRAME_SAMPLES - 1) // FRAME_SAMPLES
        probabilities = np.empty(frame_count, dtype=np.float32)
        latency: list[float] = []
        for index in range(frame_count):
            frame = np.zeros((1, FRAME_SAMPLES), dtype=np.float32)
            chunk = audio[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES]
            frame[0, : len(chunk)] = chunk
            model_input = np.concatenate((context, frame), axis=1)
            started = perf_counter()
            output, state = self.session.run(
                None,
                {"input": model_input, "state": state, "sr": np.asarray(16000, dtype=np.int64)},
            )
            latency.append((perf_counter() - started) * 1000.0)
            probabilities[index] = float(np.asarray(output).reshape(-1)[0])
            context = model_input[:, -CONTEXT_SAMPLES:]
        intervals = postprocess(probabilities, len(audio), profile)
        return VadResult(probabilities, intervals, latency)


def _postprocess_single_threshold(
    probabilities: np.ndarray, audio_length_samples: int, profile: dict
) -> list[dict]:
    threshold = float(profile["threshold"])
    negative_threshold = max(threshold - 0.15, 0.01)
    min_speech = SAMPLE_RATE * float(profile["min_speech_duration_ms"]) / 1000.0
    min_silence = SAMPLE_RATE * float(profile["min_silence_duration_ms"]) / 1000.0
    pad = SAMPLE_RATE * float(profile["speech_pad_ms"]) / 1000.0
    triggered = False
    temporary_end = 0
    current: dict | None = None
    speeches: list[dict] = []
    for index, value in enumerate(probabilities):
        current_sample = FRAME_SAMPLES * index
        if value >= threshold and temporary_end != 0:
            temporary_end = 0
        if value >= threshold and not triggered:
            triggered = True
            current = {"start": current_sample}
            continue
        if value < negative_threshold and triggered and current is not None:
            if temporary_end == 0:
                temporary_end = current_sample
            if current_sample - temporary_end < min_silence:
                continue
            current["end"] = temporary_end
            if current["end"] - current["start"] > min_speech:
                speeches.append(current)
            current = None
            temporary_end = 0
            triggered = False
    if current is not None and audio_length_samples - current["start"] > min_speech:
        current["end"] = audio_length_samples
        speeches.append(current)
    output: list[dict] = []
    for index, speech in enumerate(speeches):
        start, end = int(speech["start"]), int(speech["end"])
        if index == 0:
            start = int(max(0, start - pad))
        if index != len(speeches) - 1:
            silence = speeches[index + 1]["start"] - end
            if silence < 2 * pad:
                half = int(silence // 2)
                end += half
                speeches[index + 1]["start"] = int(max(0, speeches[index + 1]["start"] - half))
            else:
                end = int(min(audio_length_samples, end + pad))
                speeches[index + 1]["start"] = int(max(0, speeches[index + 1]["start"] - pad))
        else:
            end = int(min(audio_length_samples, end + pad))
        output.append({
            "start_sample": start,
            "end_sample": end,
            "start_seconds": start / SAMPLE_RATE,
            "end_seconds": end / SAMPLE_RATE,
        })
    return output


def _merge_intervals(intervals: list[dict], audio_length_samples: int) -> list[dict]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (int(item["start_sample"]), int(item["end_sample"])))
    merged: list[dict] = []
    for interval in ordered:
        start = max(0, int(interval["start_sample"]))
        end = min(audio_length_samples, int(interval["end_sample"]))
        if end <= start:
            continue
        if merged and start <= int(merged[-1]["end_sample"]):
            merged[-1]["end_sample"] = max(int(merged[-1]["end_sample"]), end)
            merged[-1]["end_seconds"] = int(merged[-1]["end_sample"]) / SAMPLE_RATE
            continue
        merged.append({
            "start_sample": start,
            "end_sample": end,
            "start_seconds": start / SAMPLE_RATE,
            "end_seconds": end / SAMPLE_RATE,
        })
    return merged


def _postprocess_anchored_dual_threshold(
    probabilities: np.ndarray, audio_length_samples: int, profile: dict
) -> list[dict]:
    high_profile = dict(profile)
    high_profile.pop("postprocess", None)
    high_intervals = _postprocess_single_threshold(probabilities, audio_length_samples, high_profile)

    low_profile = dict(high_profile)
    low_profile["threshold"] = float(profile["low_threshold"])
    low_candidates = _postprocess_single_threshold(probabilities, audio_length_samples, low_profile)

    high_threshold = float(profile["threshold"])
    minimum_high_frames = int(profile["minimum_high_frames"])
    minimum_candidate_samples = int(
        round(SAMPLE_RATE * float(profile["minimum_candidate_duration_ms"]) / 1000.0)
    )
    selected = list(high_intervals)
    for candidate in low_candidates:
        start = int(candidate["start_sample"])
        end = int(candidate["end_sample"])
        if end - start < minimum_candidate_samples:
            continue
        high_frames = sum(
            1
            for index, probability in enumerate(probabilities)
            if start <= index * FRAME_SAMPLES < end and float(probability) >= high_threshold
        )
        if high_frames >= minimum_high_frames:
            selected.append(candidate)
    return _merge_intervals(selected, audio_length_samples)


def _complete_confirmed_segment_edges(
    intervals: list[dict], audio_length_samples: int, profile: dict
) -> list[dict]:
    if not profile.get("edge_completion", False) or len(intervals) != 1 or audio_length_samples <= 0:
        return intervals
    interval = intervals[0]
    start, end = int(interval["start_sample"]), int(interval["end_sample"])
    coverage = (end - start) / audio_length_samples
    minimum_coverage = float(profile.get("edge_completion_minimum_coverage", 0.70))
    maximum_gap = int(round(SAMPLE_RATE * float(profile.get("edge_completion_maximum_gap_ms", 2000)) / 1000.0))
    trailing_gap = audio_length_samples - end
    if coverage < minimum_coverage or trailing_gap <= 0 or trailing_gap > maximum_gap:
        return intervals
    return [{
        "start_sample": start,
        "end_sample": audio_length_samples,
        "start_seconds": start / SAMPLE_RATE,
        "end_seconds": audio_length_samples / SAMPLE_RATE,
    }]


def postprocess(probabilities: np.ndarray, audio_length_samples: int, profile: dict) -> list[dict]:
    if profile.get("postprocess", "single_threshold") == "anchored_dual_threshold":
        intervals = _postprocess_anchored_dual_threshold(probabilities, audio_length_samples, profile)
        return _complete_confirmed_segment_edges(intervals, audio_length_samples, profile)
    return _postprocess_single_threshold(probabilities, audio_length_samples, profile)
