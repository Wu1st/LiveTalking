from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .vad import FRAME_SAMPLES, SAMPLE_RATE, _merge_intervals, _postprocess_single_threshold


@dataclass(frozen=True)
class PatentDConfig:
    high_threshold: float = 0.775
    low_threshold: float = 0.175
    min_speech_duration_ms: int = 150
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 40
    minimum_high_frames: int = 3
    minimum_candidate_duration_ms: int = 2000
    anchor_credit_frames: float = 2.0
    lookback_seconds: float = 0.25
    capacity_seconds: float = 1.0
    maturity_seconds: float = 4.0
    expiry_seconds: float = 12.0
    minimum_anchor_overlap_seconds: float = 0.032


def _profile(config: PatentDConfig, threshold: float) -> dict:
    return {
        "threshold": threshold,
        "min_speech_duration_ms": config.min_speech_duration_ms,
        "min_silence_duration_ms": config.min_silence_duration_ms,
        "speech_pad_ms": config.speech_pad_ms,
    }


def _tuples(intervals: list[dict]) -> list[tuple[int, int]]:
    return [(int(item["start_sample"]), int(item["end_sample"])) for item in intervals]


def _dicts(intervals: list[tuple[int, int]]) -> list[dict]:
    return [
        {
            "start_sample": int(start),
            "end_sample": int(end),
            "start_seconds": int(start) / SAMPLE_RATE,
            "end_seconds": int(end) / SAMPLE_RATE,
        }
        for start, end in intervals
    ]


def _merge_tuples(intervals: list[tuple[int, int]], total_samples: int) -> list[tuple[int, int]]:
    return _tuples(
        _merge_intervals(
            [
                {
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "start_seconds": int(start) / SAMPLE_RATE,
                    "end_seconds": int(end) / SAMPLE_RATE,
                }
                for start, end in intervals
            ],
            total_samples,
        )
    )


def anchored_intervals(
    low_probability: np.ndarray,
    high_probability: np.ndarray,
    total_samples: int,
    config: PatentDConfig,
) -> list[tuple[int, int]]:
    high = _postprocess_single_threshold(
        high_probability, total_samples, _profile(config, config.high_threshold)
    )
    low = _postprocess_single_threshold(
        low_probability, total_samples, _profile(config, config.low_threshold)
    )
    selected = list(high)
    minimum_samples = round(SAMPLE_RATE * config.minimum_candidate_duration_ms / 1000.0)
    for item in low:
        start, end = int(item["start_sample"]), int(item["end_sample"])
        frame_start = max(0, start // FRAME_SAMPLES)
        frame_end = min(len(high_probability), int(math.ceil(end / FRAME_SAMPLES)))
        anchors = int(np.sum(high_probability[frame_start:frame_end] >= config.high_threshold))
        if end - start >= minimum_samples and anchors >= config.minimum_high_frames:
            selected.append(item)
    return _tuples(_merge_intervals(selected, total_samples))


def apply_anchor_credit(
    raw_probability: np.ndarray,
    wiener_probability: np.ndarray,
    config: PatentDConfig,
) -> tuple[np.ndarray, dict]:
    raw = np.asarray(raw_probability, dtype=np.float32)
    wiener = np.asarray(wiener_probability, dtype=np.float32)
    if raw.ndim != 1 or raw.shape != wiener.shape:
        raise ValueError("raw and Wiener probabilities must be equal one-dimensional arrays")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(wiener)):
        raise ValueError("probabilities must be finite")

    controlled = np.minimum(raw, wiener).astype(np.float32)
    pending: deque[int] = deque()
    tokens = 0.0
    frame_seconds = FRAME_SAMPLES / SAMPLE_RATE
    capacity_frames = config.capacity_seconds / frame_seconds
    lookback_frames = int(round(config.lookback_seconds / frame_seconds))
    risk: set[int] = set()
    admitted: set[int] = set()
    retroactive = 0
    anchors = 0
    denied_expired = 0
    maximum_tokens = 0.0

    for index in range(len(controlled)):
        while pending and index - pending[0] > lookback_frames:
            pending.popleft()
            denied_expired += 1
        raw_value = float(raw[index])
        wiener_value = float(wiener[index])
        is_anchor = wiener_value >= config.high_threshold
        is_risk = (
            config.low_threshold <= wiener_value < config.high_threshold
            and raw_value < config.low_threshold
        )
        if is_anchor:
            anchors += 1
            tokens = min(capacity_frames, tokens + config.anchor_credit_frames)
            while pending and tokens + 1e-12 >= 1.0:
                pending_index = pending.popleft()
                controlled[pending_index] = wiener[pending_index]
                admitted.add(pending_index)
                tokens -= 1.0
                retroactive += 1
        elif is_risk:
            risk.add(index)
            if tokens + 1e-12 >= 1.0:
                controlled[index] = wiener[index]
                admitted.add(index)
                tokens -= 1.0
            else:
                pending.append(index)
        maximum_tokens = max(maximum_tokens, tokens)

    denied = len(risk - admitted)
    return controlled, {
        "risk_frames": len(risk),
        "anchor_frames": anchors,
        "admitted_frames": len(admitted),
        "retroactive_frames": retroactive,
        "denied_frames": denied,
        "expired_while_streaming": denied_expired,
        "maximum_tokens": maximum_tokens,
        "capacity_frames": capacity_frames,
        "pending_after_finalize": 0,
        "frames_processed": len(controlled),
    }


def _intersection(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _incremental_samples(candidate: tuple[int, int], base: list[tuple[int, int]]) -> int:
    return max(0, candidate[1] - candidate[0] - sum(_intersection(candidate, item) for item in base))


def maturity_debt_completion(
    base: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    anchors: list[tuple[int, int]],
    total_samples: int,
    config: PatentDConfig,
) -> tuple[list[tuple[int, int]], dict]:
    base = _merge_tuples(base, total_samples)
    candidates = _merge_tuples(candidates, total_samples)
    anchors = _merge_tuples(anchors, total_samples)
    maturity = round(config.maturity_seconds * SAMPLE_RATE)
    expiry = round(config.expiry_seconds * SAMPLE_RATE)
    minimum_overlap = round(config.minimum_anchor_overlap_seconds * SAMPLE_RATE)
    selected = list(base)
    rows = []
    for candidate in candidates:
        duration = candidate[1] - candidate[0]
        anchor_overlap = sum(_intersection(candidate, anchor) for anchor in anchors)
        incremental = _incremental_samples(candidate, base)
        mature = duration > maturity
        unexpired = duration <= expiry
        collateralized = anchor_overlap >= minimum_overlap
        admitted = bool(incremental and mature and unexpired and collateralized)
        if admitted:
            selected.append(candidate)
        rows.append(
            {
                "start_sample": candidate[0],
                "end_sample": candidate[1],
                "duration_samples": duration,
                "anchor_overlap_samples": anchor_overlap,
                "incremental_debt_samples": incremental,
                "mature": mature,
                "unexpired": unexpired,
                "collateralized": collateralized,
                "admitted": admitted,
            }
        )
    final = _merge_tuples(selected, total_samples)
    return final, {
        "mechanism": "ANCHOR_BACKED_BOUNDED_MATURITY_DEBT",
        "candidate_intervals": len(rows),
        "admitted_candidate_intervals": sum(int(row["admitted"]) for row in rows),
        "admitted_debt_samples": sum(
            row["incremental_debt_samples"] for row in rows if row["admitted"]
        ),
        "base_subset_final": all(
            _incremental_samples(item, final) == 0 for item in base
        ),
        "records": rows,
    }


def run_patent_d(
    raw_probability: np.ndarray,
    wiener_probability: np.ndarray,
    total_samples: int,
    config: PatentDConfig = PatentDConfig(),
) -> tuple[list[dict], dict]:
    high = _tuples(
        _postprocess_single_threshold(
            wiener_probability, total_samples, _profile(config, config.high_threshold)
        )
    )
    fixed_sensitive = anchored_intervals(
        wiener_probability, wiener_probability, total_samples, config
    )
    controlled, credit_audit = apply_anchor_credit(
        raw_probability, wiener_probability, config
    )
    base = anchored_intervals(controlled, wiener_probability, total_samples, config)
    final, maturity_audit = maturity_debt_completion(
        base, fixed_sensitive, high, total_samples, config
    )
    return _dicts(final), {
        "mechanism": "PATENT_SCHEME_D",
        "config": asdict(config),
        "high_confidence_intervals": _dicts(high),
        "fixed_sensitive_candidates": _dicts(fixed_sensitive),
        "base_intervals": _dicts(base),
        "anchor_credit": credit_audit,
        "maturity_debt": maturity_audit,
        "base_subset_final": maturity_audit["base_subset_final"],
    }
