from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .config import DEFAULT_MODEL, load_frozen
from .enhancement import wiener_decision_directed
from .patent_d import PatentDConfig, run_patent_d
from .vad import SileroV6


@dataclass(frozen=True)
class PipelineResult:
    original: np.ndarray
    enhanced: np.ndarray
    raw_probabilities: np.ndarray
    wiener_probabilities: np.ndarray
    intervals: list[dict]
    patent_audit: dict
    duration_seconds: float
    speech_duration_seconds: float
    processing_ms: float
    core_rtf: float


class SpeechFrontendEngine:
    """In-memory Scheme D engine with one reusable ONNX Runtime session."""

    def __init__(self) -> None:
        self.config, self.profiles = load_frozen()
        self.model = SileroV6(DEFAULT_MODEL)
        pipeline = self.config["pipeline"]
        self.profile_name = str(pipeline["base_profile"])
        self.profile = dict(self.profiles["profiles"][self.profile_name])
        self.profile["threshold"] = float(pipeline["high_threshold"])

    def process(self, audio: np.ndarray) -> PipelineResult:
        original = np.ascontiguousarray(
            np.clip(np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0),
            dtype=np.float32,
        )
        started = perf_counter()
        enhanced = wiener_decision_directed(original)
        raw_vad = self.model.infer(original, self.profile)
        wiener_vad = self.model.infer(enhanced, self.profile)
        intervals, patent_audit = run_patent_d(
            raw_vad.probabilities,
            wiener_vad.probabilities,
            len(original),
            PatentDConfig(),
        )
        processing_seconds = perf_counter() - started
        duration_seconds = len(original) / 16000.0
        speech_duration_seconds = sum(
            (int(item["end_sample"]) - int(item["start_sample"])) / 16000.0
            for item in intervals
        )
        return PipelineResult(
            original=original,
            enhanced=enhanced,
            raw_probabilities=raw_vad.probabilities,
            wiener_probabilities=wiener_vad.probabilities,
            intervals=intervals,
            patent_audit=patent_audit,
            duration_seconds=duration_seconds,
            speech_duration_seconds=speech_duration_seconds,
            processing_ms=processing_seconds * 1000.0,
            core_rtf=processing_seconds / max(duration_seconds, 1e-12),
        )

