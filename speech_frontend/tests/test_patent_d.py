from __future__ import annotations

import numpy as np

from speech_frontend.patent_d import PatentDConfig, run_patent_d


def test_patent_d_is_deterministic_and_monotonic() -> None:
    rng = np.random.default_rng(20260825)
    raw = rng.uniform(0, 1, 600).astype(np.float32)
    wiener = rng.uniform(0, 1, 600).astype(np.float32)
    total_samples = 600 * 512 - 117
    actual, audit = run_patent_d(raw, wiener, total_samples, PatentDConfig())
    repeated, repeated_audit = run_patent_d(raw, wiener, total_samples, PatentDConfig())
    assert actual == repeated
    assert audit == repeated_audit
    assert audit["base_subset_final"]


def test_patent_d_rejects_unbounded_candidate() -> None:
    frames = 500
    raw = np.zeros(frames, dtype=np.float32)
    wiener = np.full(frames, 0.30, dtype=np.float32)
    wiener[100:105] = 0.90
    intervals, audit = run_patent_d(raw, wiener, frames * 512)
    assert all((item["end_sample"] - item["start_sample"]) <= 12 * 16000 for item in intervals)
    assert audit["maturity_debt"]["base_subset_final"]
