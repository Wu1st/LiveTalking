from __future__ import annotations

import argparse
import csv
import json
import os
import resource
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from .analysis import analyze
from .audio import read_wav_mono_16k, write_wav
from .config import DEFAULT_MODEL, load_frozen, sha256_file
from .enhancement import wiener_decision_directed
from .patent_d import PatentDConfig, run_patent_d
from .vad import SileroV6


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def process(
    input_path: Path,
    output_dir: Path,
) -> dict:
    config, profiles = load_frozen()
    mode_config = config["pipeline"]
    output_dir.mkdir(parents=True, exist_ok=True)
    total_started = perf_counter()
    started = perf_counter()
    audio, input_metadata = read_wav_mono_16k(input_path)
    prepare_seconds = perf_counter() - started
    write_wav(output_dir / "original_16k.wav", audio)

    started = perf_counter()
    vad_input = wiener_decision_directed(audio)
    vad_preprocess_seconds = perf_counter() - started
    write_wav(output_dir / "vad_input.wav", vad_input)

    started = perf_counter()
    model = SileroV6(DEFAULT_MODEL)
    model_load_seconds = perf_counter() - started
    profile_name = mode_config["base_profile"]
    profile = dict(profiles["profiles"][profile_name])
    profile["threshold"] = mode_config["high_threshold"]
    started = perf_counter()
    raw_vad = model.infer(audio, profile)
    wiener_vad = model.infer(vad_input, profile)
    intervals, patent_audit = run_patent_d(
        raw_vad.probabilities,
        wiener_vad.probabilities,
        len(audio),
        PatentDConfig(),
    )
    vad_seconds = perf_counter() - started

    started = perf_counter()
    processed = vad_input.copy()
    enhancement_seconds = perf_counter() - started
    write_wav(output_dir / "processed.wav", processed)

    intervals_payload = {
        "schema_version": 1,
        "runtime": "python",
        "vad": "Original Silero V6",
        "pipeline": "patent_scheme_d",
        "profile": f"{profile_name}@{profile['threshold']}",
        "threshold": profile["threshold"],
        "postprocess": "anchor_credit_bounded_maturity_debt",
        "low_threshold": mode_config["low_threshold"],
        "vad_inputs": ["raw_audio", "wiener_decision_directed_audio"],
        "sample_rate": 16000,
        "sample_count": int(len(audio)),
        "intervals": intervals,
        "patent_audit": patent_audit,
    }
    (output_dir / "vad_intervals.json").write_text(
        json.dumps(intervals_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "vad_probabilities.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frame_index", "start_sample", "start_seconds", "raw_probability", "wiener_probability"]
        )
        for index, (raw_probability, wiener_probability) in enumerate(
            zip(raw_vad.probabilities, wiener_vad.probabilities, strict=True)
        ):
            writer.writerow(
                [
                    index,
                    index * 512,
                    index * 512 / 16000.0,
                    f"{float(raw_probability):.9g}",
                    f"{float(wiener_probability):.9g}",
                ]
            )
    started = perf_counter()
    analysis = {
        "schema_version": 1,
        "raw": analyze(audio),
        "processed": analyze(processed),
        "vad_input": analyze(vad_input),
        "vad_intervals": intervals,
        "patent_audit": patent_audit,
    }
    analysis_seconds = perf_counter() - started
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_seconds = perf_counter() - total_started
    duration_seconds = len(audio) / 16000.0
    summary = {
        "schema_version": 1,
        "product": config["product"],
        "runtime": "python",
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "model_sha256": config["vad"]["model_sha256"],
        "profile_sha256": config["vad"]["profile_sha256"],
        "input_metadata": input_metadata,
        "pipeline": "patent_scheme_d",
        "enhancement": "wiener_dd",
        "vad_profile": f"{profile_name}@{profile['threshold']}",
        "vad_high_threshold": mode_config["high_threshold"],
        "vad_low_threshold": mode_config["low_threshold"],
        "vad_postprocess": "anchor_credit_bounded_maturity_debt",
        "vad_inputs": ["raw_audio", "wiener_decision_directed_audio"],
        "dereverb": "off",
        "algorithmic_lookahead_ms": 12.5,
        "duration_seconds": duration_seconds,
        "speech_interval_count": len(intervals),
        "frame_count": len(wiener_vad.probabilities),
        "patent_audit": patent_audit,
        "timing": {
            "audio_prepare_seconds": prepare_seconds,
            "model_load_seconds": model_load_seconds,
            "vad_seconds": vad_seconds,
            "vad_preprocess_seconds": vad_preprocess_seconds,
            "enhancement_seconds": enhancement_seconds,
            "analysis_seconds": analysis_seconds,
            "total_seconds": total_seconds,
            "core_rtf": (vad_seconds + vad_preprocess_seconds + enhancement_seconds) / max(duration_seconds, 1e-12),
            "average_vad_frame_ms": mean(raw_vad.frame_latency_ms + wiener_vad.frame_latency_ms),
            "p95_vad_frame_ms": _percentile(raw_vad.frame_latency_ms + wiener_vad.frame_latency_ms, 95.0),
        },
        "outputs": {
            name: sha256_file(output_dir / name)
            for name in (
                "original_16k.wav", "vad_input.wav", "processed.wav", "vad_intervals.json",
                "vad_probabilities.csv", "analysis.json",
            )
        },
        "environment": {"pid": os.getpid()},
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    (output_dir / "processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speech_frontend", description="Patent Scheme D Python reference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    process_parser = subparsers.add_parser("process", help="process one WAV file")
    process_parser.add_argument("input", type=Path)
    process_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "process":
        summary = process(arguments.input, arguments.output)
        print(json.dumps({"output": str(arguments.output.resolve()), "rtf": summary["timing"]["core_rtf"]}))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
