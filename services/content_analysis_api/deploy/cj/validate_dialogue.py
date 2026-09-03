#!/usr/bin/env python3
"""Run the four consumed proxy dialogue samples against the deployed API."""

from __future__ import annotations

import json
from pathlib import Path
import time

import httpx


ROOT = Path("/data/cj/qwen3-asr")
INPUT_DIR = ROOT / "run/validation-inputs/dialogue"
OUTPUT_DIR = ROOT / "run/validation-results/dialogue"
API_KEY_FILE = ROOT / "run/content-analysis-api.key"
ENDPOINT = "http://127.0.0.1:18081/v1/conversation/analyze"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    summary: list[dict[str, object]] = []
    for source in sorted(INPUT_DIR.glob("fhr_windowed_*_v2_20260831.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))["structured_input"]
        started = time.perf_counter()
        response = httpx.post(
            ENDPOINT,
            headers={"X-API-Key": api_key},
            json=payload,
            timeout=180.0,
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        result = response.json()
        output = OUTPUT_DIR / f"{source.stem}.result.json"
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        analysis = result.get("analysis") or {}
        routing = analysis.get("dialogue_understanding_meta") or {}
        summary.append(
            {
                "sample": source.name,
                "elapsed_seconds": round(elapsed, 3),
                "model": result.get("model"),
                "route": routing.get("route"),
                "fallback_used": routing.get("fallback_used"),
                "dialogue_understanding": analysis.get("dialogue_understanding"),
            }
        )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
