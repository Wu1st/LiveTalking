from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "assets" / "configs" / "frontend.json"
DEFAULT_PROFILES = PACKAGE_ROOT / "assets" / "configs" / "diagnostic_profiles_stage_b2.json"
DEFAULT_MODEL = PACKAGE_ROOT / "assets" / "models" / "silero_vad_v6.onnx"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen(config_path: Path = DEFAULT_CONFIG) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profiles = json.loads(DEFAULT_PROFILES.read_text(encoding="utf-8"))
    expected = {
        "name": "patent_scheme_d",
        "high_threshold": 0.775,
        "low_threshold": 0.175,
        "anchor_credit_frames": 2.0,
        "lookback_seconds": 0.25,
        "capacity_seconds": 1.0,
        "maturity_seconds": 4.0,
        "expiry_seconds": 12.0,
        "minimum_anchor_overlap_seconds": 0.032,
        "base_result_monotonic": True,
    }
    if any(config.get("pipeline", {}).get(key) != value for key, value in expected.items()):
        raise RuntimeError("the fixed production pipeline is not frozen")
    if sha256_file(DEFAULT_MODEL) != config["vad"]["model_sha256"]:
        raise RuntimeError("Silero V6 model SHA256 mismatch")
    if sha256_file(DEFAULT_PROFILES) != config["vad"]["profile_sha256"]:
        raise RuntimeError("VAD profile SHA256 mismatch")
    return config, profiles
