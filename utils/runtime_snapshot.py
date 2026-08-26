"""Capture a compact, non-sensitive runtime snapshot for later comparisons."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from utils.latency import emit_latency


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


def _git_snapshot(root: Path) -> dict[str, Any]:
    status = _run(["git", "status", "--porcelain"], root)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], root),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(status),
        "changed_file_count": len(status.splitlines()) if status else 0,
    }


def _audio_snapshot(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {"configured": False}

    path = Path(str(path_value)).expanduser()
    result: dict[str, Any] = {
        "configured": True,
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return result

    try:
        result["bytes"] = path.stat().st_size
    except OSError:
        pass

    try:
        import soundfile as sf

        info = sf.info(str(path))
        result.update(
            {
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration_s": info.duration,
                "format": info.format,
                "subtype": info.subtype,
            }
        )
    except Exception as exc:
        result["inspection_error"] = type(exc).__name__
    return result


def _gpu_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        result.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            }
        )
    except Exception as exc:
        result["torch_error"] = repr(exc)

    result["nvidia_smi"] = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    return result


def log_runtime_snapshot(opt: Any, root: str | Path) -> None:
    """Write the effective startup configuration without secrets or prompts."""
    project_root = Path(root).resolve()
    fields = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "git": _git_snapshot(project_root),
        "gpu": _gpu_snapshot(),
        "runtime": {
            "model": getattr(opt, "model", None),
            "transport": getattr(opt, "transport", None),
            "avatar_id": getattr(opt, "avatar_id", None),
            "batch_size": getattr(opt, "batch_size", None),
            "fps": getattr(opt, "fps", None),
            "listenport": getattr(opt, "listenport", None),
            "tts": getattr(opt, "tts", None),
            "tts_server": getattr(opt, "TTS_SERVER", None),
            "translation_model": getattr(opt, "TRANSLATION_MODEL", None),
            "translation_server": getattr(opt, "TRANSLATION_SERVER", None),
        },
        "reference_audio": _audio_snapshot(getattr(opt, "REF_FILE", None)),
        "reference_text_chars": len(str(getattr(opt, "REF_TEXT", "") or "")),
    }
    emit_latency("runtime_snapshot", None, **fields)


def log_runtime_ready(started_at: float) -> None:
    emit_latency(
        "runtime_ready",
        None,
        startup_elapsed_ms=(time.perf_counter() - started_at) * 1000,
    )
