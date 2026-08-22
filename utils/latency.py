"""Structured latency events for the LiveTalking request pipeline.

The regular ``livetalking.log`` remains human-readable.  This module writes a
second JSONL stream which is deliberately easy to aggregate and correlate by
``trace_id``.  Metrics must never make a user request fail, so all writes are
best-effort and isolated from the application path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping, MutableMapping


TRACE_CONTEXT_KEY = "_latency_trace"
DEFAULT_METRICS_PATH = "latency_metrics.jsonl"

_LOGGER_NAME = "livetalking.latency"
_logger: logging.Logger | None = None
_logger_lock = threading.Lock()


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    with _logger_lock:
        if _logger is not None:
            return _logger

        metrics_path = Path(
            os.getenv("LIVETALKING_METRICS_LOG", DEFAULT_METRICS_PATH)
        ).expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_logger = logging.getLogger(_LOGGER_NAME)
        metrics_logger.setLevel(logging.INFO)
        metrics_logger.propagate = False

        if not metrics_logger.handlers:
            handler = RotatingFileHandler(
                metrics_path,
                maxBytes=50 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            metrics_logger.addHandler(handler)

        _logger = metrics_logger
        return metrics_logger


def _safe_value(value: Any) -> Any:
    """Convert event fields into JSON-safe, reasonably small values."""
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, str):
        return value[:2000]
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in list(value.items())[:100]
            if key != TRACE_CONTEXT_KEY
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value[:100]]
    return str(value)


def optional_float(value: Any) -> float | None:
    """Parse an optional numeric field supplied by a browser form."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def new_trace(
    session_id: str,
    source: str,
    *,
    started_monotonic: float | None = None,
    trace_id: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Create a request trace and emit its first event."""
    trace = {
        "trace_id": trace_id or uuid.uuid4().hex,
        "session_id": str(session_id),
        "source": source,
        "started_monotonic": (
            time.perf_counter() if started_monotonic is None else started_monotonic
        ),
    }
    emit_latency("request_started", trace, **metadata)
    return trace


def trace_from_id(session_id: str, trace_id: str, source: str) -> dict[str, Any]:
    """Build a lightweight trace for browser events received after a response."""
    return {
        "trace_id": str(trace_id),
        "session_id": str(session_id),
        "source": source,
    }


def attach_trace(
    datainfo: MutableMapping[str, Any] | None,
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Return metadata containing a trace context without changing the caller."""
    result = dict(datainfo or {})
    result[TRACE_CONTEXT_KEY] = dict(trace)
    return result


def get_trace(container: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(container, Mapping):
        return None
    trace = container.get(TRACE_CONTEXT_KEY)
    return dict(trace) if isinstance(trace, Mapping) else None


def ensure_trace(
    datainfo: MutableMapping[str, Any] | None,
    session_id: str,
    source: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(metadata, trace)`` and create a trace only when absent."""
    trace = get_trace(datainfo)
    if trace is None:
        trace = new_trace(session_id, source)
        datainfo = attach_trace(datainfo, trace)
    return dict(datainfo or {}), trace


def emit_latency(
    stage: str,
    trace: Mapping[str, Any] | None = None,
    **fields: Any,
) -> None:
    """Write one structured event; failures are intentionally swallowed."""
    try:
        now_monotonic = time.perf_counter()
        event: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "stage": stage,
        }

        if trace:
            for key in ("trace_id", "session_id", "source"):
                if trace.get(key) is not None:
                    event[key] = trace[key]
            started = trace.get("started_monotonic")
            if isinstance(started, (int, float)):
                event["relative_ms"] = (now_monotonic - float(started)) * 1000

        event.update(fields)
        _get_logger().info(
            json.dumps(_safe_value(event), ensure_ascii=False, separators=(",", ":"))
        )
    except Exception:
        # Observability must not alter the request result or media pipeline.
        return
