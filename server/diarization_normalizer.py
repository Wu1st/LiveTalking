"""Normalize FunASR speaker-diarization results into a stable API schema."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any


_CJK_CHARACTER = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_BOUNDARY_PUNCTUATION = "，。！？；：,.!?;:"
_TERMINAL_PUNCTUATION = "。！？.!?"


def normalize_punctuation(text: str) -> str:
    """Remove duplicated ASR punctuation without changing transcript content.

    Args:
        text: Raw or postprocessed transcript text.

    Returns:
        Text with duplicated punctuation and Chinese spacing normalized.
    """
    value = str(text or "").strip()
    if not value:
        return value

    value = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", value)

    def collapse(
        pattern: str,
        chinese: str,
        ascii_mark: str,
        *,
        keep_ellipsis: bool = False,
    ) -> None:
        nonlocal value

        def replacement(match: re.Match[str]) -> str:
            raw = match.group(0)
            if keep_ellipsis and chinese not in raw:
                return raw
            return chinese if chinese in raw else ascii_mark

        value = re.sub(pattern, replacement, value)

    collapse(r"[。.](?:\s*[。.])+", "。", ".", keep_ellipsis=True)
    collapse(r"[，,](?:\s*[，,])+", "，", ",")
    collapse(r"[！!](?:\s*[！!])+", "！", "!")
    collapse(r"[？?](?:\s*[？?])+", "？", "?")
    collapse(r"[；;](?:\s*[；;])+", "；", ";")
    collapse(r"[：:](?:\s*[：:])+", "：", ":")
    value = re.sub(
        rf"([，。！？；：])\s+(?=[{_CJK_CHARACTER}])",
        r"\1",
        value,
    )
    return value.strip()


def _join_transcript_text(left: str, right: str) -> str:
    """Join adjacent diarization segments using language-appropriate spacing."""
    left = normalize_punctuation(left)
    right = normalize_punctuation(right)
    if not left:
        return right
    if not right:
        return left

    separator = " "
    if re.search(rf"[{_CJK_CHARACTER}]$", left) and re.match(
        rf"^[{_CJK_CHARACTER}]",
        right,
    ):
        separator = ""
    return normalize_punctuation(f"{left}{separator}{right}")


def _clean_segment_boundary(
    previous_text: str,
    current_text: str,
) -> tuple[str, str]:
    """Move duplicated leading punctuation onto the preceding segment.

    SenseVoice and CT-Punc can place the same sentence-boundary mark at the
    end of one segment and the beginning of the next.  Segment-local cleanup
    cannot see that duplication, so normalize it after chronological sorting.
    """
    previous = normalize_punctuation(previous_text)
    current = normalize_punctuation(current_text)
    match = re.match(rf"^[{re.escape(_BOUNDARY_PUNCTUATION)}]+", current)
    if match is None:
        return previous, current

    marks = match.group(0)
    current = current[match.end() :].lstrip()
    if not previous:
        return previous, current

    preferred = next(
        (mark for mark in marks if mark in _TERMINAL_PUNCTUATION),
        marks[0],
    )
    trailing = previous[-1]
    if trailing not in _BOUNDARY_PUNCTUATION:
        previous = normalize_punctuation(previous + preferred)
    elif preferred in _TERMINAL_PUNCTUATION and trailing not in _TERMINAL_PUNCTUATION:
        previous = normalize_punctuation(previous[:-1] + preferred)
    return previous, current


def _as_seconds(value: Any, unit: str | None = None) -> float:
    """Convert a FunASR timestamp to seconds.

    Args:
        value: Numeric timestamp returned by FunASR.
        unit: Optional explicit unit. FunASR normally returns milliseconds.

    Returns:
        A non-negative timestamp in seconds.

    Raises:
        ValueError: If the timestamp is missing or not numeric.
    """
    if value is None:
        raise ValueError("缺少说话人时间戳")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"说话人时间戳不是数字：{value!r}") from error

    if unit and unit.lower() in {"s", "sec", "second", "seconds"}:
        return max(0.0, numeric)
    return max(0.0, numeric / 1000.0)


def _first_mapping(value: Any) -> Mapping[str, Any]:
    """Extract the first result mapping from FunASR's list response."""
    if isinstance(value, list):
        if not value or not isinstance(value[0], Mapping):
            raise ValueError("FunASR 返回结果为空或格式不正确")
        return value[0]
    if not isinstance(value, Mapping):
        raise ValueError("FunASR 返回结果不是对象")
    return value


def normalize_result(
    result: Any,
    postprocess_text: Callable[[str], str] | None = None,
    merge_gap_seconds: float = 0.25,
) -> dict[str, Any]:
    """Convert FunASR ``sentence_info`` into anonymous speaker segments.

    Args:
        result: Raw result returned by ``AutoModel.generate``.
        postprocess_text: Optional SenseVoice text cleanup callback.
        merge_gap_seconds: Maximum gap for merging adjacent segments from the
            same speaker into one interval.

    Returns:
        JSON-safe text, sentence segments, merged speaker intervals, labels,
        and speaker count.

    Raises:
        ValueError: If FunASR did not return speaker labels or timestamps.
    """
    payload = _first_mapping(result)
    sentence_info = (
        payload.get("sentence_info")
        or payload.get("segments")
        or payload.get("sentence_timestamp")
        or []
    )
    if not isinstance(sentence_info, list):
        raise ValueError("FunASR 未返回可用的 sentence_info")

    parsed_segments: list[tuple[str, float, float, str]] = []
    for index, item in enumerate(sentence_info):
        if not isinstance(item, Mapping):
            continue

        raw_speaker = item.get("spk", item.get("speaker", item.get("speaker_id")))
        if raw_speaker is None:
            raise ValueError(f"FunASR 第 {index} 个句段缺少说话人标签")

        speaker_key = str(raw_speaker)
        unit = (
            item.get("time_unit")
            or item.get("timestamp_unit")
            or payload.get("time_unit")
            or payload.get("timestamp_unit")
        )
        start_value = item.get("start", item.get("start_ms"))
        end_value = item.get("end", item.get("end_ms"))
        timestamp = item.get("timestamp")
        if (
            (start_value is None or end_value is None)
            and isinstance(timestamp, (list, tuple))
            and len(timestamp) >= 2
        ):
            start_value, end_value = timestamp[0], timestamp[1]

        start = _as_seconds(start_value, unit)
        end = _as_seconds(end_value, unit)
        if end < start:
            start, end = end, start

        text = str(item.get("text", item.get("sentence", "")) or "").strip()
        if postprocess_text is not None:
            text = postprocess_text(text).strip()
        parsed_segments.append(
            (speaker_key, start, end, normalize_punctuation(text))
        )

    # Sort before assigning labels so Speaker_00 remains chronological even
    # if a model version returns sentence_info slightly out of order.
    parsed_segments.sort(key=lambda item: (item[1], item[2]))
    speaker_map: dict[str, str] = {}
    sentence_segments: list[dict[str, Any]] = []
    for speaker_key, start, end, text in parsed_segments:
        previous_text = sentence_segments[-1]["text"] if sentence_segments else ""
        previous_text, text = _clean_segment_boundary(previous_text, text)
        if sentence_segments:
            sentence_segments[-1]["text"] = previous_text
        # Punctuation-only fragments contain no speech content and should not
        # become standalone LLM evidence segments.
        if not text:
            continue
        speaker = speaker_map.setdefault(
            speaker_key,
            f"Speaker_{len(speaker_map):02d}",
        )
        sentence_segments.append(
            {
                "speaker": speaker,
                "speaker_id": speaker_key,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )

    speaker_segments: list[dict[str, Any]] = []
    for segment in sentence_segments:
        if (
            speaker_segments
            and speaker_segments[-1]["speaker"] == segment["speaker"]
            and segment["start"] - speaker_segments[-1]["end"] <= merge_gap_seconds
        ):
            previous = speaker_segments[-1]
            previous["end"] = round(max(previous["end"], segment["end"]), 3)
            if segment["text"]:
                previous["text"] = _join_transcript_text(
                    previous["text"],
                    segment["text"],
                )
            continue

        speaker_segments.append(
            {
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            }
        )

    # Rebuild the full text from the boundary-cleaned timeline.  The payload's
    # top-level text can contain cross-segment sequences such as "。，" even
    # when each individual fragment looks locally valid.
    raw_text = ""
    for segment in sentence_segments:
        raw_text = _join_transcript_text(raw_text, segment["text"])
    if not raw_text:
        raw_text = str(payload.get("text") or "").strip()
        if postprocess_text is not None:
            raw_text = postprocess_text(raw_text).strip()
        raw_text = normalize_punctuation(raw_text)

    return {
        "text": raw_text,
        "speakers": list(speaker_map.values()),
        "speaker_count": len(speaker_map),
        "segments": sentence_segments,
        "speaker_segments": speaker_segments,
    }
