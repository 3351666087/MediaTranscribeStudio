"""Versioned, auditable voice-activity evidence for every media job."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import WorkerError


VOICE_ACTIVITY_SCHEMA_VERSION = "1.0.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLASSIFICATIONS = {
    "no-speech-candidates-detected",
    "no-lexical-speech-detected",
    "transcribable-speech-detected",
}


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkerError(
            "VOICE_ACTIVITY_INVALID",
            f"{field} must be a positive integer",
        )
    return value


def _windows(
    values: Sequence[Mapping[str, Any]],
    *,
    media_duration_ms: int,
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    cursor = 0
    seen: set[str] = set()
    for index, value in enumerate(values):
        window_id = str(value.get("id") or "").strip()
        start_ms = value.get("startMs")
        end_ms = value.get("endMs")
        if (
            not window_id
            or window_id in seen
            or isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < cursor
            or end_ms <= start_ms
            or end_ms > media_duration_ms
        ):
            raise WorkerError(
                "VOICE_ACTIVITY_INVALID",
                f"windows[{index}] is invalid or overlaps another window",
            )
        seen.add(window_id)
        cursor = end_ms
        output.append(
            {
                "id": window_id,
                "startMs": start_ms,
                "endMs": end_ms,
                "durationMs": end_ms - start_ms,
            }
        )
    return tuple(output)


def build_voice_activity(
    *,
    job_id: str,
    source_sha256: str,
    media_duration_ms: int,
    normalization_profile: str,
    provider: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    minimum_window_ms: int,
    classification: str,
    has_transcribable_speech: bool,
) -> dict[str, Any]:
    """Build and validate one canonical voice-activity artifact."""

    normalized_windows = _windows(
        windows,
        media_duration_ms=_positive_integer(
            media_duration_ms,
            "mediaDurationMs",
        ),
    )
    speech_duration_ms = sum(
        item["durationMs"] for item in normalized_windows
    )
    value = {
        "schemaVersion": VOICE_ACTIVITY_SCHEMA_VERSION,
        "artifactType": "voice-activity",
        "jobId": str(job_id),
        "sourceSha256": str(source_sha256),
        "mediaDurationMs": media_duration_ms,
        "normalizationProfile": str(normalization_profile),
        "provider": dict(provider),
        "minimumWindowMs": minimum_window_ms,
        "classification": classification,
        "hasSpeechCandidates": bool(normalized_windows),
        "hasTranscribableSpeech": has_transcribable_speech,
        "speechWindowCount": len(normalized_windows),
        "speechDurationMs": speech_duration_ms,
        "speechRatio": round(speech_duration_ms / media_duration_ms, 12),
        "windows": list(normalized_windows),
    }
    return validate_voice_activity(value)


def validate_voice_activity(value: Any) -> dict[str, Any]:
    """Reject malformed or internally contradictory voice evidence."""

    if not isinstance(value, Mapping):
        raise WorkerError(
            "VOICE_ACTIVITY_INVALID",
            "voice activity evidence must be an object",
        )
    required = {
        "schemaVersion",
        "artifactType",
        "jobId",
        "sourceSha256",
        "mediaDurationMs",
        "normalizationProfile",
        "provider",
        "minimumWindowMs",
        "classification",
        "hasSpeechCandidates",
        "hasTranscribableSpeech",
        "speechWindowCount",
        "speechDurationMs",
        "speechRatio",
        "windows",
    }
    if set(value) != required:
        raise WorkerError(
            "VOICE_ACTIVITY_INVALID",
            "voice activity evidence has missing or unsupported fields",
        )
    job_id = value.get("jobId")
    source_sha256 = value.get("sourceSha256")
    normalization_profile = value.get("normalizationProfile")
    provider = value.get("provider")
    classification = value.get("classification")
    has_candidates = value.get("hasSpeechCandidates")
    has_transcribable = value.get("hasTranscribableSpeech")
    raw_windows = value.get("windows")
    media_duration_ms = _positive_integer(
        value.get("mediaDurationMs"),
        "mediaDurationMs",
    )
    minimum_window_ms = _positive_integer(
        value.get("minimumWindowMs"),
        "minimumWindowMs",
    )
    if (
        value.get("schemaVersion") != VOICE_ACTIVITY_SCHEMA_VERSION
        or value.get("artifactType") != "voice-activity"
        or not isinstance(job_id, str)
        or not job_id.strip()
        or not isinstance(source_sha256, str)
        or _SHA256.fullmatch(source_sha256) is None
        or not isinstance(normalization_profile, str)
        or not normalization_profile.strip()
        or not isinstance(provider, Mapping)
        or set(provider) != {"id", "version"}
        or any(
            not isinstance(provider.get(key), str)
            or not str(provider.get(key)).strip()
            for key in ("id", "version")
        )
        or classification not in _CLASSIFICATIONS
        or not isinstance(has_candidates, bool)
        or not isinstance(has_transcribable, bool)
        or not isinstance(raw_windows, list)
        or any(not isinstance(item, Mapping) for item in raw_windows)
    ):
        raise WorkerError(
            "VOICE_ACTIVITY_INVALID",
            "voice activity evidence contains invalid fields",
        )
    normalized_windows = _windows(
        raw_windows,
        media_duration_ms=media_duration_ms,
    )
    speech_duration_ms = sum(
        item["durationMs"] for item in normalized_windows
    )
    expected_ratio = round(speech_duration_ms / media_duration_ms, 12)
    if (
        value.get("speechWindowCount") != len(normalized_windows)
        or value.get("speechDurationMs") != speech_duration_ms
        or value.get("speechRatio") != expected_ratio
        or has_candidates != bool(normalized_windows)
        or (
            classification == "no-speech-candidates-detected"
            and (has_candidates or has_transcribable)
        )
        or (
            classification == "no-lexical-speech-detected"
            and (not has_candidates or has_transcribable)
        )
        or (
            classification == "transcribable-speech-detected"
            and (not has_candidates or not has_transcribable)
        )
    ):
        raise WorkerError(
            "VOICE_ACTIVITY_INVALID",
            "voice activity evidence is internally inconsistent",
        )
    return {
        **dict(value),
        "jobId": job_id.strip(),
        "normalizationProfile": normalization_profile.strip(),
        "provider": {
            "id": str(provider["id"]).strip(),
            "version": str(provider["version"]).strip(),
        },
        "windows": list(normalized_windows),
    }


def with_voice_activity_classification(
    value: Mapping[str, Any],
    *,
    classification: str,
    has_transcribable_speech: bool,
) -> dict[str, Any]:
    updated = {
        **dict(value),
        "classification": classification,
        "hasTranscribableSpeech": has_transcribable_speech,
    }
    return validate_voice_activity(updated)


__all__ = [
    "VOICE_ACTIVITY_SCHEMA_VERSION",
    "build_voice_activity",
    "validate_voice_activity",
    "with_voice_activity_classification",
]
