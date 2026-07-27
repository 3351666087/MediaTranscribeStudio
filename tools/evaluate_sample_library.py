"""Score completed short sample-library runs without hiding missing evidence."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.language import normalize_language_tag
from backend.errors import WorkerError
from backend.final_adjudication import (
    validate_final_adjudicated_transcript,
)
from backend.pipeline_metrics import (
    ReferenceTurn,
    evaluate_reference_quality,
    maximum_weight_assignment,
)
from backend.speaker_timeline import (
    speaker_timeline_turns,
    validate_speaker_timeline,
)
from tools.sample_library import tokenize_for_score, word_error_rate

_MEETEVAL_VERSION = "0.4.3"
_MEETEVAL_REVISION = "badcd3c7cf82f98d2ac1f292801fbe6e9093ee2f"
_MEETEVAL_MAX_SPEAKER_STREAMS = 20
_TCPWER_COLLAR_SECONDS = 5.0
_SRT_TIMESTAMP = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$"
)
_VTT_TIMESTAMP = re.compile(
    r"^\d{2}:\d{2}(?::\d{2})?\.\d{3}\s+-->\s+"
    r"\d{2}:\d{2}(?::\d{2})?\.\d{3}$"
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _transcript_path(result: dict[str, Any]) -> Path | None:
    terminal = result.get("terminal_event")
    if not isinstance(terminal, dict):
        return None
    payload = terminal.get("payload")
    if not isinstance(payload, dict):
        return None
    artifact_paths = payload.get("artifactPaths", [])
    if not isinstance(artifact_paths, list):
        return None
    for value in artifact_paths:
        if isinstance(value, str) and value.endswith("transcript-document.v2.json"):
            return Path(value)
    for value in artifact_paths:
        if isinstance(value, str) and value.endswith(".json"):
            candidate = Path(value)
            if candidate.name == "transcript-document.v2.json":
                return candidate
    review_path = payload.get("reviewQueuePath")
    if isinstance(review_path, str):
        candidate = Path(review_path).parent.parent / "transcript-document.v2.json"
        if candidate.is_file():
            return candidate
    return None


def _subtitle_quality(root: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for suffix in (".srt", ".vtt", ".ass"):
        files = sorted(root.rglob(f"*{suffix}")) if root.exists() else []
        valid = 0
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if suffix == ".srt":
                ok = any(_SRT_TIMESTAMP.fullmatch(line) for line in lines)
            elif suffix == ".vtt":
                ok = bool(lines and lines[0].casefold() == "webvtt") and any(
                    _VTT_TIMESTAMP.fullmatch(line) for line in lines
                )
            else:
                ok = "[events]" in text.casefold() and any(
                    line.casefold().startswith("dialogue:") for line in lines
                )
            if ok:
                valid += 1
        checks[suffix[1:]] = {"files": len(files), "valid": valid}
    return checks


def _runtime_quality(transcript_path: Path) -> dict[str, Any] | None:
    metrics = _read_json(transcript_path.parent / "pipeline-metrics.v1.json")
    if metrics is None:
        return None
    runtime = metrics.get("runtime")
    cache = metrics.get("cache")
    routing = metrics.get("routing")
    resources = metrics.get("resources")
    if not all(
        isinstance(value, dict)
        for value in (runtime, cache, routing, resources)
    ):
        return None
    return {
        "metricsPath": str(transcript_path.parent / "pipeline-metrics.v1.json"),
        "elapsedMs": runtime.get("elapsedMs"),
        "rtf": runtime.get("rtf"),
        "stages": runtime.get("stages"),
        "peakRamMb": resources.get("peakRamMb"),
        "peakVramMb": resources.get("peakVramMb"),
        "cacheHitRate": cache.get("hitRate"),
        "recomputationRate": cache.get("recomputationRate"),
        "escalationRate": routing.get("escalationRate"),
        "escalatedSegments": routing.get("escalated"),
        "totalSegments": routing.get("segments"),
    }


def _review_quality(transcript_path: Path) -> dict[str, Any] | None:
    review_path = transcript_path.parent / "review" / "review-queue.json"
    review = _read_json(review_path)
    if review is None:
        return None
    items = review.get("items")
    return {
        "path": str(review_path),
        "openCount": review.get("openCount"),
        "itemCount": len(items) if isinstance(items, list) else None,
    }


def _final_adjudication_path(
    result: Mapping[str, Any],
    transcript_path: Path,
) -> Path:
    terminal = result.get("terminal_event")
    payload = terminal.get("payload") if isinstance(terminal, Mapping) else None
    paths = payload.get("artifactPaths") if isinstance(payload, Mapping) else None
    if isinstance(paths, list):
        for value in paths:
            if (
                isinstance(value, str)
                and Path(value).name
                == "final-adjudicated-transcript.v1.json"
            ):
                return Path(value)
    return transcript_path.parent / "final-adjudicated-transcript.v1.json"


def _language_root(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        normalized = normalize_language_tag(value, allow_auto=True)
    except ValueError:
        return None
    if normalized in {"auto", "mul", "und"}:
        return normalized
    return normalized.split("-", 1)[0].casefold()


def _language_quality(
    *,
    expected_language: object,
    transcript: dict[str, Any],
    segments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    expected_root = _language_root(expected_language)
    document_language = transcript.get("language")
    document_root = _language_root(document_language)
    detected: list[str] = []
    requested: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        evidence = segment.get("evidence")
        asr = evidence.get("asr") if isinstance(evidence, dict) else None
        if not isinstance(asr, dict):
            continue
        detected_root = _language_root(
            segment.get("language", asr.get("language"))
        )
        if detected_root is not None:
            detected.append(detected_root)
        requested_root = _language_root(asr.get("requestedLanguage"))
        if requested_root is not None:
            requested.append(requested_root)
    detected_counts = {
        language: detected.count(language) for language in sorted(set(detected))
    }
    requested_languages = sorted(set(requested))
    automatic_eligible = (
        expected_root not in {None, "auto", "mul", "und"}
        and bool(detected)
        and requested_languages == ["auto"]
    )
    correct = (
        sum(language == expected_root for language in detected)
        if automatic_eligible
        else None
    )
    return {
        "expectedLanguage": expected_language,
        "expectedLanguageRoot": expected_root,
        "documentLanguage": document_language,
        "documentLanguageRoot": document_root,
        "documentLanguageMatch": (
            document_root == expected_root
            if expected_root not in {None, "auto", "mul", "und"}
            else None
        ),
        "requestedLanguages": requested_languages,
        "detectedLanguageCounts": detected_counts,
        "segmentCount": len(segments),
        "scoredSegmentCount": len(detected) if automatic_eligible else 0,
        "correctSegmentCount": correct,
        "segmentAccuracy": (
            correct / len(detected)
            if automatic_eligible and correct is not None
            else None
        ),
        "undeterminedRate": (
            detected.count("und") / len(detected) if detected else None
        ),
        "automaticDetectionEligible": automatic_eligible,
    }


def _final_language_quality(
    *,
    expected_language: object,
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_root = _language_root(expected_language)
    detected = [
        root
        for segment in segments
        if (root := _language_root(segment.get("language"))) is not None
    ]
    detected_counts = {
        language: detected.count(language) for language in sorted(set(detected))
    }
    eligible = expected_root not in {None, "auto", "mul", "und"}
    correct = (
        sum(language == expected_root for language in detected)
        if eligible
        else None
    )
    return {
        "authority": "final-adjudicated-language-span",
        "expectedLanguage": expected_language,
        "expectedLanguageRoot": expected_root,
        "detectedLanguageCounts": detected_counts,
        "segmentCount": len(segments),
        "scoredSegmentCount": len(detected) if eligible else 0,
        "correctSegmentCount": correct,
        "segmentAccuracy": (
            correct / len(detected)
            if eligible and correct is not None and detected
            else None
        ),
        "undeterminedRate": (
            detected.count("und") / len(detected) if detected else None
        ),
        "eligible": eligible,
    }


def _segment_language_roots(segment: Mapping[str, Any]) -> tuple[str, ...]:
    evidence = segment.get("evidence")
    asr = evidence.get("asr") if isinstance(evidence, Mapping) else None
    candidates = (
        asr.get("languageCandidates")
        if isinstance(asr, Mapping)
        else None
    )
    roots: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            root = _language_root(candidate)
            if root not in {None, "auto", "mul", "und"} and root not in roots:
                roots.append(root)
    evidence_language = (
        asr.get("language") if isinstance(asr, Mapping) else None
    )
    direct = _language_root(segment.get("language", evidence_language))
    if direct not in {None, "auto", "mul", "und"} and direct not in roots:
        roots.append(direct)
    return tuple(roots)


def _code_switch_language_quality(
    *,
    case: Mapping[str, Any],
    transcript: Mapping[str, Any],
    segments: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    truth = case.get("languageTruth")
    if not isinstance(truth, Mapping):
        return None
    expected_raw = truth.get("expectedLanguages", case.get("expectedLanguages"))
    if not isinstance(expected_raw, list) or not expected_raw:
        return None
    expected = sorted(
        {
            root
            for value in expected_raw
            if (root := _language_root(value))
            not in {"auto", "mul", "und"}
        }
    )
    detected_by_segment = [
        {
            "startMs": segment.get("startMs"),
            "endMs": segment.get("endMs"),
            "languages": list(_segment_language_roots(segment)),
        }
        for segment in segments
        if isinstance(segment, Mapping)
    ]
    detected = sorted(
        {
            language
            for item in detected_by_segment
            for language in item["languages"]
        }
    )
    document_root = _language_root(transcript.get("language"))
    detected_set = set(detected)
    expected_set = set(expected)
    base: dict[str, Any] = {
        "qualification": truth.get("qualification"),
        "expectedLanguages": expected,
        "detectedLanguages": detected,
        "documentLanguage": transcript.get("language"),
        "documentLanguageRoot": document_root,
        "documentMarkedMultilingual": document_root == "mul",
        "expectedLanguageRecall": (
            len(expected_set & detected_set) / len(expected_set)
            if expected_set
            else None
        ),
        "expectedLanguageSetExact": detected_set == expected_set,
        "unexpectedLanguages": sorted(detected_set - expected_set),
        "missingLanguages": sorted(expected_set - detected_set),
        "segmentLanguageEvidence": detected_by_segment,
        "timeScoringEligible": truth.get("timeScoringEligible") is True,
        "expectedSwitchCount": truth.get("expectedSwitchCount"),
        "switchLevel": truth.get("switchLevel"),
        "mainLanguage": truth.get("mainLanguage"),
        "durationWeightedAccuracy": None,
        "referenceSwitchPointsMs": [],
        "predictedSwitchPointsMs": [],
        "switchPointAbsoluteErrorsMs": [],
        "switchPointMeanAbsoluteErrorMs": None,
        "switchPointMaxAbsoluteErrorMs": None,
    }
    intervals = truth.get("intervals")
    if (
        truth.get("timeScoringEligible") is not True
        or not isinstance(intervals, list)
    ):
        return base
    reference_intervals: list[tuple[int, int, str]] = []
    for interval in intervals:
        if not isinstance(interval, Mapping):
            continue
        root = _language_root(interval.get("language"))
        start = interval.get("startSeconds")
        end = interval.get("endSeconds")
        if (
            root in {None, "auto", "mul", "und"}
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
        ):
            continue
        start_ms = round(float(start) * 1000)
        end_ms = round(float(end) * 1000)
        if end_ms > start_ms:
            reference_intervals.append((start_ms, end_ms, root))
    total_ms = 0
    correct_ms = 0
    predicted_primary: list[tuple[int, int, str | None]] = []
    for segment, evidence in zip(segments, detected_by_segment):
        start = segment.get("startMs")
        end = segment.get("endMs")
        languages = evidence["languages"]
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and end > start
        ):
            primary = languages[0] if len(languages) == 1 else None
            predicted_primary.append((start, end, primary))
            for ref_start, ref_end, expected_language in reference_intervals:
                overlap = max(0, min(end, ref_end) - max(start, ref_start))
                if overlap <= 0:
                    continue
                total_ms += overlap
                if primary == expected_language:
                    correct_ms += overlap
    base["durationWeightedAccuracy"] = (
        correct_ms / total_ms if total_ms else None
    )
    reference_points_raw = truth.get("switchPointsSeconds")
    reference_points = (
        [
            round(float(value) * 1000)
            for value in reference_points_raw
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if isinstance(reference_points_raw, list)
        else []
    )
    predicted_points = [
        current[0]
        for previous, current in zip(predicted_primary, predicted_primary[1:])
        if previous[2] is not None
        and current[2] is not None
        and previous[2] != current[2]
    ]
    errors = (
        [
            min(abs(reference - predicted) for predicted in predicted_points)
            for reference in reference_points
        ]
        if predicted_points
        else []
    )
    base["referenceSwitchPointsMs"] = reference_points
    base["predictedSwitchPointsMs"] = predicted_points
    base["switchPointAbsoluteErrorsMs"] = errors
    base["switchPointMeanAbsoluteErrorMs"] = (
        statistics.fmean(errors) if errors else None
    )
    base["switchPointMaxAbsoluteErrorMs"] = max(errors) if errors else None
    base["missedReferenceSwitchCount"] = (
        len(reference_points) if not predicted_points else 0
    )
    return base


def _boundary_quality(
    segments: Sequence[dict[str, Any]],
    reference_turns: Sequence[ReferenceTurn],
    *,
    speaker_timeline: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if speaker_timeline is not None:
        predicted_boundaries = sorted(
            {
                int(turn[key])
                for turn in speaker_timeline_turns(
                    speaker_timeline,
                    mode="regular",
                )
                for key in ("startMs", "endMs")
                if isinstance(turn.get(key), int)
                and not isinstance(turn.get(key), bool)
            }
        )
        prediction_source = "speakerTimeline.regular"
    else:
        predicted_boundaries = sorted(
            {
                int(segment[key])
                for segment in segments
                if isinstance(segment, dict)
                for key in ("startMs", "endMs")
                if isinstance(segment.get(key), int)
                and not isinstance(segment.get(key), bool)
            }
        )
        prediction_source = "segments"
    reference_boundaries = sorted(
        {
            value
            for turn in reference_turns
            for value in (turn.start_ms, turn.end_ms)
        }
    )
    if not predicted_boundaries or not reference_boundaries:
        return None
    errors = [
        min(abs(reference - predicted) for predicted in predicted_boundaries)
        for reference in reference_boundaries
    ]
    ordered = sorted(errors)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        rank = (len(ordered) - 1) * fraction
        lower = int(rank)
        upper = min(len(ordered) - 1, lower + 1)
        weight = rank - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "predictionSource": prediction_source,
        "referenceBoundaryCount": len(reference_boundaries),
        "predictedBoundaryCount": len(predicted_boundaries),
        "meanAbsoluteErrorMs": round(statistics.fmean(errors), 6),
        "p50AbsoluteErrorMs": round(percentile(0.5), 6),
        "p95AbsoluteErrorMs": round(percentile(0.95), 6),
        "maxAbsoluteErrorMs": max(errors),
    }


def _diarization_quality(
    case: dict[str, Any],
    segments: Sequence[dict[str, Any]],
    *,
    speaker_timeline: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    eligibility = case.get("truthEligibility")
    turns = case.get("turns")
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("derJer") is not True
        or not isinstance(turns, list)
        or not turns
    ):
        return None, None
    reference_turns = [
        ReferenceTurn(
            start_ms=round(float(turn["startSeconds"]) * 1000),
            end_ms=round(float(turn["endSeconds"]) * 1000),
            speaker_ids=(str(turn["speakerId"]),),
        )
        for turn in turns
        if isinstance(turn, dict)
    ]
    predicted = [
        SimpleNamespace(
            start_ms=int(segment["startMs"]),
            end_ms=int(segment["endMs"]),
            speaker_id=str(segment["speakerId"]),
            overlapping=bool(
                segment.get("overlapping")
                or segment.get("overlap")
                or segment.get("isOverlap")
            ),
            evidence=(
                dict(segment["evidence"])
                if isinstance(segment.get("evidence"), dict)
                else {}
            ),
        )
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("startMs"), int)
        and isinstance(segment.get("endMs"), int)
        and isinstance(segment.get("speakerId"), str)
    ]
    quality = evaluate_reference_quality(
        predicted,
        reference_turns,
        speaker_timeline=speaker_timeline,
    )
    return (
        {key: round(value, 9) for key, value in quality.items()},
        _boundary_quality(
            segments,
            reference_turns,
            speaker_timeline=speaker_timeline,
        ),
    )


def _native_full_timeline_quality(
    *,
    case: dict[str, Any],
    segments: Sequence[dict[str, Any]],
    duration_ms: int | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Score an immutable model-native timeline before canonical mapping."""

    snapshots: list[Mapping[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        evidence = segment.get("evidence")
        overlap = evidence.get("overlap") if isinstance(evidence, Mapping) else None
        snapshot = (
            overlap.get("fullTimelineInference")
            if isinstance(overlap, Mapping)
            else None
        )
        if snapshot is not None:
            if not isinstance(snapshot, Mapping):
                raise ValueError("model-native full timeline must be an object")
            snapshots.append(snapshot)
    if not snapshots:
        return None, None
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 1
        or len(snapshots) != len(segments)
    ):
        raise ValueError(
            "model-native full timeline requires complete segment coverage "
            "and transcript duration"
        )

    serialized = {
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for snapshot in snapshots
    }
    if len(serialized) != 1:
        raise ValueError("model-native full timeline snapshots must be immutable")
    snapshot = snapshots[0]
    turns = snapshot.get("speakerTurns")
    local_speakers = snapshot.get("localSpeakers")
    declared_hash = snapshot.get("speakerTurnsSha256")
    if (
        snapshot.get("scope") != "full-normalized-timeline"
        or snapshot.get("startMs") != 0
        or snapshot.get("endMs") != duration_ms
        or not isinstance(turns, list)
        or not turns
        or snapshot.get("turnCount") != len(turns)
        or not isinstance(local_speakers, list)
        or not local_speakers
        or snapshot.get("localSpeakerCount") != len(local_speakers)
        or not isinstance(declared_hash, str)
    ):
        raise ValueError("model-native full timeline metadata is invalid")
    serialized_turns = json.dumps(
        turns,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(serialized_turns).hexdigest() != declared_hash:
        raise ValueError("model-native full timeline hash is invalid")

    predicted: list[dict[str, Any]] = []
    observed_speakers: set[str] = set()
    previous_start = -1
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            raise ValueError("model-native full timeline turn is invalid")
        start_ms = turn.get("startMs")
        end_ms = turn.get("endMs")
        speaker_id = turn.get("localSpeaker")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < previous_start
            or start_ms < 0
            or end_ms <= start_ms
            or end_ms > duration_ms
            or not isinstance(speaker_id, str)
            or not speaker_id.strip()
        ):
            raise ValueError(
                f"model-native full timeline turn {index} is invalid"
            )
        previous_start = start_ms
        observed_speakers.add(speaker_id.strip())
        predicted.append(
            {
                "startMs": start_ms,
                "endMs": end_ms,
                "speakerId": speaker_id.strip(),
            }
        )
    if sorted(observed_speakers) != local_speakers:
        raise ValueError(
            "model-native full timeline speaker inventory is invalid"
        )

    diarization, boundary = _diarization_quality(case, predicted)
    expected_count = case.get("expectedSpeakerCount")
    count_eligible = (
        isinstance(expected_count, int)
        and not isinstance(expected_count, bool)
        and expected_count > 0
        and (
            not isinstance(case.get("truthEligibility"), Mapping)
            or case["truthEligibility"].get("speakerCount") is not False
        )
    )
    quality = {
        "authority": "model-native-full-timeline-unmapped",
        "localSpeakerCount": len(local_speakers),
        "expectedSpeakerCount": expected_count,
        "speakerCountAbsoluteError": (
            abs(len(local_speakers) - expected_count)
            if count_eligible
            else None
        ),
        "speakerCountMatch": (
            len(local_speakers) == expected_count if count_eligible else None
        ),
        "turnCount": len(turns),
        "speakerTurnsSha256": declared_hash,
        "speakerCountConstraints": snapshot.get("speakerCountConstraints"),
        **(diarization or {}),
    }
    return quality, boundary


def _metric_transcript_text(segment: Mapping[str, Any]) -> str:
    for field in ("rawText", "normalizedText", "displayText"):
        value = segment.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _normalized_literal(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _factual_integrity_quality(
    *,
    case: Mapping[str, Any],
    final_text: str,
) -> dict[str, Any]:
    truth = case.get("factualTruth")
    if not isinstance(truth, Mapping):
        return {
            "eligible": False,
            "scored": False,
            "reason": "annotated-factual-truth-missing",
        }
    required = truth.get("requiredLiterals")
    forbidden = truth.get("forbiddenLiterals", [])
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(value, str) and value for value in required)
        or not isinstance(forbidden, list)
        or not all(isinstance(value, str) and value for value in forbidden)
    ):
        raise ValueError(
            "factualTruth requires non-empty requiredLiterals and an optional "
            "forbiddenLiterals string array"
        )
    normalized = _normalized_literal(final_text)
    required_results = [
        {
            "literal": value,
            "preserved": _normalized_literal(value) in normalized,
        }
        for value in required
    ]
    forbidden_results = [
        {
            "literal": value,
            "hallucinated": _normalized_literal(value) in normalized,
        }
        for value in forbidden
    ]
    preserved = sum(item["preserved"] for item in required_results)
    hallucinated = sum(item["hallucinated"] for item in forbidden_results)
    return {
        "eligible": True,
        "scored": True,
        "requiredLiteralCount": len(required_results),
        "preservedRequiredLiteralCount": preserved,
        "requiredLiteralRecall": preserved / len(required_results),
        "forbiddenLiteralCount": len(forbidden_results),
        "hallucinatedForbiddenLiteralCount": hallucinated,
        "passed": preserved == len(required_results) and hallucinated == 0,
        "requiredLiterals": required_results,
        "forbiddenLiterals": forbidden_results,
    }


def _scoring_unit(texts: Sequence[str]) -> str:
    uses_character_tokens = [
        any("\u3400" <= char <= "\u9fff" for char in text)
        for text in texts
        if text.strip()
    ]
    if uses_character_tokens and all(uses_character_tokens):
        return "character"
    if any(uses_character_tokens):
        return "mixed-character-word"
    return "word"


def _joint_metric_labels(scoring_unit: str) -> dict[str, str]:
    if scoring_unit == "character":
        return {
            "cpWer": "cpCER",
            "tcpWer": "tcpCER",
            "speakerAttributedWer": "speaker-attributed CER",
        }
    if scoring_unit == "word":
        return {
            "cpWer": "cpWER",
            "tcpWer": "tcpWER",
            "speakerAttributedWer": "SA-WER",
        }
    return {
        "cpWer": "concatenated-permutation mixed-token error rate",
        "tcpWer": "time-constrained mixed-token error rate",
        "speakerAttributedWer": "speaker-attributed mixed-token error rate",
    }


def _scoring_segment(
    *,
    speaker: str,
    text: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    return {
        "speaker": speaker,
        "words": tokenize_for_score(text),
        "start_time": start_ms / 1000.0,
        "end_time": end_ms / 1000.0,
    }


def _reference_transcript_segments(
    case: Mapping[str, Any],
    *,
    duration_ms: int | None,
) -> tuple[list[dict[str, Any]], str | None]:
    raw_turns = case.get("referenceTranscriptTurns")
    if not isinstance(raw_turns, list) or not raw_turns:
        return [], "reference-transcript-turns-missing"
    segments: list[dict[str, Any]] = []
    for index, turn in enumerate(raw_turns):
        if not isinstance(turn, Mapping):
            raise ValueError(
                f"reference transcript turn {index} must be an object"
            )
        speaker = turn.get("speakerId")
        start_seconds = turn.get("startSeconds")
        end_seconds = turn.get("endSeconds")
        transcript = turn.get("transcript")
        if (
            not isinstance(speaker, str)
            or not speaker.strip()
            or isinstance(start_seconds, bool)
            or not isinstance(start_seconds, (int, float))
            or isinstance(end_seconds, bool)
            or not isinstance(end_seconds, (int, float))
            or not isinstance(transcript, str)
            or not transcript.strip()
        ):
            raise ValueError(
                f"reference transcript turn {index} is incomplete"
            )
        start_ms = round(float(start_seconds) * 1000)
        end_ms = round(float(end_seconds) * 1000)
        if (
            start_ms < 0
            or end_ms <= start_ms
            or (
                isinstance(duration_ms, int)
                and end_ms > duration_ms
            )
        ):
            raise ValueError(
                f"reference transcript turn {index} has invalid boundaries"
            )
        segment = _scoring_segment(
            speaker=speaker.strip(),
            text=transcript,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if not segment["words"]:
            raise ValueError(
                f"reference transcript turn {index} has no scoring tokens"
            )
        segments.append(segment)
    segments.sort(
        key=lambda item: (
            item["start_time"],
            item["end_time"],
            item["speaker"],
        )
    )
    expected_speakers = case.get("speakerSet")
    if isinstance(expected_speakers, list):
        normalized_expected = {
            str(item).strip()
            for item in expected_speakers
            if isinstance(item, str) and item.strip()
        }
        observed = {str(item["speaker"]) for item in segments}
        if normalized_expected != observed:
            raise ValueError(
                "reference transcript speakers do not match the truth speaker set"
            )
    return segments, None


def _hypothesis_transcript_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    text_field: str = "rawText",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        start_ms = segment.get("startMs")
        end_ms = segment.get("endMs")
        speaker = segment.get("speakerId")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < 0
            or end_ms <= start_ms
            or not isinstance(speaker, str)
            or not speaker.strip()
        ):
            raise ValueError(
                f"hypothesis transcript segment {index} has invalid attribution"
            )
        text = segment.get(text_field)
        if not isinstance(text, str) or not text.strip():
            if text_field == "rawText":
                derived_text_available = any(
                    isinstance(segment.get(field), str)
                    and bool(str(segment[field]).strip())
                    for field in ("normalizedText", "displayText")
                )
                if derived_text_available:
                    raise ValueError(
                        "joint transcription scoring requires immutable rawText"
                    )
            continue
        scoring = _scoring_segment(
            speaker=speaker.strip(),
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if scoring["words"]:
            output.append(scoring)
    output.sort(
        key=lambda item: (
            item["start_time"],
            item["end_time"],
            item["speaker"],
        )
    )
    return output


def _acoustic_prediction_turns(
    *,
    segments: Sequence[Mapping[str, Any]],
    speaker_timeline: Mapping[str, Any] | None,
) -> list[tuple[int, int, str]]:
    if speaker_timeline is not None:
        timeline_turns = speaker_timeline_turns(
            speaker_timeline,
            mode="regular",
        )
        if timeline_turns:
            return [
                (
                    int(turn["startMs"]),
                    int(turn["endMs"]),
                    str(turn["speakerId"]),
                )
                for turn in timeline_turns
            ]
    return [
        (
            int(segment["startMs"]),
            int(segment["endMs"]),
            str(segment["speakerId"]),
        )
        for segment in segments
        if isinstance(segment.get("startMs"), int)
        and not isinstance(segment.get("startMs"), bool)
        and isinstance(segment.get("endMs"), int)
        and not isinstance(segment.get("endMs"), bool)
        and int(segment["endMs"]) > int(segment["startMs"])
        and isinstance(segment.get("speakerId"), str)
        and str(segment["speakerId"]).strip()
    ]


def _speaker_overlap_assignment(
    *,
    reference: Sequence[Mapping[str, Any]],
    hypothesis_turns: Sequence[tuple[int, int, str]],
    hypothesis_speakers: set[str],
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[str],
]:
    reference_speakers = sorted(
        {str(segment["speaker"]) for segment in reference}
    )
    predicted_speakers = sorted(
        {
            *(str(speaker) for _, _, speaker in hypothesis_turns),
            *hypothesis_speakers,
        }
    )
    reference_index = {
        speaker: index for index, speaker in enumerate(reference_speakers)
    }
    predicted_index = {
        speaker: index for index, speaker in enumerate(predicted_speakers)
    }
    weights = [
        [0.0 for _ in predicted_speakers]
        for _ in reference_speakers
    ]
    for segment in reference:
        reference_start = round(float(segment["start_time"]) * 1000)
        reference_end = round(float(segment["end_time"]) * 1000)
        reference_id = str(segment["speaker"])
        for predicted_start, predicted_end, predicted_id in hypothesis_turns:
            overlap_ms = min(reference_end, predicted_end) - max(
                reference_start,
                predicted_start,
            )
            if overlap_ms > 0:
                weights[reference_index[reference_id]][
                    predicted_index[predicted_id]
                ] += float(overlap_ms)
    mapping: list[dict[str, Any]] = []
    mapped_reference: set[str] = set()
    mapped_hypothesis: set[str] = set()
    for reference_row, predicted_column in maximum_weight_assignment(weights):
        overlap_ms = weights[reference_row][predicted_column]
        if overlap_ms <= 0.0:
            continue
        reference_id = reference_speakers[reference_row]
        hypothesis_id = predicted_speakers[predicted_column]
        mapping.append(
            {
                "referenceSpeaker": reference_id,
                "hypothesisSpeaker": hypothesis_id,
                "overlapMs": round(overlap_ms, 6),
            }
        )
        mapped_reference.add(reference_id)
        mapped_hypothesis.add(hypothesis_id)
    return (
        mapping,
        sorted(set(reference_speakers) - mapped_reference),
        sorted(set(predicted_speakers) - mapped_hypothesis),
    )


def _error_rate_payload(value: Any) -> dict[str, Any]:
    payload = dataclasses.asdict(value)
    assignment = payload.pop("assignment", None)
    reference_self_overlap = payload.pop("reference_self_overlap", None)
    hypothesis_self_overlap = payload.pop("hypothesis_self_overlap", None)
    output = {
        "errorRate": (
            round(float(payload["error_rate"]), 9)
            if payload.get("error_rate") is not None
            else None
        ),
        "errors": int(payload["errors"]),
        "referenceTokens": int(payload["length"]),
        "insertions": int(payload["insertions"]),
        "deletions": int(payload["deletions"]),
        "substitutions": int(payload["substitutions"]),
    }
    if "missed_speaker" in payload:
        output.update(
            {
                "missedSpeakers": int(payload["missed_speaker"]),
                "falseAlarmSpeakers": int(payload["falarm_speaker"]),
                "scoredSpeakers": int(payload["scored_speaker"]),
            }
        )
    if assignment is not None:
        output["assignment"] = [
            {
                "referenceSpeaker": pair[0],
                "hypothesisSpeaker": pair[1],
            }
            for pair in assignment
        ]
    if reference_self_overlap is not None:
        output["referenceSelfOverlap"] = {
            "overlapSeconds": float(reference_self_overlap["overlap_time"]),
            "totalSeconds": float(reference_self_overlap["total_time"]),
            "rate": round(
                float(reference_self_overlap["overlap_rate"]),
                9,
            ),
        }
    if hypothesis_self_overlap is not None:
        output["hypothesisSelfOverlap"] = {
            "overlapSeconds": float(hypothesis_self_overlap["overlap_time"]),
            "totalSeconds": float(hypothesis_self_overlap["total_time"]),
            "rate": round(
                float(hypothesis_self_overlap["overlap_rate"]),
                9,
            ),
        }
    return output


def _speaker_attributed_error_rate(
    *,
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    hypothesis_turns: Sequence[tuple[int, int, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from meeteval.wer.wer.error_rate import combine_error_rates
    from meeteval.wer.wer.siso import siso_word_error_rate

    reference_tokens: dict[str, list[str]] = defaultdict(list)
    hypothesis_tokens: dict[str, list[str]] = defaultdict(list)
    for segment in reference:
        reference_tokens[str(segment["speaker"])].extend(segment["words"])
    for segment in hypothesis:
        hypothesis_tokens[str(segment["speaker"])].extend(segment["words"])
    mapping, missed_reference, false_alarm_hypothesis = (
        _speaker_overlap_assignment(
            reference=reference,
            hypothesis_turns=hypothesis_turns,
            hypothesis_speakers=set(hypothesis_tokens),
        )
    )
    rates = []
    for item in mapping:
        rates.append(
            siso_word_error_rate(
                [{"words": reference_tokens[item["referenceSpeaker"]]}],
                [{"words": hypothesis_tokens[item["hypothesisSpeaker"]]}],
            )
        )
    for speaker in missed_reference:
        rates.append(
            siso_word_error_rate(
                [{"words": reference_tokens[speaker]}],
                [{"words": []}],
            )
        )
    for speaker in false_alarm_hypothesis:
        if not hypothesis_tokens[speaker]:
            continue
        rates.append(
            siso_word_error_rate(
                [{"words": []}],
                [{"words": hypothesis_tokens[speaker]}],
            )
        )
    if not rates:
        raise ValueError("speaker-attributed scoring has no token streams")
    quality = _error_rate_payload(combine_error_rates(*rates))
    quality.update(
        {
            "mappingPolicy": "time-overlap-max-weight-hungarian-v1",
            "missedReferenceSpeakers": missed_reference,
            "falseAlarmHypothesisSpeakers": false_alarm_hypothesis,
        }
    )
    return quality, mapping


def _joint_transcription_quality(
    *,
    case: Mapping[str, Any],
    transcript: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    speaker_timeline: Mapping[str, Any] | None,
    text_field: str = "rawText",
) -> dict[str, Any]:
    eligibility = case.get("truthEligibility")
    if (
        not isinstance(eligibility, Mapping)
        or eligibility.get("asr") is not True
    ):
        return {
            "eligible": False,
            "scored": False,
            "reason": "asr-truth-ineligible",
        }
    source = transcript.get("source")
    duration_ms = (
        source.get("durationMs") if isinstance(source, Mapping) else None
    )
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 1
    ):
        raise ValueError(
            "joint transcription scoring requires transcript duration"
        )
    reference, unavailable_reason = _reference_transcript_segments(
        case,
        duration_ms=duration_ms,
    )
    if unavailable_reason is not None:
        return {
            "eligible": False,
            "scored": False,
            "reason": unavailable_reason,
        }
    hypothesis = _hypothesis_transcript_segments(
        segments,
        text_field=text_field,
    )
    if not hypothesis:
        return {
            "eligible": False,
            "scored": False,
            "reason": "hypothesis-transcript-empty",
        }
    installed_version = importlib.metadata.version("meeteval")
    if installed_version != _MEETEVAL_VERSION:
        raise RuntimeError(
            "meeting metrics require exactly "
            f"meeteval=={_MEETEVAL_VERSION}; found {installed_version}"
        )

    from meeteval.wer.wer.cp import cp_word_error_rate
    from meeteval.wer.wer.time_constrained import tcp_word_error_rate

    timing_eligible = eligibility.get("turnBoundaries") is True
    attribution_eligible = eligibility.get("derJer") is True
    reference_speakers = {
        str(segment["speaker"]) for segment in reference
    }
    hypothesis_speakers = {
        str(segment["speaker"]) for segment in hypothesis
    }
    backend = {
        "package": "meeteval",
        "version": installed_version,
        "upstreamRevision": _MEETEVAL_REVISION,
        "maximumSpeakerStreams": _MEETEVAL_MAX_SPEAKER_STREAMS,
    }
    if (
        max(len(reference_speakers), len(hypothesis_speakers))
        > _MEETEVAL_MAX_SPEAKER_STREAMS
    ):
        return {
            "eligible": True,
            "scored": False,
            "reason": "meeteval-speaker-stream-limit",
            "backend": backend,
            "referenceSpeakerCount": len(reference_speakers),
            "hypothesisSpeakerCount": len(hypothesis_speakers),
            "metricEligibility": {
                "cpWer": True,
                "tcpWer": timing_eligible,
                "speakerAttributedWer": attribution_eligible,
            },
            "cpWer": None,
            "tcpWer": None,
            "speakerAttributedWer": None,
        }

    cp_error = cp_word_error_rate(reference, hypothesis)
    tcp_error = (
        tcp_word_error_rate(
            reference,
            hypothesis,
            collar=_TCPWER_COLLAR_SECONDS,
        )
        if timing_eligible
        else None
    )
    hypothesis_turns = _acoustic_prediction_turns(
        segments=segments,
        speaker_timeline=speaker_timeline,
    )
    speaker_attributed, acoustic_mapping = (
        _speaker_attributed_error_rate(
            reference=reference,
            hypothesis=hypothesis,
            hypothesis_turns=hypothesis_turns,
        )
        if attribution_eligible
        else (None, [])
    )
    reference_turns = case["referenceTranscriptTurns"]
    scoring_unit = _scoring_unit(
        [
            str(turn["transcript"])
            for turn in reference_turns
            if isinstance(turn, Mapping)
        ]
    )
    return {
        "eligible": True,
        "scored": True,
        "backend": backend,
        "tokenization": "mts-language-aware-nfkc-v1",
        "scoringUnit": scoring_unit,
        "metricLabels": _joint_metric_labels(scoring_unit),
        "hypothesisTextAuthority": text_field,
        "referenceSegmentCount": len(reference),
        "hypothesisSegmentCount": len(hypothesis),
        "referenceSpeakerCount": len(reference_speakers),
        "hypothesisSpeakerCount": len(hypothesis_speakers),
        "referenceTokenCount": sum(
            len(segment["words"]) for segment in reference
        ),
        "hypothesisTokenCount": sum(
            len(segment["words"]) for segment in hypothesis
        ),
        "cpWer": _error_rate_payload(cp_error),
        "tcpWer": (
            {
                **_error_rate_payload(tcp_error),
                "collarSeconds": _TCPWER_COLLAR_SECONDS,
                "referencePseudoWordTiming": "character_based",
                "hypothesisPseudoWordTiming": "character_based_points",
            }
            if tcp_error is not None
            else None
        ),
        "speakerAttributedWer": speaker_attributed,
        "acousticSpeakerMapping": acoustic_mapping,
        "metricEligibility": {
            "cpWer": True,
            "tcpWer": timing_eligible,
            "speakerAttributedWer": attribution_eligible,
        },
    }


def _blocked_post_semantic_acceptance(
    *,
    final_path: Path,
    reasons: Sequence[str],
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "authority": "final-adjudicated-transcript.v1",
        "status": "blocked",
        "releaseApproved": False,
        "artifactPath": str(final_path),
        "blockingReasons": list(dict.fromkeys(reasons)),
        "metrics": None,
    }
    if error is not None:
        output["error"] = dict(error)
    return output


def _post_semantic_acceptance(
    *,
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    transcript_path: Path,
    transcript: Mapping[str, Any],
) -> dict[str, Any]:
    final_path = _final_adjudication_path(result, transcript_path)
    review_path = transcript_path.parent / "review" / "review-queue.json"
    semantic_path = (
        transcript_path.parent / "semantic" / "semantic-suggestions.v1.json"
    )
    if not final_path.is_file():
        reasons = ["final-adjudicated-transcript-missing"]
        review = _read_json(review_path)
        if (
            isinstance(review, Mapping)
            and isinstance(review.get("openCount"), int)
            and review.get("openCount") != 0
        ):
            reasons.append("open-review-items")
        semantic = _read_json(semantic_path)
        if (
            not isinstance(semantic, Mapping)
            or semantic.get("status") != "completed"
        ):
            reasons.append("semantic-processing-incomplete")
        return _blocked_post_semantic_acceptance(
            final_path=final_path,
            reasons=reasons,
        )

    final = _read_json(final_path)
    review = _read_json(review_path)
    semantic = _read_json(semantic_path)
    missing_dependencies = [
        label
        for label, value in (
            ("final-adjudicated-transcript-invalid-json", final),
            ("review-queue-missing-or-invalid", review),
            ("semantic-artifact-missing-or-invalid", semantic),
        )
        if value is None
    ]
    if missing_dependencies:
        return _blocked_post_semantic_acceptance(
            final_path=final_path,
            reasons=missing_dependencies,
        )
    assert final is not None
    assert review is not None
    assert semantic is not None
    try:
        validated = validate_final_adjudicated_transcript(
            final,
            expected_document=transcript,
            expected_review_queue=review,
            expected_semantic_artifact=semantic,
        )
    except WorkerError as exc:
        return _blocked_post_semantic_acceptance(
            final_path=final_path,
            reasons=["final-artifact-binding-invalid"],
            error=exc.as_payload(),
        )

    raw_segments = validated.get("segments")
    if not isinstance(raw_segments, list):
        return _blocked_post_semantic_acceptance(
            final_path=final_path,
            reasons=["final-artifact-segments-invalid"],
        )
    segments = [
        segment for segment in raw_segments if isinstance(segment, Mapping)
    ]
    if len(segments) != len(raw_segments):
        return _blocked_post_semantic_acceptance(
            final_path=final_path,
            reasons=["final-artifact-segments-invalid"],
        )
    reference = str(
        case.get("scoringTranscript")
        or case.get("expectedTranscript")
        or ""
    )
    truth_eligibility = case.get("truthEligibility")
    text_eligible = (
        reference.strip() != ""
        and (
            not isinstance(truth_eligibility, Mapping)
            or truth_eligibility.get("asr") is not False
        )
    )
    final_text = " ".join(
        str(segment["finalText"]).strip() for segment in segments
    )
    final_text_quality = {
        "referenceAvailable": text_eligible,
        "werOrCer": (
            word_error_rate(reference, final_text)
            if text_eligible
            else None
        ),
        "referenceCharacters": len(reference),
        "hypothesisCharacters": len(final_text),
        "hypothesisAuthority": "finalText",
        "scoringUnit": (
            _scoring_unit([reference]) if text_eligible else None
        ),
    }
    expected_count = case.get("expectedSpeakerCount")
    speaker_count_eligible = (
        isinstance(expected_count, int)
        and not isinstance(expected_count, bool)
        and expected_count > 0
        and (
            not isinstance(truth_eligibility, Mapping)
            or truth_eligibility.get("speakerCount") is not False
        )
    )
    policy = validated["speakerPolicy"]
    resolved_count = policy["resolvedCount"]
    distinct_speakers = sorted(
        {str(segment["speakerId"]) for segment in segments}
    )
    speaker_count_quality = {
        "authority": "final-adjudicated-speaker-namespace",
        "eligible": speaker_count_eligible,
        "expectedSpeakerCount": expected_count,
        "resolvedSpeakerCount": resolved_count,
        "distinctAssignedSpeakerCount": len(distinct_speakers),
        "speakerCountAbsoluteError": (
            abs(resolved_count - expected_count)
            if speaker_count_eligible
            else None
        ),
        "speakerCountMatch": (
            resolved_count == expected_count
            and len(distinct_speakers) == expected_count
            if speaker_count_eligible
            else None
        ),
    }
    diarization, boundary = _diarization_quality(
        dict(case),
        [dict(segment) for segment in segments],
        speaker_timeline=None,
    )
    joint = _joint_transcription_quality(
        case=case,
        transcript={
            "source": validated["source"],
        },
        segments=segments,
        speaker_timeline=None,
        text_field="finalText",
    )
    language = _final_language_quality(
        expected_language=case.get("language"),
        segments=segments,
    )
    code_switch = _code_switch_language_quality(
        case=case,
        transcript={},
        segments=[dict(segment) for segment in segments],
    )
    factual = _factual_integrity_quality(
        case=case,
        final_text=final_text,
    )
    missing_truth: list[str] = []
    if not speaker_count_eligible:
        missing_truth.append("speaker-count-truth")
    if diarization is None:
        missing_truth.append("speaker-turn-truth")
    if not text_eligible:
        missing_truth.append("serialized-text-truth")
    if joint.get("scored") is not True:
        missing_truth.append("speaker-attributed-text-truth")
    language_truth_available = (
        language.get("eligible") is True
        or (
            isinstance(case.get("languageTruth"), Mapping)
            and isinstance(code_switch, Mapping)
        )
    )
    if not language_truth_available:
        missing_truth.append("language-truth")
    if (
        isinstance(case.get("languageTruth"), Mapping)
        and code_switch is None
    ):
        missing_truth.append("code-switch-truth")
    if factual.get("scored") is not True:
        missing_truth.append("annotated-factual-truth")
    runtime = _runtime_quality(transcript_path)
    metrics = {
        "speakerCount": speaker_count_quality,
        "diarization": diarization,
        "boundary": boundary,
        "finalText": final_text_quality,
        "jointTranscription": joint,
        "language": language,
        "codeSwitch": code_switch,
        "factualIntegrity": factual,
        "overlap": (
            {
                "f1": diarization.get("overlapF1"),
                "authority": "final-adjudicated-segment-overlap",
            }
            if isinstance(diarization, Mapping)
            else None
        ),
        "review": validated["review"],
        "runtimeResources": runtime,
    }
    status = (
        "not-scored-missing-reference-truth"
        if missing_truth
        else "not-approved-threshold-profile-missing"
    )
    return {
        "authority": "final-adjudicated-transcript.v1",
        "status": status,
        "releaseApproved": False,
        "artifactPath": str(final_path),
        "artifactValid": True,
        "acceptanceSubject": validated["acceptanceSubject"],
        "finalTextAuthority": validated["finalTextAuthority"],
        "semanticStatus": validated["semantic"]["status"],
        "openReviewCount": validated["review"]["openCount"],
        "missingReferenceTruth": missing_truth,
        "blockingReasons": (
            ["missing-reference-truth"]
            if missing_truth
            else ["acceptance-threshold-profile-missing"]
        ),
        "metrics": metrics,
    }


def evaluate_case(
    *,
    case: dict[str, Any],
    result_path: Path,
    results_root: Path,
    worker_output_root: Path,
    artifact_id: str,
) -> dict[str, Any]:
    result = _read_json(result_path)
    base: dict[str, Any] = {
        "id": case["id"],
        "sourceId": case.get("sourceId"),
        "language": case.get("language"),
        "region": case.get("region"),
        "evaluationSplit": case.get("evaluationSplit"),
        "scenario": case.get("scenario"),
        "resultPath": str(result_path),
        "status": "missing-result",
        "qualityPolicy": {
            "formalAuthority": "postSemanticAcceptance",
            "frontModelMetricsRole": "diagnostic-only",
            "nonCompensatingDomains": True,
        },
        "postSemanticAcceptance": _blocked_post_semantic_acceptance(
            final_path=(
                worker_output_root
                / artifact_id
                / "final-adjudicated-transcript.v1.json"
            ),
            reasons=["result-or-final-adjudication-missing"],
        ),
    }
    if result is None:
        return base
    base["status"] = result.get("status", "unknown")
    base["terminalType"] = result.get("terminal_type")
    if result.get("error"):
        base["errorCode"] = (
            result.get("error", {}).get("code")
            if isinstance(result.get("error"), dict)
            else "unknown"
        )
    transcript_path = _transcript_path(result)
    if transcript_path is None or not transcript_path.is_file():
        base["evidence"] = {"transcript": "missing"}
        base["subtitleQuality"] = _subtitle_quality(
            worker_output_root / artifact_id
        )
        return base
    transcript = _read_json(transcript_path)
    if transcript is None:
        base["evidence"] = {"transcript": "invalid-json"}
        return base
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        base["evidence"] = {"transcript": "missing-segments"}
        return base
    speaker_timeline = None
    raw_speaker_timeline = transcript.get("speakerTimeline")
    if raw_speaker_timeline is not None:
        source = transcript.get("source")
        policy = transcript.get("speakerPolicy")
        duration_ms = (
            source.get("durationMs")
            if isinstance(source, Mapping)
            else None
        )
        canonical_ids = (
            policy.get("speakerIds")
            if isinstance(policy, Mapping)
            else None
        )
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
            or not isinstance(canonical_ids, list)
        ):
            raise ValueError(
                "speakerTimeline requires transcript duration and canonical speaker IDs"
            )
        speaker_timeline = validate_speaker_timeline(
            raw_speaker_timeline,
            duration_ms=duration_ms,
            canonical_speaker_ids=tuple(canonical_ids),
        )
    resolved_count = transcript.get("speakerPolicy", {}).get("resolvedCount")
    actual_sequence = [
        str(segment.get("speakerId"))
        for segment in segments
        if isinstance(segment, dict) and segment.get("speakerId")
    ]
    actual_distinct_sequence = list(dict.fromkeys(actual_sequence))
    expected_turn_count = case.get("expectedTurnCount")
    if not isinstance(expected_turn_count, int) and isinstance(
        case.get("turns"),
        list,
    ):
        expected_turn_count = len(case["turns"])
    expected_count = case.get("expectedSpeakerCount")
    truth_eligibility = case.get("truthEligibility")
    speaker_count_eligible = (
        isinstance(expected_count, int)
        and not isinstance(expected_count, bool)
        and expected_count > 0
        and (
            not isinstance(truth_eligibility, dict)
            or truth_eligibility.get("speakerCount") is not False
        )
    )
    diarization, boundary = _diarization_quality(
        case,
        segments,
        speaker_timeline=speaker_timeline,
    )
    source = transcript.get("source")
    duration_ms = (
        source.get("durationMs") if isinstance(source, Mapping) else None
    )
    native_diarization, native_boundary = _native_full_timeline_quality(
        case=case,
        segments=segments,
        duration_ms=duration_ms,
    )
    base["evidence"] = {
        "transcript": str(transcript_path),
        "segmentCount": len(segments),
        "expectedTurnCount": expected_turn_count,
        "turnCountMatch": (
            len(segments) == expected_turn_count
            if isinstance(expected_turn_count, int)
            else None
        ),
        "resolvedSpeakerCount": resolved_count,
        "expectedSpeakerCount": expected_count,
        "speakerCountAbsoluteError": (
            abs(int(resolved_count) - int(expected_count))
            if isinstance(resolved_count, int) and speaker_count_eligible
            else None
        ),
        "speakerCountMatch": (
            resolved_count == expected_count
            if speaker_count_eligible
            else None
        ),
        "actualDistinctSpeakerSequence": actual_distinct_sequence,
        "speakerTimelineAuthority": (
            speaker_timeline.get("authority")
            if speaker_timeline is not None
            else None
        ),
        "expectedDistinctSpeakerCount": expected_count,
        "distinctSpeakerCountMatch": (
            len(actual_distinct_sequence) == expected_count
            if speaker_count_eligible
            else None
        ),
    }
    reference = str(
        case.get("scoringTranscript")
        or case.get("expectedTranscript")
        or ""
    )
    text_eligible = (
        not isinstance(case.get("truthEligibility"), dict)
        or case["truthEligibility"].get("asr") is not False
    )
    if reference.strip() and text_eligible:
        hypothesis_parts: list[str] = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, Mapping):
                continue
            raw_text = segment.get("rawText")
            if not isinstance(raw_text, str):
                raise ValueError(
                    f"ASR scoring segment {index} is missing immutable rawText"
                )
            hypothesis_parts.append(raw_text)
        hypothesis = " ".join(hypothesis_parts)
        hypothesis_authority = "rawText"
    else:
        hypothesis = " ".join(
            _metric_transcript_text(segment)
            for segment in segments
            if isinstance(segment, Mapping)
        )
        hypothesis_authority = "best-available-unscored"
    base["textQuality"] = {
        "referenceAvailable": bool(reference.strip()) and text_eligible,
        "werOrCer": (
            word_error_rate(reference, hypothesis)
            if reference.strip() and text_eligible
            else None
        ),
        "referenceCharacters": len(reference),
        "hypothesisCharacters": len(hypothesis),
        "hypothesisAuthority": hypothesis_authority,
        "scoringUnit": (
            _scoring_unit([reference])
            if reference.strip() and text_eligible
            else None
        ),
    }
    base["languageQuality"] = _language_quality(
        expected_language=case.get("language"),
        transcript=transcript,
        segments=segments,
    )
    base["codeSwitchLanguageQuality"] = _code_switch_language_quality(
        case=case,
        transcript=transcript,
        segments=segments,
    )
    base["diarizationQuality"] = diarization
    base["boundaryQuality"] = boundary
    base["nativeDiarizationQuality"] = native_diarization
    base["nativeBoundaryQuality"] = native_boundary
    base["jointTranscriptionQuality"] = _joint_transcription_quality(
        case=case,
        transcript=transcript,
        segments=segments,
        speaker_timeline=speaker_timeline,
    )
    base["runtimeQuality"] = _runtime_quality(transcript_path)
    base["reviewQuality"] = _review_quality(transcript_path)
    base["subtitleQuality"] = _subtitle_quality(
        worker_output_root / artifact_id
    )
    base["postSemanticAcceptance"] = _post_semantic_acceptance(
        case=case,
        result=result,
        transcript_path=transcript_path,
        transcript=transcript,
    )
    return base


def _post_metric(
    report: Mapping[str, Any],
    *path: str,
) -> Any:
    value: Any = report.get("postSemanticAcceptance")
    for field in ("metrics", *path):
        if not isinstance(value, Mapping):
            return None
        value = value.get(field)
    return value


def _bucket_summary(
    reports: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        raw = (
            report.get("evidence", {}).get("expectedSpeakerCount")
            if field == "speakerCount"
            else report.get(field)
        )
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            key = str(value) if value is not None else "unknown"
            grouped.setdefault(key, []).append(report)
    output: dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        text_values = [
            value
            for item in items
            if isinstance(
                (value := _post_metric(item, "finalText", "werOrCer")),
                (int, float),
            )
        ]
        der_values = [
            value
            for item in items
            if isinstance(
                (value := _post_metric(item, "diarization", "der")),
                (int, float),
            )
        ]
        jer_values = [
            value
            for item in items
            if isinstance(
                (value := _post_metric(item, "diarization", "jer")),
                (int, float),
            )
        ]
        rtf_values = [
            value
            for item in items
            if isinstance(
                (value := _post_metric(item, "runtimeResources", "rtf")),
                (int, float),
            )
        ]
        language_accuracy_values = [
            value
            for item in items
            if isinstance(
                (value := _post_metric(item, "language", "segmentAccuracy")),
                (int, float),
            )
        ]
        cp_wer_values = [
            value
            for item in items
            if isinstance(
                (
                    value := _post_metric(
                        item,
                        "jointTranscription",
                        "cpWer",
                        "errorRate",
                    )
                ),
                (int, float),
            )
        ]
        tcp_wer_values = [
            value
            for item in items
            if isinstance(
                (
                    value := _post_metric(
                        item,
                        "jointTranscription",
                        "tcpWer",
                        "errorRate",
                    )
                ),
                (int, float),
            )
        ]
        sa_wer_values = [
            value
            for item in items
            if isinstance(
                (
                    value := _post_metric(
                        item,
                        "jointTranscription",
                        "speakerAttributedWer",
                        "errorRate",
                    )
                ),
                (int, float),
            )
        ]
        speaker_matches = [
            value
            for item in items
            if isinstance(
                (
                    value := _post_metric(
                        item,
                        "speakerCount",
                        "speakerCountMatch",
                    )
                ),
                bool,
            )
        ]
        output[key] = {
            "total": len(items),
            "observed": sum(item.get("status") == "observed" for item in items),
            "speakerCountMatchRate": (
                sum(speaker_matches) / len(speaker_matches)
                if speaker_matches
                else None
            ),
            "meanWerOrCer": statistics.fmean(text_values) if text_values else None,
            "meanDer": statistics.fmean(der_values) if der_values else None,
            "meanJer": statistics.fmean(jer_values) if jer_values else None,
            "meanRtf": statistics.fmean(rtf_values) if rtf_values else None,
            "meanLanguageSegmentAccuracy": (
                statistics.fmean(language_accuracy_values)
                if language_accuracy_values
                else None
            ),
            "meanCpWer": (
                statistics.fmean(cp_wer_values) if cp_wer_values else None
            ),
            "meanTcpWer": (
                statistics.fmean(tcp_wer_values) if tcp_wer_values else None
            ),
            "meanSpeakerAttributedWer": (
                statistics.fmean(sa_wer_values) if sa_wer_values else None
            ),
        }
    return output


def _value_counts(
    reports: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in reports:
        value = report.get(field)
        key = str(value) if value is not None else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--worker-output-root",
        type=Path,
        default=Path(".runtime_cache/outputs/sample-library"),
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--run-suffix",
        default="",
        help="suffix between the case id and -result.json, e.g. -manual",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _select_result(
    results_root: Path,
    case_id: str,
    run_suffix: str,
) -> tuple[Path, str]:
    if run_suffix:
        return (
            results_root / f"{case_id}{run_suffix}-result.json",
            f"{case_id}{run_suffix}",
        )
    exact = results_root / f"{case_id}-result.json"
    candidates = [exact, *results_root.glob(f"{case_id}-run*-result.json")]
    existing = [path for path in candidates if path.is_file()]
    if existing:
        observed: list[Path] = []
        for path in existing:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                value = {}
            if isinstance(value, dict) and value.get("status") == "observed":
                observed.append(path)
        pool = observed or existing
        selected = max(pool, key=lambda path: path.stat().st_mtime)
        return selected, selected.name.removesuffix("-result.json")
    return exact, case_id


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = resolved.get("cases", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("resolved manifest is missing non-empty cases")
    selected = set(args.case)
    reports = []
    for case in rows:
        if not isinstance(case, dict) or (selected and case.get("id") not in selected):
            continue
        result_path, artifact_id = _select_result(
            args.results_root,
            str(case["id"]),
            args.run_suffix,
        )
        reports.append(
            evaluate_case(
                case=case,
                result_path=result_path,
                results_root=args.results_root,
                worker_output_root=args.worker_output_root,
                artifact_id=artifact_id,
            )
        )
    report = {
        "schemaVersion": "1.3.0",
        "libraryId": resolved.get("libraryId"),
        "cases": reports,
        "summary": {
            "total": len(reports),
            "observed": sum(item.get("status") == "observed" for item in reports),
            "statuses": _value_counts(reports, "status"),
            "terminalTypes": _value_counts(reports, "terminalType"),
            "errorCodes": _value_counts(reports, "errorCode"),
            "languageScored": sum(
                _post_metric(item, "language", "eligible") is True
                for item in reports
            ),
            "codeSwitchDocumentScored": sum(
                isinstance(_post_metric(item, "codeSwitch"), Mapping)
                for item in reports
            ),
            "codeSwitchTimingScored": sum(
                _post_metric(item, "codeSwitch", "timeScoringEligible") is True
                and isinstance(
                    _post_metric(
                        item,
                        "codeSwitch",
                        "durationWeightedAccuracy",
                    ),
                    (int, float),
                )
                for item in reports
            ),
            "jointTranscriptionScored": sum(
                _post_metric(item, "jointTranscription", "scored") is True
                for item in reports
            ),
            "postSemanticAcceptanceStatuses": {
                status: sum(
                    item.get("postSemanticAcceptance", {}).get("status")
                    == status
                    for item in reports
                    if isinstance(
                        item.get("postSemanticAcceptance"),
                        Mapping,
                    )
                )
                for status in sorted(
                    {
                        str(item["postSemanticAcceptance"]["status"])
                        for item in reports
                        if isinstance(
                            item.get("postSemanticAcceptance"),
                            Mapping,
                        )
                        and item["postSemanticAcceptance"].get("status")
                        is not None
                    }
                )
            },
            "validFinalAdjudicatedArtifacts": sum(
                item.get("postSemanticAcceptance", {}).get("artifactValid")
                is True
                for item in reports
                if isinstance(
                    item.get("postSemanticAcceptance"),
                    Mapping,
                )
            ),
            "blockedBeforeFormalScoring": sum(
                item.get("postSemanticAcceptance", {}).get("status")
                == "blocked"
                for item in reports
                if isinstance(
                    item.get("postSemanticAcceptance"),
                    Mapping,
                )
            ),
            "releaseApproved": bool(reports)
            and all(
                item.get("postSemanticAcceptance", {}).get(
                    "releaseApproved"
                )
                is True
                for item in reports
                if isinstance(
                    item.get("postSemanticAcceptance"),
                    Mapping,
                )
            ),
            "releaseApprovedCases": sum(
                item.get("postSemanticAcceptance", {}).get(
                    "releaseApproved"
                )
                is True
                for item in reports
                if isinstance(
                    item.get("postSemanticAcceptance"),
                    Mapping,
                )
            ),
            "missingEvidence": sum(
                item.get("evidence", {}).get("transcript")
                in {None, "missing", "invalid-json", "missing-segments"}
                for item in reports
            ),
        },
        "buckets": {
            field: _bucket_summary(reports, field)
            for field in (
                "sourceId",
                "evaluationSplit",
                "language",
                "region",
                "speakerCount",
                "scenario",
            )
        },
    }
    output = args.output or args.results_root / "quality-report.v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
