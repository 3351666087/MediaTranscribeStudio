"""Score completed short sample-library runs without hiding missing evidence."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.language import normalize_language_tag
from backend.pipeline_metrics import ReferenceTurn, evaluate_reference_quality
from tools.sample_library import word_error_rate

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
) -> dict[str, Any] | None:
    predicted_boundaries = sorted(
        {
            int(segment[key])
            for segment in segments
            if isinstance(segment, dict)
            for key in ("startMs", "endMs")
            if isinstance(segment.get(key), int)
        }
    )
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
    quality = evaluate_reference_quality(predicted, reference_turns)
    return (
        {key: round(value, 9) for key, value in quality.items()},
        _boundary_quality(segments, reference_turns),
    )


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
        "language": case.get("language"),
        "region": case.get("region"),
        "evaluationSplit": case.get("evaluationSplit"),
        "scenario": case.get("scenario"),
        "resultPath": str(result_path),
        "status": "missing-result",
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
    hypothesis = " ".join(
        str(segment.get("displayText") or segment.get("normalizedText") or "")
        for segment in segments
        if isinstance(segment, dict)
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
    diarization, boundary = _diarization_quality(case, segments)
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
    base["textQuality"] = {
        "referenceAvailable": bool(reference.strip()) and text_eligible,
        "werOrCer": (
            word_error_rate(reference, hypothesis)
            if reference.strip() and text_eligible
            else None
        ),
        "referenceCharacters": len(reference),
        "hypothesisCharacters": len(hypothesis),
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
    base["runtimeQuality"] = _runtime_quality(transcript_path)
    base["reviewQuality"] = _review_quality(transcript_path)
    base["subtitleQuality"] = _subtitle_quality(
        worker_output_root / artifact_id
    )
    return base


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
            item.get("textQuality", {}).get("werOrCer")
            for item in items
            if isinstance(item.get("textQuality"), dict)
            and isinstance(item["textQuality"].get("werOrCer"), (int, float))
        ]
        der_values = [
            item.get("diarizationQuality", {}).get("der")
            for item in items
            if isinstance(item.get("diarizationQuality"), dict)
            and isinstance(item["diarizationQuality"].get("der"), (int, float))
        ]
        jer_values = [
            item.get("diarizationQuality", {}).get("jer")
            for item in items
            if isinstance(item.get("diarizationQuality"), dict)
            and isinstance(item["diarizationQuality"].get("jer"), (int, float))
        ]
        rtf_values = [
            item.get("runtimeQuality", {}).get("rtf")
            for item in items
            if isinstance(item.get("runtimeQuality"), dict)
            and isinstance(item["runtimeQuality"].get("rtf"), (int, float))
        ]
        language_accuracy_values = [
            item.get("languageQuality", {}).get("segmentAccuracy")
            for item in items
            if isinstance(item.get("languageQuality"), dict)
            and isinstance(
                item["languageQuality"].get("segmentAccuracy"),
                (int, float),
            )
        ]
        speaker_matches = [
            item.get("evidence", {}).get("speakerCountMatch")
            for item in items
            if isinstance(item.get("evidence"), dict)
            and isinstance(item["evidence"].get("speakerCountMatch"), bool)
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
        }
    return output


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
        "schemaVersion": "1.0.0",
        "libraryId": resolved.get("libraryId"),
        "cases": reports,
        "summary": {
            "total": len(reports),
            "observed": sum(item.get("status") == "observed" for item in reports),
            "languageScored": sum(
                isinstance(item.get("languageQuality"), dict)
                and item["languageQuality"].get("automaticDetectionEligible") is True
                for item in reports
            ),
            "codeSwitchDocumentScored": sum(
                isinstance(item.get("codeSwitchLanguageQuality"), dict)
                for item in reports
            ),
            "codeSwitchTimingScored": sum(
                isinstance(item.get("codeSwitchLanguageQuality"), dict)
                and item["codeSwitchLanguageQuality"].get("timeScoringEligible") is True
                and isinstance(
                    item["codeSwitchLanguageQuality"].get(
                        "durationWeightedAccuracy"
                    ),
                    (int, float),
                )
                for item in reports
            ),
            "missingEvidence": sum(
                item.get("evidence", {}).get("transcript")
                in {None, "missing", "invalid-json", "missing-segments"}
                for item in reports
            ),
        },
        "buckets": {
            field: _bucket_summary(reports, field)
            for field in ("language", "region", "speakerCount", "scenario")
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
