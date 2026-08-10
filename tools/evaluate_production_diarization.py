#!/usr/bin/env python3
"""Score production speaker timelines against frozen diarization truth."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import (
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)


class DiarizationEvaluationError(ValueError):
    """Raised when frozen truth or production evidence is incomplete."""


@dataclass(frozen=True)
class Turn:
    start_ms: int
    end_ms: int
    speaker: str


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        raise DiarizationEvaluationError("metric denominator must be positive")
    return round(float(numerator) / float(denominator), 12)


def _milliseconds(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiarizationEvaluationError(f"{field} must be numeric")
    if not math.isfinite(float(value)):
        raise DiarizationEvaluationError(f"{field} must be finite")
    return round(float(value) * 1_000)


def _validate_turns(
    turns: Iterable[Turn],
    *,
    duration_ms: int,
    label: str,
) -> tuple[Turn, ...]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for index, turn in enumerate(turns):
        if not turn.speaker or len(turn.speaker) > 256:
            raise DiarizationEvaluationError(
                f"{label} turn {index} has an invalid speaker"
            )
        if not 0 <= turn.start_ms < turn.end_ms <= duration_ms:
            raise DiarizationEvaluationError(
                f"{label} turn {index} is outside the evaluated duration"
            )
        grouped.setdefault(turn.speaker, []).append(
            (turn.start_ms, turn.end_ms)
        )
    merged: list[Turn] = []
    for speaker in sorted(grouped):
        intervals = sorted(grouped[speaker])
        start_ms, end_ms = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end_ms:
                end_ms = max(end_ms, next_end)
            else:
                merged.append(Turn(start_ms, end_ms, speaker))
                start_ms, end_ms = next_start, next_end
        merged.append(Turn(start_ms, end_ms, speaker))
    if not merged:
        raise DiarizationEvaluationError(f"{label} contains no turns")
    return tuple(
        sorted(merged, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker))
    )


def _reference_turns(case: Mapping[str, Any]) -> tuple[int, tuple[Turn, ...]]:
    audio = case.get("audio")
    if not isinstance(audio, Mapping):
        raise DiarizationEvaluationError("case.audio is missing")
    duration_ms = _milliseconds(
        audio.get("durationSeconds"), field="case.audio.durationSeconds"
    )
    if duration_ms <= 0:
        raise DiarizationEvaluationError("case duration must be positive")
    rows = case.get("turns")
    if not isinstance(rows, list) or not rows:
        raise DiarizationEvaluationError("case.turns is missing")
    turns: list[Turn] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DiarizationEvaluationError(
                f"case.turns[{index}] must be an object"
            )
        turns.append(
            Turn(
                _milliseconds(
                    row.get("startSeconds"),
                    field=f"case.turns[{index}].startSeconds",
                ),
                _milliseconds(
                    row.get("endSeconds"),
                    field=f"case.turns[{index}].endSeconds",
                ),
                str(row.get("speakerId", "")),
            )
        )
    normalized = _validate_turns(turns, duration_ms=duration_ms, label="reference")
    expected = case.get("expectedSpeakerCount")
    observed = len({turn.speaker for turn in normalized})
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or not 1 <= expected <= 16
    ):
        raise DiarizationEvaluationError("case.expectedSpeakerCount is invalid")
    if expected != observed:
        raise DiarizationEvaluationError(
            "reference speaker set does not match expectedSpeakerCount"
        )
    return duration_ms, normalized


def _hypothesis_turns(
    document: Mapping[str, Any], *, duration_ms: int
) -> tuple[Turn, ...]:
    timeline = document.get("speakerTimeline")
    if not isinstance(timeline, Mapping):
        raise DiarizationEvaluationError(
            "transcript has no authoritative speakerTimeline"
        )
    mapping = timeline.get("mapping")
    if not isinstance(mapping, Mapping) or mapping.get("accepted") is not True:
        raise DiarizationEvaluationError(
            "speakerTimeline canonical mapping was not accepted"
        )
    regular = timeline.get("regular")
    if (
        not isinstance(regular, Mapping)
        or regular.get("native") is not True
        or regular.get("semantics") != "overlap-preserving"
    ):
        raise DiarizationEvaluationError(
            "speakerTimeline regular timeline is not native overlap-preserving evidence"
        )
    rows = regular.get("turns")
    if not isinstance(rows, list) or not rows:
        raise DiarizationEvaluationError("speakerTimeline.regular.turns is empty")
    turns: list[Turn] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DiarizationEvaluationError(
                f"speakerTimeline.regular.turns[{index}] must be an object"
            )
        start_ms, end_ms = row.get("startMs"), row.get("endMs")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
        ):
            raise DiarizationEvaluationError(
                f"speakerTimeline.regular.turns[{index}] has invalid bounds"
            )
        turns.append(Turn(start_ms, end_ms, str(row.get("speakerId", ""))))
    return _validate_turns(turns, duration_ms=duration_ms, label="hypothesis")


def _active(turns: Sequence[Turn], midpoint: float) -> frozenset[str]:
    return frozenset(
        turn.speaker
        for turn in turns
        if turn.start_ms <= midpoint < turn.end_ms
    )


def _intervals(
    reference: Sequence[Turn],
    hypothesis: Sequence[Turn],
    *,
    duration_ms: int,
) -> tuple[tuple[int, frozenset[str], frozenset[str]], ...]:
    boundaries = {0, duration_ms}
    for turn in (*reference, *hypothesis):
        boundaries.update((turn.start_ms, turn.end_ms))
    ordered = sorted(boundaries)
    result: list[tuple[int, frozenset[str], frozenset[str]]] = []
    for start_ms, end_ms in zip(ordered, ordered[1:]):
        if end_ms <= start_ms:
            continue
        midpoint = (start_ms + end_ms) / 2
        result.append(
            (
                end_ms - start_ms,
                _active(reference, midpoint),
                _active(hypothesis, midpoint),
            )
        )
    return tuple(result)


def _overlap_matrix(
    intervals: Sequence[tuple[int, frozenset[str], frozenset[str]]],
    reference_speakers: Sequence[str],
    hypothesis_speakers: Sequence[str],
) -> dict[tuple[str, str], int]:
    matrix = {
        (reference, hypothesis): 0
        for reference in reference_speakers
        for hypothesis in hypothesis_speakers
    }
    for duration_ms, reference_active, hypothesis_active in intervals:
        for reference in reference_active:
            for hypothesis in hypothesis_active:
                matrix[(reference, hypothesis)] += duration_ms
    return matrix


def _optimal_pairs(
    reference_speakers: Sequence[str],
    hypothesis_speakers: Sequence[str],
    matrix: Mapping[tuple[str, str], int],
) -> tuple[tuple[str, str], ...]:
    """Maximum-overlap matching with a bitmask over the smaller side."""
    if not reference_speakers or not hypothesis_speakers:
        return ()
    reference_is_small = len(reference_speakers) <= len(hypothesis_speakers)
    small = tuple(reference_speakers if reference_is_small else hypothesis_speakers)
    large = tuple(hypothesis_speakers if reference_is_small else reference_speakers)
    states: dict[int, tuple[int, tuple[tuple[int, int], ...]]] = {0: (0, ())}
    for large_index, large_label in enumerate(large):
        updated = dict(states)
        for mask, (score, path) in states.items():
            for small_index, small_label in enumerate(small):
                bit = 1 << small_index
                if mask & bit:
                    continue
                weight = (
                    matrix[(small_label, large_label)]
                    if reference_is_small
                    else matrix[(large_label, small_label)]
                )
                candidate = (score + weight, (*path, (small_index, large_index)))
                next_mask = mask | bit
                current = updated.get(next_mask)
                if (
                    current is None
                    or candidate[0] > current[0]
                    or (candidate[0] == current[0] and candidate[1] < current[1])
                ):
                    updated[next_mask] = candidate
        states = updated
    full_mask = (1 << len(small)) - 1
    if full_mask not in states:
        raise DiarizationEvaluationError("speaker mapping could not be resolved")
    pairs: list[tuple[str, str]] = []
    for small_index, large_index in states[full_mask][1]:
        if reference_is_small:
            pairs.append((small[small_index], large[large_index]))
        else:
            pairs.append((large[large_index], small[small_index]))
    return tuple(sorted(pairs))


def _percentile_nearest_rank(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _boundary_metrics(
    reference: Sequence[Turn],
    hypothesis: Sequence[Turn],
    *,
    duration_ms: int,
    tolerance_ms: int,
) -> dict[str, Any]:
    reference_boundaries = sorted(
        {
            value
            for turn in reference
            for value in (turn.start_ms, turn.end_ms)
            if value not in {0, duration_ms}
        }
    )
    hypothesis_boundaries = sorted(
        {
            value
            for turn in hypothesis
            for value in (turn.start_ms, turn.end_ms)
            if value not in {0, duration_ms}
        }
    )
    errors = [
        min(abs(boundary - candidate) for candidate in hypothesis_boundaries)
        if hypothesis_boundaries
        else duration_ms
        for boundary in reference_boundaries
    ]
    matched = sum(error <= tolerance_ms for error in errors)
    hypothesis_matched = sum(
        bool(reference_boundaries)
        and min(abs(boundary - candidate) for candidate in reference_boundaries)
        <= tolerance_ms
        for boundary in hypothesis_boundaries
    )
    return {
        "toleranceMs": tolerance_ms,
        "referenceCount": len(reference_boundaries),
        "hypothesisCount": len(hypothesis_boundaries),
        "matchedReferenceCount": matched,
        "missedReferenceCount": len(reference_boundaries) - matched,
        "spuriousHypothesisCount": len(hypothesis_boundaries) - hypothesis_matched,
        "recall": (
            _ratio(matched, len(reference_boundaries))
            if reference_boundaries
            else 1.0
        ),
        "precision": (
            _ratio(hypothesis_matched, len(hypothesis_boundaries))
            if hypothesis_boundaries
            else (1.0 if not reference_boundaries else 0.0)
        ),
        "meanAbsoluteErrorMs": round(sum(errors) / len(errors), 6) if errors else 0.0,
        "p95AbsoluteErrorMs": _percentile_nearest_rank(errors, 0.95) or 0,
        "referenceNearestErrorsMs": errors,
    }


def _review_metrics(queue: Mapping[str, Any] | None) -> dict[str, Any]:
    if queue is None:
        return {
            "queuePresent": False,
            "itemCount": None,
            "openCount": None,
            "decisionCount": None,
            "reasonCounts": {},
        }
    items, decisions = queue.get("items"), queue.get("decisions")
    if not isinstance(items, list) or not isinstance(decisions, list):
        raise DiarizationEvaluationError("review queue is malformed")
    open_count = queue.get("openCount")
    if (
        isinstance(open_count, bool)
        or not isinstance(open_count, int)
        or not 0 <= open_count <= len(items)
    ):
        raise DiarizationEvaluationError("review queue openCount is invalid")
    reasons: dict[str, int] = {}
    for item in items:
        if isinstance(item, Mapping) and isinstance(item.get("reasonCode"), str):
            reason = str(item["reasonCode"])
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "queuePresent": True,
        "itemCount": len(items),
        "openCount": open_count,
        "decisionCount": len(decisions),
        "reasonCounts": dict(sorted(reasons.items())),
    }


def evaluate_case(
    case: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    boundary_tolerance_ms: int = 250,
    review_queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    duration_ms, reference = _reference_turns(case)
    source, media = document.get("source"), case.get("media")
    case_sha = media.get("sha256") if isinstance(media, Mapping) else case.get("sha256")
    if not isinstance(source, Mapping) or source.get("sha256") != case_sha:
        raise DiarizationEvaluationError(
            "transcript source SHA-256 does not match frozen case media"
        )
    if source.get("durationMs") != duration_ms:
        raise DiarizationEvaluationError("transcript source duration does not match frozen case")
    hypothesis = _hypothesis_turns(document, duration_ms=duration_ms)
    reference_speakers = sorted({turn.speaker for turn in reference})
    hypothesis_speakers = sorted({turn.speaker for turn in hypothesis})
    intervals = _intervals(reference, hypothesis, duration_ms=duration_ms)
    matrix = _overlap_matrix(intervals, reference_speakers, hypothesis_speakers)
    pairs = _optimal_pairs(reference_speakers, hypothesis_speakers, matrix)
    hypothesis_to_reference = {hyp: ref for ref, hyp in pairs}
    reference_to_hypothesis = {ref: hyp for ref, hyp in pairs}

    reference_ms = correct_ms = missed_ms = false_alarm_ms = confusion_ms = 0
    overlap_ref_ms = overlap_correct_ms = overlap_ref_region_ms = 0
    overlap_hyp_region_ms = overlap_intersection_region_ms = 0
    for duration, ref_active, hyp_active in intervals:
        ref_count, hyp_count = len(ref_active), len(hyp_active)
        correct = sum(
            hypothesis_to_reference.get(hyp) in ref_active for hyp in hyp_active
        )
        reference_ms += ref_count * duration
        correct_ms += correct * duration
        missed_ms += max(0, ref_count - hyp_count) * duration
        false_alarm_ms += max(0, hyp_count - ref_count) * duration
        confusion_ms += (min(ref_count, hyp_count) - correct) * duration
        if ref_count >= 2:
            overlap_ref_ms += ref_count * duration
            overlap_correct_ms += correct * duration
            overlap_ref_region_ms += duration
        if hyp_count >= 2:
            overlap_hyp_region_ms += duration
        if ref_count >= 2 and hyp_count >= 2:
            overlap_intersection_region_ms += duration

    ref_durations = {
        speaker: sum(duration for duration, active, _ in intervals if speaker in active)
        for speaker in reference_speakers
    }
    hyp_durations = {
        speaker: sum(duration for duration, _, active in intervals if speaker in active)
        for speaker in hypothesis_speakers
    }
    speaker_jaccard_errors: list[dict[str, Any]] = []
    for ref in reference_speakers:
        hyp = reference_to_hypothesis.get(ref)
        if hyp is None:
            intersection_ms, union_ms, error = 0, ref_durations[ref], 1.0
        else:
            intersection_ms = matrix[(ref, hyp)]
            union_ms = ref_durations[ref] + hyp_durations[hyp] - intersection_ms
            error = round(1.0 - _ratio(intersection_ms, union_ms), 12)
        speaker_jaccard_errors.append(
            {
                "referenceSpeaker": ref,
                "hypothesisSpeaker": hyp,
                "intersectionMs": intersection_ms,
                "unionMs": union_ms,
                "error": error,
            }
        )
    boundary = _boundary_metrics(
        reference, hypothesis, duration_ms=duration_ms, tolerance_ms=boundary_tolerance_ms
    )
    review = _review_metrics(review_queue)
    evaluation_split = case.get("evaluationSplit")
    if not isinstance(evaluation_split, str) or not evaluation_split:
        evaluation_split = "unspecified"
    return {
        "caseId": case.get("id"),
        "evaluationSplit": evaluation_split,
        "language": case.get("language"),
        "scenarios": case.get("scenario", []),
        "durationMs": duration_ms,
        "speakerCount": {
            "reference": len(reference_speakers),
            "hypothesis": len(hypothesis_speakers),
            "absoluteError": abs(len(reference_speakers) - len(hypothesis_speakers)),
            "exact": len(reference_speakers) == len(hypothesis_speakers),
        },
        "mapping": [
            {"referenceSpeaker": ref, "hypothesisSpeaker": hyp, "overlapMs": matrix[(ref, hyp)]}
            for ref, hyp in pairs
        ],
        "diarization": {
            "collarMs": 0,
            "overlapIncluded": True,
            "referenceSpeakerMs": reference_ms,
            "correctSpeakerMs": correct_ms,
            "missedSpeakerMs": missed_ms,
            "falseAlarmSpeakerMs": false_alarm_ms,
            "speakerConfusionMs": confusion_ms,
            "errorSpeakerMs": missed_ms + false_alarm_ms + confusion_ms,
            "der": _ratio(missed_ms + false_alarm_ms + confusion_ms, reference_ms),
            "jer": round(
                sum(item["error"] for item in speaker_jaccard_errors)
                / len(speaker_jaccard_errors),
                12,
            ),
            "speakerJaccardErrors": speaker_jaccard_errors,
        },
        "overlap": {
            "referenceSpeakerMs": overlap_ref_ms,
            "correctSpeakerMs": overlap_correct_ms,
            "missedOrConfusedSpeakerMs": overlap_ref_ms - overlap_correct_ms,
            "speakerRecall": _ratio(overlap_correct_ms, overlap_ref_ms) if overlap_ref_ms else 1.0,
            "referenceRegionMs": overlap_ref_region_ms,
            "hypothesisRegionMs": overlap_hyp_region_ms,
            "intersectionRegionMs": overlap_intersection_region_ms,
            "regionRecall": (
                _ratio(overlap_intersection_region_ms, overlap_ref_region_ms)
                if overlap_ref_region_ms
                else 1.0
            ),
            "regionPrecision": (
                _ratio(overlap_intersection_region_ms, overlap_hyp_region_ms)
                if overlap_hyp_region_ms
                else (1.0 if not overlap_ref_region_ms else 0.0)
            ),
        },
        "boundary": boundary,
        "review": review,
        "_aggregate": {
            "referenceSpeakerMs": reference_ms,
            "missedSpeakerMs": missed_ms,
            "falseAlarmSpeakerMs": false_alarm_ms,
            "speakerConfusionMs": confusion_ms,
            "jaccardErrorSum": sum(item["error"] for item in speaker_jaccard_errors),
            "referenceSpeakerCount": len(reference_speakers),
            "overlapReferenceSpeakerMs": overlap_ref_ms,
            "overlapCorrectSpeakerMs": overlap_correct_ms,
            "boundaryErrorsMs": boundary["referenceNearestErrorsMs"],
        },
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise DiarizationEvaluationError("cannot aggregate zero cases")
    reference_ms = sum(int(row["_aggregate"]["referenceSpeakerMs"]) for row in rows)
    missed_ms = sum(int(row["_aggregate"]["missedSpeakerMs"]) for row in rows)
    false_alarm_ms = sum(int(row["_aggregate"]["falseAlarmSpeakerMs"]) for row in rows)
    confusion_ms = sum(int(row["_aggregate"]["speakerConfusionMs"]) for row in rows)
    speaker_count = sum(int(row["_aggregate"]["referenceSpeakerCount"]) for row in rows)
    jaccard_sum = sum(float(row["_aggregate"]["jaccardErrorSum"]) for row in rows)
    overlap_ref_ms = sum(int(row["_aggregate"]["overlapReferenceSpeakerMs"]) for row in rows)
    overlap_correct_ms = sum(int(row["_aggregate"]["overlapCorrectSpeakerMs"]) for row in rows)
    boundary_errors = [
        int(error)
        for row in rows
        for error in row["_aggregate"]["boundaryErrorsMs"]
    ]
    review_rows = [row["review"] for row in rows if row["review"]["queuePresent"]]
    exact_count = sum(bool(row["speakerCount"]["exact"]) for row in rows)
    error_ms = missed_ms + false_alarm_ms + confusion_ms
    return {
        "caseCount": len(rows),
        "speakerCountExactCases": exact_count,
        "speakerCountExactAccuracy": _ratio(exact_count, len(rows)),
        "meanAbsoluteSpeakerCountError": round(
            sum(int(row["speakerCount"]["absoluteError"]) for row in rows)
            / len(rows),
            12,
        ),
        "diarization": {
            "referenceSpeakerMs": reference_ms,
            "missedSpeakerMs": missed_ms,
            "falseAlarmSpeakerMs": false_alarm_ms,
            "speakerConfusionMs": confusion_ms,
            "errorSpeakerMs": error_ms,
            "der": _ratio(error_ms, reference_ms),
            "jer": round(jaccard_sum / speaker_count, 12),
        },
        "overlap": {
            "referenceSpeakerMs": overlap_ref_ms,
            "correctSpeakerMs": overlap_correct_ms,
            "missedOrConfusedSpeakerMs": overlap_ref_ms - overlap_correct_ms,
            "speakerRecall": _ratio(overlap_correct_ms, overlap_ref_ms) if overlap_ref_ms else 1.0,
        },
        "boundary": {
            "referenceCount": len(boundary_errors),
            "meanAbsoluteErrorMs": (
                round(sum(boundary_errors) / len(boundary_errors), 6)
                if boundary_errors
                else 0.0
            ),
            "p95AbsoluteErrorMs": _percentile_nearest_rank(boundary_errors, 0.95) or 0,
        },
        "review": {
            "queuesPresent": len(review_rows),
            "itemCount": sum(int(row["itemCount"]) for row in review_rows),
            "openCount": sum(int(row["openCount"]) for row in review_rows),
            "decisionCount": sum(int(row["decisionCount"]) for row in review_rows),
        },
    }


def build_report(
    *, manifests: Sequence[Path], case_outputs: Mapping[str, Path], boundary_tolerance_ms: int = 250
) -> dict[str, Any]:
    if not manifests or not case_outputs:
        raise DiarizationEvaluationError("manifest and case outputs are required")
    if not 0 <= boundary_tolerance_ms <= 5_000:
        raise DiarizationEvaluationError("boundary tolerance is invalid")
    case_index: dict[str, Mapping[str, Any]] = {}
    manifest_evidence: list[dict[str, Any]] = []
    for path in manifests:
        resolved = path.resolve(strict=True)
        document = read_json_strict(resolved)
        cases = document.get("cases")
        if not isinstance(cases, list):
            raise DiarizationEvaluationError("manifest.cases is missing")
        for case in cases:
            if not isinstance(case, Mapping) or not isinstance(case.get("id"), str):
                raise DiarizationEvaluationError("manifest case is invalid")
            case_id = str(case["id"])
            if case_id in case_index:
                raise DiarizationEvaluationError(f"duplicate case across manifests: {case_id}")
            case_index[case_id] = case
        manifest_evidence.append({
            "path": str(resolved),
            "fileSha256": sha256_file(resolved),
            "canonicalSha256": canonical_json_sha256(document),
        })
    rows: list[dict[str, Any]] = []
    input_evidence: list[dict[str, Any]] = []
    for case_id in sorted(case_outputs):
        case = case_index.get(case_id)
        if case is None:
            raise DiarizationEvaluationError(f"case is not frozen: {case_id}")
        supplied = case_outputs[case_id].resolve(strict=True)
        transcript_path = (
            supplied / "transcript-document.v2.json"
            if supplied.is_dir()
            else supplied
        ).resolve(strict=True)
        review_path = transcript_path.parent / "review" / "review-queue.json"
        document = read_json_strict(transcript_path)
        queue = read_json_strict(review_path) if review_path.is_file() else None
        row = evaluate_case(
            case,
            document,
            boundary_tolerance_ms=boundary_tolerance_ms,
            review_queue=queue,
        )
        rows.append(row)
        input_evidence.append({
            "caseId": case_id,
            "transcriptPath": str(transcript_path),
            "transcriptFileSha256": sha256_file(transcript_path),
            "transcriptCanonicalSha256": canonical_json_sha256(document),
            "reviewQueuePath": str(review_path.resolve()) if queue is not None else None,
            "reviewQueueFileSha256": sha256_file(review_path) if queue is not None else None,
        })
    public_rows = [
        {key: value for key, value in row.items() if key != "_aggregate"}
        for row in rows
    ]
    by_split = {
        split: _aggregate([row for row in rows if row["evaluationSplit"] == split])
        for split in sorted({row["evaluationSplit"] for row in rows})
    }
    body = {
        "schemaVersion": "1.0.0",
        "artifactType": "production-diarization-quality-report",
        "status": "completed",
        "evaluationPolicy": {
            "speakerMapping": "maximum-overlap-one-to-one-v1",
            "timeResolution": "exact-piecewise-integer-milliseconds",
            "collarMs": 0,
            "overlapIncluded": True,
            "jerDefinition": "DIHARD-reference-speaker-macro-jaccard-v1",
            "boundaryDefinition": "nearest-native-turn-edge-v1",
            "boundaryToleranceMs": boundary_tolerance_ms,
            "automaticPassThresholdsApplied": False,
        },
        "inputs": {"manifests": manifest_evidence, "caseOutputs": input_evidence},
        "caseResults": public_rows,
        "aggregates": {"overall": _aggregate(rows), "bySplit": by_split},
        "limitations": [
            "Metrics describe only the listed frozen cases and do not "
            "generalize to unmeasured recordings.",
            "Boundary precision and recall use nearest native turn edges "
            "and are reported separately from DER/JER.",
            "Review counts quantify queued production review items; they "
            "do not substitute for Codex listening adjudication.",
        ],
    }
    return {**body, "canonicalSha256": canonical_json_sha256(body)}


def _assignment(value: str) -> tuple[str, Path]:
    case_id, separator, raw_path = value.partition("=")
    if not separator or not case_id or not raw_path:
        raise argparse.ArgumentTypeError("expected CASE_ID=PATH")
    return case_id, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument(
        "--case-output",
        action="append",
        required=True,
        type=_assignment,
        metavar="CASE_ID=PATH",
    )
    parser.add_argument("--boundary-tolerance-ms", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case_outputs: dict[str, Path] = {}
    for case_id, path in args.case_output:
        if case_id in case_outputs:
            raise DiarizationEvaluationError(f"case output was repeated: {case_id}")
        case_outputs[case_id] = path
    report = build_report(
        manifests=args.manifest,
        case_outputs=case_outputs,
        boundary_tolerance_ms=args.boundary_tolerance_ms,
    )
    atomic_write_json_no_replace(args.output.resolve(), report)
    print(f"caseCount={len(report['caseResults'])}")
    print(f"der={report['aggregates']['overall']['diarization']['der']}")
    print(f"jer={report['aggregates']['overall']['diarization']['jer']}")
    print(f"canonicalSha256={report['canonicalSha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
