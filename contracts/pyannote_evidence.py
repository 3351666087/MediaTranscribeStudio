"""Shared validation for auditable Pyannote canonical speaker revisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _value(item: Any, attribute: str, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, attribute, None)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def is_verified_pyannote_speaker_revision(
    revision: Any,
    evidence: Mapping[str, Any],
    canonical_speakers: set[str],
) -> bool:
    """Return true only for a complete acoustic track-mapping proof."""

    if (
        _value(revision, "revision_type", "type") != "speaker"
        or _value(revision, "source", "source") != "acoustic"
        or _value(revision, "reason_code", "reasonCode")
        != "PYANNOTE_CANONICAL_TRACK_MAPPING"
    ):
        return False
    before = _value(revision, "before", "before")
    after = _value(revision, "after", "after")
    refs = _value(revision, "evidence_refs", "evidenceRefs")
    if (
        not isinstance(before, str)
        or before not in canonical_speakers
        or not isinstance(after, str)
        or after not in canonical_speakers
        or not isinstance(refs, Sequence)
        or isinstance(refs, (str, bytes, bytearray))
        or not any(
            isinstance(ref, str) and ref.startswith("pyannote-mapping:")
            for ref in refs
        )
    ):
        return False

    proof = evidence.get("pyannoteCanonicalMapping")
    if not isinstance(proof, Mapping):
        return False
    provider = proof.get("provider")
    blockers = proof.get("blockers")
    mapping = proof.get("mapping")
    weights = proof.get("weights")
    margin = proof.get("mappingMargin")
    threshold = proof.get("mappingMarginThreshold")
    optimal_score = proof.get("optimalScore")
    alternative_score = proof.get("alternativeScore")
    total_track_ms = proof.get("totalTrackMs")
    dominance = proof.get("dominance")
    dominance_threshold = proof.get("primaryDominanceThreshold")
    local_durations = proof.get("localDurationsMs")
    dominant = proof.get("dominantLocalSpeaker")
    if (
        not isinstance(provider, Mapping)
        or provider.get("id") != "pyannote-community-1"
        or not isinstance(provider.get("version"), str)
        or not provider["version"].strip()
        or proof.get("method")
        != "global-duration-weighted-acoustic-hungarian-v1"
        or proof.get("accepted") is not True
        or proof.get("applied") is not True
        or blockers != []
        or proof.get("beforeSpeakerId") != before
        or proof.get("afterSpeakerId") != after
        or not isinstance(mapping, Mapping)
        or len(mapping) != len(canonical_speakers)
        or set(mapping.values()) != canonical_speakers
        or not all(
            isinstance(local, str)
            and local.strip()
            and isinstance(canonical, str)
            for local, canonical in mapping.items()
        )
        or not isinstance(weights, Mapping)
        or set(weights) != set(mapping)
        or not isinstance(dominant, str)
        or mapping.get(dominant) != after
        or not isinstance(local_durations, Mapping)
        or set(local_durations) - set(mapping)
        or dominant not in local_durations
    ):
        return False

    normalized_margin = _finite_number(margin)
    normalized_threshold = _finite_number(threshold)
    normalized_optimal = _finite_number(optimal_score)
    normalized_alternative = _finite_number(alternative_score)
    normalized_dominance = _finite_number(dominance)
    normalized_dominance_threshold = _finite_number(dominance_threshold)
    if (
        normalized_margin is None
        or normalized_threshold is None
        or normalized_optimal is None
        or normalized_alternative is None
        or normalized_dominance is None
        or normalized_dominance_threshold is None
        or isinstance(total_track_ms, bool)
        or not isinstance(total_track_ms, int)
        or total_track_ms <= 0
        or normalized_margin < normalized_threshold
        or normalized_threshold < 0.0
        or not 0.0 <= normalized_dominance_threshold <= 1.0
        or not normalized_dominance_threshold <= normalized_dominance <= 1.0
    ):
        return False

    normalized_weights: dict[str, dict[str, float]] = {}
    for local_speaker, raw_row in weights.items():
        if (
            not isinstance(local_speaker, str)
            or not isinstance(raw_row, Mapping)
            or set(raw_row) != canonical_speakers
        ):
            return False
        row: dict[str, float] = {}
        for canonical_speaker, raw_weight in raw_row.items():
            weight = _finite_number(raw_weight)
            if weight is None:
                return False
            row[str(canonical_speaker)] = weight
        normalized_weights[local_speaker] = row
    assigned_score = sum(
        normalized_weights[local][canonical]
        for local, canonical in mapping.items()
    )
    expected_margin = max(
        0.0,
        normalized_optimal - normalized_alternative,
    ) / total_track_ms
    if (
        not math.isclose(assigned_score, normalized_optimal, abs_tol=1e-5)
        or not math.isclose(
            normalized_margin,
            expected_margin,
            abs_tol=1e-7,
        )
    ):
        return False

    normalized_durations: dict[str, int] = {}
    for local_speaker, raw_duration in local_durations.items():
        if (
            not isinstance(local_speaker, str)
            or isinstance(raw_duration, bool)
            or not isinstance(raw_duration, int)
            or raw_duration <= 0
        ):
            return False
        normalized_durations[local_speaker] = raw_duration
    tracked_segment_ms = sum(normalized_durations.values())
    expected_dominant = sorted(
        normalized_durations,
        key=lambda speaker: (-normalized_durations[speaker], speaker),
    )[0]
    if (
        tracked_segment_ms > total_track_ms
        or dominant != expected_dominant
        or not math.isclose(
            normalized_dominance,
            normalized_durations[dominant] / tracked_segment_ms,
            abs_tol=1e-9,
        )
    ):
        return False

    overlap = evidence.get("overlap")
    canonical_turns = (
        overlap.get("canonicalSpeakerTurns")
        if isinstance(overlap, Mapping)
        else None
    )
    if not isinstance(canonical_turns, Sequence) or isinstance(
        canonical_turns,
        (str, bytes, bytearray),
    ):
        return False
    turn_durations: dict[str, int] = {}
    for turn in canonical_turns:
        if not isinstance(turn, Mapping):
            return False
        local_speaker = turn.get("localSpeaker")
        start_ms = turn.get("startMs")
        end_ms = turn.get("endMs")
        if (
            not isinstance(local_speaker, str)
            or local_speaker not in mapping
            or turn.get("speakerId") != mapping[local_speaker]
            or isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or end_ms <= start_ms
        ):
            return False
        turn_durations[local_speaker] = (
            turn_durations.get(local_speaker, 0) + end_ms - start_ms
        )
    return (
        turn_durations == normalized_durations
        and turn_durations.get(dominant, 0) > 0
    )


__all__ = ["is_verified_pyannote_speaker_revision"]
