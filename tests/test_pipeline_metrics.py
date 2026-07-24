from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.pipeline_metrics import ReferenceTurn, evaluate_reference_quality


def _segment(*, overlap_evidence=None, overlapping: bool = False):
    evidence = {}
    if overlap_evidence is not None:
        evidence["overlap"] = {"overlapIntervals": overlap_evidence}
    return SimpleNamespace(
        start_ms=0,
        end_ms=1000,
        speaker_id="speaker-1",
        overlapping=overlapping,
        evidence=evidence,
    )


def _reference():
    return (
        ReferenceTurn(0, 1000, ("reference-1",)),
        ReferenceTurn(400, 600, ("reference-2",)),
    )


def test_reference_quality_uses_exact_overlap_evidence() -> None:
    metrics = evaluate_reference_quality(
        [
            _segment(
                overlap_evidence=[
                    {
                        "startMs": 400,
                        "endMs": 600,
                        "localSpeakers": ["LOCAL_A", "LOCAL_B"],
                    }
                ],
                overlapping=True,
            )
        ],
        _reference(),
    )

    assert metrics["overlapF1"] == pytest.approx(1.0)


def test_reference_quality_retains_legacy_boolean_overlap_fallback() -> None:
    metrics = evaluate_reference_quality(
        [_segment(overlapping=True)],
        _reference(),
    )

    assert metrics["overlapF1"] == pytest.approx(1.0 / 3.0)


def test_reference_quality_ignores_boolean_span_when_exact_evidence_is_empty() -> None:
    metrics = evaluate_reference_quality(
        [_segment(overlap_evidence=[], overlapping=True)],
        _reference(),
    )

    assert metrics["overlapF1"] == pytest.approx(0.0)


def test_reference_quality_scores_canonical_overlapping_speaker_tracks() -> None:
    segment = _segment(
        overlap_evidence=[
            {
                "startMs": 400,
                "endMs": 600,
                "localSpeakers": ["LOCAL_A", "LOCAL_B"],
            }
        ],
        overlapping=True,
    )
    segment.evidence["overlap"]["canonicalSpeakerTurns"] = [
        {
            "startMs": 0,
            "endMs": 1000,
            "speakerId": "speaker-1",
            "localSpeaker": "LOCAL_A",
        },
        {
            "startMs": 400,
            "endMs": 600,
            "speakerId": "speaker-2",
            "localSpeaker": "LOCAL_B",
        },
    ]

    metrics = evaluate_reference_quality([segment], _reference())

    assert metrics == {
        "der": pytest.approx(0.0),
        "jer": pytest.approx(0.0),
        "speakerConfusion": pytest.approx(0.0),
        "overlapF1": pytest.approx(1.0),
    }
