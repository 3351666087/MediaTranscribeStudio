from __future__ import annotations

import copy

import pytest

from backend.documents import validate_segments
from backend.errors import WorkerError
from backend.models import Revision, SpeakerScore, TranscriptSegment
from contracts.pyannote_evidence import is_verified_pyannote_speaker_revision


def mapping_evidence() -> dict[str, object]:
    return {
        "overlap": {
            "canonicalSpeakerTurns": [
                {
                    "startMs": 0,
                    "endMs": 1_000,
                    "speakerId": "speaker-2",
                    "localSpeaker": "LOCAL_B",
                }
            ]
        },
        "pyannoteCanonicalMapping": {
            "provider": {
                "id": "pyannote-community-1",
                "version": "2.0.0",
            },
            "method": "global-duration-weighted-acoustic-hungarian-v1",
            "mapping": {
                "LOCAL_A": "speaker-1",
                "LOCAL_B": "speaker-2",
            },
            "weights": {
                "LOCAL_A": {
                    "speaker-1": 900.0,
                    "speaker-2": 100.0,
                },
                "LOCAL_B": {
                    "speaker-1": 100.0,
                    "speaker-2": 900.0,
                },
            },
            "optimalScore": 1_800.0,
            "alternativeScore": 200.0,
            "totalTrackMs": 2_000,
            "mappingMargin": 0.8,
            "mappingMarginThreshold": 0.2,
            "primaryDominanceThreshold": 0.6,
            "accepted": True,
            "localDurationsMs": {"LOCAL_B": 1_000},
            "dominantLocalSpeaker": "LOCAL_B",
            "dominance": 1.0,
            "beforeSpeakerId": "speaker-1",
            "afterSpeakerId": "speaker-2",
            "blockers": [],
            "applied": True,
            "reviewStatus": "RESOLVED",
        },
    }


def mapping_revision() -> Revision:
    return Revision(
        revision_id="segment-001:speaker:1",
        revision_type="speaker",
        source="acoustic",
        before="speaker-1",
        after="speaker-2",
        reason_code="PYANNOTE_CANONICAL_TRACK_MAPPING",
        confidence=0.8,
        evidence_refs=("pyannote-mapping:segment-001",),
    )


def transcript_segments(
    evidence: dict[str, object] | None = None,
) -> tuple[TranscriptSegment, ...]:
    return (
        TranscriptSegment(
            segment_id="segment-001",
            start_ms=0,
            end_ms=1_000,
            speaker_id="speaker-2",
            raw_text="one",
            normalized_text="one",
            display_text="one",
            confidence=0.9,
            speaker_scores=(
                SpeakerScore("speaker-1", 0.9),
                SpeakerScore("speaker-2", 0.1),
            ),
            speaker_margin=0.8,
            overlapping=True,
            revisions=(mapping_revision(),),
            evidence=evidence or mapping_evidence(),
        ),
        TranscriptSegment(
            segment_id="segment-002",
            start_ms=1_000,
            end_ms=2_000,
            speaker_id="speaker-1",
            raw_text="two",
            normalized_text="two",
            display_text="two",
            confidence=0.9,
            speaker_scores=(
                SpeakerScore("speaker-1", 0.9),
                SpeakerScore("speaker-2", 0.1),
            ),
            speaker_margin=0.8,
        ),
    )


def test_complete_pyannote_mapping_proof_allows_audited_override() -> None:
    evidence = mapping_evidence()
    assert is_verified_pyannote_speaker_revision(
        mapping_revision(),
        evidence,
        {"speaker-1", "speaker-2"},
    )

    validate_segments(
        transcript_segments(evidence),
        speaker_count=2,
        duration_ms=2_000,
        high_margin_threshold=0.4,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("blockers", ["PYANNOTE_MAPPING_MARGIN_BELOW_THRESHOLD"]),
        ("mappingMargin", 0.1),
        ("totalTrackMs", 1_000),
        ("dominance", 0.5),
        ("weights", {"LOCAL_A": {}, "LOCAL_B": {}}),
    ),
)
def test_incomplete_or_inconsistent_mapping_proof_fails_closed(
    field: str,
    value: object,
) -> None:
    evidence = copy.deepcopy(mapping_evidence())
    proof = evidence["pyannoteCanonicalMapping"]
    assert isinstance(proof, dict)
    proof[field] = value

    assert not is_verified_pyannote_speaker_revision(
        mapping_revision(),
        evidence,
        {"speaker-1", "speaker-2"},
    )
    with pytest.raises(WorkerError) as caught:
        validate_segments(
            transcript_segments(evidence),
            speaker_count=2,
            duration_ms=2_000,
            high_margin_threshold=0.4,
        )
    assert caught.value.code == "OVERLAP_OVERRIDE_FORBIDDEN"


def test_human_lock_cannot_be_bypassed_by_valid_mapping_proof() -> None:
    segments = transcript_segments()
    locked = TranscriptSegment(
        **{
            **segments[0].__dict__,
            "human_locked": True,
        }
    )
    with pytest.raises(WorkerError) as caught:
        validate_segments(
            (locked, segments[1]),
            speaker_count=2,
            duration_ms=2_000,
            high_margin_threshold=0.4,
        )
    assert caught.value.code == "HUMAN_LOCK_OVERRIDE_FORBIDDEN"
