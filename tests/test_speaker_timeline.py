from __future__ import annotations

import copy

import pytest

from backend.errors import WorkerError
from backend.speaker_timeline import (
    build_speaker_timeline,
    validate_speaker_timeline,
)


def _timeline():
    return build_speaker_timeline(
        provider_version="4.0.4",
        local_to_canonical={
            "LOCAL_A": "speaker-1",
            "LOCAL_B": "speaker-2",
        },
        mapping_margin=0.42,
        regular_turns=[
            {
                "startMs": 0,
                "endMs": 1_000,
                "speakerId": "speaker-1",
                "localSpeaker": "LOCAL_A",
            },
            {
                "startMs": 400,
                "endMs": 800,
                "speakerId": "speaker-2",
                "localSpeaker": "LOCAL_B",
            },
            {
                "startMs": 800,
                "endMs": 1_500,
                "speakerId": "speaker-2",
                "localSpeaker": "LOCAL_B",
            },
        ],
        exclusive_turns=[
            {
                "startMs": 0,
                "endMs": 600,
                "speakerId": "speaker-1",
                "localSpeaker": "LOCAL_A",
            },
            {
                "startMs": 600,
                "endMs": 1_500,
                "speakerId": "speaker-2",
                "localSpeaker": "LOCAL_B",
            },
        ],
        duration_ms=2_000,
        canonical_speaker_ids=("speaker-1", "speaker-2"),
    )


def test_authoritative_timeline_preserves_overlap_and_exclusive_ownership() -> None:
    timeline = _timeline()

    assert timeline["regular"]["turns"] == [
        {
            "startMs": 0,
            "endMs": 1_000,
            "speakerId": "speaker-1",
            "localSpeaker": "LOCAL_A",
        },
        {
            "startMs": 400,
            "endMs": 1_500,
            "speakerId": "speaker-2",
            "localSpeaker": "LOCAL_B",
        },
    ]
    assert timeline["exclusive"]["turns"] == [
        {
            "startMs": 0,
            "endMs": 600,
            "speakerId": "speaker-1",
            "localSpeaker": "LOCAL_A",
        },
        {
            "startMs": 600,
            "endMs": 1_500,
            "speakerId": "speaker-2",
            "localSpeaker": "LOCAL_B",
        },
    ]
    assert timeline["textAlignment"] == {
        "status": "segment-level-exclusive-dominance",
        "wordTimestampsAvailable": False,
        "sourceTextMutable": False,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "hash",
        "out-of-range",
        "exclusive-overlap",
        "mapping-cardinality",
    ),
)
def test_authoritative_timeline_fails_closed_on_invalid_contract(
    mutation: str,
) -> None:
    timeline = copy.deepcopy(_timeline())
    if mutation == "hash":
        timeline["regular"]["turns"][0]["endMs"] = 999
    elif mutation == "out-of-range":
        timeline["exclusive"]["turns"][-1]["endMs"] = 2_001
    elif mutation == "exclusive-overlap":
        timeline["exclusive"]["turns"][1]["startMs"] = 500
    else:
        timeline["mapping"]["localToCanonical"].pop("LOCAL_B")

    with pytest.raises(WorkerError) as raised:
        validate_speaker_timeline(
            timeline,
            duration_ms=2_000,
            canonical_speaker_ids=("speaker-1", "speaker-2"),
        )

    assert raised.value.code == "SPEAKER_TIMELINE_INVALID"


def test_builder_rejects_exclusive_overlap_instead_of_choosing_a_speaker() -> None:
    with pytest.raises(WorkerError) as raised:
        build_speaker_timeline(
            provider_version="4.0.4",
            local_to_canonical={
                "LOCAL_A": "speaker-1",
                "LOCAL_B": "speaker-2",
            },
            mapping_margin=0.42,
            regular_turns=[
                {
                    "startMs": 0,
                    "endMs": 900,
                    "speakerId": "speaker-1",
                    "localSpeaker": "LOCAL_A",
                },
                {
                    "startMs": 500,
                    "endMs": 1_000,
                    "speakerId": "speaker-2",
                    "localSpeaker": "LOCAL_B",
                },
            ],
            exclusive_turns=[
                {
                    "startMs": 0,
                    "endMs": 700,
                    "speakerId": "speaker-1",
                    "localSpeaker": "LOCAL_A",
                },
                {
                    "startMs": 600,
                    "endMs": 1_000,
                    "speakerId": "speaker-2",
                    "localSpeaker": "LOCAL_B",
                },
            ],
            duration_ms=1_000,
            canonical_speaker_ids=("speaker-1", "speaker-2"),
        )

    assert raised.value.code == "SPEAKER_TIMELINE_INVALID"
