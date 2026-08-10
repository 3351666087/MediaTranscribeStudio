from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.evaluate_production_diarization import (
    DiarizationEvaluationError,
    build_report,
    evaluate_case,
)


MEDIA_SHA = "a" * 64


def _case(
    *,
    turns: list[tuple[float, float, str]],
    duration: float = 4.0,
    case_id: str = "case-1",
    split: str = "development",
) -> dict[str, object]:
    speakers = sorted({speaker for _, _, speaker in turns})
    return {
        "id": case_id,
        "evaluationSplit": split,
        "language": "en",
        "scenario": ["overlap"],
        "sha256": MEDIA_SHA,
        "audio": {"durationSeconds": duration},
        "expectedSpeakerCount": len(speakers),
        "turns": [
            {
                "startSeconds": start,
                "endSeconds": end,
                "speakerId": speaker,
            }
            for start, end, speaker in turns
        ],
    }


def _document(
    *,
    turns: list[tuple[int, int, str]],
    duration_ms: int = 4_000,
    source_sha: str = MEDIA_SHA,
) -> dict[str, object]:
    speakers = sorted({speaker for _, _, speaker in turns})
    return {
        "schemaVersion": "2.0.0",
        "source": {"sha256": source_sha, "durationMs": duration_ms},
        "speakerTimeline": {
            "mapping": {
                "accepted": True,
                "localToCanonical": {
                    f"local-{index}": speaker
                    for index, speaker in enumerate(speakers)
                },
            },
            "regular": {
                "native": True,
                "semantics": "overlap-preserving",
                "turns": [
                    {
                        "startMs": start,
                        "endMs": end,
                        "speakerId": speaker,
                    }
                    for start, end, speaker in turns
                ],
            },
        },
    }


def test_perfect_overlap_timeline_has_zero_der_jer_and_count_error() -> None:
    case = _case(
        turns=[(0, 3, "ref-a"), (1, 4, "ref-b")],
    )
    document = _document(
        turns=[(0, 3_000, "hyp-x"), (1_000, 4_000, "hyp-y")]
    )

    result = evaluate_case(case, document)

    assert result["speakerCount"] == {
        "reference": 2,
        "hypothesis": 2,
        "absoluteError": 0,
        "exact": True,
    }
    assert result["diarization"]["der"] == 0
    assert result["diarization"]["jer"] == 0
    assert result["overlap"]["speakerRecall"] == 1
    assert result["boundary"]["meanAbsoluteErrorMs"] == 0


def test_overlap_missed_speaker_and_confusion_are_reported_separately() -> None:
    case = _case(
        turns=[(0, 4, "ref-a"), (1, 3, "ref-b")],
    )
    document = _document(turns=[(0, 4_000, "hyp-x")])

    result = evaluate_case(case, document)

    assert result["speakerCount"]["absoluteError"] == 1
    assert result["diarization"]["referenceSpeakerMs"] == 6_000
    assert result["diarization"]["missedSpeakerMs"] == 2_000
    assert result["diarization"]["speakerConfusionMs"] == 0
    assert result["diarization"]["der"] == pytest.approx(1 / 3)
    assert result["diarization"]["jer"] == 0.5
    assert result["overlap"]["referenceSpeakerMs"] == 4_000
    assert result["overlap"]["missedOrConfusedSpeakerMs"] == 2_000
    assert result["overlap"]["speakerRecall"] == 0.5


def test_wrong_equal_count_labels_are_speaker_confusion() -> None:
    case = _case(
        turns=[(0, 2, "ref-a"), (2, 4, "ref-b")],
    )
    document = _document(
        turns=[(0, 1_000, "hyp-x"), (1_000, 4_000, "hyp-y")]
    )

    result = evaluate_case(case, document)

    assert result["diarization"]["missedSpeakerMs"] == 0
    assert result["diarization"]["falseAlarmSpeakerMs"] == 0
    assert result["diarization"]["speakerConfusionMs"] == 1_000
    assert result["diarization"]["der"] == 0.25
    assert result["boundary"]["referenceNearestErrorsMs"] == [1_000]
    assert result["boundary"]["spuriousHypothesisCount"] == 1


def test_oversegmentation_adds_false_alarm_and_jaccard_union_error() -> None:
    case = _case(turns=[(0, 4, "ref-a")])
    document = _document(
        turns=[(0, 4_000, "hyp-x"), (2_000, 4_000, "hyp-y")]
    )

    result = evaluate_case(case, document)

    assert result["diarization"]["falseAlarmSpeakerMs"] == 2_000
    assert result["diarization"]["der"] == 0.5
    # DIHARD JER is reference-speaker macro averaged; an unmapped extra
    # system speaker is reflected by DER but not by the mapped Jaccard pair.
    assert result["diarization"]["jer"] == 0


def test_review_queue_counts_and_reasons_are_preserved() -> None:
    case = _case(turns=[(0, 4, "ref-a")])
    document = _document(turns=[(0, 4_000, "hyp-x")])
    queue = {
        "items": [
            {"reasonCode": "LOW_MARGIN"},
            {"reasonCode": "LOW_MARGIN"},
            {"reasonCode": "OVERLAP"},
        ],
        "decisions": [{"decisionId": "one"}],
        "openCount": 2,
    }

    result = evaluate_case(case, document, review_queue=queue)

    assert result["review"] == {
        "queuePresent": True,
        "itemCount": 3,
        "openCount": 2,
        "decisionCount": 1,
        "reasonCounts": {"LOW_MARGIN": 2, "OVERLAP": 1},
    }


def test_review_queue_open_count_is_strict() -> None:
    case = _case(turns=[(0, 4, "ref-a")])
    document = _document(turns=[(0, 4_000, "hyp-x")])

    with pytest.raises(DiarizationEvaluationError, match="openCount"):
        evaluate_case(
            case,
            document,
            review_queue={"items": [], "decisions": [], "openCount": 1},
        )


def test_source_hash_and_authoritative_timeline_fail_closed() -> None:
    case = _case(turns=[(0, 4, "ref-a")])
    with pytest.raises(DiarizationEvaluationError, match="SHA-256"):
        evaluate_case(
            case,
            _document(turns=[(0, 4_000, "hyp-x")], source_sha="b" * 64),
        )

    document = _document(turns=[(0, 4_000, "hyp-x")])
    document["speakerTimeline"]["mapping"]["accepted"] = False
    with pytest.raises(DiarizationEvaluationError, match="not accepted"):
        evaluate_case(case, document)


def test_reference_speaker_count_is_bounded() -> None:
    turns = [(0, 4, f"ref-{index}") for index in range(17)]
    case = _case(turns=turns)
    document = _document(turns=[(0, 4_000, "hyp-x")])

    with pytest.raises(DiarizationEvaluationError, match="expectedSpeakerCount"):
        evaluate_case(case, document)


def test_build_report_is_hash_bound_and_aggregates_splits(tmp_path: Path) -> None:
    development = _case(
        turns=[(0, 4, "ref-a")],
        case_id="dev",
        split="development",
    )
    held_out = _case(
        turns=[(0, 2, "ref-a"), (2, 4, "ref-b")],
        case_id="held",
        split="held-out",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"cases": [development, held_out]}), encoding="utf-8"
    )
    outputs: dict[str, Path] = {}
    for case_id, turns in {
        "dev": [(0, 4_000, "hyp-x")],
        "held": [(0, 2_000, "hyp-x"), (2_000, 4_000, "hyp-y")],
    }.items():
        output = tmp_path / case_id
        output.mkdir()
        (output / "transcript-document.v2.json").write_text(
            json.dumps(_document(turns=turns)), encoding="utf-8"
        )
        outputs[case_id] = output

    report = build_report(manifests=[manifest], case_outputs=outputs)

    assert report["aggregates"]["overall"]["caseCount"] == 2
    assert set(report["aggregates"]["bySplit"]) == {
        "development",
        "held-out",
    }
    assert report["inputs"]["manifests"][0]["fileSha256"] == sha256_file(
        manifest
    )
    body = {key: value for key, value in report.items() if key != "canonicalSha256"}
    assert report["canonicalSha256"] == canonical_json_sha256(body)


def test_missing_legacy_split_is_reported_as_unspecified(tmp_path: Path) -> None:
    case = _case(turns=[(0, 4, "ref-a")])
    case.pop("evaluationSplit")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    output = tmp_path / "case-1"
    output.mkdir()
    (output / "transcript-document.v2.json").write_text(
        json.dumps(_document(turns=[(0, 4_000, "hyp-x")])),
        encoding="utf-8",
    )

    report = build_report(
        manifests=[manifest],
        case_outputs={"case-1": output},
    )

    assert report["caseResults"][0]["evaluationSplit"] == "unspecified"
    assert report["aggregates"]["bySplit"]["unspecified"]["caseCount"] == 1
