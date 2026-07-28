import json
from pathlib import Path

import pytest

from tools.build_code_switch_sample_library import (
    RESOLVED_SCHEMA_VERSION,
    _load_cached_dataset_evidence,
    _load_cached_source_row,
    load_manifest,
    parse_liva_turns,
    parse_liva_turns_with_issues,
    parse_tagged_transcript,
    select_switch_window,
)
from tools.evaluate_sample_library import _code_switch_language_quality
from tools.global_sample_library import GlobalSampleLibraryError


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sample_library" / "code-switch-manifest.v1.json"


def test_checked_in_manifest_covers_real_multispeaker_and_timed_switches() -> None:
    assert RESOLVED_SCHEMA_VERSION == "1.1.0"
    manifest = load_manifest(MANIFEST)
    cases = manifest["cases"]
    assert len(cases) == 19
    assert {case["acquisition"]["kind"] for case in cases} == {
        "whole-utterance",
        "speaker-window",
        "switch-window",
    }
    assert {case["evaluationSplit"] for case in cases} == {
        "development",
        "regression",
        "held-out",
    }
    assert {case.get("targetSpeakerCount") for case in cases} >= {
        None,
        2,
        3,
        4,
        5,
    }
    assert {
        language
        for case in cases
        for language in case["expectedLanguages"]
    } >= {
        "zh",
        "en",
        "ko",
        "ja",
        "sw",
        "pcm",
        "yo",
        "tl",
        "es",
        "pt",
        "it",
        "de",
        "fr",
    }
    assert {
        case["maxDurationSeconds"]
        for case in cases
        if case["acquisition"]["kind"] == "switch-window"
    } == {40}


def test_liva_parser_preserves_overlap_and_speaker_identity() -> None:
    turns = parse_liva_turns(
        "Speaker 2 [00:00:00.000 - 00:00:04.930]: First turn.\n\n"
        "Speaker 3 [00:00:03.500 - 00:00:05.100]: Overlap turn.\n"
    )
    assert turns == [
        {
            "speakerId": "source-speaker-2",
            "startSeconds": 0.0,
            "endSeconds": 4.93,
            "transcript": "First turn.",
        },
        {
            "speakerId": "source-speaker-3",
            "startSeconds": 3.5,
            "endSeconds": 5.1,
            "transcript": "Overlap turn.",
        },
    ]


def test_liva_parser_audits_zero_duration_source_annotations() -> None:
    turns, issues = parse_liva_turns_with_issues(
        "Speaker 1 [00:00:01 - 00:00:01]: Brief marker.\n\n"
        "Speaker 2 [00:00:02 - 00:00:03]: Timed speech.\n"
    )
    assert turns == [
        {
            "speakerId": "source-speaker-2",
            "startSeconds": 2.0,
            "endSeconds": 3.0,
            "transcript": "Timed speech.",
        }
    ]
    assert issues == [
        {
            "reasonCode": "ZERO_DURATION_SOURCE_ANNOTATION",
            "speakerId": "source-speaker-1",
            "sourceTimestampSeconds": 1.0,
            "transcript": "Brief marker.",
        }
    ]


def test_tagged_window_uses_full_chunks_around_requested_switch() -> None:
    chunks = parse_tagged_transcript(
        "<en><start:0.00>first<end:10.00>"
        "<en><start:10.00>second<end:20.00>"
        "<es><start:20.00>tercero<end:35.00>"
        "<es><start:35.00>cuarto<end:50.00>"
        "<en><start:50.00>fifth<end:60.00>"
    )
    window = select_switch_window(chunks, switch_index=0, maximum_seconds=45)
    assert window["sourceStartSeconds"] == 0.0
    assert window["sourceEndSeconds"] == 35.0
    assert window["switchPointsSeconds"] == [20.0]
    assert [chunk["transcript"] for chunk in window["chunks"]] == [
        "first",
        "second",
        "tercero",
    ]


def test_tagged_parser_rejects_gap_that_would_fake_timing_truth() -> None:
    with pytest.raises(GlobalSampleLibraryError, match="not contiguous"):
        parse_tagged_transcript(
            "<en><start:0.00>first<end:10.00>"
            "<es><start:11.00>segundo<end:20.00>"
        )


def test_case_duration_cap_cannot_exceed_manifest_limit(
    tmp_path: Path,
) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["cases"][0]["maxDurationSeconds"] = (
        value["maxDurationSeconds"] + 1
    )
    manifest = tmp_path / "invalid-code-switch-manifest.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        GlobalSampleLibraryError,
        match="maxDurationSeconds must be between",
    ):
        load_manifest(manifest)


def test_cached_source_row_is_bound_to_dataset_revision_and_locator(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(MANIFEST)
    case = manifest["cases"][0]
    source = next(
        item for item in manifest["sources"] if item["id"] == case["sourceId"]
    )
    metadata = tmp_path / "source-metadata" / f"{case['id']}.json"
    metadata.parent.mkdir(parents=True)
    payload = {
        "dataset": source["dataset"],
        "revision": source["revision"],
        "config": case["acquisition"]["config"],
        "split": case["acquisition"]["split"],
        "rowIndex": case["acquisition"]["rowIndex"],
        "row": {
            "transcription": "cached text",
            "audioAsset": "https://example.invalid/pinned.wav",
        },
    }
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    cached = _load_cached_source_row(
        tmp_path,
        case["id"],
        source=source,
        case=case,
    )
    assert cached is not None
    row, path = cached
    assert path == metadata
    assert row["transcription"] == "cached text"
    assert row["audio"] == [{"src": "https://example.invalid/pinned.wav"}]

    payload["revision"] = "0" * 40
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        GlobalSampleLibraryError,
        match="does not match the pinned row",
    ):
        _load_cached_source_row(
            tmp_path,
            case["id"],
            source=source,
            case=case,
        )


def test_cached_dataset_evidence_is_bound_to_pinned_source(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(MANIFEST)
    source = manifest["sources"][0]
    evidence = tmp_path / "source-evidence" / f"{source['id']}.json"
    evidence.parent.mkdir(parents=True)
    payload = {
        "schemaVersion": "1.0.0",
        "verifiedAt": "2026-07-24T12:00:00+00:00",
        "source": {
            **source,
            "lastModified": "2026-07-24T11:00:00.000Z",
        },
    }
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    cached = _load_cached_dataset_evidence(tmp_path, source=source)
    assert cached is not None
    value, path = cached
    assert path == evidence
    assert value["revision"] == source["revision"]

    payload["source"]["license"] = "apache-2.0"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        GlobalSampleLibraryError,
        match="does not match the pinned source",
    ):
        _load_cached_dataset_evidence(tmp_path, source=source)


def test_code_switch_quality_separates_document_and_timing_evidence() -> None:
    case = {
        "expectedLanguages": ["en", "es"],
        "languageTruth": {
            "qualification": "synthetic-concatenation-exact-chunk-timestamps",
            "expectedLanguages": ["en", "es"],
            "timeScoringEligible": True,
            "intervals": [
                {"language": "en", "startSeconds": 0.0, "endSeconds": 10.0},
                {"language": "es", "startSeconds": 10.0, "endSeconds": 20.0},
            ],
            "switchPointsSeconds": [10.0],
        },
    }
    segments = [
        {
            "startMs": 0,
            "endMs": 11000,
            "language": "en",
            "evidence": {
                "asr": {
                    "requestedLanguage": "auto",
                    "languageCandidates": ["en"],
                }
            },
        },
        {
            "startMs": 11000,
            "endMs": 20000,
            "language": "es",
            "evidence": {
                "asr": {
                    "requestedLanguage": "auto",
                    "languageCandidates": ["es"],
                }
            },
        },
    ]
    quality = _code_switch_language_quality(
        case=case,
        transcript={"language": "mul"},
        segments=segments,
    )
    assert quality is not None
    assert quality["expectedLanguageSetExact"] is True
    assert quality["documentMarkedMultilingual"] is True
    assert quality["durationWeightedAccuracy"] == pytest.approx(0.95)
    assert quality["predictedSwitchPointsMs"] == [11000]
    assert quality["switchPointAbsoluteErrorsMs"] == [1000]


def test_document_only_truth_never_produces_switch_timing_score() -> None:
    quality = _code_switch_language_quality(
        case={
            "languageTruth": {
                "qualification": "document-language-pair-without-time-alignment",
                "expectedLanguages": ["ko", "ja"],
                "timeScoringEligible": False,
                "expectedSwitchCount": 2,
            }
        },
        transcript={"language": "ko"},
        segments=[
            {
                "startMs": 0,
                "endMs": 5000,
                "language": "ko",
                "evidence": {"asr": {"languageCandidates": ["ko"]}},
            }
        ],
    )
    assert quality is not None
    assert quality["expectedLanguageRecall"] == 0.5
    assert quality["timeScoringEligible"] is False
    assert quality["durationWeightedAccuracy"] is None
    assert quality["referenceSwitchPointsMs"] == []
