from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sample_library import (
    SampleLibraryError,
    edit_distance,
    load_manifest,
    tokenize_for_score,
    word_error_rate,
)
from tools.evaluate_sample_library import _subtitle_quality, evaluate_case


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "sample_library" / "manifest.v1.json"


def test_manifest_covers_languages_and_scenarios() -> None:
    manifest = load_manifest(SPEC)
    assert manifest.max_duration_seconds <= 30
    assert len(manifest.cases) >= 8
    assert len({case.language for case in manifest.cases}) >= 6
    assert {
        "clean-single",
        "telephony-noise",
        "two-speaker-turns",
        "overlap",
        "video-scene-cuts",
    } <= {case.scenario for case in manifest.cases}


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    raw = json.loads(SPEC.read_text(encoding="utf-8"))
    raw["cases"][0]["output"] = "../outside.wav"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SampleLibraryError, match="relative path"):
        load_manifest(path)


def test_manifest_requires_pinned_remote_hash(tmp_path: Path) -> None:
    raw = json.loads(SPEC.read_text(encoding="utf-8"))
    raw["cases"][-1]["remote"]["sha256"] = "not-a-hash"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SampleLibraryError, match="SHA-256"):
        load_manifest(path)


def test_language_aware_tokenization_and_error_rate() -> None:
    assert tokenize_for_score("Hello, world!") == ["hello", "world"]
    assert tokenize_for_score("字幕质量") == ["字", "幕", "质", "量"]
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1
    assert word_error_rate("字幕质量", "字幕") == 0.5


def test_subtitle_quality_accepts_standard_webvtt_timestamps(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.vtt").write_text(
        "WEBVTT\n\n"
        "1\n"
        "00:00:00.000 --> 00:00:01.470\n"
        "Hello\n",
        encoding="utf-8",
    )
    quality = _subtitle_quality(tmp_path)
    assert quality["vtt"] == {"files": 1, "valid": 1}


def test_unknown_speaker_truth_does_not_report_false_count_result(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "outputs" / "unknown"
    artifact_root.mkdir(parents=True)
    transcript = artifact_root / "transcript-document.v2.json"
    transcript.write_text(
        json.dumps(
            {
                "speakerPolicy": {"resolvedCount": 2},
                "segments": [
                    {
                        "startMs": 0,
                        "endMs": 1000,
                        "speakerId": "speaker-1",
                        "displayText": "hello",
                    },
                    {
                        "startMs": 1000,
                        "endMs": 2000,
                        "speakerId": "speaker-2",
                        "displayText": "world",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "unknown-result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "observed",
                "terminal_event": {
                    "payload": {
                        "artifactPaths": [str(transcript)],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_case(
        case={
            "id": "unknown",
            "expectedSpeakerCount": None,
            "truthEligibility": {"speakerCount": False, "asr": False},
        },
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="unknown",
    )

    assert report["evidence"]["resolvedSpeakerCount"] == 2
    assert report["evidence"]["speakerCountAbsoluteError"] is None
    assert report["evidence"]["speakerCountMatch"] is None
    assert report["evidence"]["distinctSpeakerCountMatch"] is None


def test_language_quality_scores_only_automatic_detection(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "outputs" / "language"
    artifact_root.mkdir(parents=True)
    transcript = artifact_root / "transcript-document.v2.json"
    transcript.write_text(
        json.dumps(
            {
                "language": "mul",
                "speakerPolicy": {"resolvedCount": 1},
                "segments": [
                    {
                        "startMs": 0,
                        "endMs": 1000,
                        "speakerId": "speaker-1",
                        "language": "fr",
                        "displayText": "bonjour",
                        "evidence": {
                            "asr": {
                                "language": "stale",
                                "requestedLanguage": "auto",
                            }
                        },
                    },
                    {
                        "startMs": 1000,
                        "endMs": 2000,
                        "speakerId": "speaker-1",
                        "language": "und",
                        "displayText": "encore",
                        "evidence": {
                            "asr": {
                                "language": "und",
                                "requestedLanguage": "auto",
                            }
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "language-result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "observed",
                "terminal_event": {
                    "payload": {"artifactPaths": [str(transcript)]}
                },
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_case(
        case={
            "id": "language",
            "language": "fr-FR",
            "expectedSpeakerCount": 1,
        },
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="language",
    )

    assert report["languageQuality"] == {
        "expectedLanguage": "fr-FR",
        "expectedLanguageRoot": "fr",
        "documentLanguage": "mul",
        "documentLanguageRoot": "mul",
        "documentLanguageMatch": False,
        "requestedLanguages": ["auto"],
        "detectedLanguageCounts": {"fr": 1, "und": 1},
        "segmentCount": 2,
        "scoredSegmentCount": 2,
        "correctSegmentCount": 1,
        "segmentAccuracy": 0.5,
        "undeterminedRate": 0.5,
        "automaticDetectionEligible": True,
    }


def test_reference_language_prompt_is_not_scored_as_detection(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "outputs" / "language"
    artifact_root.mkdir(parents=True)
    transcript = artifact_root / "transcript-document.v2.json"
    transcript.write_text(
        json.dumps(
            {
                "language": "fr-FR",
                "speakerPolicy": {"resolvedCount": 1},
                "segments": [
                    {
                        "startMs": 0,
                        "endMs": 1000,
                        "speakerId": "speaker-1",
                        "displayText": "bonjour",
                        "evidence": {
                            "asr": {
                                "language": "fr-FR",
                                "requestedLanguage": "fr-FR",
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "language-result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "observed",
                "terminal_event": {
                    "payload": {"artifactPaths": [str(transcript)]}
                },
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_case(
        case={"id": "language", "language": "fr-FR"},
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="language",
    )

    assert report["languageQuality"]["automaticDetectionEligible"] is False
    assert report["languageQuality"]["segmentAccuracy"] is None
