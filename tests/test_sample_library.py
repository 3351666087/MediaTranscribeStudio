from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend import (
    MappingLocalLLMProvider,
    SemanticProcessingRunner,
    build_final_adjudicated_transcript,
    build_final_no_speech_adjudication,
)
from backend.voice_activity import build_voice_activity
from tools.sample_library import (
    SampleLibraryError,
    edit_distance,
    load_manifest,
    tokenize_for_score,
    word_error_rate,
)
from tools.evaluate_sample_library import (
    _bucket_summary,
    _joint_metric_labels,
    _joint_transcription_quality,
    _scoring_unit,
    _subtitle_quality,
    _value_counts,
    evaluate_case,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "sample_library" / "manifest.v1.json"


def _write_adjudicated_fixture(
    tmp_path: Path,
    *,
    tamper_final_text: bool = False,
    include_final: bool = True,
    open_review: bool = False,
    single_speaker_multilingual: bool = False,
) -> tuple[Path, Path]:
    output = tmp_path / "outputs" / "adjudicated"
    (output / "review").mkdir(parents=True)
    (output / "semantic").mkdir()
    speaker_ids = (
        ["speaker-1"]
        if single_speaker_multilingual
        else ["speaker-1", "speaker-2"]
    )
    second_speaker = (
        "speaker-1" if single_speaker_multilingual else "speaker-2"
    )
    document = {
        "schemaVersion": "2.0.0",
        "documentId": "doc-adjudicated",
        "jobId": "adjudicated",
        "generatedAt": "2026-07-28T00:00:00Z",
        "language": "mul",
        "source": {
            "fileName": "fixture.wav",
            "sha256": "a" * 64,
            "durationMs": 2_000,
        },
        "speakerPolicy": {
            "mode": "manual",
            "resolvedCount": len(speaker_ids),
            "speakerIds": speaker_ids,
        },
        "speakers": [{"id": speaker_id} for speaker_id in speaker_ids],
        "segments": [
            {
                "id": "segment-1",
                "startMs": 0,
                "endMs": 1_000,
                "speakerId": "speaker-1",
                "rawText": "wrong one",
                "normalizedText": "alpha",
                "displayText": "alpha",
                "confidence": 0.9,
                "speakerScores": [
                    {"speakerId": "speaker-1", "score": 0.9},
                    *(
                        []
                        if single_speaker_multilingual
                        else [{"speakerId": "speaker-2", "score": 0.1}]
                    ),
                ],
                "speakerMargin": 0.8,
                "overlapping": False,
                "humanLocked": False,
                "revisions": [],
                "language": "en",
                "evidence": {"asr": {"provider": "fixture-asr"}},
            },
            {
                "id": "segment-2",
                "startMs": 1_000,
                "endMs": 2_000,
                "speakerId": second_speaker,
                "rawText": "wrong two",
                "normalizedText": "beta",
                "displayText": "beta",
                "confidence": 0.9,
                "speakerScores": [
                    {
                        "speakerId": "speaker-1",
                        "score": (
                            0.9 if single_speaker_multilingual else 0.1
                        ),
                    },
                    *(
                        []
                        if single_speaker_multilingual
                        else [{"speakerId": "speaker-2", "score": 0.9}]
                    ),
                ],
                "speakerMargin": 0.8,
                "overlapping": False,
                "humanLocked": False,
                "revisions": [],
                "language": "es",
                "evidence": {"asr": {"provider": "fixture-asr"}},
            },
        ],
        "provenance": {"offline": True, "models": []},
    }
    semantic = SemanticProcessingRunner(
        provider=MappingLocalLLMProvider(
            [
                {
                    "results": [
                        {
                            "segmentId": "segment-1",
                            "decision": "abstain",
                            "confidence": 0.9,
                        },
                        {
                            "segmentId": "segment-2",
                            "decision": "abstain",
                            "confidence": 0.9,
                        },
                    ]
                }
            ]
        ),
        model="qwen3.5:9b",
    ).run(document)
    review = {
        "schemaVersion": "2.0.0",
        "jobId": "adjudicated",
        "openCount": 1 if open_review else 0,
        "items": (
            [{"id": "review-1", "status": "open"}] if open_review else []
        ),
        "decisions": [],
    }
    transcript_path = output / "transcript-document.v2.json"
    transcript_path.write_text(json.dumps(document), encoding="utf-8")
    (output / "review" / "review-queue.json").write_text(
        json.dumps(review),
        encoding="utf-8",
    )
    (output / "semantic" / "semantic-suggestions.v1.json").write_text(
        json.dumps(semantic),
        encoding="utf-8",
    )
    artifact_paths = [str(transcript_path)]
    if include_final:
        final = build_final_adjudicated_transcript(
            document,
            review,
            semantic,
        )
        if tamper_final_text:
            final["segments"][0]["finalText"] = "tampered"
        final_path = output / "final-adjudicated-transcript.v1.json"
        final_path.write_text(json.dumps(final), encoding="utf-8")
        artifact_paths.append(str(final_path))
    result_path = tmp_path / "adjudicated-result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "observed",
                "terminal_event": {
                    "payload": {"artifactPaths": artifact_paths}
                },
            }
        ),
        encoding="utf-8",
    )
    return result_path, output


def _write_no_speech_adjudicated_fixture(
    tmp_path: Path,
    *,
    tamper_voice_hash: bool = False,
) -> tuple[Path, Path]:
    output = tmp_path / "outputs" / "no-speech"
    output.mkdir(parents=True)
    voice = build_voice_activity(
        job_id="no-speech",
        source_sha256="b" * 64,
        media_duration_ms=5_000,
        normalization_profile="mono-16khz-f32-v1",
        provider={"id": "FunASR", "version": "1.2.0"},
        windows=(),
        minimum_window_ms=120,
        classification="no-speech-candidates-detected",
        has_transcribable_speech=False,
    )
    final = build_final_no_speech_adjudication(voice)
    if tamper_voice_hash:
        final["input"]["voiceActivitySha256"] = "0" * 64
    voice_path = output / "voice-activity.v1.json"
    final_path = output / "final-adjudicated-transcript.v1.json"
    voice_path.write_text(json.dumps(voice), encoding="utf-8")
    final_path.write_text(json.dumps(final), encoding="utf-8")
    result_path = tmp_path / "no-speech-result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "observed",
                "terminal_type": "job.completed",
                "terminal_event": {
                    "payload": {
                        "hasTranscribableSpeech": False,
                        "artifactPaths": [
                            str(voice_path),
                            str(final_path),
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return result_path, output


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
    assert _scoring_unit(["字幕质量"]) == "character"
    assert _joint_metric_labels("character")["cpWer"] == "cpCER"
    assert _scoring_unit(["hello", "字幕"]) == "mixed-character-word"


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


def test_formal_acceptance_scores_only_hash_bound_final_segments(
    tmp_path: Path,
) -> None:
    result_path, output = _write_adjudicated_fixture(tmp_path)
    report = evaluate_case(
        case={
            "id": "adjudicated",
            "language": "mul",
            "expectedSpeakerCount": 2,
            "speakerSet": ["truth-a", "truth-b"],
            "scoringTranscript": "alpha beta",
            "turns": [
                {
                    "startSeconds": 0.0,
                    "endSeconds": 1.0,
                    "speakerId": "truth-a",
                },
                {
                    "startSeconds": 1.0,
                    "endSeconds": 2.0,
                    "speakerId": "truth-b",
                },
            ],
            "referenceTranscriptTurns": [
                {
                    "startSeconds": 0.0,
                    "endSeconds": 1.0,
                    "speakerId": "truth-a",
                    "transcript": "alpha",
                },
                {
                    "startSeconds": 1.0,
                    "endSeconds": 2.0,
                    "speakerId": "truth-b",
                    "transcript": "beta",
                },
            ],
            "languageTruth": {
                "qualification": "exact-fixture",
                "expectedLanguages": ["en", "es"],
                "timeScoringEligible": True,
                "intervals": [
                    {
                        "language": "en",
                        "startSeconds": 0.0,
                        "endSeconds": 1.0,
                    },
                    {
                        "language": "es",
                        "startSeconds": 1.0,
                        "endSeconds": 2.0,
                    },
                ],
                "switchPointsSeconds": [1.0],
            },
            "factualTruth": {
                "requiredLiterals": ["alpha", "beta"],
                "forbiddenLiterals": ["gamma"],
            },
            "truthEligibility": {
                "speakerCount": True,
                "turnBoundaries": True,
                "derJer": True,
                "asr": True,
            },
        },
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="adjudicated",
    )

    assert report["qualityPolicy"]["frontModelMetricsRole"] == (
        "diagnostic-only"
    )
    assert report["textQuality"]["hypothesisAuthority"] == "rawText"
    assert report["textQuality"]["werOrCer"] > 0.0
    acceptance = report["postSemanticAcceptance"]
    assert acceptance["artifactValid"] is True
    assert acceptance["disposition"] == "transcribable-speech"
    assert acceptance["status"] == "not-approved-threshold-profile-missing"
    assert acceptance["releaseApproved"] is False
    metrics = acceptance["metrics"]
    assert metrics["speechPresence"]["match"] is True
    assert metrics["finalText"]["hypothesisAuthority"] == "finalText"
    assert metrics["finalText"]["werOrCer"] == 0.0
    assert metrics["jointTranscription"]["hypothesisTextAuthority"] == (
        "finalText"
    )
    assert metrics["jointTranscription"]["cpWer"]["errorRate"] == 0.0
    assert metrics["jointTranscription"]["tcpWer"]["errorRate"] == 0.0
    assert metrics["jointTranscription"]["speakerAttributedWer"][
        "errorRate"
    ] == 0.0
    assert metrics["speakerCount"]["speakerCountMatch"] is True
    assert metrics["diarization"]["der"] == 0.0
    assert metrics["language"]["detectedLanguageCounts"] == {
        "en": 1,
        "es": 1,
    }
    assert metrics["codeSwitch"]["durationWeightedAccuracy"] == 1.0
    assert metrics["factualIntegrity"]["passed"] is True
    assert acceptance["artifactPath"] == str(
        output / "final-adjudicated-transcript.v1.json"
    )


def test_formal_acceptance_preserves_one_speaker_across_language_spans(
    tmp_path: Path,
) -> None:
    result_path, output = _write_adjudicated_fixture(
        tmp_path,
        single_speaker_multilingual=True,
    )
    report = evaluate_case(
        case={
            "id": "adjudicated",
            "language": "mul",
            "expectedSpeakerCount": 1,
            "speakerSet": ["truth-a"],
            "scoringTranscript": "alpha beta",
            "turns": [
                {
                    "startSeconds": 0.0,
                    "endSeconds": 1.0,
                    "speakerId": "truth-a",
                },
                {
                    "startSeconds": 1.0,
                    "endSeconds": 2.0,
                    "speakerId": "truth-a",
                },
            ],
            "referenceTranscriptTurns": [
                {
                    "startSeconds": 0.0,
                    "endSeconds": 1.0,
                    "speakerId": "truth-a",
                    "transcript": "alpha",
                },
                {
                    "startSeconds": 1.0,
                    "endSeconds": 2.0,
                    "speakerId": "truth-a",
                    "transcript": "beta",
                },
            ],
            "languageTruth": {
                "qualification": "single-speaker-code-switch-fixture",
                "expectedLanguages": ["en", "es"],
                "timeScoringEligible": True,
                "intervals": [
                    {
                        "language": "en",
                        "startSeconds": 0.0,
                        "endSeconds": 1.0,
                    },
                    {
                        "language": "es",
                        "startSeconds": 1.0,
                        "endSeconds": 2.0,
                    },
                ],
                "switchPointsSeconds": [1.0],
            },
            "factualTruth": {
                "requiredLiterals": ["alpha", "beta"],
                "forbiddenLiterals": ["gamma"],
            },
            "truthEligibility": {
                "speakerCount": True,
                "turnBoundaries": True,
                "derJer": True,
                "asr": True,
            },
        },
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="adjudicated",
    )

    acceptance = report["postSemanticAcceptance"]
    assert acceptance["artifactValid"] is True
    assert acceptance["metrics"]["speakerCount"]["speakerCountMatch"] is True
    assert acceptance["metrics"]["codeSwitch"][
        "durationWeightedAccuracy"
    ] == 1.0
    final = json.loads(
        (
            output / "final-adjudicated-transcript.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert [segment["speakerId"] for segment in final["segments"]] == [
        "speaker-1",
        "speaker-1",
    ]
    assert [segment["language"] for segment in final["segments"]] == [
        "en",
        "es",
    ]


def test_formal_acceptance_blocks_open_review_without_final_artifact(
    tmp_path: Path,
) -> None:
    result_path, _ = _write_adjudicated_fixture(
        tmp_path,
        include_final=False,
        open_review=True,
    )
    report = evaluate_case(
        case={"id": "adjudicated"},
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="adjudicated",
    )

    acceptance = report["postSemanticAcceptance"]
    assert acceptance["status"] == "blocked"
    assert acceptance["releaseApproved"] is False
    assert acceptance["blockingReasons"] == [
        "final-adjudicated-transcript-missing",
        "open-review-items",
    ]


def test_formal_acceptance_blocks_final_artifact_hash_tampering(
    tmp_path: Path,
) -> None:
    result_path, _ = _write_adjudicated_fixture(
        tmp_path,
        tamper_final_text=True,
    )
    report = evaluate_case(
        case={"id": "adjudicated"},
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="adjudicated",
    )

    acceptance = report["postSemanticAcceptance"]
    assert acceptance["status"] == "blocked"
    assert acceptance["blockingReasons"] == [
        "final-artifact-binding-invalid"
    ]
    assert acceptance["error"]["code"] == (
        "FINAL_ADJUDICATION_BINDING_INVALID"
    )


def test_formal_acceptance_scores_hash_bound_no_speech_disposition(
    tmp_path: Path,
) -> None:
    result_path, output = _write_no_speech_adjudicated_fixture(tmp_path)

    report = evaluate_case(
        case={
            "id": "no-speech",
            "expectedLexicalSpeech": False,
            "truthEligibility": {"voiceActivity": True},
        },
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="no-speech",
    )

    assert report["evidence"]["transcript"] == (
        "not-applicable-no-speech"
    )
    acceptance = report["postSemanticAcceptance"]
    assert acceptance["artifactValid"] is True
    assert acceptance["disposition"] == "no-transcribable-speech"
    assert acceptance["acceptanceSubject"] == "lexical-speech-presence"
    assert acceptance["semanticStatus"] == "not-applicable-no-speech"
    assert acceptance["openReviewCount"] == 0
    assert acceptance["status"] == "not-approved-threshold-profile-missing"
    assert acceptance["missingReferenceTruth"] == []
    assert acceptance["metrics"]["speechPresence"] == {
        "authority": "final-adjudicated-disposition",
        "eligible": True,
        "truthSource": "expectedLexicalSpeech",
        "expectedLexicalSpeech": False,
        "detectedTranscribableSpeech": False,
        "classification": "no-speech-candidates-detected",
        "match": True,
    }
    assert acceptance["metrics"]["speakerCount"] is None
    assert acceptance["metrics"]["finalText"] is None
    assert acceptance["artifactPath"] == str(
        output / "final-adjudicated-transcript.v1.json"
    )


def test_formal_acceptance_rejects_no_speech_voice_hash_tampering(
    tmp_path: Path,
) -> None:
    result_path, _ = _write_no_speech_adjudicated_fixture(
        tmp_path,
        tamper_voice_hash=True,
    )

    report = evaluate_case(
        case={"id": "no-speech", "expectedLexicalSpeech": False},
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="no-speech",
    )

    acceptance = report["postSemanticAcceptance"]
    assert acceptance["status"] == "blocked"
    assert acceptance["blockingReasons"] == [
        "final-artifact-binding-invalid"
    ]
    assert acceptance["error"]["code"] == (
        "FINAL_ADJUDICATION_BINDING_INVALID"
    )


def test_formal_acceptance_hard_fails_missed_lexical_speech(
    tmp_path: Path,
) -> None:
    result_path, _ = _write_no_speech_adjudicated_fixture(tmp_path)

    report = evaluate_case(
        case={"id": "no-speech", "expectedLexicalSpeech": True},
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="no-speech",
    )

    acceptance = report["postSemanticAcceptance"]
    assert acceptance["artifactValid"] is True
    assert acceptance["status"] == "not-approved-hard-domain-failure"
    assert acceptance["releaseApproved"] is False
    assert acceptance["missingReferenceTruth"] == []
    assert acceptance["blockingReasons"] == ["speech-presence-mismatch"]
    assert acceptance["metrics"]["speechPresence"]["match"] is False


def test_formal_acceptance_hard_fails_false_lexical_speech(
    tmp_path: Path,
) -> None:
    result_path, _ = _write_adjudicated_fixture(tmp_path)

    report = evaluate_case(
        case={
            "id": "adjudicated",
            "expectedLexicalSpeech": False,
            "expectedSpeakerCount": 2,
            "scoringTranscript": "alpha beta",
        },
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="adjudicated",
    )

    acceptance = report["postSemanticAcceptance"]
    assert acceptance["artifactValid"] is True
    assert acceptance["status"] == "not-approved-hard-domain-failure"
    assert acceptance["releaseApproved"] is False
    assert acceptance["blockingReasons"] == ["speech-presence-mismatch"]
    assert acceptance["metrics"]["speechPresence"]["match"] is False


def test_evaluator_preserves_source_and_split_buckets() -> None:
    reports = [
        {
            "sourceId": "fleurs",
            "evaluationSplit": "development",
            "status": "observed",
        },
        {
            "sourceId": "fleurs",
            "evaluationSplit": "held-out",
            "status": "harness-failed",
            "errorCode": "JOB_TIMEOUT",
        },
        {
            "sourceId": "ami",
            "evaluationSplit": "held-out",
            "status": "harness-failed",
            "errorCode": "BATCH_ABORTED",
        },
    ]

    assert _bucket_summary(reports, "sourceId") == {
        "ami": {
            "total": 1,
            "observed": 0,
            "speakerCountMatchRate": None,
            "meanWerOrCer": None,
            "meanDer": None,
            "meanJer": None,
            "meanRtf": None,
            "meanLanguageSegmentAccuracy": None,
            "meanCpWer": None,
            "meanTcpWer": None,
            "meanSpeakerAttributedWer": None,
        },
        "fleurs": {
            "total": 2,
            "observed": 1,
            "speakerCountMatchRate": None,
            "meanWerOrCer": None,
            "meanDer": None,
            "meanJer": None,
            "meanRtf": None,
            "meanLanguageSegmentAccuracy": None,
            "meanCpWer": None,
            "meanTcpWer": None,
            "meanSpeakerAttributedWer": None,
        },
    }
    assert _bucket_summary(reports, "evaluationSplit")["held-out"][
        "observed"
    ] == 0
    assert _value_counts(reports, "errorCode") == {
        "BATCH_ABORTED": 1,
        "JOB_TIMEOUT": 1,
        "unknown": 1,
    }


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


def test_evaluator_scores_hash_bound_unmapped_model_native_timeline(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "outputs" / "native-timeline"
    artifact_root.mkdir(parents=True)
    turns = [
        {"startMs": 0, "endMs": 1000, "localSpeaker": "SPEAKER_00"},
        {"startMs": 1000, "endMs": 2000, "localSpeaker": "SPEAKER_01"},
    ]
    turns_hash = hashlib.sha256(
        json.dumps(
            turns,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    full_timeline = {
        "scope": "full-normalized-timeline",
        "startMs": 0,
        "endMs": 2000,
        "turnCount": 2,
        "speakerTurns": turns,
        "speakerTurnsSha256": turns_hash,
        "localSpeakerCount": 2,
        "localSpeakers": ["SPEAKER_00", "SPEAKER_01"],
        "speakerCountConstraints": {"numSpeakers": 2},
    }
    segments = [
        {
            "startMs": start_ms,
            "endMs": end_ms,
            "speakerId": speaker_id,
            "displayText": text,
            "evidence": {
                "overlap": {"fullTimelineInference": full_timeline}
            },
        }
        for start_ms, end_ms, speaker_id, text in (
            (0, 1000, "speaker-1", "hello"),
            (1000, 2000, "speaker-2", "world"),
        )
    ]
    transcript = artifact_root / "transcript-document.v2.json"
    transcript.write_text(
        json.dumps(
            {
                "source": {"durationMs": 2000},
                "speakerPolicy": {"resolvedCount": 2},
                "segments": segments,
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "native-timeline-result.json"
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
            "id": "native-timeline",
            "expectedSpeakerCount": 2,
            "turns": [
                {
                    "startSeconds": 0.0,
                    "endSeconds": 1.0,
                    "speakerId": "truth-a",
                },
                {
                    "startSeconds": 1.0,
                    "endSeconds": 2.0,
                    "speakerId": "truth-b",
                },
            ],
            "truthEligibility": {
                "speakerCount": True,
                "derJer": True,
                "asr": False,
            },
        },
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="native-timeline",
    )

    assert report["nativeDiarizationQuality"] == {
        "authority": "model-native-full-timeline-unmapped",
        "localSpeakerCount": 2,
        "expectedSpeakerCount": 2,
        "speakerCountAbsoluteError": 0,
        "speakerCountMatch": True,
        "turnCount": 2,
        "speakerTurnsSha256": turns_hash,
        "speakerCountConstraints": {"numSpeakers": 2},
        "der": 0.0,
        "jer": 0.0,
        "speakerConfusion": 0.0,
        "overlapF1": 1.0,
    }
    assert report["nativeBoundaryQuality"]["maxAbsoluteErrorMs"] == 0


def test_evaluator_rejects_tampered_model_native_timeline_hash(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "outputs" / "tampered-native-timeline"
    artifact_root.mkdir(parents=True)
    full_timeline = {
        "scope": "full-normalized-timeline",
        "startMs": 0,
        "endMs": 1000,
        "turnCount": 1,
        "speakerTurns": [
            {
                "startMs": 0,
                "endMs": 1000,
                "localSpeaker": "SPEAKER_00",
            }
        ],
        "speakerTurnsSha256": "0" * 64,
        "localSpeakerCount": 1,
        "localSpeakers": ["SPEAKER_00"],
        "speakerCountConstraints": None,
    }
    transcript = artifact_root / "transcript-document.v2.json"
    transcript.write_text(
        json.dumps(
            {
                "source": {"durationMs": 1000},
                "speakerPolicy": {"resolvedCount": 1},
                "segments": [
                    {
                        "startMs": 0,
                        "endMs": 1000,
                        "speakerId": "speaker-1",
                        "displayText": "hello",
                        "evidence": {
                            "overlap": {
                                "fullTimelineInference": full_timeline
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "tampered-native-timeline-result.json"
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

    with pytest.raises(
        ValueError,
        match="model-native full timeline hash is invalid",
    ):
        evaluate_case(
            case={
                "id": "tampered-native-timeline",
                "expectedSpeakerCount": 1,
            },
            result_path=result_path,
            results_root=tmp_path,
            worker_output_root=tmp_path / "outputs",
            artifact_id="tampered-native-timeline",
        )


def test_joint_transcription_separates_cp_tcp_and_speaker_attribution(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "outputs" / "joint-transcription"
    artifact_root.mkdir(parents=True)
    transcript = artifact_root / "transcript-document.v2.json"
    transcript.write_text(
        json.dumps(
            {
                "source": {"durationMs": 22_000},
                "speakerPolicy": {"resolvedCount": 2},
                "segments": [
                    {
                        "startMs": 0,
                        "endMs": 1000,
                        "speakerId": "speaker-1",
                        "rawText": "beta",
                        "normalizedText": "mutated beta",
                        "displayText": "mutated beta",
                    },
                    {
                        "startMs": 20_000,
                        "endMs": 21_000,
                        "speakerId": "speaker-2",
                        "rawText": "alpha",
                        "normalizedText": "mutated alpha",
                        "displayText": "mutated alpha",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "joint-transcription-result.json"
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
            "id": "joint-transcription",
            "expectedSpeakerCount": 2,
            "speakerSet": ["truth-a", "truth-b"],
            "scoringTranscript": "alpha beta",
            "turns": [
                {
                    "startSeconds": 0.0,
                    "endSeconds": 1.0,
                    "speakerId": "truth-a",
                },
                {
                    "startSeconds": 20.0,
                    "endSeconds": 21.0,
                    "speakerId": "truth-b",
                },
            ],
            "referenceTranscriptTurns": [
                {
                    "startSeconds": 0.0,
                    "endSeconds": 1.0,
                    "speakerId": "truth-a",
                    "transcript": "alpha",
                },
                {
                    "startSeconds": 20.0,
                    "endSeconds": 21.0,
                    "speakerId": "truth-b",
                    "transcript": "beta",
                },
            ],
            "truthEligibility": {
                "speakerCount": True,
                "turnBoundaries": True,
                "derJer": True,
                "asr": True,
            },
        },
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="joint-transcription",
    )

    quality = report["jointTranscriptionQuality"]
    assert quality["eligible"] is True
    assert quality["backend"]["package"] == "meeteval"
    assert quality["backend"]["version"] == "0.4.3"
    assert quality["hypothesisTextAuthority"] == "rawText"
    assert quality["scored"] is True
    assert quality["scoringUnit"] == "word"
    assert quality["metricLabels"]["cpWer"] == "cpWER"
    assert quality["cpWer"]["errorRate"] == 0.0
    assert quality["tcpWer"]["errorRate"] == 1.0
    assert quality["tcpWer"]["collarSeconds"] == 5.0
    assert quality["speakerAttributedWer"]["errorRate"] == 1.0
    assert quality["speakerAttributedWer"]["mappingPolicy"] == (
        "time-overlap-max-weight-hungarian-v1"
    )
    assert quality["acousticSpeakerMapping"] == [
        {
            "referenceSpeaker": "truth-a",
            "hypothesisSpeaker": "speaker-1",
            "overlapMs": 1000.0,
        },
        {
            "referenceSpeaker": "truth-b",
            "hypothesisSpeaker": "speaker-2",
            "overlapMs": 1000.0,
        },
    ]
    assert report["textQuality"]["werOrCer"] == 1.0
    assert report["textQuality"]["hypothesisAuthority"] == "rawText"


def test_joint_transcription_requires_per_turn_reference_truth(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "outputs" / "serialized-only"
    artifact_root.mkdir(parents=True)
    transcript = artifact_root / "transcript-document.v2.json"
    transcript.write_text(
        json.dumps(
            {
                "source": {"durationMs": 1000},
                "speakerPolicy": {"resolvedCount": 2},
                "segments": [
                    {
                        "startMs": 0,
                        "endMs": 1000,
                        "speakerId": "speaker-1",
                        "rawText": "hello",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "serialized-only-result.json"
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
            "id": "serialized-only",
            "scoringTranscript": "hello",
            "truthEligibility": {
                "asr": True,
                "turnBoundaries": True,
                "derJer": True,
            },
        },
        result_path=result_path,
        results_root=tmp_path,
        worker_output_root=tmp_path / "outputs",
        artifact_id="serialized-only",
    )

    assert report["textQuality"]["werOrCer"] == 0.0
    assert report["jointTranscriptionQuality"] == {
        "eligible": False,
        "scored": False,
        "reason": "reference-transcript-turns-missing",
    }


def test_joint_transcription_preserves_meeteval_speaker_limit() -> None:
    reference_turns = [
        {
            "startSeconds": float(index),
            "endSeconds": float(index + 1),
            "speakerId": f"truth-{index + 1}",
            "transcript": f"token{index + 1}",
        }
        for index in range(21)
    ]
    hypothesis = [
        {
            "startMs": index * 1000,
            "endMs": (index + 1) * 1000,
            "speakerId": f"speaker-{index + 1}",
            "rawText": f"token{index + 1}",
        }
        for index in range(21)
    ]

    quality = _joint_transcription_quality(
        case={
            "referenceTranscriptTurns": reference_turns,
            "truthEligibility": {
                "asr": True,
                "turnBoundaries": True,
                "derJer": True,
            },
        },
        transcript={"source": {"durationMs": 21_000}},
        segments=hypothesis,
        speaker_timeline=None,
    )

    assert quality["eligible"] is True
    assert quality["scored"] is False
    assert quality["reason"] == "meeteval-speaker-stream-limit"
    assert quality["backend"]["maximumSpeakerStreams"] == 20
    assert quality["referenceSpeakerCount"] == 21
    assert quality["hypothesisSpeakerCount"] == 21
    assert quality["cpWer"] is None
