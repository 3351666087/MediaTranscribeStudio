from __future__ import annotations

import json

import pytest

from tools.run_first_principles_llm_arbitration import (
    LocalLLMArbitrationError,
    _apply_deterministic_constraints,
    _overlapping_speakers,
    _validate_proposal,
    run,
)


def test_validate_proposal_accepts_empty_negative_decision() -> None:
    _validate_proposal(
        {
            "lexicalSpeechPresent": False,
            "speakerCount": 0,
            "languages": [],
            "transcriptSource": "none",
            "transcriptText": "",
            "translationText": "",
            "speakerAttribution": [],
            "decision": "no lexical speech",
            "uncertainties": ["ASR hallucination candidate"],
        },
        whisper_segment_count=1,
        allowed_speakers_by_segment=[set()],
    )


def test_validate_proposal_rejects_unknown_speaker() -> None:
    with pytest.raises(LocalLLMArbitrationError, match="speaker attribution 0"):
        _validate_proposal(
            {
                "lexicalSpeechPresent": True,
                "speakerCount": 1,
                "languages": ["en"],
                "transcriptSource": "whisper",
                "transcriptText": "hello",
                "translationText": "你好",
                "speakerAttribution": [
                    {"segmentIndex": 0, "speakers": ["invented"], "language": "en"}
                ],
                "decision": "speech",
                "uncertainties": [],
            },
            whisper_segment_count=1,
            allowed_speakers_by_segment=[{"SPEAKER_00"}],
        )


def test_validate_proposal_requires_attribution_for_every_whisper_segment() -> None:
    with pytest.raises(LocalLLMArbitrationError, match="cover every Whisper segment"):
        _validate_proposal(
            {
                "lexicalSpeechPresent": True,
                "speakerCount": 1,
                "languages": ["en"],
                "transcriptSource": "whisper",
                "transcriptText": "hello",
                "translationText": "你好",
                "speakerAttribution": [],
                "decision": "speech",
                "uncertainties": [],
            },
            whisper_segment_count=1,
            allowed_speakers_by_segment=[{"SPEAKER_00"}],
        )


def test_overlapping_speakers_limits_each_segment_to_acoustic_candidates() -> None:
    assert _overlapping_speakers(
        [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
        ],
        [
            {
                "startSeconds": 0.25,
                "endSeconds": 1.25,
                "localSpeaker": "SPEAKER_00",
            },
            {
                "startSeconds": 1.5,
                "endSeconds": 1.75,
                "localSpeaker": "SPEAKER_01",
            },
        ],
    ) == [{"SPEAKER_00"}, {"SPEAKER_00", "SPEAKER_01"}]


def test_same_language_chinese_translation_is_deterministic_copy() -> None:
    proposal = {
        "lexicalSpeechPresent": True,
        "languages": ["zh"],
        "transcriptText": "原文",
        "translationText": "被改写的中文",
    }

    corrections = _apply_deterministic_constraints(proposal)

    assert proposal["translationText"] == "原文"
    assert corrections == [
        {
            "field": "translationText",
            "reason": "same-language Chinese translation must be an exact copy",
        }
    ]


def test_missing_speaker_attributions_are_deterministically_completed() -> None:
    proposal = {
        "lexicalSpeechPresent": True,
        "languages": ["en"],
        "transcriptText": "hello",
        "translationText": "你好",
        "speakerAttribution": [
            {"segmentIndex": 1, "speakers": ["SPEAKER_01"], "language": "en"}
        ],
    }

    corrections = _apply_deterministic_constraints(
        proposal,
        allowed_speakers_by_segment=[{"SPEAKER_00"}, {"SPEAKER_01"}],
    )

    assert proposal["speakerAttribution"] == [
        {"segmentIndex": 0, "speakers": ["SPEAKER_00"], "language": "en"},
        {"segmentIndex": 1, "speakers": ["SPEAKER_01"], "language": "en"},
    ]
    assert corrections[-1]["field"] == "speakerAttribution"


def _write_report(path, artifact_type: str, audit_case_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "artifactType": artifact_type,
                "truthAccessed": False,
                "cases": [{"auditCaseId": audit_case_id}],
            }
        ),
        encoding="utf-8",
    )


def test_run_resumes_failed_report_bound_to_all_inputs(tmp_path, monkeypatch) -> None:
    audit_case_id = "case-1"
    blind = tmp_path / "blind.json"
    blind.write_text(
        json.dumps(
            {
                "artifactType": "first-principles-blind-media-batch",
                "batchId": "batch-1",
                "cases": [{"auditCaseId": audit_case_id, "durationSeconds": 5.0}],
            }
        ),
        encoding="utf-8",
    )
    inputs = {}
    for name in ("qwen", "whisper", "vad", "diarization"):
        path = tmp_path / f"{name}.json"
        _write_report(path, f"evidence-{name}", audit_case_id)
        inputs[name] = path
    output = tmp_path / "llm.json"

    def proposal(**_kwargs):
        return (
            {
                "lexicalSpeechPresent": False,
                "speakerCount": 0,
                "languages": [],
                "transcriptSource": "none",
                "transcriptText": "",
                "translationText": "",
                "speakerAttribution": [],
                "decision": "no lexical speech",
                "uncertainties": [],
            },
            0.01,
            "response-sha",
            {"generatedTokens": 1},
        )

    monkeypatch.setattr(
        "tools.run_first_principles_llm_arbitration._request", proposal
    )
    first = run(
        blind_path=blind,
        qwen_path=inputs["qwen"],
        whisper_path=inputs["whisper"],
        vad_path=inputs["vad"],
        diarization_path=inputs["diarization"],
        output_path=output,
        model="qwen3.5:9b",
        timeout_seconds=1,
        maximum=1,
    )
    first["cases"][0].update(
        {"status": "failed", "proposal": None, "failure": "prior failure"}
    )
    output.write_text(json.dumps(first), encoding="utf-8")

    resumed = run(
        blind_path=blind,
        qwen_path=inputs["qwen"],
        whisper_path=inputs["whisper"],
        vad_path=inputs["vad"],
        diarization_path=inputs["diarization"],
        output_path=output,
        model="qwen3.5:9b",
        timeout_seconds=1,
    )

    assert resumed["model"]["maximumCases"] is None
    assert resumed["cases"][0]["status"] == "completed"
    assert resumed["cases"][0]["priorFailures"] == [
        {
            "failure": "prior failure",
            "wallSeconds": 0.01,
            "promptSha256": resumed["cases"][0]["promptSha256"],
        }
    ]


def test_run_rejects_resume_after_evidence_changes(tmp_path, monkeypatch) -> None:
    audit_case_id = "case-1"
    blind = tmp_path / "blind.json"
    blind.write_text(
        json.dumps(
            {
                "artifactType": "first-principles-blind-media-batch",
                "batchId": "batch-1",
                "cases": [{"auditCaseId": audit_case_id, "durationSeconds": 5.0}],
            }
        ),
        encoding="utf-8",
    )
    inputs = {}
    for name in ("qwen", "whisper", "vad", "diarization"):
        path = tmp_path / f"{name}.json"
        _write_report(path, f"evidence-{name}", audit_case_id)
        inputs[name] = path
    output = tmp_path / "llm.json"
    monkeypatch.setattr(
        "tools.run_first_principles_llm_arbitration._request",
        lambda **_kwargs: (
            {
                "lexicalSpeechPresent": False,
                "speakerCount": 0,
                "languages": [],
                "transcriptSource": "none",
                "transcriptText": "",
                "translationText": "",
                "speakerAttribution": [],
                "decision": "no lexical speech",
                "uncertainties": [],
            },
            0.01,
            "response-sha",
            {"generatedTokens": 1},
        ),
    )
    run(
        blind_path=blind,
        qwen_path=inputs["qwen"],
        whisper_path=inputs["whisper"],
        vad_path=inputs["vad"],
        diarization_path=inputs["diarization"],
        output_path=output,
        model="qwen3.5:9b",
        timeout_seconds=1,
    )
    inputs["whisper"].write_text(
        inputs["whisper"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(LocalLLMArbitrationError, match="cannot be resumed"):
        run(
            blind_path=blind,
            qwen_path=inputs["qwen"],
            whisper_path=inputs["whisper"],
            vad_path=inputs["vad"],
            diarization_path=inputs["diarization"],
            output_path=output,
            model="qwen3.5:9b",
            timeout_seconds=1,
        )


def test_run_preserves_rejected_proposal_for_diagnosis(tmp_path, monkeypatch) -> None:
    audit_case_id = "case-1"
    blind = tmp_path / "blind.json"
    blind.write_text(
        json.dumps(
            {
                "artifactType": "first-principles-blind-media-batch",
                "batchId": "batch-1",
                "cases": [{"auditCaseId": audit_case_id, "durationSeconds": 5.0}],
            }
        ),
        encoding="utf-8",
    )
    inputs = {}
    for name in ("qwen", "whisper", "vad", "diarization"):
        path = tmp_path / f"{name}.json"
        _write_report(path, f"evidence-{name}", audit_case_id)
        inputs[name] = path
    rejected = {"unexpected": "proposal"}
    monkeypatch.setattr(
        "tools.run_first_principles_llm_arbitration._request",
        lambda **_kwargs: (
            rejected,
            0.01,
            "response-sha",
            {"generatedTokens": 1},
        ),
    )

    report = run(
        blind_path=blind,
        qwen_path=inputs["qwen"],
        whisper_path=inputs["whisper"],
        vad_path=inputs["vad"],
        diarization_path=inputs["diarization"],
        output_path=tmp_path / "llm.json",
        model="qwen3.5:9b",
        timeout_seconds=1,
    )

    assert report["cases"][0]["status"] == "failed"
    assert report["cases"][0]["proposal"] == rejected
    assert report["cases"][0]["rawProposal"] == rejected
    assert report["cases"][0]["responseSha256"] == "response-sha"
    assert "proposal fields are invalid" in report["cases"][0]["failure"]
