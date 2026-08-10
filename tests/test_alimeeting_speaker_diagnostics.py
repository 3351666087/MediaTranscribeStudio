from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.persistence import canonical_json_sha256
import tools.build_alimeeting_speaker_diagnostics as diagnostics
from tools.build_alimeeting_speaker_diagnostics import (
    AliMeetingDiagnosticError,
    USAGE_POLICY,
    _scope_partition_manifest,
    audit_identity_scope,
    validate_partition_manifest,
    validate_related_diarization_cases,
)


def _session(session_id: str, speakers: list[str]) -> dict[str, object]:
    turns = [
        {
            "speakerId": speakers[0],
            "startSeconds": 0.0,
            "endSeconds": 1.0,
            "transcript": "甲",
        },
        {
            "speakerId": speakers[1],
            "startSeconds": 2.0,
            "endSeconds": 3.0,
            "transcript": "乙",
        },
        {
            "speakerId": speakers[0],
            "startSeconds": 6.0,
            "endSeconds": 7.0,
            "transcript": "丙",
        },
        {
            "speakerId": speakers[1],
            "startSeconds": 8.0,
            "endSeconds": 9.0,
            "transcript": "丁",
        },
    ]
    return {
        "sessionId": session_id,
        "speakerSet": speakers,
        "speakerCount": len(speakers),
        "turns": turns,
        "farAudio": Path(f"{session_id}-far.wav"),
        "nearAudio": [Path(f"{session_id}-{speaker}.wav") for speaker in speakers],
        "farAudioEvidence": {
            "path": f"{session_id}-far.wav",
            "bytes": 1,
            "sha256": "a" * 64,
        },
        "nearAudioEvidence": [
            {
                "path": f"{session_id}-{speaker}.wav",
                "bytes": 1,
                "sha256": "b" * 64,
            }
            for speaker in speakers
        ],
        "farTextGridEvidence": {
            "path": f"{session_id}.TextGrid",
            "bytes": 1,
            "sha256": "c" * 64,
        },
    }


def test_identity_audit_proves_no_cross_session_positive_identity() -> None:
    sessions = [
        _session("R0001_M0001", ["N_SPK0001", "N_SPK0002"]),
        _session("R0002_M0002", ["N_SPK0003", "N_SPK0004"]),
    ]

    audit = audit_identity_scope(sessions)

    assert audit["sourceSessionCount"] == 2
    assert audit["speakerOccurrenceCount"] == 4
    assert audit["uniqueSpeakerCount"] == 4
    assert audit["crossSessionRepeatedSpeakerIds"] == []
    assert audit["crossSessionSameSpeakerTrialsAvailable"] is False
    assert audit["nearFarMayCountAsIndependentRecordings"] is False

    sessions[1]["speakerSet"] = ["N_SPK0001", "N_SPK0004"]
    with pytest.raises(AliMeetingDiagnosticError, match="cross-session"):
        audit_identity_scope(sessions)


def test_related_diarization_cases_reject_split_or_recording_leakage() -> None:
    sessions = ["R0001_M0001", "R0002_M0002"]
    cases = [
        {
            "sourceId": "alimeeting",
            "sourceRevision": "locked",
            "sourceSessionId": session,
            "sourceModality": modality,
            "evaluationSplit": "held-out",
        }
        for session in sessions
        for modality in ("far-field-array", "synchronized-near-field-mixture")
    ]

    evidence = validate_related_diarization_cases(
        cases,
        expected_sessions=sessions,
        source_revision="locked",
    )

    assert evidence["caseCount"] == 4
    assert evidence["recordingIsolationVerified"] is True
    cases[-1]["evaluationSplit"] = "development"
    with pytest.raises(AliMeetingDiagnosticError, match="source isolation"):
        validate_related_diarization_cases(
            cases,
            expected_sessions=sessions,
            source_revision="locked",
        )


def test_partition_policy_is_fail_closed_for_near_far_recording_identity() -> None:
    session = _session("R0001_M0001", ["N_SPK0001", "N_SPK0002"])
    base = {
        "schemaVersion": "1.0.0",
        "libraryId": "base",
        "randomSeed": 1,
        "source": {
            "recordingId": "R0001_M0001",
            "audioPath": "audio.wav",
            "annotationPath": "truth.json",
        },
        "selection": {},
        "counts": {
            "speakers": 2,
            "clips": 3,
            "sameSpeakerTrials": 1,
            "differentSpeakerTrials": 1,
            "totalTrials": 2,
        },
        "clips": [
            {
                "clipId": "a-1",
                "speakerId": "N_SPK0001",
                "sourceRecordingId": "R0001_M0001",
                "evaluationSplit": "held-out",
            },
            {
                "clipId": "a-2",
                "speakerId": "N_SPK0001",
                "sourceRecordingId": "R0001_M0001",
                "evaluationSplit": "held-out",
            },
            {
                "clipId": "b-1",
                "speakerId": "N_SPK0002",
                "sourceRecordingId": "R0001_M0001",
                "evaluationSplit": "held-out",
            },
        ],
        "trials": [
            {
                "trialId": "same",
                "enrollmentClipId": "a-1",
                "testClipId": "a-2",
                "sameSpeaker": True,
                "evaluationSplit": "held-out",
            },
            {
                "trialId": "different",
                "enrollmentClipId": "a-1",
                "testClipId": "b-1",
                "sameSpeaker": False,
                "evaluationSplit": "held-out",
            },
        ],
    }
    scoped = _scope_partition_manifest(
        base,
        session=session,
        modality="synchronized-near-field-mixture",
        related_diarization={"path": "related.json", "sha256": "d" * 64},
    )

    assert scoped["usagePolicy"] == USAGE_POLICY
    assert scoped["source"]["recordingId"] == "R0001_M0001"
    assert scoped["source"]["sourceSessionId"] == "R0001_M0001"
    assert scoped["source"]["nearFarAreIndependentRecordings"] is False
    assert scoped["speakerCoverage"] == {
        "sourceSpeakerIds": ["N_SPK0001", "N_SPK0002"],
        "trialSpeakerIds": ["N_SPK0001", "N_SPK0002"],
        "excludedSpeakerIds": [],
        "allOfficialSessionSpeakersCovered": True,
    }
    assert all(
        trial["recordingRelation"] == "within-recording"
        for trial in scoped["trials"]
    )
    body = dict(scoped)
    digest = body.pop("canonicalSha256")
    assert digest == canonical_json_sha256(body)

    scoped["source"]["recordingId"] = "fake-independent-near-recording"
    with pytest.raises(AliMeetingDiagnosticError, match="recording identity"):
        validate_partition_manifest(scoped)


def test_full_builder_emits_isolated_diagnostic_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = [
        _session("R0001_M0001", ["N_SPK0001", "N_SPK0002"]),
        _session("R0002_M0002", ["N_SPK0003", "N_SPK0004"]),
    ]
    plan = {
        "dataset": "SLR119/AliMeeting",
        "revision": "locked",
        "license": "cc-by-sa-4.0",
        "licenseDecision": "strict official license",
        "archiveSha256": "e" * 64,
        "extractedTreeSha256": "f" * 64,
        "evaluationSplit": "held-out",
        "split": "Eval",
    }
    global_manifest = tmp_path / "global.json"
    global_manifest.write_text("{}", encoding="utf-8")
    related = tmp_path / "related.json"
    related.write_text(
        json.dumps(
            {
                "libraryId": "related",
                "sources": [
                    {
                        "sourceId": "alimeeting",
                        "revision": "locked",
                        "license": "cc-by-sa-4.0",
                        "archive": {"sha256": "e" * 64},
                        "extractedTree": {"sha256": "f" * 64},
                    }
                ],
                "cases": [
                    {
                        "sourceId": "alimeeting",
                        "sourceRevision": "locked",
                        "sourceSessionId": session["sessionId"],
                        "sourceModality": modality,
                        "evaluationSplit": "held-out",
                    }
                    for session in sessions
                    for modality in (
                        "far-field-array",
                        "synchronized-near-field-mixture",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        diagnostics,
        "load_global_manifest",
        lambda _path: SimpleNamespace(planned_real_diarization_sources=()),
    )
    monkeypatch.setattr(diagnostics, "_planned_by_id", lambda _manifest, _id: plan)
    monkeypatch.setattr(diagnostics, "_validate_alimeeting_plan", lambda _plan: None)
    monkeypatch.setattr(
        diagnostics,
        "_resolve_alimeeting_corpus",
        lambda **_kwargs: (
            tmp_path,
            {"treeSha256": "f" * 64},
            True,
        ),
    )
    monkeypatch.setattr(
        diagnostics,
        "_discover_alimeeting_sessions",
        lambda _root, _plan, _tree: sessions,
    )

    def fake_source_audio(
        _session_value: object,
        _modality: str,
        output: Path,
        *,
        cached_audio: Path | None = None,
    ) -> dict[str, object]:
        assert cached_audio is None
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 16_000 * 10)
        return diagnostics._mono_wave_evidence(output)

    monkeypatch.setattr(diagnostics, "_source_audio", fake_source_audio)
    output_root = tmp_path / "output"
    destination = diagnostics.build_diagnostics(
        global_manifest_path=global_manifest,
        related_diarization_path=related,
        output_root=output_root,
        clip_duration_ms=500,
        maximum_clips_per_speaker=2,
        minimum_spacing_ms=2_000,
        minimum_pair_separation_ms=4_000,
        maximum_pairs_per_class=4,
        random_seed=7,
    )
    index = json.loads(destination.read_text(encoding="utf-8"))

    assert index["counts"]["sourceSessions"] == 2
    assert index["counts"]["partitions"] == 4
    assert index["counts"]["sourceSpeakerIdentities"] == 4
    assert index["counts"]["trialSpeakerIdentities"] == 4
    assert index["usagePolicy"] == USAGE_POLICY
    assert index["splitIsolation"]["recordingLeakageCheckPassed"] is True
    assert index["splitIsolation"]["recordingIdsBySplit"] == {
        "development": [],
        "regression": [],
        "held-out": ["R0001_M0001", "R0002_M0002"],
    }
    assert {row["sourceRecordingId"] for row in index["partitions"]} == {
        "R0001_M0001",
        "R0002_M0002",
    }
    assert all(row["diagnosticOnly"] for row in index["partitions"])
    assert all(
        row["speakerCoverage"]["allOfficialSessionSpeakersCovered"]
        for row in index["partitions"]
    )
    for row in index["partitions"]:
        partition = json.loads((output_root / row["path"]).read_text(encoding="utf-8"))
        validate_partition_manifest(partition)
