from __future__ import annotations

import threading
from pathlib import Path

from backend import (
    AdapterContext,
    ProductionSemanticCandidateRegistry,
    build_semantic_candidate_lattice_from_document,
    build_semantic_job_arbitration,
)
from backend.asr_evidence import build_asr_candidate_set
from backend.persistence import atomic_write_json
from backend.speaker_pipeline import AsrHypothesis
from backend.voice_activity import build_voice_activity


REQUEST_KIND = {
    "speech-disposition": "speech-disposition-challenger",
    "speaker-cardinality-timeline": "timeline-challenger",
    "speaker-assignment": "speaker-assignment-challenger",
    "language-span": "open-set-lid",
    "asr-text": "provider-native-nbest",
}


def _document(audio_path: Path) -> dict:
    return {
        "schemaVersion": "2.0.0",
        "documentId": "doc-production-semantic",
        "jobId": "production-semantic",
        "generatedAt": "2026-07-28T06:00:00Z",
        "language": "en",
        "source": {
            "fileName": "fixture.wav",
            "sha256": "a" * 64,
            "durationMs": 1_000,
        },
        "speakerPolicy": {
            "mode": "manual",
            "resolvedCount": 1,
            "speakerIds": ["speaker-1"],
        },
        "speakers": [{"id": "speaker-1"}],
        "segments": [
            {
                "id": "segment-1",
                "startMs": 0,
                "endMs": 1_000,
                "speakerId": "speaker-1",
                "rawText": "Hello",
                "normalizedText": "Hello",
                "displayText": "Hello",
                "confidence": 0.8,
                "speakerScores": [
                    {"speakerId": "speaker-1", "score": 0.8}
                ],
                "speakerMargin": 1.0,
                "overlapping": False,
                "humanLocked": False,
                "revisions": [],
                "language": "en",
                "evidence": {
                    "preparation": {
                        "audioPath": str(audio_path),
                        "normalizationProfile": "mono-16khz-f32-v1",
                    },
                    "asr": {"provider": "fixture-asr"},
                },
            }
        ],
        "provenance": {
            "offline": True,
            "workerVersion": "2.0.0",
            "transcriptionAdapter": {"id": "fixture", "version": "1"},
            "models": [],
        },
    }


def _request_response(lattice: dict) -> dict:
    selections = []
    requests = []
    for domain in lattice["domains"]:
        for group in domain["groups"]:
            refs = [
                f"candidate-lattice:{lattice['latticeId']}",
                f"candidate-group:{group['groupId']}",
            ]
            if group["status"] == "available":
                selections.append(
                    {
                        "groupId": group["groupId"],
                        "rankedCandidateIds": [
                            candidate["candidateId"]
                            for candidate in group["candidates"]
                            if candidate["selectionEligible"]
                        ],
                        "reasonCodes": ["AVAILABLE_EVIDENCE"],
                        "evidenceRefs": refs,
                    }
                )
            else:
                requests.append(
                    {
                        "domain": domain["domain"],
                        "groupId": group["groupId"],
                        "scopeId": group["scopeId"],
                        "requestKind": REQUEST_KIND[domain["domain"]],
                        "minimumAlternativeCount": 2,
                        "reasonCodes": ["INSUFFICIENT_ALTERNATIVES"],
                        "evidenceRefs": refs,
                    }
                )
    return {
        "latticeId": lattice["latticeId"],
        "latticeSha256": lattice["latticeSha256"],
        "selections": selections,
        "candidateGenerationRequests": requests,
    }


def test_production_registry_generates_all_five_real_evidence_domains(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(b"fixture-audio")
    output = tmp_path / "output"
    output.mkdir()
    document = _document(audio_path)
    voice = build_voice_activity(
        job_id=document["jobId"],
        source_sha256=document["source"]["sha256"],
        media_duration_ms=1_000,
        normalization_profile="mono-16khz-f32-v1",
        provider={"id": "fixture-vad", "version": "1"},
        windows=[{"id": "vad-1", "startMs": 0, "endMs": 950}],
        minimum_window_ms=100,
        classification="transcribable-speech-detected",
        has_transcribable_speech=True,
    )
    atomic_write_json(output / "voice-activity.v1.json", voice)

    class FakeAsr:
        def __init__(self) -> None:
            self.calls = 0
            self.release_calls = 0

        def transcribe_batch(
            self,
            prepared,
            windows,
            context,
            *,
            requested_language,
            max_generated_tokens,
        ):
            self.calls += 1
            return [
                AsrHypothesis(
                    window_id=window.window_id,
                    text="Hello.",
                    confidence=0.9,
                    evidence=build_asr_candidate_set(
                        model_id="fixture-qwen3-asr",
                        model_revision="revision-1",
                        model_manifest_sha256="b" * 64,
                        model_identity_status="manifest-bound",
                        source_audio_sha256=document["source"]["sha256"],
                        normalization_profile="mono-16khz-f32-v1",
                        source_window_id=window.window_id,
                        start_ms=window.start_ms,
                        end_ms=window.end_ms,
                        hypotheses=[
                            {
                                "text": "Hello.",
                                "language": "en",
                                "tokens": [],
                                "acousticScore": None,
                                "acousticScoreStatus": "provider-unavailable",
                                "decodeScore": None,
                                "decodeScoreStatus": "provider-unavailable",
                            }
                        ],
                    ),
                )
                for window in windows
            ]

        def release_resources(self) -> None:
            self.release_calls += 1

    class FakePyannote:
        def __init__(self) -> None:
            self.calls = 0
            self.release_calls = 0

        def timeline_challenger(
            self,
            path,
            *,
            duration_ms,
            context,
        ):
            self.calls += 1
            turns = [
                {
                    "startMs": 0,
                    "endMs": duration_ms,
                    "localSpeaker": "SPEAKER_00",
                }
            ]
            return {
                "speakerTurns": turns,
                "exclusiveSpeakerTurns": turns,
                "modelId": "pyannote-community-1",
                "modelRevision": "revision-1",
                "modelManifestSha256": "c" * 64,
            }

        def release_resources(self) -> None:
            self.release_calls += 1

    lattice = build_semantic_candidate_lattice_from_document(document)
    arbitration = build_semantic_job_arbitration(
        job_id=document["jobId"],
        lattice=lattice,
        response=_request_response(lattice),
        model="fixture-9b",
        provider={
            "id": "fixture-loopback",
            "version": "1",
            "networkPolicy": "loopback-only",
        },
    )
    asr = FakeAsr()
    pyannote = FakePyannote()
    registry = ProductionSemanticCandidateRegistry(
        asr_adapter=asr,  # type: ignore[arg-type]
        pyannote_adapter=pyannote,  # type: ignore[arg-type]
        context=AdapterContext(
            job_id=document["jobId"],
            output_directory=output,
            cancellation=threading.Event(),
        ),
    )

    generation = registry.fulfill(document, lattice, arbitration)

    assert generation["status"] == "completed"
    assert generation["metrics"]["unfulfilledRequestCount"] == 0
    assert generation["outputLattice"]["availability"][
        "allRequiredDomainsAvailable"
    ] is True
    assert {
        item["domain"] for item in generation["fulfilledRequests"]
    } == set(REQUEST_KIND)
    assert asr.calls == 1
    assert pyannote.calls == 1
    registry.release_resources()
    assert asr.release_calls == 1
    assert pyannote.release_calls == 1
