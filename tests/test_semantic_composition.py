from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from backend import (
    BusinessProcessingConfig,
    BusinessProcessingRunner,
    MappingLocalLLMProvider,
    SemanticCandidateGenerationRegistry,
    SemanticCompositionError,
    SemanticCompositionOrchestrator,
    SemanticJobArbitrationRunner,
    build_semantic_candidate_lattice,
    build_semantic_candidate_lattice_from_document,
    build_semantic_composition,
    build_semantic_job_arbitration,
    build_asr_text_challenger_result,
    build_final_composed_transcript,
    build_open_set_lid_challenger_result,
    build_timeline_challenger_result,
    build_voice_activity_challenger_result,
    compose_transcript_document,
    extend_semantic_candidate_lattice,
    semantic_job_prompt_context,
    validate_semantic_composition,
    validate_semantic_candidate_generation,
    validate_semantic_job_arbitration,
    validate_final_composed_transcript,
)
from backend.persistence import canonical_json_sha256, read_json_strict
from backend.asr_evidence import build_asr_candidate_set
from backend.errors import WorkerError
from backend.semantic_composition import (
    _bounded_transcript_context,
    _complete_structural_continuity_requests,
)
from backend.voice_activity import build_voice_activity


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = {
    "producerType": "model",
    "systemId": "fixture-challenger",
    "revision": "revision-1",
    "artifactSha256": "c" * 64,
    "modelManifestSha256": "d" * 64,
    "identityStatus": "manifest-bound",
}
REQUEST_KIND = {
    "speech-disposition": "speech-disposition-challenger",
    "speaker-cardinality-timeline": "timeline-challenger",
    "speaker-assignment": "speaker-assignment-challenger",
    "language-span": "open-set-lid",
    "asr-text": "provider-native-nbest",
}
_SEGMENT_ATOMIC_DOMAINS_FOR_TEST = {
    "speaker-assignment",
    "language-span",
    "asr-text",
}


def _segment(
    segment_id: str,
    *,
    start_ms: int,
    speaker_id: str,
    text: str,
) -> dict:
    return {
        "id": segment_id,
        "startMs": start_ms,
        "endMs": start_ms + 1_000,
        "speakerId": speaker_id,
        "rawText": text,
        "normalizedText": text,
        "displayText": text,
        "confidence": 0.8,
        "speakerScores": [
            {"speakerId": "speaker-1", "score": 0.6},
            {"speakerId": "speaker-2", "score": 0.4},
        ],
        "speakerMargin": 0.2,
        "overlapping": False,
        "humanLocked": False,
        "revisions": [],
        "language": "en",
        "evidence": {"asr": {"provider": "fixture-asr"}},
    }


def _document() -> dict:
    return {
        "schemaVersion": "2.0.0",
        "documentId": "doc-semantic-composition",
        "jobId": "job-semantic-composition",
        "generatedAt": "2026-07-28T00:00:00Z",
        "language": "en",
        "source": {
            "fileName": "fixture.wav",
            "sha256": "a" * 64,
            "durationMs": 2_000,
        },
        "speakerPolicy": {
            "mode": "manual",
            "resolvedCount": 2,
            "speakerIds": ["speaker-1", "speaker-2"],
        },
        "speakers": [{"id": "speaker-1"}, {"id": "speaker-2"}],
        "segments": [
            _segment(
                "segment-1",
                start_ms=0,
                speaker_id="speaker-1",
                text="Hello",
            ),
            _segment(
                "segment-2",
                start_ms=1_000,
                speaker_id="speaker-2",
                text="World",
            ),
        ],
        "provenance": {"offline": True, "models": []},
    }


def _candidate(payload: dict, *, current: bool) -> dict:
    return {
        "payload": payload,
        "producers": [PRODUCER],
        "selectionEligible": True,
        "eligibilityReason": "eligible",
        "isCurrent": current,
    }


def _full_lattice(document: dict) -> dict:
    timeline_current = {
        "speakerCount": 2,
        "speakerIds": ["speaker-1", "speaker-2"],
        "timelineKind": "current-transcript",
        "startMs": 0,
        "endMs": 2_000,
        "turns": [
            {
                "startMs": 0,
                "endMs": 1_000,
                "speakerId": "speaker-1",
                "overlap": False,
            },
            {
                "startMs": 1_000,
                "endMs": 2_000,
                "speakerId": "speaker-2",
                "overlap": False,
            },
        ],
    }
    timeline_challenger = {
        "speakerCount": 3,
        "speakerIds": ["speaker-1", "speaker-2", "speaker-3"],
        "timelineKind": "challenger",
        "startMs": 0,
        "endMs": 2_000,
        "turns": [
            {
                "startMs": 0,
                "endMs": 500,
                "speakerId": "speaker-1",
                "overlap": False,
            },
            {
                "startMs": 500,
                "endMs": 1_000,
                "speakerId": "speaker-3",
                "overlap": False,
            },
            {
                "startMs": 1_000,
                "endMs": 2_000,
                "speakerId": "speaker-2",
                "overlap": False,
            },
        ],
    }
    candidate_groups = {
        "speech-disposition": [
            {
                "scopeId": "media",
                "candidates": [
                    _candidate(
                        {
                            "classification": "transcribable-speech",
                            "startMs": 0,
                            "endMs": 2_000,
                        },
                        current=True,
                    ),
                    _candidate(
                        {
                            "classification": "no-transcribable-speech",
                            "startMs": 0,
                            "endMs": 2_000,
                        },
                        current=False,
                    ),
                ],
            }
        ],
        "speaker-cardinality-timeline": [
            {
                "scopeId": "media",
                "candidates": [
                    _candidate(timeline_current, current=True),
                    _candidate(timeline_challenger, current=False),
                ],
            }
        ],
        "speaker-assignment": [
            {
                "scopeId": "segment:segment-1",
                "candidates": [
                    _candidate(
                        {
                            "segmentId": "segment-1",
                            "startMs": 0,
                            "endMs": 1_000,
                            "speakerId": "speaker-1",
                            "score": 0.6,
                        },
                        current=True,
                    ),
                    _candidate(
                        {
                            "segmentId": "segment-1",
                            "startMs": 0,
                            "endMs": 1_000,
                            "speakerId": "speaker-3",
                            "score": 0.7,
                        },
                        current=False,
                    ),
                ],
            },
            {
                "scopeId": "segment:segment-2",
                "candidates": [
                    _candidate(
                        {
                            "segmentId": "segment-2",
                            "startMs": 1_000,
                            "endMs": 2_000,
                            "speakerId": "speaker-2",
                            "score": 0.8,
                        },
                        current=True,
                    ),
                    _candidate(
                        {
                            "segmentId": "segment-2",
                            "startMs": 1_000,
                            "endMs": 2_000,
                            "speakerId": "speaker-1",
                            "score": 0.2,
                        },
                        current=False,
                    ),
                ],
            },
        ],
        "language-span": [
            {
                "scopeId": "segment:segment-1",
                "candidates": [
                    _candidate(
                        {
                            "segmentId": "segment-1",
                            "startMs": 0,
                            "endMs": 1_000,
                            "language": "en",
                            "confidence": 0.6,
                        },
                        current=True,
                    ),
                    _candidate(
                        {
                            "segmentId": "segment-1",
                            "startMs": 0,
                            "endMs": 1_000,
                            "language": "es",
                            "confidence": 0.9,
                        },
                        current=False,
                    ),
                ],
            },
            {
                "scopeId": "segment:segment-2",
                "candidates": [
                    _candidate(
                        {
                            "segmentId": "segment-2",
                            "startMs": 1_000,
                            "endMs": 2_000,
                            "language": "en",
                            "confidence": 0.8,
                        },
                        current=True,
                    ),
                    _candidate(
                        {
                            "segmentId": "segment-2",
                            "startMs": 1_000,
                            "endMs": 2_000,
                            "language": "fr",
                            "confidence": 0.2,
                        },
                        current=False,
                    ),
                ],
            },
        ],
        "asr-text": [
            {
                "scopeId": "segment:segment-1",
                "candidates": [
                    _candidate(
                        {
                            "segmentId": "segment-1",
                            "startMs": 0,
                            "endMs": 1_000,
                            "text": "Hello",
                            "language": "en",
                            "sourceCandidateId": "asr-segment-1-rank-1",
                            "candidateSetSha256": "e" * 64,
                        },
                        current=True,
                    ),
                    _candidate(
                        {
                            "segmentId": "segment-1",
                            "startMs": 0,
                            "endMs": 1_000,
                            "text": "Hola",
                            "language": "es",
                            "sourceCandidateId": "asr-segment-1-rank-2",
                            "candidateSetSha256": "e" * 64,
                        },
                        current=False,
                    ),
                ],
            },
            {
                "scopeId": "segment:segment-2",
                "candidates": [
                    _candidate(
                        {
                            "segmentId": "segment-2",
                            "startMs": 1_000,
                            "endMs": 2_000,
                            "text": "World",
                            "language": "en",
                            "sourceCandidateId": "asr-segment-2-rank-1",
                            "candidateSetSha256": "f" * 64,
                        },
                        current=True,
                    ),
                    _candidate(
                        {
                            "segmentId": "segment-2",
                            "startMs": 1_000,
                            "endMs": 2_000,
                            "text": "Monde",
                            "language": "fr",
                            "sourceCandidateId": "asr-segment-2-rank-2",
                            "candidateSetSha256": "f" * 64,
                        },
                        current=False,
                    ),
                ],
            },
        ],
    }
    return build_semantic_candidate_lattice(
        source_media_sha256=document["source"]["sha256"],
        transcript_sha256=canonical_json_sha256(document),
        transcript_schema_version=document["schemaVersion"],
        source_duration_ms=document["source"]["durationMs"],
        candidate_groups=candidate_groups,
    )


def _speaker_continuity_regression_document(fixture: dict) -> dict:
    visible = fixture["visibleInput"]
    binding = fixture["inputBinding"]
    document = _document()
    document.update(
        {
            "documentId": f"doc-{fixture['fixtureId']}",
            "jobId": f"job-{fixture['fixtureId']}",
            "language": visible["language"],
            "source": {
                "fileName": "frozen-visible-speaker-continuity.wav",
                "sha256": binding["sourceMediaSha256"],
                "durationMs": binding["sourceDurationMs"],
            },
            "speakerPolicy": {
                "mode": "auto",
                "resolvedCount": visible["speakerCount"],
                "speakerIds": visible["speakerIds"],
            },
            "speakers": [
                {"id": speaker_id}
                for speaker_id in visible["speakerIds"]
            ],
        }
    )
    document["segments"] = []
    for item in visible["segments"]:
        segment = _segment(
            item["segmentId"],
            start_ms=item["startMs"],
            speaker_id=item["speakerId"],
            text=item["text"],
        )
        segment.update(
            {
                "endMs": item["endMs"],
                "language": item["language"],
                "speakerScores": [
                    {
                        "speakerId": speaker_id,
                        "score": (
                            0.9 if speaker_id == item["speakerId"] else 0.6
                        ),
                    }
                    for speaker_id in visible["speakerIds"]
                ],
                "speakerMargin": 0.3,
            }
        )
        document["segments"].append(segment)
    document["provenance"] = {
        "offline": True,
        "models": [],
        "frozenTranscriptDocumentSha256": binding[
            "transcriptDocumentSha256"
        ],
        "frozenCandidateLatticeSha256": binding[
            "candidateLatticeSha256"
        ],
    }
    return document


def _speaker_continuity_regression_lattice(document: dict) -> dict:
    segments = document["segments"]
    speaker_ids = document["speakerPolicy"]["speakerIds"]
    source_duration_ms = document["source"]["durationMs"]
    current_turns = [
        {
            "startMs": segment["startMs"],
            "endMs": segment["endMs"],
            "speakerId": segment["speakerId"],
            "overlap": False,
        }
        for segment in segments
    ]
    consolidated_turns = [
        {**turn, "speakerId": speaker_ids[0]}
        for turn in current_turns
    ]
    candidate_groups: dict[str, list[dict]] = {
        "speech-disposition": [
            {
                "scopeId": "media",
                "candidates": [
                    _candidate(
                        {
                            "classification": "transcribable-speech",
                            "startMs": 0,
                            "endMs": source_duration_ms,
                        },
                        current=True,
                    )
                ],
            }
        ],
        "speaker-cardinality-timeline": [
            {
                "scopeId": "media",
                "candidates": [
                    _candidate(
                        {
                            "speakerCount": len(speaker_ids),
                            "speakerIds": speaker_ids,
                            "timelineKind": "current-transcript",
                            "startMs": 0,
                            "endMs": source_duration_ms,
                            "turns": current_turns,
                        },
                        current=True,
                    ),
                    _candidate(
                        {
                            "speakerCount": 1,
                            "speakerIds": [speaker_ids[0]],
                            "timelineKind": "overlap-preserving",
                            "startMs": 0,
                            "endMs": source_duration_ms,
                            "turns": consolidated_turns,
                        },
                        current=False,
                    ),
                    _candidate(
                        {
                            "speakerCount": 1,
                            "speakerIds": [speaker_ids[0]],
                            "timelineKind": "single-speaker",
                            "startMs": 0,
                            "endMs": source_duration_ms,
                            "turns": consolidated_turns,
                        },
                        current=False,
                    ),
                ],
            }
        ],
        "speaker-assignment": [],
        "language-span": [],
        "asr-text": [],
    }
    base_language = document["language"].split("-", 1)[0]
    for segment in segments:
        scope_id = f"segment:{segment['id']}"
        candidate_groups["speaker-assignment"].append(
            {
                "scopeId": scope_id,
                "candidates": [
                    _candidate(
                        {
                            "segmentId": segment["id"],
                            "startMs": segment["startMs"],
                            "endMs": segment["endMs"],
                            "speakerId": speaker_id,
                            "score": (
                                0.9
                                if speaker_id == segment["speakerId"]
                                else 0.6
                            ),
                        },
                        current=speaker_id == segment["speakerId"],
                    )
                    for speaker_id in speaker_ids
                ],
            }
        )
        candidate_groups["language-span"].append(
            {
                "scopeId": scope_id,
                "candidates": [
                    _candidate(
                        {
                            "segmentId": segment["id"],
                            "startMs": segment["startMs"],
                            "endMs": segment["endMs"],
                            "language": language,
                            "confidence": None,
                        },
                        current=language == segment["language"],
                    )
                    for language in (segment["language"], base_language)
                ],
            }
        )
        candidate_set_sha256 = hashlib.sha256(
            segment["id"].encode("utf-8")
        ).hexdigest()
        candidate_groups["asr-text"].append(
            {
                "scopeId": scope_id,
                "candidates": [
                    _candidate(
                        {
                            "segmentId": segment["id"],
                            "startMs": segment["startMs"],
                            "endMs": segment["endMs"],
                            "text": segment["normalizedText"],
                            "language": segment["language"],
                            "sourceCandidateId": (
                                f"asr-{candidate_set_sha256[:24]}"
                            ),
                            "candidateSetSha256": candidate_set_sha256,
                        },
                        current=True,
                    )
                ],
            }
        )
    return build_semantic_candidate_lattice(
        source_media_sha256=document["source"]["sha256"],
        transcript_sha256=canonical_json_sha256(document),
        transcript_schema_version=document["schemaVersion"],
        source_duration_ms=source_duration_ms,
        candidate_groups=candidate_groups,
    )


def _candidate_for(group: dict, predicate) -> str:
    return next(
        item["candidateId"] for item in group["candidates"] if predicate(item)
    )


def _ready_response(lattice: dict) -> dict:
    selections = []
    for domain in lattice["domains"]:
        for group in domain["groups"]:
            if domain["domain"] == "speaker-cardinality-timeline":
                selected = _candidate_for(
                    group,
                    lambda item: item["payload"]["speakerCount"] == 3,
                )
            elif (
                domain["domain"] == "speaker-assignment"
                and group["scopeId"] == "segment:segment-1"
            ):
                selected = _candidate_for(
                    group,
                    lambda item: item["payload"]["speakerId"] == "speaker-3",
                )
            elif (
                domain["domain"] == "language-span"
                and group["scopeId"] == "segment:segment-1"
            ):
                selected = _candidate_for(
                    group,
                    lambda item: item["payload"]["language"] == "es",
                )
            elif (
                domain["domain"] == "asr-text"
                and group["scopeId"] == "segment:segment-1"
            ):
                selected = _candidate_for(
                    group,
                    lambda item: item["payload"]["text"] == "Hola",
                )
            else:
                selected = group["currentCandidateId"]
            eligible = [
                item["candidateId"]
                for item in group["candidates"]
                if item["selectionEligible"]
            ]
            selections.append(
                {
                    "groupId": group["groupId"],
                    "rankedCandidateIds": [
                        selected,
                        *(item for item in eligible if item != selected),
                    ],
                    "reasonCodes": ["CROSS_DOMAIN_EVIDENCE"],
                    "evidenceRefs": [
                        f"candidate-lattice:{lattice['latticeId']}",
                        f"candidate-group:{group['groupId']}",
                        f"candidate:{selected}",
                    ],
                }
            )
    return {
        "latticeId": lattice["latticeId"],
        "latticeSha256": lattice["latticeSha256"],
        "selections": selections,
        "candidateGenerationRequests": [],
    }


def _request_or_select_response(
    lattice: dict,
    *,
    force_request_domains: frozenset[str] = frozenset(),
) -> dict:
    selections = []
    requests = []
    for domain in lattice["domains"]:
        if not domain["groups"]:
            requests.append(
                {
                    "domain": domain["domain"],
                    "groupId": None,
                    "scopeId": "domain",
                    "requestKind": REQUEST_KIND[domain["domain"]],
                    "minimumAlternativeCount": 2,
                    "reasonCodes": ["MISSING_DOMAIN"],
                    "evidenceRefs": [
                        f"candidate-lattice:{lattice['latticeId']}"
                    ],
                }
            )
        for group in domain["groups"]:
            refs = [
                f"candidate-lattice:{lattice['latticeId']}",
                f"candidate-group:{group['groupId']}",
            ]
            if (
                group["status"] == "available"
                and domain["domain"] not in force_request_domains
            ):
                eligible = [
                    item["candidateId"]
                    for item in group["candidates"]
                    if item["selectionEligible"]
                ]
                selections.append(
                    {
                        "groupId": group["groupId"],
                        "rankedCandidateIds": eligible,
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


def _select_current_response(lattice: dict) -> dict:
    selections = []
    for domain in lattice["domains"]:
        for group in domain["groups"]:
            assert group["status"] == "available"
            current = group["currentCandidateId"]
            eligible = [
                item["candidateId"]
                for item in group["candidates"]
                if item["selectionEligible"]
            ]
            selections.append(
                {
                    "groupId": group["groupId"],
                    "rankedCandidateIds": [
                        current,
                        *(item for item in eligible if item != current),
                    ],
                    "reasonCodes": ["CURRENT_CROSS_DOMAIN_CONSISTENT"],
                    "evidenceRefs": [
                        f"candidate-lattice:{lattice['latticeId']}",
                        f"candidate-group:{group['groupId']}",
                    ],
                }
            )
    return {
        "latticeId": lattice["latticeId"],
        "latticeSha256": lattice["latticeSha256"],
        "selections": selections,
        "candidateGenerationRequests": [],
    }


def _artifact(lattice: dict, response: dict) -> dict:
    return build_semantic_job_arbitration(
        job_id="job-semantic-composition",
        lattice=lattice,
        response=response,
        model="fixture-9b",
        provider={
            "id": "fixture-provider",
            "version": "1",
            "networkPolicy": "loopback-only",
        },
        generated_at="2026-07-28T01:00:00Z",
    )


def _positional_response(
    lattice: dict,
    *,
    target_group_ids: list[str],
    selected_by_group: dict[str, str | None],
    translation_texts: list[str | None] | None = None,
) -> dict:
    groups = {
        group["groupId"]: group
        for domain in lattice["domains"]
        for group in domain["groups"]
    }
    choices = []
    for group_id in target_group_ids:
        selected = selected_by_group[group_id]
        if selected is None:
            choices.append(-1)
            continue
        eligible = [
            candidate["candidateId"]
            for candidate in groups[group_id]["candidates"]
            if candidate["selectionEligible"] is True
        ]
        choices.append(eligible.index(selected))
    response: dict = {
        "choiceByPosition": {
            str(index): choice for index, choice in enumerate(choices)
        }
    }
    if translation_texts is not None:
        response["translationTexts"] = translation_texts
    return response


def _prompt_target_group_ids(lattice: dict, prompt: dict) -> list[str]:
    by_domain_scope = {
        (domain["domain"], group["scopeId"]): group["groupId"]
        for domain in lattice["domains"]
        for group in domain["groups"]
    }
    target_groups = sorted(
        prompt["candidateLattice"]["targetGroups"],
        key=lambda item: item["groupPosition"],
    )
    return [
        by_domain_scope[(group["domain"], group["scopeId"])]
        for group in target_groups
    ]


def test_job_arbitration_and_composer_apply_all_high_authority_domains() -> None:
    document = _document()
    lattice = _full_lattice(document)
    arbitration = _artifact(lattice, _ready_response(lattice))

    assert arbitration["status"] == "ready-to-compose"
    assert arbitration["metrics"]["selectedGroupCount"] == 8

    composition = build_semantic_composition(
        document,
        lattice,
        arbitration,
        generated_at="2026-07-28T02:00:00Z",
    )

    assert composition["speakerPolicy"] == {
        "resolvedCount": 3,
        "speakerIds": ["speaker-1", "speaker-2", "speaker-3"],
    }
    assert len(composition["timeline"]["turns"]) == 3
    assert composition["segments"][0]["speakerId"] == "speaker-3"
    assert composition["segments"][0]["language"] == "es"
    assert composition["segments"][0]["finalText"] == "Hola"
    assert (
        composition["binding"]["selectedLatticeSha256"]
        != lattice["latticeSha256"]
    )
    assert composition["humanLocksPreserved"] is True


def test_job_runner_uses_one_complete_high_authority_response() -> None:
    document = _document()
    lattice = _full_lattice(document)
    response = _ready_response(lattice)
    selected_by_group = {
        selection["groupId"]: selection["rankedCandidateIds"][0]
        for selection in response["selections"]
    }

    class CapturingProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.requests: list[dict] = []

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            prompt = json.loads(kwargs["user_prompt"])
            return _positional_response(
                lattice,
                target_group_ids=_prompt_target_group_ids(lattice, prompt),
                selected_by_group=selected_by_group,
            )

    provider = CapturingProvider()

    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "ready-to-compose"
    assert artifact["promptVersion"] == "semantic-job-candidate-arbitration-v17"
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert "total speaker count and complete timeline" in request["system_prompt"]
    assert "human locks" in request["system_prompt"]
    assert "current has no default priority" in request["system_prompt"]
    assert "one speaker cannot produce two independent overlapping" in request[
        "system_prompt"
    ]
    assert (
        "More or fewer speakers, turns, boundaries, or words are not quality evidence"
        in request["system_prompt"]
    )
    assert "multilingual-fidelity calibration rubric" in request["system_prompt"]
    assert "same speaker and language" in request["system_prompt"]
    assert "joined wording still exposes a concrete lexical" in request[
        "system_prompt"
    ]
    assert "request provider-native N-best for every ASR group" in request[
        "system_prompt"
    ]
    assert "singleton candidate is not itself a defect" in request[
        "system_prompt"
    ]
    assert "visibly incompatible with the source language's normal" in request[
        "system_prompt"
    ]
    assert "never invent the replacement text yourself" in request[
        "system_prompt"
    ]
    assert "incumbent speaker labels as truth" in request["system_prompt"]
    assert "Request speaker-cardinality-timeline plus every affected" in request[
        "system_prompt"
    ]
    assert "Never substitute language-span, ASR-text" in request[
        "system_prompt"
    ]
    assert "For speaker Apply" not in request["system_prompt"]
    assert "For speaker The host" not in request["system_prompt"]
    assert "flagged. decisions" not in request["system_prompt"]
    assert "For structural decisions, timestamps are hard evidence" in request[
        "system_prompt"
    ]
    assert "smallest domain that can repair that defect" in request["system_prompt"]
    assert "Evaluate every target domain independently" in request["system_prompt"]
    assert "does not resolve or suppress an independent ASR" in request[
        "system_prompt"
    ]
    assert "final lexical, grammatical, and syntactic-slot" in request[
        "system_prompt"
    ]
    user_prompt = json.loads(request["user_prompt"])
    assert user_prompt["decisionPhase"] == "joint-final"
    assert user_prompt["outputRules"]["currentCandidateHasDefaultPriority"] is False
    assert user_prompt["outputRules"]["requireCrossDomainConsistency"] is True
    assert user_prompt["outputRules"][
        "visibleOrthographyOrScriptDefectRequiresAsrRequest"
    ] is True
    assert user_prompt["semanticCalibration"] == {
        "rubricVersion": "multilingual-fidelity-v2",
        "referenceTranscriptVisible": False,
        "modelIdentityHasPriority": False,
        "challengerPolicy": "smallest-evidence-backed-domain-only",
        "preserve": [
            "complete-spoken-meaning",
            "source-script-and-diacritics",
            "code-switch-boundaries",
            "named-entities-numbers-units-dates-negation",
        ],
        "forbid": [
            "translation-of-source-candidate",
            "style-only-rewrite",
            "fluency-only-correction",
            "unrelated-domain-reopening",
        ],
        "inspect": [
            "segment-internal-lexical-and-grammatical-coherence",
            "same-speaker-same-language-adjacent-joined-coherence",
            "cross-boundary-omission-duplication-or-dangling-phrase",
            "candidate-coverage-and-repairability",
            "source-orthography-script-and-syntactic-slot-compatibility",
            "timestamp-ordered-lexical-continuity-across-speaker-labels",
            "speaker-switches-versus-visible-turn-taking-cues",
            "independent-domain-defect-sweep-after-structural-review",
        ],
        "crossSegmentAsrPolicy": {
            "joinOnlyImmediateTimestampAdjacentSameSpeakerSameLanguage": True,
            "individualFragmentIncompletenessAloneIsDefect": False,
            "joinedConcreteLexicalOrGrammaticalDefectRequiresAsrRequestWhenUnresolved": True,
            "requestEveryAffectedAsrGroup": True,
            "singletonCandidateAloneIsDefect": False,
            "reopenUnrelatedDomains": False,
        },
        "speakerContinuityPolicy": {
            "incumbentSpeakerLabelsAreEvidenceNotTruth": True,
            "inspectCompleteTimestampOrderedVisibleText": True,
            "stableLanguageSingleUtteranceAcrossRapidSpeakerSwitchesSignalsSpeakerChallengerNeed": True,
            "lexicalContinuityAloneProvesSingleSpeaker": False,
            "lexicalContinuityAloneAuthorizesAutomaticMerge": False,
            "resolveTimelineBeforeAssignments": True,
            "requestTimelineAndEveryAffectedAssignmentWhenUnresolved": True,
            "languageAsrOrDispositionCanSubstituteForSpeakerRepair": False,
        },
        "crossDomainReviewPolicy": {
            "evaluateEveryTargetDomainIndependently": True,
            "structuralRequestSuppressesIndependentAsrDefect": False,
            "asrRequestSuppressesIndependentSpeakerDefect": False,
            "allowMultipleBoundedRequestsPerScope": True,
            "finalLexicalSweepAfterStructuralReview": True,
        },
    }
    assert user_prompt["outputRules"]["requestOnlyForConcreteVisibleDefect"] is True
    assert user_prompt["outputRules"]["requestSmallestRelevantDomain"] is True
    assert user_prompt["outputRules"][
        "inspectAdjacentSameSpeakerLanguageAsrContinuity"
    ] is True
    assert user_prompt["outputRules"][
        "requestAllAsrGroupsContributingToJoinedDefect"
    ] is True
    assert user_prompt["outputRules"][
        "singletonAsrCandidateAloneDoesNotAuthorizeRequest"
    ] is True
    assert user_prompt["outputRules"][
        "inspectTranscriptIndependentOfIncumbentSpeakerLabels"
    ] is True
    assert user_prompt["outputRules"][
        "speakerContinuityDefectRequiresTimelineAndAssignmentResolution"
    ] is True
    assert user_prompt["outputRules"][
        "languageAsrOrDispositionCannotSubstituteForSpeakerRepair"
    ] is True
    assert user_prompt["outputRules"][
        "lexicalContinuityAloneDoesNotProveSameSpeaker"
    ] is True
    assert user_prompt["outputRules"][
        "evaluateEveryTargetDomainIndependently"
    ] is True
    assert user_prompt["outputRules"][
        "allowMultipleBoundedRequestsPerScope"
    ] is True
    assert user_prompt["outputRules"][
        "finalLexicalSweepAfterStructuralReview"
    ] is True
    assert user_prompt["decisionProtocol"][
        "choiceByPositionKeysAlignWithGroupPositions"
    ] is True
    assert user_prompt["decisionProtocol"]["responseChoiceField"] == (
        "choiceByPosition"
    )
    assert user_prompt["decisionProtocol"]["candidateChoiceField"] == (
        "choiceIndex"
    )
    assert user_prompt["decisionProtocol"][
        "requestDefaultChallengerIndex"
    ] == -1
    assert all(
        bound["minimum"] == -1 and bound["maximum"] >= 0
        for bound in user_prompt["decisionProtocol"][
            "choiceIndexBoundsByGroupPosition"
        ]
    )
    assert request["response_schema"]["required"] == ["choiceByPosition"]
    assert all(
        item["minimum"] == -1
        for item in request["response_schema"]["properties"][
            "choiceByPosition"
        ]["properties"].values()
    )


def test_prompt_exposes_visible_language_calibration_as_advisory_evidence() -> None:
    document = _document()
    document["segments"][0]["language"] = "yue-Hant-HK"
    document["segments"][0]["normalizedText"] = "from our department"
    lattice = _full_lattice(document)
    context = semantic_job_prompt_context(lattice, document=document)

    segment = context["transcriptSegments"][0]
    calibration = segment["visibleLanguageCalibration"]
    assert calibration["claimedLanguageConflict"] is True
    assert calibration["recommendedDomains"] == ["language-span", "asr-text"]
    assert context["visibleLanguageCalibration"] == {
        "schemaVersion": "1.0.0",
        "heuristicOnly": True,
        "flaggedSegmentCount": 1,
        "flaggedSegmentIds": ["segment-1"],
        "recommendedDomainCounts": {"language-span": 1, "asr-text": 1},
    }
    assert context["visibleSpeakerContinuity"] == {
        "schemaVersion": "1.0.0",
        "heuristicOnly": True,
        "segmentCount": 2,
        "distinctSpeakerCount": 2,
        "speakerLabelRunCount": 2,
        "speakerSwitchCount": 1,
        "distinctLanguageCount": 2,
        "languageSwitchCount": 1,
        "overlapSegmentCount": 0,
        "adjacentPairCount": 1,
        "zeroGapAdjacentPairCount": 1,
        "positiveGapAdjacentPairCount": 0,
        "largestAdjacentGapMs": 0,
        "visibleSpanMs": 2000,
    }


def test_frozen_arabic_cross_segment_lexical_major_requests_both_asr_groups() -> None:
    fixture_path = (
        ROOT
        / "benchmarks"
        / "product_reviews"
        / "development-20260809"
        / "fleurs_ar_eg_validation_090.semantic-v13-major-regression.v1.json"
    )
    fixture = read_json_strict(fixture_path)
    fixture_body = copy.deepcopy(fixture)
    fixture_hash = fixture_body.pop("canonicalSha256")
    assert canonical_json_sha256(fixture_body) == fixture_hash
    assert fixture["visibilityPolicy"] == {
        "caseIdIsAuditOnly": True,
        "referenceTranscriptVisibleToArbitrator": False,
        "expectedDecisionVisibleToArbitrator": False,
        "modelIdentityVisibleToArbitrator": False,
    }

    visible = fixture["visibleInput"]
    binding = fixture["inputBinding"]
    document = _document()
    document.update(
        {
            "documentId": "doc-frozen-visible-arabic-regression",
            "jobId": "job-frozen-visible-arabic-regression",
            "language": visible["language"],
            "source": {
                "fileName": "frozen-visible-arabic.wav",
                "sha256": binding["sourceMediaSha256"],
                "durationMs": binding["sourceDurationMs"],
            },
            "speakerPolicy": {
                "mode": "manual",
                "resolvedCount": visible["speakerCount"],
                "speakerIds": ["speaker-1"],
            },
            "speakers": [{"id": "speaker-1"}],
        }
    )
    document["segments"] = []
    for item in visible["segments"]:
        segment = _segment(
            item["segmentId"],
            start_ms=item["startMs"],
            speaker_id=item["speakerId"],
            text=item["text"],
        )
        segment.update(
            {
                "endMs": item["endMs"],
                "language": item["language"],
                "speakerScores": [
                    {"speakerId": item["speakerId"], "score": 0.9}
                ],
                "speakerMargin": 2.0,
            }
        )
        document["segments"].append(segment)
    document["provenance"] = {
        "offline": True,
        "models": [],
        "frozenTranscriptDocumentSha256": binding[
            "transcriptDocumentSha256"
        ],
        "frozenCandidateLatticeSha256": binding[
            "candidateLatticeSha256"
        ],
    }
    lattice = build_semantic_candidate_lattice_from_document(document)

    class VisibleEvidenceProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.request: dict | None = None

        def generate_json(self, **kwargs):
            self.request = dict(kwargs)
            prompt = json.loads(kwargs["user_prompt"])
            return {
                "choiceByPosition": {
                    str(group["groupPosition"]): (
                        -1
                        if group["domain"] == "asr-text"
                        else next(
                            candidate["choiceIndex"]
                            for candidate in group["candidates"]
                            if candidate["current"] is True
                        )
                    )
                    for group in prompt["candidateLattice"]["targetGroups"]
                }
            }

    provider = VisibleEvidenceProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-semantic-arbitrator",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(document, candidate_lattice=lattice)

    assert provider.request is not None
    serialized_prompt = provider.request["user_prompt"]
    system_prompt = provider.request["system_prompt"]
    prompt = json.loads(serialized_prompt)
    assert fixture["auditBinding"]["caseId"] not in serialized_prompt
    assert fixture["blindReviewFinding"]["reason"] not in serialized_prompt
    assert "provider-native N-best for every ASR group" in system_prompt
    assert prompt["semanticCalibration"]["rubricVersion"] == (
        "multilingual-fidelity-v2"
    )
    assert [
        {
            key: segment[key]
            for key in (
                "segmentId",
                "startMs",
                "endMs",
                "speakerId",
                "language",
                "text",
            )
        }
        for segment in prompt["candidateLattice"]["transcriptSegments"]
    ] == [
        {
            key: segment[key]
            for key in (
                "segmentId",
                "startMs",
                "endMs",
                "speakerId",
                "language",
                "text",
            )
        }
        for segment in visible["segments"]
    ]
    asr_groups = [
        group
        for group in prompt["candidateLattice"]["targetGroups"]
        if group["domain"] == "asr-text"
    ]
    assert {
        group["scopeId"]: group["candidateCoverage"]
        for group in asr_groups
    } == {
        f"segment:{segment['segmentId']}": {
            "selectableCandidateCount": segment[
                "asrSelectableCandidateCount"
            ],
            "distinctSummaryCount": segment["asrDistinctSummaryCount"],
            "singleSelectableCandidate": True,
            "hasDistinctAlternative": False,
        }
        for segment in visible["segments"]
    }
    assert artifact["status"] == "candidate-generation-required"
    assert [
        {
            key: request[key]
            for key in ("domain", "scopeId", "requestKind")
        }
        for request in artifact["candidateGenerationRequests"]
    ] == fixture["blindReviewFinding"]["expectedRequests"]
    assert {
        request["domain"]
        for request in artifact["candidateGenerationRequests"]
    } == {"asr-text"}


@pytest.mark.parametrize(
    "fixture_name",
    [
        "minds_fr_fr_089.semantic-v13-major-regression.v1.json",
        "minds_zh_cn_258.semantic-v13-major-regression.v1.json",
    ],
)
def test_frozen_speaker_continuity_majors_request_structural_domains(
    fixture_name: str,
) -> None:
    fixture_path = (
        ROOT
        / "benchmarks"
        / "product_reviews"
        / "development-20260809"
        / fixture_name
    )
    fixture = read_json_strict(fixture_path)
    fixture_body = copy.deepcopy(fixture)
    fixture_hash = fixture_body.pop("canonicalSha256")
    assert canonical_json_sha256(fixture_body) == fixture_hash
    visible = fixture["visibleInput"]
    binding = fixture["inputBinding"]
    document = _speaker_continuity_regression_document(fixture)
    lattice = _speaker_continuity_regression_lattice(document)
    context = semantic_job_prompt_context(lattice, document=document)
    continuity = context["visibleSpeakerContinuity"]
    assert continuity["segmentCount"] == len(visible["segments"])
    assert continuity["distinctSpeakerCount"] == visible["speakerCount"]
    assert continuity["speakerSwitchCount"] == len(visible["segments"]) - 1
    assert continuity["languageSwitchCount"] == 0
    assert continuity["overlapSegmentCount"] == 0
    assert continuity["zeroGapAdjacentPairCount"] == (
        len(visible["segments"]) - 1
        - (1 if fixture["auditBinding"]["caseId"] == "minds_fr_fr_089" else 0)
    )
    assert continuity["visibleSpanMs"] == (
        visible["segments"][-1]["endMs"]
        - visible["segments"][0]["startMs"]
    )

    class StructuralReviewProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.request: dict | None = None

        def generate_json(self, **kwargs):
            self.request = dict(kwargs)
            prompt = json.loads(kwargs["user_prompt"])
            choices = {}
            for group in prompt["candidateLattice"]["targetGroups"]:
                if group["domain"] in {
                    "speaker-cardinality-timeline",
                    "speaker-assignment",
                }:
                    choices[str(group["groupPosition"])] = -1
                else:
                    choices[str(group["groupPosition"])] = next(
                        candidate["choiceIndex"]
                        for candidate in group["candidates"]
                        if candidate["current"] is True
                    )
            return {"choiceByPosition": choices}

    provider = StructuralReviewProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-semantic-arbitrator",
        context_tokens=65_536,
        output_tokens=4_096,
        batch_size=32,
    ).run(document, candidate_lattice=lattice)

    assert provider.request is not None
    serialized_prompt = provider.request["user_prompt"]
    prompt = json.loads(serialized_prompt)
    system_prompt = provider.request["system_prompt"]
    assert fixture["auditBinding"]["caseId"] not in serialized_prompt
    assert fixture["blindReviewFinding"]["reason"] not in serialized_prompt
    assert "incumbent speaker labels as truth" in system_prompt
    assert "do not automatically merge speakers" in system_prompt
    assert "Never substitute language-span, ASR-text" in system_prompt
    assert prompt["semanticCalibration"]["speakerContinuityPolicy"] == {
        "incumbentSpeakerLabelsAreEvidenceNotTruth": True,
        "inspectCompleteTimestampOrderedVisibleText": True,
        "stableLanguageSingleUtteranceAcrossRapidSpeakerSwitchesSignalsSpeakerChallengerNeed": True,
        "lexicalContinuityAloneProvesSingleSpeaker": False,
        "lexicalContinuityAloneAuthorizesAutomaticMerge": False,
        "resolveTimelineBeforeAssignments": True,
        "requestTimelineAndEveryAffectedAssignmentWhenUnresolved": True,
        "languageAsrOrDispositionCanSubstituteForSpeakerRepair": False,
    }
    prompt_segments = prompt["candidateLattice"]["transcriptSegments"]
    assert [
        (
            segment["segmentId"],
            segment["startMs"],
            segment["endMs"],
            segment["speakerId"],
            segment["language"],
            segment["text"],
        )
        for segment in prompt_segments
    ] == [
        (
            segment["segmentId"],
            segment["startMs"],
            segment["endMs"],
            segment["speakerId"],
            segment["language"],
            segment["text"],
        )
        for segment in visible["segments"]
    ]
    assert prompt["candidateLattice"]["transcriptContextPolicy"][
        "allTargetSegmentsIncluded"
    ] is True
    timeline_groups = [
        group
        for group in prompt["candidateLattice"]["targetGroups"]
        if group["domain"] == "speaker-cardinality-timeline"
    ]
    assert len(timeline_groups) == 1
    assert timeline_groups[0]["candidateCoverage"] == {
        "selectableCandidateCount": visible["timelineSelectableCandidateCount"],
        "distinctSummaryCount": visible["timelineSelectableCandidateCount"],
        "singleSelectableCandidate": False,
        "hasDistinctAlternative": True,
    }
    assignment_groups = [
        group
        for group in prompt["candidateLattice"]["targetGroups"]
        if group["domain"] == "speaker-assignment"
    ]
    assert {
        group["scopeId"]: group["candidateCoverage"][
            "selectableCandidateCount"
        ]
        for group in assignment_groups
    } == {
        f"segment:{segment['segmentId']}": visible[
            "speakerAssignmentSelectableCandidateCountPerSegment"
        ]
        for segment in visible["segments"]
    }
    assert artifact["status"] == "candidate-generation-required"
    actual_requests = {
        (item["domain"], item["scopeId"], item["requestKind"])
        for item in artifact["candidateGenerationRequests"]
    }
    expected_requests = {
        (item["domain"], item["scopeId"], item["requestKind"])
        for item in fixture["blindReviewFinding"]["expectedRequests"]
    }
    assert actual_requests == expected_requests
    assert {
        item["domain"] for item in artifact["candidateGenerationRequests"]
    } == {"speaker-cardinality-timeline", "speaker-assignment"}
    assert {
        item["domain"] for item in artifact["selections"]
        if "domain" in item
    }.issubset({
        "speech-disposition",
        "language-span",
        "asr-text",
    })


def test_runner_restricts_frozen_human_locked_choices_before_provider() -> None:
    document = _document()
    document["segments"][0]["humanLocked"] = True
    lattice = _full_lattice(document)
    locked_language_lattice_group = next(
        group
        for domain in lattice["domains"]
        if domain["domain"] == "language-span"
        for group in domain["groups"]
        if group["scopeId"] == "segment:segment-1"
    )
    assert locked_language_lattice_group["eligibleCandidateCount"] == 2

    class CapturingCurrentProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.requests: list[dict] = []

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            prompt = json.loads(kwargs["user_prompt"])
            return {
                "choiceByPosition": {
                    str(group["groupPosition"]): next(
                        candidate["choiceIndex"]
                        for candidate in group["candidates"]
                        if candidate["current"] is True
                    )
                    for group in prompt["candidateLattice"]["targetGroups"]
                }
            }

    provider = CapturingCurrentProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "ready-to-compose"
    assert len(provider.requests) == 1
    request = provider.requests[0]
    prompt = json.loads(request["user_prompt"])
    groups = prompt["candidateLattice"]["targetGroups"]
    schema_by_position = request["response_schema"]["properties"][
        "choiceByPosition"
    ]["properties"]
    restricted = {
        (group["domain"], group["scopeId"]): group
        for group in groups
        if group.get("humanLockRestricted") is True
    }
    assert set(restricted) == {
        ("speech-disposition", "media"),
        ("speaker-cardinality-timeline", "media"),
        *{
            (domain, "segment:segment-1")
            for domain in _SEGMENT_ATOMIC_DOMAINS_FOR_TEST
        },
    }
    for group in restricted.values():
        assert group["requestDefaultChallengerAllowed"] is False
        assert len(group["candidates"]) == 1
        candidate = group["candidates"][0]
        assert candidate["current"] is True
        assert group["humanLockAllowedCandidateChoiceIndexes"] == [
            candidate["choiceIndex"]
        ]
        assert schema_by_position[str(group["groupPosition"])]["enum"] == [
            candidate["choiceIndex"]
        ]

    unlocked = [
        group for group in groups if group["scopeId"] == "segment:segment-2"
    ]
    assert {group["domain"] for group in unlocked} == (
        _SEGMENT_ATOMIC_DOMAINS_FOR_TEST
    )
    for group in unlocked:
        assert "humanLockRestricted" not in group
        assert group["requestDefaultChallengerAllowed"] is True
        assert len(group["candidates"]) == 2
        assert -1 in schema_by_position[str(group["groupPosition"])]["enum"]

    composition = build_semantic_composition(document, lattice, artifact)
    assert composition["humanLocksPreserved"] is True
    assert composition["segments"][0]["speakerId"] == "speaker-1"
    assert composition["segments"][0]["language"] == "en"
    assert composition["segments"][0]["finalText"] == "Hello"


def test_runner_keeps_timeline_challenger_that_preserves_locked_turn() -> None:
    document = _document()
    document["segments"][1]["humanLocked"] = True
    lattice = _full_lattice(document)

    class CapturingCurrentProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.requests: list[dict] = []

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            prompt = json.loads(kwargs["user_prompt"])
            return {
                "choiceByPosition": {
                    str(group["groupPosition"]): next(
                        candidate["choiceIndex"]
                        for candidate in group["candidates"]
                        if candidate["current"] is True
                    )
                    for group in prompt["candidateLattice"]["targetGroups"]
                }
            }

    provider = CapturingCurrentProvider()
    SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(document, candidate_lattice=lattice)

    request = provider.requests[0]
    prompt = json.loads(request["user_prompt"])
    timeline = next(
        group
        for group in prompt["candidateLattice"]["targetGroups"]
        if group["domain"] == "speaker-cardinality-timeline"
    )
    assert timeline["humanLockRestricted"] is True
    assert timeline["requestDefaultChallengerAllowed"] is False
    assert {candidate["summary"]["timelineKind"] for candidate in timeline["candidates"]} == {
        "current-transcript",
        "challenger",
    }
    allowed = timeline["humanLockAllowedCandidateChoiceIndexes"]
    assert allowed == sorted(
        candidate["choiceIndex"] for candidate in timeline["candidates"]
    )
    timeline_schema = request["response_schema"]["properties"][
        "choiceByPosition"
    ]["properties"][str(timeline["groupPosition"])]
    assert timeline_schema["enum"] == allowed
    assert -1 not in timeline_schema["enum"]


def test_runner_enforces_assignments_supported_by_committed_timeline() -> None:
    document = _document()
    lattice = _full_lattice(document)

    class TimelineAwareProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.requests: list[dict] = []
            self.segment_two_attempts = 0

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            prompt = json.loads(kwargs["user_prompt"])
            groups = sorted(
                prompt["candidateLattice"]["targetGroups"],
                key=lambda item: item["groupPosition"],
            )
            choices: dict[str, int] = {}
            for group in groups:
                candidates = group["candidates"]
                if group["domain"] == "speaker-cardinality-timeline":
                    choice = next(
                        candidate["choiceIndex"]
                        for candidate in candidates
                        if candidate["summary"]["timelineKind"] == "challenger"
                    )
                elif (
                    group["domain"] == "speaker-assignment"
                    and group["scopeId"] == "segment:segment-2"
                ):
                    key = "structurallyCompatibleWithCommittedTimeline"
                    if self.segment_two_attempts == 0:
                        choice = next(
                            candidate["choiceIndex"]
                            for candidate in candidates
                            if candidate[key] is False
                        )
                    else:
                        choice = next(
                            candidate["choiceIndex"]
                            for candidate in candidates
                            if candidate[key] is True
                        )
                    self.segment_two_attempts += 1
                else:
                    choice = 0
                choices[str(group["groupPosition"])] = choice
            return {"choiceByPosition": choices}

    provider = TimelineAwareProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=3,
        max_batch_attempts=2,
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "ready-to-compose"
    assert len(provider.requests) == 4
    segment_two_requests = [
        request
        for request in provider.requests
        if "segment:segment-2"
        in json.loads(request["user_prompt"])["targetScopeIds"]
    ]
    assert len(segment_two_requests) == 2
    first_request = segment_two_requests[0]
    first_prompt = json.loads(first_request["user_prompt"])
    assignment_group = next(
        group
        for group in first_prompt["candidateLattice"]["targetGroups"]
        if group["domain"] == "speaker-assignment"
    )
    compatible = assignment_group[
        "structurallyAllowedCandidateChoiceIndexes"
    ]
    incompatible = [
        candidate["choiceIndex"]
        for candidate in assignment_group["candidates"]
        if candidate["structurallyCompatibleWithCommittedTimeline"] is False
    ]
    assert len(compatible) == 1
    assert len(incompatible) == 1
    position = str(assignment_group["groupPosition"])
    choice_schema = first_request["response_schema"]["properties"][
        "choiceByPosition"
    ]["properties"][position]
    assert choice_schema["enum"] == [-1, *compatible]
    assert incompatible[0] not in choice_schema["enum"]
    bound = first_prompt["decisionProtocol"][
        "choiceIndexBoundsByGroupPosition"
    ][assignment_group["groupPosition"]]
    assert bound["allowedChoiceIndexes"] == [-1, *compatible]
    retry_prompt = json.loads(segment_two_requests[1]["user_prompt"])
    assert retry_prompt["correction"]["validationFailureCode"] == (
        "COMMITTED_TIMELINE_CONFLICT"
    )

    composition = build_semantic_composition(document, lattice, artifact)
    assert composition["segments"][1]["speakerId"] == "speaker-2"


def test_runner_retries_same_batch_timeline_assignment_conflict() -> None:
    document = _document()
    lattice = _full_lattice(document)

    class ConflictingThenConsistentProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.requests: list[dict] = []

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            prompt = json.loads(kwargs["user_prompt"])
            groups = sorted(
                prompt["candidateLattice"]["targetGroups"],
                key=lambda item: item["groupPosition"],
            )
            first_attempt = len(self.requests) == 1
            choices: dict[str, int] = {}
            for group in groups:
                candidates = group["candidates"]
                if first_attempt and group["domain"] == (
                    "speaker-cardinality-timeline"
                ):
                    choice = next(
                        item["choiceIndex"]
                        for item in candidates
                        if item["summary"]["timelineKind"] == "challenger"
                    )
                elif (
                    first_attempt
                    and group["domain"] == "speaker-assignment"
                    and group["scopeId"] == "segment:segment-2"
                ):
                    choice = next(
                        item["choiceIndex"]
                        for item in candidates
                        if item["summary"]["speakerId"] == "speaker-1"
                    )
                else:
                    choice = next(
                        item["choiceIndex"]
                        for item in candidates
                        if item["current"] is True
                    )
                choices[str(group["groupPosition"])] = choice
            return {"choiceByPosition": choices}

    provider = ConflictingThenConsistentProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
        max_batch_attempts=2,
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "ready-to-compose"
    assert len(provider.requests) == 2
    retry_prompt = json.loads(provider.requests[1]["user_prompt"])
    assert retry_prompt["correction"]["validationFailureCode"] == (
        "COMMITTED_TIMELINE_CONFLICT"
    )
    composition = build_semantic_composition(document, lattice, artifact)
    assert [segment["speakerId"] for segment in composition["segments"]] == [
        "speaker-1",
        "speaker-2",
    ]


def test_runner_rejects_legacy_live_full_response_that_bypasses_consistency() -> None:
    document = _document()
    lattice = _full_lattice(document)

    with pytest.raises(WorkerError) as caught:
        SemanticJobArbitrationRunner(
            provider=MappingLocalLLMProvider([_ready_response(lattice)]),
            model="fixture-9b",
            context_tokens=32_768,
            output_tokens=4_096,
            batch_size=8,
            max_batch_attempts=1,
        ).run(document, candidate_lattice=lattice)

    assert caught.value.code == "SEMANTIC_JOB_PROVIDER_FAILED"
    assert "STRICT_JSON_OR_SCHEMA_INVALID" in caught.value.details["reason"]
    assert caught.value.details["attemptDiagnostics"][0][
        "validationFailureCode"
    ] == "STRICT_JSON_OR_SCHEMA_INVALID"


def test_runner_rearbitrates_segment_triads_when_carried_timeline_is_missing() -> None:
    document = _document()
    lattice = _full_lattice(document)
    previous = _artifact(
        lattice,
        _request_or_select_response(
            lattice,
            force_request_domains=frozenset(
                {"speaker-cardinality-timeline"}
            ),
        ),
    )
    selected_by_group = {
        group["groupId"]: group["currentCandidateId"]
        for domain in lattice["domains"]
        for group in domain["groups"]
        if group["status"] == "available"
    }

    class CapturingProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.prompts: list[dict] = []

        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            self.prompts.append(prompt)
            target_ids = _prompt_target_group_ids(lattice, prompt)
            return _positional_response(
                lattice,
                target_group_ids=target_ids,
                selected_by_group=selected_by_group,
            )

    provider = CapturingProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(
        document,
        candidate_lattice=lattice,
        carried_lattice=lattice,
        carried_arbitration=previous,
    )

    assert artifact["status"] == "ready-to-compose"
    assert len(provider.prompts) == 1
    prompt = provider.prompts[0]
    target_groups = prompt["candidateLattice"]["targetGroups"]
    assert {
        (group["domain"], group["scopeId"]) for group in target_groups
    } == {
        ("speaker-cardinality-timeline", "media"),
        *{
            (domain, f"segment:segment-{segment_number}")
            for segment_number in (1, 2)
            for domain in _SEGMENT_ATOMIC_DOMAINS_FOR_TEST
        },
    }
    assert [
        item["domain"] for item in prompt["committedSelections"]
    ] == ["speech-disposition"]
    assert prompt["committedRequests"] == [
        {
            "domain": "speaker-cardinality-timeline",
            "scopeId": "media",
            "requestKind": "timeline-challenger",
        }
    ]
    assert prompt["outputRules"][
        "committedRequestsAreUnresolvedEvidenceGaps"
    ] is True
    build_semantic_composition(document, lattice, artifact)


def test_challenger_request_does_not_raise_minimum_above_generator_contract() -> None:
    document = _document()
    lattice = _full_lattice(document)

    class RequestTimelineProvider(MappingLocalLLMProvider):
        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            return {
                "choiceIndexes": [
                    -1 if group["domain"] == "speaker-cardinality-timeline" else 0
                    for group in prompt["candidateLattice"]["targetGroups"]
                ]
            }

    artifact = SemanticJobArbitrationRunner(
        provider=RequestTimelineProvider([]),
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(document, candidate_lattice=lattice)

    request = artifact["candidateGenerationRequests"][0]
    timeline_group = next(
        domain["groups"][0]
        for domain in lattice["domains"]
        if domain["domain"] == "speaker-cardinality-timeline"
    )
    assert timeline_group["eligibleCandidateCount"] > 1
    assert request["domain"] == "speaker-cardinality-timeline"
    assert request["minimumAlternativeCount"] == 2


def test_singleton_groups_remain_model_visible_with_translation() -> None:
    document = _document()
    for segment in document["segments"]:
        segment["speakerScores"] = [
            {"speakerId": segment["speakerId"], "score": 1.0}
        ]
        segment["speakerMargin"] = 1.0
    lattice = build_semantic_candidate_lattice_from_document(document)

    class TranslationProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.prompts: list[dict] = []

        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            self.prompts.append(prompt)
            return {
                "choiceIndexes": [0] * prompt["targetGroupCount"],
                "translationTexts": ["你好", "世界"],
            }

    provider = TranslationProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
        translation_targets=("zh-CN",),
    ).run(document, candidate_lattice=lattice)

    assert len(provider.prompts) == 1
    target_groups = provider.prompts[0]["candidateLattice"]["targetGroups"]
    available_group_count = sum(
        domain["availableGroupCount"] for domain in lattice["domains"]
    )
    assert len(target_groups) == available_group_count
    assert sum(group["domain"] == "asr-text" for group in target_groups) == 2
    assert [item["text"] for item in artifact["translations"]] == [
        "你好",
        "世界",
    ]


def test_job_runner_decides_structure_before_scope_atomic_segment_triads() -> None:
    document = _document()
    lattice = _full_lattice(document)
    ready = _ready_response(lattice)
    selection_by_group = {
        selection["groupId"]: selection["rankedCandidateIds"][0]
        for selection in ready["selections"]
    }
    group_domain = {
        group["groupId"]: domain["domain"]
        for domain in lattice["domains"]
        for group in domain["groups"]
    }

    class ScopeAwareProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.requests: list[dict] = []

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            prompt = json.loads(kwargs["user_prompt"])
            target_group_ids = _prompt_target_group_ids(lattice, prompt)
            return {
                "latticeId": lattice["latticeId"],
                "latticeSha256": lattice["latticeSha256"],
                "decisions": [
                    {
                        "groupId": group_id,
                        "action": "select",
                        "selectedCandidateId": selection_by_group[group_id],
                        "requestKind": None,
                    }
                    for group_id in target_group_ids
                ],
            }

    provider = ScopeAwareProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=2,
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "ready-to-compose"
    assert len(provider.requests) == 3
    prompts = [
        json.loads(request["user_prompt"])
        for request in provider.requests
    ]
    assert prompts[0]["decisionPhase"] == "global-structure"
    assert prompts[0]["targetScopeIds"] == ["media"]
    assert {
        group_domain[group_id]
        for group_id in _prompt_target_group_ids(lattice, prompts[0])
    } == {
        "speech-disposition",
        "speaker-cardinality-timeline",
    }
    for prompt in prompts[1:]:
        assert prompt["decisionPhase"] == "segment-joint"
        assert len(prompt["targetScopeIds"]) == 1
        assert {
            group_domain[group_id]
            for group_id in _prompt_target_group_ids(lattice, prompt)
        } == {
            "speaker-assignment",
            "language-span",
            "asr-text",
        }
        assert {
            item["domain"] for item in prompt["committedSelections"]
        }.issuperset(
            {
                "speech-disposition",
                "speaker-cardinality-timeline",
            }
        )
        assert (
            prompt["outputRules"][
                "committedSelectionsAreAuthoritativeContext"
            ]
            is True
        )


def test_job_runner_carries_unresolved_requests_across_batches() -> None:
    """A structural challenger request must remain visible to later batches."""

    document = _document()
    lattice = _full_lattice(document)
    selected_by_group = {
        group["groupId"]: group["currentCandidateId"]
        for domain in lattice["domains"]
        for group in domain["groups"]
        if group["status"] == "available"
    }

    class RequestThenSelectProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.prompts: list[dict] = []
            self.system_prompts: list[str] = []

        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            self.prompts.append(prompt)
            self.system_prompts.append(str(kwargs["system_prompt"]))
            target_ids = _prompt_target_group_ids(lattice, prompt)
            choices = dict(selected_by_group)
            if len(self.prompts) == 1:
                timeline_id = next(
                    group_id
                    for group_id in target_ids
                    if next(
                        domain["domain"]
                        for domain in lattice["domains"]
                        if any(
                            group["groupId"] == group_id
                            for group in domain["groups"]
                        )
                    )
                    == "speaker-cardinality-timeline"
                )
                choices[timeline_id] = None
            return _positional_response(
                lattice,
                target_group_ids=target_ids,
                selected_by_group=choices,
            )

    provider = RequestThenSelectProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=2,
    ).run(document, candidate_lattice=lattice)

    assert len(provider.prompts) == 3
    assert "committedRequests" not in provider.prompts[0]
    expected = {
        "domain": "speaker-cardinality-timeline",
        "scopeId": "media",
        "requestKind": "timeline-challenger",
    }
    assert expected in provider.prompts[1]["committedRequests"]
    assert expected in provider.prompts[2]["committedRequests"]
    assert provider.prompts[1]["outputRules"][
        "committedRequestsAreUnresolvedEvidenceGaps"
    ] is True
    assert provider.prompts[1]["outputRules"][
        "committedRequestsAreNotFacts"
    ] is True
    transcript_policy = provider.prompts[1]["candidateLattice"][
        "transcriptContextPolicy"
    ]
    assert transcript_policy["committedStructuralFullTranscriptRequested"] is True
    assert transcript_policy["allTargetSegmentsIncluded"] is True
    assert transcript_policy["includedSegmentCount"] == transcript_policy[
        "totalSegmentCount"
    ]
    assert (
        "treat every speaker-assignment group in that run as affected"
        in provider.system_prompts[1]
    )
    assert artifact["status"] == "candidate-generation-required"


def test_batch_eight_exposes_complete_active_speaker_continuity_run() -> None:
    fixture = read_json_strict(
        ROOT
        / "benchmarks"
        / "product_reviews"
        / "development-20260809"
        / "minds_fr_fr_089.semantic-v13-major-regression.v1.json"
    )
    document = _speaker_continuity_regression_document(fixture)
    lattice = _speaker_continuity_regression_lattice(document)
    expected_scopes = [
        f"segment:{segment['segmentId']}"
        for segment in fixture["visibleInput"]["segments"]
    ]

    class RunAwareProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.prompts: list[dict] = []
            self.native_assignment_requests: list[str] = []

        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            self.prompts.append(prompt)
            active_scopes = {
                scope_id
                for run in prompt["candidateLattice"][
                    "activeSpeakerContinuityRuns"
                ]
                for scope_id in run["unresolvedAssignmentScopes"]
            }
            choices: dict[str, int] = {}
            for group in prompt["candidateLattice"]["targetGroups"]:
                choice = next(
                    candidate["choiceIndex"]
                    for candidate in group["candidates"]
                    if candidate["current"] is True
                )
                if group["domain"] == "speaker-cardinality-timeline":
                    choice = -1
                elif (
                    group["domain"] == "speaker-assignment"
                    and group["scopeId"] in active_scopes
                ):
                    choice = -1
                    self.native_assignment_requests.append(group["scopeId"])
                choices[str(group["groupPosition"])] = choice
            return {"choiceByPosition": choices}

    provider = RunAwareProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-semantic-arbitrator",
        context_tokens=65_536,
        output_tokens=4_096,
        batch_size=8,
    ).run(document, candidate_lattice=lattice)

    assert provider.prompts[0]["decisionPhase"] == "global-structure"
    assert provider.prompts[0]["candidateLattice"][
        "activeSpeakerContinuityRuns"
    ] == []
    segment_prompts = provider.prompts[1:]
    assert len(segment_prompts) > 1
    for prompt in segment_prompts:
        runs = prompt["candidateLattice"]["activeSpeakerContinuityRuns"]
        assert len(runs) == 1
        run = runs[0]
        assert run["orderedScopes"] == expected_scopes
        assert run["barriers"] == []
        assert run["unresolvedTimelineScopes"] == ["media"]
        assert run["unresolvedAssignmentScopes"] == expected_scopes
        assert set(run["requestedAssignmentScopes"]).issubset(expected_scopes)
        membership = run["currentBatchMembership"]
        assert membership["targetAssignmentScopes"] == prompt[
            "targetScopeIds"
        ]
        assert prompt["outputRules"][
            "activeSpeakerContinuityRunsAreAdvisory"
        ] is True
        assert prompt["outputRules"][
            "activeRunContextDoesNotProveSpeakerIdentity"
        ] is True

    assert provider.native_assignment_requests == expected_scopes
    assert {
        request["scopeId"]
        for request in artifact["candidateGenerationRequests"]
        if request["domain"] == "speaker-assignment"
    } == set(expected_scopes)
    assert artifact["status"] == "candidate-generation-required"


def test_batch_eight_keeps_independent_asr_request_with_speaker_requests() -> None:
    fixture = read_json_strict(
        ROOT
        / "benchmarks"
        / "product_reviews"
        / "development-20260809"
        / "fleurs_id_id_validation_003.semantic-v14-major-regression.v1.json"
    )
    document = _speaker_continuity_regression_document(fixture)
    lattice = _speaker_continuity_regression_lattice(document)
    defective_scope = "segment:window-000001.speaker-run-03"

    class IndependentDomainProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.prompts: list[dict] = []
            self.native_request_decisions: list[tuple[str, str, str]] = []

        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            self.prompts.append(prompt)
            choices: dict[str, int] = {}
            for group in prompt["candidateLattice"]["targetGroups"]:
                choice = next(
                    candidate["choiceIndex"]
                    for candidate in group["candidates"]
                    if candidate["current"] is True
                )
                if group["domain"] == "speaker-cardinality-timeline":
                    choice = -1
                elif (
                    group["domain"] == "speaker-assignment"
                    and prompt["candidateLattice"][
                        "activeSpeakerContinuityRuns"
                    ]
                ):
                    choice = -1
                elif (
                    group["domain"] == "asr-text"
                    and group["scopeId"] == defective_scope
                ):
                    choice = -1
                if choice == -1:
                    request_kind = {
                        "speaker-cardinality-timeline": "timeline-challenger",
                        "speaker-assignment": "speaker-assignment-challenger",
                        "asr-text": "provider-native-nbest",
                    }[group["domain"]]
                    self.native_request_decisions.append(
                        (group["domain"], group["scopeId"], request_kind)
                    )
                choices[str(group["groupPosition"])] = choice
            return {"choiceByPosition": choices}

    provider = IndependentDomainProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-semantic-arbitrator",
        context_tokens=65_536,
        output_tokens=4_096,
        batch_size=8,
    ).run(document, candidate_lattice=lattice)

    expected = {
        (item["domain"], item["scopeId"], item["requestKind"])
        for item in fixture["blindReviewFinding"]["expectedRequests"]
    }
    actual = {
        (item["domain"], item["scopeId"], item["requestKind"])
        for item in artifact["candidateGenerationRequests"]
    }
    assert actual == expected
    assert set(provider.native_request_decisions) == expected

    defective_prompt = next(
        prompt
        for prompt in provider.prompts
        if any(
            group["domain"] == "asr-text"
            and group["scopeId"] == defective_scope
            for group in prompt["candidateLattice"]["targetGroups"]
        )
    )
    target_domains = {
        group["domain"]
        for group in defective_prompt["candidateLattice"]["targetGroups"]
        if group["scopeId"] == defective_scope
    }
    assert target_domains == {
        "speaker-assignment",
        "language-span",
        "asr-text",
    }
    assert defective_prompt["semanticCalibration"][
        "crossDomainReviewPolicy"
    ] == {
        "evaluateEveryTargetDomainIndependently": True,
        "structuralRequestSuppressesIndependentAsrDefect": False,
        "asrRequestSuppressesIndependentSpeakerDefect": False,
        "allowMultipleBoundedRequestsPerScope": True,
        "finalLexicalSweepAfterStructuralReview": True,
    }
    assert defective_prompt["outputRules"][
        "allowMultipleBoundedRequestsPerScope"
    ] is True
    assert artifact["status"] == "candidate-generation-required"


def _continuity_guard_document(rows: list[dict]) -> dict:
    document = _document()
    speaker_ids = list(dict.fromkeys(str(row["speakerId"]) for row in rows))
    document.update(
        {
            "language": "fr-FR",
            "source": {
                **document["source"],
                "durationMs": max(int(row["endMs"]) for row in rows),
            },
            "speakerPolicy": {
                "mode": "auto",
                "resolvedCount": len(speaker_ids),
                "speakerIds": speaker_ids,
            },
            "speakers": [{"id": speaker_id} for speaker_id in speaker_ids],
        }
    )
    document["segments"] = []
    for row in rows:
        segment = _segment(
            str(row["id"]),
            start_ms=int(row["startMs"]),
            speaker_id=str(row["speakerId"]),
            text=str(row["text"]),
        )
        segment.update(
            {
                "endMs": int(row["endMs"]),
                "language": str(row.get("language") or "fr-FR"),
                "overlapping": bool(row.get("overlapping", False)),
                "speakerScores": [
                    {
                        "speakerId": speaker_id,
                        "score": (
                            0.9 if speaker_id == row["speakerId"] else 0.6
                        ),
                    }
                    for speaker_id in speaker_ids
                ],
            }
        )
        document["segments"].append(segment)
    return document


def _continuity_guard_response(
    lattice: dict,
    *,
    assignment_anchor_scopes: tuple[str, ...],
    timeline_request_kind: str = "timeline-challenger",
) -> dict:
    response = _select_current_response(lattice)
    groups = {
        (domain["domain"], group["scopeId"]): group
        for domain in lattice["domains"]
        for group in domain["groups"]
    }
    requested = [
        (
            groups[("speaker-cardinality-timeline", "media")],
            "speaker-cardinality-timeline",
            timeline_request_kind,
        ),
        *[
            (
                groups[("speaker-assignment", scope_id)],
                "speaker-assignment",
                "speaker-assignment-challenger",
            )
            for scope_id in assignment_anchor_scopes
        ],
    ]
    requested_group_ids = {str(group["groupId"]) for group, _, _ in requested}
    response["selections"] = [
        selection
        for selection in response["selections"]
        if selection["groupId"] not in requested_group_ids
    ]
    response["candidateGenerationRequests"] = [
        {
            "domain": domain,
            "groupId": group["groupId"],
            "scopeId": group["scopeId"],
            "requestKind": request_kind,
            "minimumAlternativeCount": 2,
            "reasonCodes": ["SEMANTIC_REQUESTED_CHALLENGER"],
            "evidenceRefs": [
                f"candidate-lattice:{lattice['latticeId']}",
                f"candidate-group:{group['groupId']}",
            ],
        }
        for group, domain, request_kind in requested
    ]
    return response


def _assignment_group_ids(lattice: dict) -> set[str]:
    return {
        str(group["groupId"])
        for domain in lattice["domains"]
        if domain["domain"] == "speaker-assignment"
        for group in domain["groups"]
    }


def test_structural_continuity_guard_completes_assignment_evidence_gap() -> None:
    """One anchored assignment request expands to its complete visible run."""

    document = _continuity_guard_document(
        [
            {
                "id": "segment-1",
                "startMs": 0,
                "endMs": 900,
                "speakerId": "speaker-1",
                "text": "Bonjour je vous",
            },
            {
                "id": "segment-2",
                "startMs": 900,
                "endMs": 1_800,
                "speakerId": "speaker-2",
                "text": "contactais pour savoir",
            },
            {
                "id": "segment-3",
                "startMs": 1_800,
                "endMs": 2_700,
                "speakerId": "speaker-3",
                "text": "si la carte",
            },
        ]
    )
    lattice = _speaker_continuity_regression_lattice(document)
    response = _continuity_guard_response(
        lattice,
        assignment_anchor_scopes=("segment:segment-1",),
    )

    _complete_structural_continuity_requests(
        response,
        lattice=lattice,
        document=document,
        requestable_group_ids=_assignment_group_ids(lattice),
    )
    artifact = _artifact(lattice, response)

    assert artifact["status"] == "candidate-generation-required"
    assert {
        (item["domain"], item["scopeId"], item["requestKind"])
        for item in artifact["candidateGenerationRequests"]
    } == {
        (
            "speaker-cardinality-timeline",
            "media",
            "timeline-challenger",
        ),
        *{
            (
                "speaker-assignment",
                f"segment:segment-{index}",
                "speaker-assignment-challenger",
            )
            for index in (1, 2, 3)
        },
    }
    assert {
        item["scopeId"]
        for item in artifact["candidateGenerationRequests"]
        if item["reasonCodes"] == ["DETERMINISTIC_SPEAKER_CONTINUITY_GAP"]
    } == {"segment:segment-2", "segment:segment-3"}


def test_structural_continuity_guard_accepts_repeated_label_inside_run() -> None:
    document = _continuity_guard_document(
        [
            {
                "id": f"segment-{index}",
                "startMs": (index - 1) * 900,
                "endMs": index * 900,
                "speakerId": speaker_id,
                "text": text,
            }
            for index, (speaker_id, text) in enumerate(
                (
                    ("speaker-1", "Bonjour"),
                    ("speaker-1", "je vous"),
                    ("speaker-2", "contactais"),
                    ("speaker-1", "pour savoir"),
                ),
                start=1,
            )
        ]
    )
    lattice = _speaker_continuity_regression_lattice(document)
    response = _continuity_guard_response(
        lattice,
        assignment_anchor_scopes=("segment:segment-3",),
    )

    _complete_structural_continuity_requests(
        response,
        lattice=lattice,
        document=document,
        requestable_group_ids=_assignment_group_ids(lattice),
    )
    artifact = _artifact(lattice, response)

    assert {
        item["scopeId"]
        for item in artifact["candidateGenerationRequests"]
        if item["domain"] == "speaker-assignment"
    } == {f"segment:segment-{index}" for index in range(1, 5)}


def test_structural_continuity_guard_only_completes_anchored_local_run() -> None:
    document = _continuity_guard_document(
        [
            {
                "id": "prefix",
                "startMs": 0,
                "endMs": 600,
                "speakerId": "speaker-1",
                "language": "en-US",
                "text": "Welcome.",
            },
            {
                "id": "run-1",
                "startMs": 600,
                "endMs": 1_200,
                "speakerId": "speaker-1",
                "text": "Bonjour je",
            },
            {
                "id": "run-2",
                "startMs": 1_200,
                "endMs": 1_800,
                "speakerId": "speaker-2",
                "text": "vous contacte",
            },
            {
                "id": "run-3",
                "startMs": 1_800,
                "endMs": 2_400,
                "speakerId": "speaker-3",
                "text": "pour savoir.",
            },
            {
                "id": "suffix-1",
                "startMs": 2_400,
                "endMs": 3_000,
                "speakerId": "speaker-2",
                "text": "Merci",
            },
            {
                "id": "suffix-2",
                "startMs": 3_000,
                "endMs": 3_600,
                "speakerId": "speaker-1",
                "text": "beaucoup",
            },
        ]
    )
    lattice = _speaker_continuity_regression_lattice(document)
    response = _continuity_guard_response(
        lattice,
        assignment_anchor_scopes=("segment:run-2",),
    )

    _complete_structural_continuity_requests(
        response,
        lattice=lattice,
        document=document,
        requestable_group_ids=_assignment_group_ids(lattice),
    )
    artifact = _artifact(lattice, response)

    assert {
        item["scopeId"]
        for item in artifact["candidateGenerationRequests"]
        if item["domain"] == "speaker-assignment"
    } == {"segment:run-1", "segment:run-2", "segment:run-3"}


@pytest.mark.parametrize(
    "barrier",
    [
        "unknown-language",
        "multiple-language",
        "explicit-overlap",
        "timestamp-overlap",
        "language-change",
        "long-gap",
        "strong-terminal",
    ],
)
def test_structural_continuity_guard_does_not_cross_run_barrier(
    barrier: str,
) -> None:
    document = _continuity_guard_document(
        [
            {
                "id": f"segment-{index}",
                "startMs": (index - 1) * 900,
                "endMs": index * 900,
                "speakerId": f"speaker-{((index - 1) % 3) + 1}",
                "text": f"fragment {index}",
            }
            for index in range(1, 6)
        ]
    )
    lattice = _speaker_continuity_regression_lattice(document)
    response = _continuity_guard_response(
        lattice,
        assignment_anchor_scopes=("segment:segment-1",),
    )
    before = copy.deepcopy(response)
    if barrier == "unknown-language":
        document["segments"][2]["language"] = "und"
    elif barrier == "multiple-language":
        document["segments"][2]["language"] = "mul"
    elif barrier == "explicit-overlap":
        document["segments"][2]["overlapping"] = True
    elif barrier == "timestamp-overlap":
        document["segments"][2]["startMs"] = 1_700
    elif barrier == "language-change":
        document["segments"][2]["language"] = "es-ES"
    elif barrier == "long-gap":
        for segment in document["segments"][2:]:
            segment["startMs"] += 1_301
            segment["endMs"] += 1_301
    else:
        document["segments"][1]["normalizedText"] = "fragment 2."

    _complete_structural_continuity_requests(
        response,
        lattice=lattice,
        document=document,
        requestable_group_ids=_assignment_group_ids(lattice),
    )

    assert response == before
    _artifact(lattice, response)


@pytest.mark.parametrize(
    ("assignment_anchor_scopes", "timeline_request_kind"),
    [
        ((), "timeline-challenger"),
        (("segment:segment-1",), "boundary-recompute"),
    ],
)
def test_structural_continuity_guard_requires_exact_request_anchors(
    assignment_anchor_scopes: tuple[str, ...],
    timeline_request_kind: str,
) -> None:
    document = _continuity_guard_document(
        [
            {
                "id": f"segment-{index}",
                "startMs": (index - 1) * 900,
                "endMs": index * 900,
                "speakerId": f"speaker-{index}",
                "text": f"fragment {index}",
            }
            for index in range(1, 4)
        ]
    )
    lattice = _speaker_continuity_regression_lattice(document)
    response = _continuity_guard_response(
        lattice,
        assignment_anchor_scopes=assignment_anchor_scopes,
        timeline_request_kind=timeline_request_kind,
    )
    before = copy.deepcopy(response)

    _complete_structural_continuity_requests(
        response,
        lattice=lattice,
        document=document,
        requestable_group_ids=_assignment_group_ids(lattice),
    )

    assert response == before
    _artifact(lattice, response)


def test_semantic_transcript_context_keeps_targets_and_bounds_global_text() -> None:
    segments = [
        {"segmentId": f"segment-{index}", "text": str(index)}
        for index in range(40)
    ]

    local, local_policy = _bounded_transcript_context(
        segments,
        scope_ids=["segment:segment-10"],
    )
    global_sample, global_policy = _bounded_transcript_context(
        segments,
        scope_ids=["media"],
    )

    assert [item["segmentId"] for item in local] == [
        "segment-8",
        "segment-9",
        "segment-10",
        "segment-11",
        "segment-12",
    ]
    assert local_policy["allTargetSegmentsIncluded"] is True
    assert local_policy["mode"] == "target-segments-with-adjacent-context"
    assert len(global_sample) == 8
    assert global_sample[0]["segmentId"] == "segment-0"
    assert global_sample[-1]["segmentId"] == "segment-39"
    assert global_policy["mode"] == "uniform-global-sample"


def test_semantic_transcript_context_keeps_compact_complete_utterances() -> None:
    segments = [
        {"segmentId": f"segment-{index}", "text": text}
        for index, text in enumerate(
            [
                "Bonjour je vous",
                "contactais pour savoir",
                "si la",
                "carte que",
                "j'ai",
                "dans votre banque",
                "pourrait marcher à l'étranger notamment",
                "si je pars en vacances ou si",
                "je pars faire des études",
            ]
        )
    ]

    selected, policy = _bounded_transcript_context(
        segments,
        scope_ids=["media"],
    )

    assert [item["segmentId"] for item in selected] == [
        item["segmentId"] for item in segments
    ]
    assert policy["mode"] == "complete-compact-transcript"
    assert policy["includedSegmentCount"] == 9


def test_job_runner_co_generates_translation_and_business_reuses_without_llm(
    tmp_path: Path,
) -> None:
    document = _document()
    lattice = _full_lattice(document)
    ready = _ready_response(lattice)
    groups = {
        group["groupId"]: {**group, "domain": domain["domain"]}
        for domain in lattice["domains"]
        for group in domain["groups"]
    }
    candidates = {
        candidate["candidateId"]: candidate
        for group in groups.values()
        for candidate in group["candidates"]
    }
    selected_by_group = {
        selection["groupId"]: selection["rankedCandidateIds"][0]
        for selection in ready["selections"]
    }
    translated_text = {
        "segment-1": "你好",
        "segment-2": "世界",
    }
    class CapturingProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.requests: list[dict] = []

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            prompt = json.loads(kwargs["user_prompt"])
            target_group_ids = _prompt_target_group_ids(lattice, prompt)
            translation_texts = []
            for slot in prompt["translationSlots"]:
                group_id = target_group_ids[slot["groupPosition"]]
                selected = candidates[selected_by_group[group_id]]
                translation_texts.append(
                    translated_text[selected["payload"]["segmentId"]]
                )
            return _positional_response(
                lattice,
                target_group_ids=target_group_ids,
                selected_by_group=selected_by_group,
                translation_texts=translation_texts,
            )

    semantic_provider = CapturingProvider()
    arbitration = SemanticJobArbitrationRunner(
        provider=semantic_provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
        translation_targets=("zh-CN",),
    ).run(document, candidate_lattice=lattice)

    assert len(semantic_provider.requests) == 1
    request = semantic_provider.requests[0]
    prompt = json.loads(request["user_prompt"])
    assert prompt["translationTargets"] == ["zh-CN"]
    assert prompt["outputRules"]["translateSelectedAsrInSameResponse"] is True
    assert request["response_schema"]["required"] == [
        "choiceByPosition",
        "translationTexts",
    ]
    assert set(request["response_schema"]["properties"]) == {
        "choiceByPosition",
        "translationTexts",
    }
    assert prompt["translationSlots"]
    assert all(
        set(slot) == {
            "groupPosition",
            "targetLanguage",
            "sourceCandidates",
        }
        and slot["targetLanguage"] == "zh-CN"
        and slot["sourceCandidates"]
        for slot in prompt["translationSlots"]
    )
    assert arbitration["translationTargets"] == ["zh-CN"]
    assert {
        (item["segmentId"], item["text"])
        for item in arbitration["translations"]
    } == {
        ("segment-1", "你好"),
        ("segment-2", "世界"),
    }
    assert {
        item["sourceTextSha256"]
        for item in arbitration["translations"]
    } == {
        hashlib.sha256(
            candidates[item["selectedCandidateId"]]["payload"]["text"].encode(
                "utf-8"
            )
        ).hexdigest()
        for item in arbitration["translations"]
    }

    composition = build_semantic_composition(document, lattice, arbitration)
    delivery = compose_transcript_document(
        document,
        composition,
        input_lattice=lattice,
        arbitration_artifact=arbitration,
    )

    class NoSecondCallProvider(MappingLocalLLMProvider):
        def generate_json(self, **kwargs):
            raise AssertionError("translation must not make a second LLM call")

    artifacts = BusinessProcessingRunner(
        provider=NoSecondCallProvider([])
    ).run(
        delivery,
        output_directory=tmp_path,
        config=BusinessProcessingConfig(
            translation_targets=("zh-CN",),
            model="fixture-9b",
        ),
        semantic_arbitration=arbitration,
    )
    translation_path = next(
        path for path in artifacts if path.name == "translation-zh-CN.v1.json"
    )
    translation = json.loads(translation_path.read_text(encoding="utf-8"))
    assert translation["promptVersion"] == (
        "semantic-job-candidate-arbitration-v17"
    )
    assert [item["text"] for item in translation["segments"]] == [
        "你好",
        "世界",
    ]
    manifest = json.loads(
        (tmp_path / "business" / "business-manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["completeness"]["translationExecution"] == {
        "zh-CN": "semantic-co-generation"
    }


def test_positional_response_rejects_out_of_range_choice() -> None:
    document = _document()
    lattice = _full_lattice(document)

    class InvalidProvider(MappingLocalLLMProvider):
        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            return {
                "choiceIndexes": [99] * prompt["targetGroupCount"]
            }

    with pytest.raises(WorkerError) as captured:
        SemanticJobArbitrationRunner(
            provider=InvalidProvider([]),
            model="fixture-9b",
            context_tokens=32_768,
            output_tokens=4_096,
            batch_size=32,
            max_batch_attempts=1,
        ).run(document, candidate_lattice=lattice)

    assert captured.value.code == "SEMANTIC_JOB_PROVIDER_FAILED"
    assert captured.value.details["attemptDiagnostics"][0][
        "validationFailureCode"
    ] == "STRICT_JSON_OR_SCHEMA_INVALID"


def test_positional_response_requires_every_translation_slot() -> None:
    document = _document()
    lattice = _full_lattice(document)
    selected_by_group = {
        selection["groupId"]: selection["rankedCandidateIds"][0]
        for selection in _ready_response(lattice)["selections"]
    }

    class MissingTranslationProvider(MappingLocalLLMProvider):
        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            return _positional_response(
                lattice,
                target_group_ids=_prompt_target_group_ids(lattice, prompt),
                selected_by_group=selected_by_group,
            )

    with pytest.raises(WorkerError) as captured:
        SemanticJobArbitrationRunner(
            provider=MissingTranslationProvider([]),
            model="fixture-9b",
            context_tokens=32_768,
            output_tokens=4_096,
            batch_size=32,
            max_batch_attempts=1,
            translation_targets=("zh-CN",),
        ).run(document, candidate_lattice=lattice)

    diagnostic = captured.value.details["attemptDiagnostics"][0]
    assert diagnostic["validationFailureCode"] == "TRANSLATION_INVALID"
    assert diagnostic["responseFields"] == ["choiceByPosition"]


def test_positional_response_discards_unbound_translation_for_candidate_request() -> None:
    document = _document()
    lattice = _full_lattice(document)
    selected_by_group = {
        selection["groupId"]: selection["rankedCandidateIds"][0]
        for selection in _ready_response(lattice)["selections"]
    }

    class InvalidRequestTranslationProvider(MappingLocalLLMProvider):
        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            response = _positional_response(
                lattice,
                target_group_ids=_prompt_target_group_ids(lattice, prompt),
                selected_by_group=selected_by_group,
                translation_texts=[
                    "占位译文" for _slot in prompt["translationSlots"]
                ],
            )
            response["choiceByPosition"][
                str(prompt["translationSlots"][0]["groupPosition"])
            ] = -1
            return response

    artifact = SemanticJobArbitrationRunner(
        provider=InvalidRequestTranslationProvider([]),
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=32,
        max_batch_attempts=1,
        translation_targets=("zh-CN",),
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "candidate-generation-required"
    assert len(artifact["translations"]) == 1


def test_positional_same_language_translation_uses_null_and_host_copy() -> None:
    document = _document()
    lattice = _full_lattice(document)
    selected_by_group = {
        selection["groupId"]: selection["rankedCandidateIds"][0]
        for selection in _select_current_response(lattice)["selections"]
    }

    class SameLanguageProvider(MappingLocalLLMProvider):
        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            return _positional_response(
                lattice,
                target_group_ids=_prompt_target_group_ids(lattice, prompt),
                selected_by_group=selected_by_group,
                translation_texts=[
                    "host-discarded" for _slot in prompt["translationSlots"]
                ],
            )

    artifact = SemanticJobArbitrationRunner(
        provider=SameLanguageProvider([]),
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=32,
        translation_targets=("en",),
    ).run(document, candidate_lattice=lattice)

    assert {
        (item["segmentId"], item["text"]) for item in artifact["translations"]
    } == {
        ("segment-1", "Hello"),
        ("segment-2", "World"),
    }


def test_positional_translation_retry_receives_missing_protected_literals() -> None:
    document = _document()
    for field in ("rawText", "normalizedText", "displayText"):
        document["segments"][0][field] = "Version 2"
    lattice = build_semantic_candidate_lattice_from_document(document)

    class LiteralRetryProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.prompts: list[dict] = []

        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            self.prompts.append(prompt)
            return {
                "choiceIndexes": [0] * prompt["targetGroupCount"],
                "translationTexts": (
                    ["版本 2", "世界"]
                    if "correction" in prompt
                    else ["版本", "世界"]
                ),
            }

    provider = LiteralRetryProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=32,
        translation_targets=("zh-CN",),
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "ready-to-compose"
    assert provider.prompts[1]["correction"][
        "translationValidationDetails"
    ] == {"missingProtectedLiterals": ["2"]}


def test_job_runner_retries_one_invalid_batch_with_fixed_feedback() -> None:
    document = _document()
    lattice = _full_lattice(document)
    compact = {
        "latticeId": lattice["latticeId"],
        "latticeSha256": lattice["latticeSha256"],
        "decisions": [
            {
                "groupId": selection["groupId"],
                "action": "select",
                "selectedCandidateId": selection["rankedCandidateIds"][0],
                "requestKind": None,
            }
            for selection in _ready_response(lattice)["selections"]
        ],
    }
    invalid = copy.deepcopy(compact)
    invalid["latticeSha256"] = "0" * 64

    class CapturingProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([invalid, compact])
            self.requests: list[dict] = []

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            return super().generate_json(**kwargs)

    provider = CapturingProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "ready-to-compose"
    assert len(provider.requests) == 2
    first_prompt = json.loads(provider.requests[0]["user_prompt"])
    retry_prompt = json.loads(provider.requests[1]["user_prompt"])
    assert "correction" not in first_prompt
    assert retry_prompt["correction"] == {
        "attempt": 2,
        "previousResponseRejected": True,
        "validationFailureCode": "LATTICE_BINDING_INVALID",
        "requiredLatticeId": lattice["latticeId"],
        "requiredLatticeSha256": lattice["latticeSha256"],
        "requiredTargetGroupCount": first_prompt["targetGroupCount"],
    }
    assert "previousResponse" not in retry_prompt


def test_job_runner_fails_closed_after_batch_retry_bound() -> None:
    document = _document()
    lattice = _full_lattice(document)
    invalid = {
        "latticeId": lattice["latticeId"],
        "latticeSha256": "0" * 64,
        "decisions": [],
    }

    with pytest.raises(WorkerError) as captured:
        SemanticJobArbitrationRunner(
            provider=MappingLocalLLMProvider([invalid, invalid]),
            model="fixture-9b",
            context_tokens=32_768,
            output_tokens=4_096,
            batch_size=8,
        ).run(document, candidate_lattice=lattice)

    assert captured.value.code == "SEMANTIC_JOB_PROVIDER_FAILED"
    assert "after 2 attempts" in captured.value.details["reason"]
    assert captured.value.details["responseContentPersisted"] is False
    assert captured.value.details["attemptDiagnostics"] == [
        {
            "batchIndex": 0,
            "attempt": attempt,
            "targetGroupCount": 8,
            "validationFailureCode": "LATTICE_BINDING_INVALID",
            "failureStage": "semantic-response-validation",
            "responseFields": [
                "decisions",
                "latticeId",
                "latticeSha256",
            ],
            "decisionCount": 0,
            "translationCount": None,
            "schemaErrorPath": None,
            "translationFailureRule": None,
            "responseContentPersisted": False,
        }
        for attempt in (1, 2)
    ]


def test_translation_mode_allows_non_asr_batch_to_omit_translations() -> None:
    document = _document()
    lattice = _full_lattice(document)
    complete = _request_or_select_response(
        lattice,
        force_request_domains=frozenset({"asr-text"}),
    )
    compact = {
        "latticeId": lattice["latticeId"],
        "latticeSha256": lattice["latticeSha256"],
        "decisions": [
            {
                "groupId": selection["groupId"],
                "action": "select",
                "selectedCandidateId": selection["rankedCandidateIds"][0],
                "requestKind": None,
            }
            for selection in complete["selections"]
        ]
        + [
            {
                "groupId": request["groupId"],
                "action": "request-candidates",
                "selectedCandidateId": None,
                "requestKind": request["requestKind"],
            }
            for request in complete["candidateGenerationRequests"]
        ],
    }

    artifact = SemanticJobArbitrationRunner(
        provider=MappingLocalLLMProvider([compact]),
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
        translation_targets=("zh-CN",),
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "candidate-generation-required"
    assert artifact["translationTargets"] == ["zh-CN"]
    assert artifact["translations"] == []
    assert any(
        request["domain"] == "asr-text"
        for request in artifact["candidateGenerationRequests"]
    )


def test_translation_mode_retries_when_selected_asr_has_no_translation() -> None:
    document = _document()
    lattice = _full_lattice(document)
    ready = _ready_response(lattice)
    compact = {
        "latticeId": lattice["latticeId"],
        "latticeSha256": lattice["latticeSha256"],
        "decisions": [
            {
                "groupId": selection["groupId"],
                "action": "select",
                "selectedCandidateId": selection["rankedCandidateIds"][0],
                "requestKind": None,
            }
            for selection in ready["selections"]
        ],
    }

    with pytest.raises(WorkerError) as captured:
        SemanticJobArbitrationRunner(
            provider=MappingLocalLLMProvider([compact, compact]),
            model="fixture-9b",
            context_tokens=32_768,
            output_tokens=4_096,
            batch_size=8,
            translation_targets=("zh-CN",),
        ).run(document, candidate_lattice=lattice)

    assert captured.value.code == "SEMANTIC_JOB_PROVIDER_FAILED"
    diagnostics = captured.value.details["attemptDiagnostics"]
    assert [item["validationFailureCode"] for item in diagnostics] == [
        "TRANSLATION_INVALID",
        "TRANSLATION_INVALID",
    ]
    assert all(item["translationCount"] is None for item in diagnostics)
    assert all(item["responseContentPersisted"] is False for item in diagnostics)
    assert all(
        item["translationFailureRule"] == "COVERAGE_MISSING"
        for item in diagnostics
    )


def test_translation_retry_supplies_selected_candidate_bindings() -> None:
    document = _document()
    lattice = _full_lattice(document)
    ready = _ready_response(lattice)
    groups = {
        group["groupId"]: {**group, "domain": domain["domain"]}
        for domain in lattice["domains"]
        for group in domain["groups"]
    }
    candidates = {
        candidate["candidateId"]: candidate
        for group in groups.values()
        for candidate in group["candidates"]
    }
    decisions = [
        {
            "groupId": selection["groupId"],
            "action": "select",
            "selectedCandidateId": selection["rankedCandidateIds"][0],
            "requestKind": None,
        }
        for selection in ready["selections"]
    ]
    translations = []
    for decision in decisions:
        group = groups[decision["groupId"]]
        if group["domain"] != "asr-text":
            continue
        selected = candidates[decision["selectedCandidateId"]]
        translated_text = ("第一段译文", "第二段译文")[len(translations)]
        translations.append(
            {
                "segmentId": selected["payload"]["segmentId"],
                "selectedCandidateId": selected["candidateId"],
                "targetLanguage": "zh-CN",
                "text": translated_text,
            }
        )
    invalid_translations = copy.deepcopy(translations)
    first_asr_group = next(
        group
        for group in groups.values()
        if group["domain"] == "asr-text"
        and group["scopeId"]
        == "segment:" + invalid_translations[0]["segmentId"]
    )
    invalid_translations[0]["selectedCandidateId"] = next(
        candidate["candidateId"]
        for candidate in first_asr_group["candidates"]
        if candidate["candidateId"]
        != invalid_translations[0]["selectedCandidateId"]
        and candidate["selectionEligible"] is True
    )
    invalid = {
        "latticeId": lattice["latticeId"],
        "latticeSha256": lattice["latticeSha256"],
        "decisions": decisions,
        "translations": invalid_translations,
    }
    valid = {
        **invalid,
        "translations": translations,
    }

    class CapturingProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([invalid, valid])
            self.requests: list[dict] = []

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            return super().generate_json(**kwargs)

    provider = CapturingProvider()
    arbitration = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
        translation_targets=("zh-CN",),
    ).run(document, candidate_lattice=lattice)

    assert arbitration["status"] == "ready-to-compose"
    correction = json.loads(provider.requests[1]["user_prompt"])[
        "correction"
    ]
    assert correction["validationFailureCode"] == "TRANSLATION_INVALID"
    assert correction["requiredTranslationBindings"] == [
        {
            "segmentId": item["segmentId"],
            "selectedCandidateId": item["selectedCandidateId"],
            "targetLanguage": "zh-CN",
        }
        for item in translations
    ]
    assert all(
        "sourceTextSha256" not in item
        for item in correction["requiredTranslationBindings"]
    )


@pytest.mark.parametrize(
    "prompt_version",
    [
        "semantic-job-candidate-arbitration-v1",
        "semantic-job-candidate-arbitration-v2",
    ],
)
def test_job_arbitration_validator_keeps_older_artifacts_readable(
    prompt_version: str,
) -> None:
    document = _document()
    lattice = _full_lattice(document)
    artifact = _artifact(lattice, _ready_response(lattice))
    artifact["promptVersion"] = prompt_version

    assert (
        validate_semantic_job_arbitration(
            artifact,
            expected_job_id=document["jobId"],
            expected_lattice=lattice,
        )["promptVersion"]
        == prompt_version
    )


def test_model_can_request_bounded_candidate_generation_for_risky_domains() -> None:
    document = _document()
    lattice = build_semantic_candidate_lattice_from_document(document)
    response = _request_or_select_response(
        lattice,
        force_request_domains=frozenset(
            {
                "speech-disposition",
                "speaker-cardinality-timeline",
                "language-span",
                "asr-text",
            }
        ),
    )
    artifact = _artifact(lattice, response)

    assert artifact["status"] == "candidate-generation-required"
    assert artifact["candidateGenerationRequests"]
    assert {
        item["domain"] for item in artifact["candidateGenerationRequests"]
    } >= {
        "speech-disposition",
        "speaker-cardinality-timeline",
        "language-span",
        "asr-text",
    }
    with pytest.raises(
        SemanticCompositionError,
        match="candidate-generation requests",
    ):
        build_semantic_composition(document, lattice, artifact)


def test_runner_preserves_model_requested_bounded_candidates() -> None:
    document = _document()
    lattice = _full_lattice(document)
    complete_response = _request_or_select_response(
        lattice,
        force_request_domains=frozenset(
            {
                "speech-disposition",
                "speaker-cardinality-timeline",
                "language-span",
                "asr-text",
            }
        ),
    )
    compact_model_response = {
        "latticeId": lattice["latticeId"],
        "latticeSha256": lattice["latticeSha256"],
        "decisions": [
            {
                "groupId": selection["groupId"],
                "action": "select",
                "selectedCandidateId": selection["rankedCandidateIds"][0],
                "requestKind": None,
            }
            for selection in complete_response["selections"]
        ]
        + [
            {
                "groupId": request["groupId"],
                "action": "request-candidates",
                "selectedCandidateId": None,
                "requestKind": request["requestKind"],
            }
            for request in complete_response["candidateGenerationRequests"]
        ],
    }

    artifact = SemanticJobArbitrationRunner(
        provider=MappingLocalLLMProvider([compact_model_response]),
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "candidate-generation-required"
    assert artifact["candidateGenerationRequests"]
    assert {
        request["requestKind"]
        for request in artifact["candidateGenerationRequests"]
    } >= {
        "speech-disposition-challenger",
        "timeline-challenger",
        "open-set-lid",
        "provider-native-nbest",
    }
    assert all(
        request["reasonCodes"] == ["SEMANTIC_REQUESTED_CHALLENGER"]
        for request in artifact["candidateGenerationRequests"]
    )


def test_runner_carries_unchanged_groups_when_cross_domain_lattice_changes() -> None:
    document = _document()
    initial = build_semantic_candidate_lattice_from_document(document)
    previous = _artifact(initial, _request_or_select_response(initial))
    timeline = next(
        domain
        for domain in initial["domains"]
        if domain["domain"] == "speaker-cardinality-timeline"
    )["groups"][0]
    challenger_payload = copy.deepcopy(timeline["candidates"][0]["payload"])
    challenger_payload["timelineKind"] = "challenger"
    extended = extend_semantic_candidate_lattice(
        initial,
        supplemental_groups=[
            {
                "domain": "speaker-cardinality-timeline",
                "groupId": timeline["groupId"],
                "scopeId": timeline["scopeId"],
                "candidates": [
                    {
                        "payload": challenger_payload,
                        "producers": [PRODUCER],
                        "selectionEligible": True,
                        "eligibilityReason": "eligible",
                    }
                ],
            }
        ],
    )
    extended_timeline = next(
        domain
        for domain in extended["domains"]
        if domain["domain"] == "speaker-cardinality-timeline"
    )["groups"][0]
    challenger_id = next(
        candidate["candidateId"]
        for candidate in extended_timeline["candidates"]
        if candidate["candidateId"]
        != extended_timeline["currentCandidateId"]
    )
    complete = _request_or_select_response(extended)
    response = {
        "latticeId": extended["latticeId"],
        "latticeSha256": extended["latticeSha256"],
        "decisions": [
            {
                "groupId": selection["groupId"],
                "action": "select",
                "selectedCandidateId": (
                    challenger_id
                    if selection["groupId"] == extended_timeline["groupId"]
                    else selection["rankedCandidateIds"][0]
                ),
                "requestKind": None,
            }
            for selection in complete["selections"]
        ],
    }

    class CapturingProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.requests: list[dict] = []

        def generate_json(self, **kwargs):
            self.requests.append(dict(kwargs))
            prompt = json.loads(kwargs["user_prompt"])
            target_ids = set(_prompt_target_group_ids(extended, prompt))
            return {
                **response,
                "decisions": [
                    decision
                    for decision in response["decisions"]
                    if decision["groupId"] in target_ids
                ],
            }

    provider = CapturingProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(
        document,
        candidate_lattice=extended,
        carried_lattice=initial,
        carried_arbitration=previous,
    )

    assert artifact["metrics"]["selectedGroupCount"] == 8
    first_prompt = json.loads(provider.requests[0]["user_prompt"])
    all_group_ids = {
        group["groupId"]
        for domain in extended["domains"]
        for group in domain["groups"]
        if group["status"] == "available"
    }
    speech_group_id = next(
        group["groupId"]
        for domain in extended["domains"]
        if domain["domain"] == "speech-disposition"
        for group in domain["groups"]
    )
    # The global speech decision is carried across the timeline extension;
    # changing the structural timeline intentionally reopens every segment
    # atomic triad for one consistent joint decision.
    assert set(_prompt_target_group_ids(extended, first_prompt)) == (
        all_group_ids - {speech_group_id}
    )
    selected = {
        item["groupId"]: item["selectedCandidateId"]
        for item in artifact["selections"]
    }
    assert selected[extended_timeline["groupId"]] == challenger_id


def test_runner_retries_incompatible_language_asr_pair() -> None:
    document = _document()
    lattice = _full_lattice(document)

    class MismatchThenCurrentProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.prompts: list[dict] = []

        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            self.prompts.append(prompt)
            choices: list[int] = []
            for group in prompt["candidateLattice"]["targetGroups"]:
                if (
                    len(self.prompts) == 1
                    and group["domain"] == "language-span"
                    and group["scopeId"] == "segment:segment-2"
                ):
                    choices.append(1)
                else:
                    choices.append(0)
            return {"choiceIndexes": choices}

    provider = MismatchThenCurrentProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=32,
        max_batch_attempts=2,
    ).run(document, candidate_lattice=lattice)

    assert artifact["status"] == "ready-to-compose"
    assert len(provider.prompts) == 2
    assert provider.prompts[1]["correction"]["validationFailureCode"] == (
        "CROSS_DOMAIN_INCONSISTENCY"
    )


def test_arbitration_rejects_incompatible_language_asr_pair() -> None:
    document = _document()
    lattice = _full_lattice(document)
    response = _select_current_response(lattice)
    language_group = next(
        group
        for domain in lattice["domains"]
        if domain["domain"] == "language-span"
        for group in domain["groups"]
        if group["scopeId"] == "segment:segment-2"
    )
    french = _candidate_for(
        language_group,
        lambda item: item["payload"]["language"] == "fr",
    )
    selection = next(
        item
        for item in response["selections"]
        if item["groupId"] == language_group["groupId"]
    )
    selection["rankedCandidateIds"] = [
        french,
        *[
            candidate_id
            for candidate_id in selection["rankedCandidateIds"]
            if candidate_id != french
        ],
    ]
    selection["evidenceRefs"] = [
        f"candidate-lattice:{lattice['latticeId']}",
        f"candidate-group:{language_group['groupId']}",
        f"candidate:{french}",
    ]

    with pytest.raises(
        SemanticCompositionError,
        match="selected language and ASR text candidates disagree",
    ):
        _artifact(lattice, response)


def test_composition_accepts_primary_and_regional_language_tags() -> None:
    document = _document()
    lattice = _full_lattice(document)
    language_group = next(
        group
        for domain in lattice["domains"]
        if domain["domain"] == "language-span"
        for group in domain["groups"]
        if group["scopeId"] == "segment:segment-2"
    )
    asr_group = next(
        group
        for domain in lattice["domains"]
        if domain["domain"] == "asr-text"
        for group in domain["groups"]
        if group["scopeId"] == "segment:segment-2"
    )
    # The fixture is English; rebuild only the two payloads through the
    # existing immutable challenger extension so candidate hashes stay bound.
    regional_language = copy.deepcopy(
        next(
            item
            for item in language_group["candidates"]
            if item["candidateId"] == language_group["currentCandidateId"]
        )["payload"]
    )
    regional_language["language"] = "en-US"
    supplemental = [
        {
            "domain": "language-span",
            "groupId": language_group["groupId"],
            "scopeId": language_group["scopeId"],
            "candidates": [
                {
                    "payload": regional_language,
                    "producers": [PRODUCER],
                    "selectionEligible": True,
                    "eligibilityReason": "eligible",
                }
            ],
        }
    ]
    extended = extend_semantic_candidate_lattice(
        lattice,
        supplemental_groups=supplemental,
    )
    response = _select_current_response(extended)
    regional_id = _candidate_for(
        next(
            group
            for domain in extended["domains"]
            if domain["domain"] == "language-span"
            for group in domain["groups"]
            if group["scopeId"] == language_group["scopeId"]
        ),
        lambda item: item["payload"]["language"] == "en-US",
    )
    selection = next(
        item
        for item in response["selections"]
        if item["groupId"] == language_group["groupId"]
    )
    selection["rankedCandidateIds"] = [
        regional_id,
        *[
            candidate_id
            for candidate_id in selection["rankedCandidateIds"]
            if candidate_id != regional_id
        ],
    ]
    selection["evidenceRefs"] = [
        f"candidate-lattice:{extended['latticeId']}",
        f"candidate-group:{language_group['groupId']}",
        f"candidate:{regional_id}",
    ]
    arbitration = _artifact(extended, response)
    composition = build_semantic_composition(document, extended, arbitration)
    assert composition["status"] == "composition-complete"


def test_runner_retries_repeat_request_after_challenger_was_fulfilled() -> None:
    document = _document()
    initial = build_semantic_candidate_lattice_from_document(document)
    previous = _artifact(
        initial,
        _request_or_select_response(
            initial,
            force_request_domains=frozenset({"language-span"}),
        ),
    )
    language_groups = next(
        domain
        for domain in initial["domains"]
        if domain["domain"] == "language-span"
    )["groups"]
    supplemental_groups = []
    for index, group in enumerate(language_groups):
        payload = copy.deepcopy(group["candidates"][0]["payload"])
        payload["language"] = ("fr" if index == 0 else "de")
        payload["confidence"] = None
        supplemental_groups.append(
            {
                "domain": "language-span",
                "groupId": group["groupId"],
                "scopeId": group["scopeId"],
                "candidates": [
                    {
                        "payload": payload,
                        "producers": [PRODUCER],
                        "selectionEligible": True,
                        "eligibilityReason": "eligible",
                    }
                ],
            }
        )
    extended = extend_semantic_candidate_lattice(
        initial,
        supplemental_groups=supplemental_groups,
    )

    class RepeatThenSelectProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.prompts: list[dict] = []

        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            self.prompts.append(prompt)
            choice = -1 if len(self.prompts) == 1 else 0
            return {
                "choiceIndexes": [choice] * prompt["targetGroupCount"]
            }

    provider = RepeatThenSelectProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(
        document,
        candidate_lattice=extended,
        carried_lattice=initial,
        carried_arbitration=previous,
    )

    assert artifact["status"] == "ready-to-compose"
    assert len(provider.prompts) == 2
    assert any(
        group["requestDefaultChallengerAllowed"] is False
        for group in provider.prompts[0]["candidateLattice"]["targetGroups"]
    )
    assert provider.prompts[1]["correction"][
        "validationFailureCode"
    ] == "CANDIDATE_REQUEST_EXHAUSTED"
    assert all(
        bound["minimum"]
        == (
            -1
            if group["requestDefaultChallengerAllowed"]
            else 0
        )
        for bound, group in zip(
            provider.prompts[1]["correction"][
                "requiredChoiceIndexBounds"
            ],
            provider.prompts[0]["candidateLattice"]["targetGroups"],
            strict=True,
        )
    )


def test_runner_retries_repeat_request_after_duplicate_challenger_exhaustion() -> None:
    document = _document()
    lattice = build_semantic_candidate_lattice_from_document(document)
    previous = _artifact(
        lattice,
        _request_or_select_response(
            lattice,
            force_request_domains=frozenset({"language-span"}),
        ),
    )

    class RepeatThenSelectProvider(MappingLocalLLMProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.prompts: list[dict] = []

        def generate_json(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            self.prompts.append(prompt)
            choice = -1 if len(self.prompts) == 1 else 0
            return {
                "choiceIndexes": [choice] * prompt["targetGroupCount"]
            }

    provider = RepeatThenSelectProvider()
    artifact = SemanticJobArbitrationRunner(
        provider=provider,
        model="fixture-9b",
        context_tokens=32_768,
        output_tokens=4_096,
        batch_size=8,
    ).run(
        document,
        candidate_lattice=lattice,
        carried_lattice=lattice,
        carried_arbitration=previous,
    )

    assert artifact["status"] == "ready-to-compose"
    assert len(provider.prompts) == 2
    first_groups = provider.prompts[0]["candidateLattice"]["targetGroups"]
    assert all(
        group["requestDefaultChallengerAllowed"]
        is (group["domain"] != "language-span")
        for group in first_groups
    )
    assert provider.prompts[1]["correction"][
        "validationFailureCode"
    ] == "CANDIDATE_REQUEST_EXHAUSTED"
    assert all(
        bound["minimum"] == (-1 if group["domain"] != "language-span" else 0)
        for bound, group in zip(
            provider.prompts[1]["correction"][
                "requiredChoiceIndexBounds"
            ],
            first_groups,
            strict=True,
        )
    )


def test_request_kind_cannot_cross_domain_and_available_ranking_is_complete() -> None:
    document = _document()
    lattice = build_semantic_candidate_lattice_from_document(document)
    response = _request_or_select_response(
        lattice,
        force_request_domains=frozenset({"speech-disposition"}),
    )
    response["candidateGenerationRequests"][0]["requestKind"] = "open-set-lid"

    with pytest.raises(SemanticCompositionError, match="requestKind"):
        _artifact(lattice, response)

    full = _ready_response(_full_lattice(document))
    full["selections"][0]["rankedCandidateIds"].pop()
    with pytest.raises(SemanticCompositionError, match="exactly rank"):
        _artifact(_full_lattice(document), full)


def test_human_lock_blocks_candidate_and_timeline_changes() -> None:
    document = _document()
    document["segments"][0]["humanLocked"] = True
    lattice = _full_lattice(document)
    arbitration = _artifact(lattice, _ready_response(lattice))

    with pytest.raises(SemanticCompositionError, match="human-locked"):
        build_semantic_composition(document, lattice, arbitration)


def test_arbitration_and_composition_reject_hash_and_payload_tampering() -> None:
    document = _document()
    lattice = _full_lattice(document)
    arbitration = _artifact(lattice, _ready_response(lattice))
    composition = build_semantic_composition(
        document,
        lattice,
        arbitration,
        generated_at="2026-07-28T02:00:00Z",
    )

    damaged_arbitration = copy.deepcopy(arbitration)
    damaged_arbitration["selections"][0]["selectedCandidateId"] = (
        "candidate-" + "f" * 24
    )
    with pytest.raises(SemanticCompositionError):
        validate_semantic_job_arbitration(
            damaged_arbitration,
            expected_job_id=document["jobId"],
            expected_lattice=lattice,
        )

    damaged_composition = copy.deepcopy(composition)
    damaged_composition["segments"][0]["finalText"] = "Invented"
    with pytest.raises(SemanticCompositionError):
        validate_semantic_composition(
            damaged_composition,
            expected_document=document,
            expected_lattice=lattice,
            expected_arbitration=arbitration,
        )


def test_registered_generators_fulfill_requests_and_enable_second_round() -> None:
    document = _document()
    lattice = build_semantic_candidate_lattice_from_document(document)
    arbitration = _artifact(
        lattice,
        _request_or_select_response(
            lattice,
            force_request_domains=frozenset(
                {
                    "speech-disposition",
                    "speaker-cardinality-timeline",
                    "language-span",
                    "asr-text",
                }
            ),
        ),
    )

    def generate(request: dict, _document: dict, current_lattice: dict) -> dict:
        group = next(
            group
            for domain in current_lattice["domains"]
            for group in domain["groups"]
            if group["groupId"] == request["groupId"]
        )
        current = next(
            candidate
            for candidate in group["candidates"]
            if candidate["candidateId"] == group["currentCandidateId"]
        )
        payload = copy.deepcopy(current["payload"])
        if request["domain"] == "speech-disposition":
            payload["classification"] = "no-transcribable-speech"
        elif request["domain"] == "speaker-cardinality-timeline":
            payload["timelineKind"] = "challenger"
        elif request["domain"] == "language-span":
            payload["language"] = "es"
            payload["confidence"] = 0.7
        elif request["domain"] == "asr-text":
            payload["text"] = payload["text"] + "."
            payload["sourceCandidateId"] = (
                "generated-" + str(request["groupId"])
            )
            payload["candidateSetSha256"] = canonical_json_sha256(request)
        else:
            raise AssertionError("unexpected request domain")
        return {
            "producer": {
                "producerType": "model",
                "systemId": "registered-" + request["requestKind"],
                "revision": "revision-1",
                "artifactSha256": canonical_json_sha256(
                    {"request": request, "payload": payload}
                ),
                "modelManifestSha256": "d" * 64,
                "identityStatus": "manifest-bound",
            },
            "candidates": [{"payload": payload}],
        }

    registry = SemanticCandidateGenerationRegistry(
        {
            request_kind: generate
            for request_kind in set(REQUEST_KIND.values())
        }
    )
    partial = SemanticCandidateGenerationRegistry(
        {"timeline-challenger": generate}
    ).fulfill(
        document,
        lattice,
        arbitration,
        generated_at="2026-07-28T02:30:00Z",
    )
    assert partial["status"] == "partial"
    assert partial["metrics"]["fulfilledRequestCount"] == 1
    assert partial["metrics"]["unfulfilledRequestCount"] == (
        len(arbitration["candidateGenerationRequests"]) - 1
    )
    assert (
        next(
            domain
            for domain in partial["outputLattice"]["domains"]
            if domain["domain"] == "speaker-cardinality-timeline"
        )["status"]
        == "available"
    )
    generation = registry.fulfill(
        document,
        lattice,
        arbitration,
        generated_at="2026-07-28T03:00:00Z",
    )

    assert generation["status"] == "completed"
    assert generation["metrics"]["fulfilledRequestCount"] == len(
        arbitration["candidateGenerationRequests"]
    )
    assert generation["metrics"]["generatedCandidateCount"] == len(
        arbitration["candidateGenerationRequests"]
    )
    generation_schema = json.loads(
        (
            ROOT
            / "contracts"
            / "semantic-candidate-generation.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(generation_schema)
    Draft202012Validator(generation_schema).validate(generation)
    extended = generation["outputLattice"]
    assert extended["availability"]["allRequiredDomainsAvailable"] is True

    second_arbitration = _artifact(
        extended,
        _select_current_response(extended),
    )
    composition = build_semantic_composition(
        document,
        extended,
        second_arbitration,
        generated_at="2026-07-28T04:00:00Z",
    )
    assert composition["status"] == "composition-complete"
    assert composition["segments"][0]["finalText"] == "Hello"

    damaged = copy.deepcopy(generation)
    damaged["fulfilledRequests"][0]["generatedCandidateIds"][0] = (
        "candidate-" + "f" * 24
    )
    with pytest.raises(SemanticCompositionError):
        validate_semantic_candidate_generation(
            damaged,
            expected_document=document,
            expected_lattice=lattice,
            expected_arbitration=arbitration,
        )


def test_real_evidence_candidate_builders_preserve_independent_bindings() -> None:
    voice_activity = build_voice_activity(
        job_id="job-semantic-composition",
        source_sha256="a" * 64,
        media_duration_ms=2_000,
        normalization_profile="mono-16khz-f32-v1",
        provider={"id": "fixture-vad", "version": "1.0.0"},
        windows=[{"id": "vad-1", "startMs": 0, "endMs": 1_500}],
        minimum_window_ms=100,
        classification="transcribable-speech-detected",
        has_transcribable_speech=True,
    )
    speech = build_voice_activity_challenger_result(
        voice_activity=voice_activity,
        artifact_sha256=canonical_json_sha256(voice_activity),
    )
    assert speech["candidates"][0]["payload"] == {
        "classification": "transcribable-speech",
        "startMs": 0,
        "endMs": 2_000,
        "speechDurationMs": 1_500,
        "speechRatio": 0.75,
        "speechWindowCount": 1,
    }

    lid = build_open_set_lid_challenger_result(
        segment_id="segment-1",
        start_ms=0,
        end_ms=1_000,
        language="en",
        confidence=0.91,
        system_id="fixture-open-lid",
        revision="revision-1",
        artifact_sha256="b" * 64,
        model_manifest_sha256="c" * 64,
    )
    assert lid["candidates"][0]["payload"]["confidence"] == 0.91
    assert lid["producer"]["identityStatus"] == "manifest-bound"

    candidate_set = build_asr_candidate_set(
        model_id="fixture-redecode",
        model_revision="revision-2",
        model_manifest_sha256="d" * 64,
        model_identity_status="manifest-bound",
        source_audio_sha256="a" * 64,
        normalization_profile="mono-16khz-f32-v1",
        source_window_id="segment-1-redecode",
        start_ms=0,
        end_ms=1_000,
        hypotheses=[
            {
                "text": "Hello world.",
                "language": "en",
                "tokens": [
                    {"text": "Hello", "startMs": 0, "endMs": 400},
                    {"text": "world.", "startMs": 500, "endMs": 900},
                ],
                "acousticScore": None,
                "acousticScoreStatus": "provider-unavailable",
                "decodeScore": None,
                "decodeScoreStatus": "provider-unavailable",
            }
        ],
    )
    asr = build_asr_text_challenger_result(
        segment_id="segment-1",
        start_ms=0,
        end_ms=1_000,
        candidate_set=candidate_set,
    )
    assert asr["producer"]["artifactSha256"] == (
        candidate_set["candidateSetSha256"]
    )
    assert asr["candidates"][0]["payload"]["text"] == "Hello world."


def test_candidate_generation_fails_closed_without_registered_handler() -> None:
    document = _document()
    lattice = build_semantic_candidate_lattice_from_document(document)
    arbitration = _artifact(
        lattice,
        _request_or_select_response(
            lattice,
            force_request_domains=frozenset({"asr-text"}),
        ),
    )
    registry = SemanticCandidateGenerationRegistry(
        {"boundary-recompute": lambda request, document, lattice: {}}
    )

    with pytest.raises(
        SemanticCompositionError,
        match="no registered candidate generator matches",
    ):
        registry.fulfill(document, lattice, arbitration)


def test_timeline_challenger_normalizes_local_labels_and_overlap() -> None:
    result = build_timeline_challenger_result(
        turns=[
            {
                "startMs": 0,
                "endMs": 1_200,
                "speaker": "S02",
                "text": "First turn",
            },
            {"startMs": 800, "endMs": 2_000, "speaker": "S01"},
        ],
        source_duration_ms=2_000,
        system_id="fixture-diarizer",
        revision="revision-1",
        artifact_sha256="e" * 64,
        model_manifest_sha256="f" * 64,
        local_speaker_field="speaker",
    )

    payload = result["candidates"][0]["payload"]
    assert payload["speakerCount"] == 2
    assert payload["speakerIds"] == ["speaker-1", "speaker-2"]
    assert [turn["speakerId"] for turn in payload["turns"]] == [
        "speaker-2",
        "speaker-1",
    ]
    assert all(turn["overlap"] for turn in payload["turns"])
    assert payload["turns"][0]["text"] == "First turn"
    assert result["producer"]["identityStatus"] == "manifest-bound"


@pytest.mark.parametrize(
    "name",
    [
        "semantic-job-arbitration.schema.json",
        "semantic-composition.schema.json",
    ],
)
def test_public_semantic_composition_schemas_validate_real_artifacts(
    name: str,
) -> None:
    document = _document()
    lattice = _full_lattice(document)
    arbitration = _artifact(lattice, _ready_response(lattice))
    artifact = (
        arbitration
        if name.startswith("semantic-job")
        else build_semantic_composition(
            document,
            lattice,
            arbitration,
            generated_at="2026-07-28T02:00:00Z",
        )
    )
    schema = json.loads((ROOT / "contracts" / name).read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(artifact)


def test_persistent_orchestrator_resumes_a_bounded_two_round_loop(
    tmp_path: Path,
) -> None:
    document = _document()

    class DeterministicArbitrator:
        def __init__(self) -> None:
            self.calls = 0
            self.release_calls = 0

        def run(
            self,
            current_document: dict,
            *,
            candidate_lattice: dict,
            carried_lattice: dict | None = None,
            carried_arbitration: dict | None = None,
            exhausted_request_group_ids=frozenset(),
        ) -> dict:
            self.calls += 1
            assert current_document == document
            if self.calls == 1:
                response = _request_or_select_response(
                    candidate_lattice,
                    force_request_domains=frozenset(
                        {
                            "speech-disposition",
                            "speaker-cardinality-timeline",
                            "language-span",
                            "asr-text",
                        }
                    ),
                )
            else:
                response = _select_current_response(candidate_lattice)
            return _artifact(candidate_lattice, response)

        def release_resources(self) -> None:
            self.release_calls += 1

    def generate(
        request: dict,
        _document: dict,
        current_lattice: dict,
    ) -> dict:
        group = next(
            group
            for domain in current_lattice["domains"]
            for group in domain["groups"]
            if group["groupId"] == request["groupId"]
        )
        current = next(
            candidate
            for candidate in group["candidates"]
            if candidate["candidateId"] == group["currentCandidateId"]
        )
        payload = copy.deepcopy(current["payload"])
        if request["domain"] == "speech-disposition":
            payload["speechDurationMs"] = 1_900
            payload["speechRatio"] = 0.95
            payload["speechWindowCount"] = 2
        elif request["domain"] == "speaker-cardinality-timeline":
            payload["timelineKind"] = "challenger"
        elif request["domain"] == "language-span":
            payload["confidence"] = 0.9
        elif request["domain"] == "asr-text":
            payload["sourceCandidateId"] = "fixture-redecode"
            payload["candidateSetSha256"] = canonical_json_sha256(request)
        else:
            raise AssertionError("unexpected request domain")
        return {
            "producer": {
                **PRODUCER,
                "systemId": "orchestrator-" + request["requestKind"],
                "artifactSha256": canonical_json_sha256(
                    {"request": request, "payload": payload}
                ),
            },
            "candidates": [{"payload": payload}],
        }

    arbitrator = DeterministicArbitrator()
    registry = SemanticCandidateGenerationRegistry(
        {
            request_kind: generate
            for request_kind in set(REQUEST_KIND.values())
        }
    )
    orchestrator = SemanticCompositionOrchestrator(
        arbitrator=arbitrator,  # type: ignore[arg-type]
        generators=registry,
        max_rounds=2,
    )
    artifact_root = tmp_path / "semantic"
    first = orchestrator.run(document, artifact_root=artifact_root)

    assert first.round_count == 2
    assert first.resumed_artifact_count == 0
    assert first.composition["status"] == "composition-complete"
    assert first.final_lattice["availability"][
        "allRequiredDomainsAvailable"
    ] is True
    assert arbitrator.calls == 2
    assert all(path.is_file() for path in first.artifact_paths)

    resumed = orchestrator.run(document, artifact_root=artifact_root)
    assert resumed.composition == first.composition
    assert resumed.resumed_artifact_count == len(first.artifact_paths)
    assert arbitrator.calls == 2

    projected = compose_transcript_document(
        document,
        resumed.composition,
        input_lattice=resumed.final_lattice,
        arbitration_artifact=resumed.arbitration,
    )
    assert projected["provenance"]["semanticComposition"][
        "applicationPolicy"
    ] == "mandatory-candidate-selection"
    assert projected["semanticTimeline"] == resumed.composition["timeline"]


def test_orchestrator_releases_generators_before_next_arbitration(
    tmp_path: Path,
) -> None:
    document = _document()
    events: list[str] = []

    class TwoRoundArbitrator:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, current_document, *, candidate_lattice, **_kwargs):
            assert current_document == document
            self.calls += 1
            events.append(f"arbitrate-{self.calls}")
            response = (
                _request_or_select_response(
                    candidate_lattice,
                    force_request_domains=frozenset(
                        {"speaker-cardinality-timeline"}
                    ),
                )
                if self.calls == 1
                else _select_current_response(candidate_lattice)
            )
            return _artifact(candidate_lattice, response)

        def release_resources(self) -> None:
            events.append("release-arbitrator")

    def generate_timeline(request, _document, current_lattice):
        events.append("generate-timeline")
        group = next(
            group
            for domain in current_lattice["domains"]
            for group in domain["groups"]
            if group["groupId"] == request["groupId"]
        )
        current = next(
            candidate
            for candidate in group["candidates"]
            if candidate["candidateId"] == group["currentCandidateId"]
        )
        payload = copy.deepcopy(current["payload"])
        payload["timelineKind"] = "challenger"
        return {
            "producer": PRODUCER,
            "candidates": [{"payload": payload}],
        }

    class TrackingRegistry(SemanticCandidateGenerationRegistry):
        def release_resources(self) -> None:
            events.append("release-generators")

    result = SemanticCompositionOrchestrator(
        arbitrator=TwoRoundArbitrator(),  # type: ignore[arg-type]
        generators=TrackingRegistry(
            {"timeline-challenger": generate_timeline}
        ),
        max_rounds=1,
    ).run(document, artifact_root=tmp_path / "semantic")

    assert result.round_count == 2
    assert events == [
        "arbitrate-1",
        "release-arbitrator",
        "generate-timeline",
        "release-generators",
        "arbitrate-2",
    ]


def test_orchestrator_fails_closed_when_between_round_release_fails(
    tmp_path: Path,
) -> None:
    document = _document()

    class RequestingArbitrator:
        def run(self, current_document, *, candidate_lattice, **_kwargs):
            assert current_document == document
            return _artifact(
                candidate_lattice,
                _request_or_select_response(
                    candidate_lattice,
                    force_request_domains=frozenset(
                        {"speaker-cardinality-timeline"}
                    ),
                ),
            )

        def release_resources(self) -> None:
            return None

    def generate_timeline(request, _document, current_lattice):
        group = next(
            group
            for domain in current_lattice["domains"]
            for group in domain["groups"]
            if group["groupId"] == request["groupId"]
        )
        current = next(
            candidate
            for candidate in group["candidates"]
            if candidate["candidateId"] == group["currentCandidateId"]
        )
        payload = copy.deepcopy(current["payload"])
        payload["timelineKind"] = "challenger"
        return {
            "producer": PRODUCER,
            "candidates": [{"payload": payload}],
        }

    class FailingReleaseRegistry(SemanticCandidateGenerationRegistry):
        def release_resources(self) -> None:
            raise RuntimeError("fixture release failure")

    orchestrator = SemanticCompositionOrchestrator(
        arbitrator=RequestingArbitrator(),  # type: ignore[arg-type]
        generators=FailingReleaseRegistry(
            {"timeline-challenger": generate_timeline}
        ),
        max_rounds=1,
    )

    with pytest.raises(WorkerError) as captured:
        orchestrator.run(document, artifact_root=tmp_path / "semantic")

    assert captured.value.code == (
        "SEMANTIC_CANDIDATE_RESOURCE_RELEASE_FAILED"
    )
    assert captured.value.retryable is True
    assert captured.value.details == {"exceptionType": "RuntimeError"}


def test_orchestrator_fails_closed_when_arbitrator_release_fails(
    tmp_path: Path,
) -> None:
    document = _document()

    class FailingArbitrator:
        def run(self, current_document, *, candidate_lattice, **_kwargs):
            assert current_document == document
            return _artifact(
                candidate_lattice,
                _request_or_select_response(
                    candidate_lattice,
                    force_request_domains=frozenset(
                        {"speaker-cardinality-timeline"}
                    ),
                ),
            )

        def release_resources(self) -> None:
            raise RuntimeError("fixture arbitrator release failure")

    class UnusedRegistry(SemanticCandidateGenerationRegistry):
        def fulfill(self, *_args, **_kwargs):
            pytest.fail("candidate generation must not start after release failure")

    def unused_timeline_challenger(*_args, **_kwargs):
        pytest.fail("candidate generation must not start after release failure")

    orchestrator = SemanticCompositionOrchestrator(
        arbitrator=FailingArbitrator(),  # type: ignore[arg-type]
        generators=UnusedRegistry({"timeline-challenger": unused_timeline_challenger}),
        max_rounds=1,
    )

    with pytest.raises(WorkerError) as captured:
        orchestrator.run(document, artifact_root=tmp_path / "semantic")

    assert captured.value.code == "SEMANTIC_ARBITRATOR_RESOURCE_RELEASE_FAILED"
    assert captured.value.retryable is True
    assert captured.value.details == {"exceptionType": "RuntimeError"}


def test_orchestrator_final_generation_receives_arbitration_only_pass(
    tmp_path: Path,
) -> None:
    document = _document()

    class FinalGenerationArbitrator:
        def __init__(self) -> None:
            self.calls = 0
            self.exhausted_by_call: list[set[str]] = []

        def run(self, current_document, *, candidate_lattice, **kwargs):
            assert current_document == document
            self.calls += 1
            self.exhausted_by_call.append(
                set(kwargs.get("exhausted_request_group_ids", ()))
            )
            response = (
                _request_or_select_response(
                    candidate_lattice,
                    force_request_domains=frozenset(
                        {"speech-disposition"}
                    ),
                )
                if self.calls == 1
                else _select_current_response(candidate_lattice)
            )
            return _artifact(candidate_lattice, response)

        def release_resources(self) -> None:
            return None

    def generate_speech(request, _document, current_lattice):
        group = next(
            group
            for domain in current_lattice["domains"]
            for group in domain["groups"]
            if group["groupId"] == request["groupId"]
        )
        current = next(
            candidate
            for candidate in group["candidates"]
            if candidate["candidateId"] == group["currentCandidateId"]
        )
        payload = copy.deepcopy(current["payload"])
        payload.update(
            {
                "speechDurationMs": 1_900,
                "speechRatio": 0.95,
                "speechWindowCount": 2,
            }
        )
        return {
            "producer": PRODUCER,
            "candidates": [{"payload": payload}],
        }

    arbitrator = FinalGenerationArbitrator()
    result = SemanticCompositionOrchestrator(
        arbitrator=arbitrator,  # type: ignore[arg-type]
        generators=SemanticCandidateGenerationRegistry(
            {"speech-disposition-challenger": generate_speech}
        ),
        max_rounds=1,
    ).run(document, artifact_root=tmp_path / "semantic")

    assert result.round_count == 2
    assert arbitrator.calls == 2
    assert len(result.generation_paths) == 1
    assert result.arbitration["status"] == "ready-to-compose"
    assert result.composition["status"] == "composition-complete"
    assert result.arbitration_path.parent.name == "round-02"
    available_group_ids = {
        group["groupId"]
        for domain in result.final_lattice["domains"]
        for group in domain["groups"]
        if group["status"] == "available"
    }
    assert arbitrator.exhausted_by_call == [set(), available_group_ids]


def test_orchestrator_carries_duplicate_one_shot_request_as_exhausted(
    tmp_path: Path,
) -> None:
    document = _document()

    class ExhaustionAwareArbitrator:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, current_document, *, candidate_lattice, **_kwargs):
            assert current_document == document
            self.calls += 1
            response = (
                _request_or_select_response(
                    candidate_lattice,
                    force_request_domains=frozenset(
                        {"speaker-cardinality-timeline"}
                    ),
                )
                if self.calls == 1
                else _select_current_response(candidate_lattice)
            )
            return _artifact(candidate_lattice, response)

        def release_resources(self) -> None:
            return None

    def duplicate_timeline(request, _document, current_lattice):
        group = next(
            group
            for domain in current_lattice["domains"]
            for group in domain["groups"]
            if group["groupId"] == request["groupId"]
        )
        current = next(
            candidate
            for candidate in group["candidates"]
            if candidate["candidateId"] == group["currentCandidateId"]
        )
        return {
            "producer": PRODUCER,
            "candidates": [{"payload": copy.deepcopy(current["payload"])}],
        }

    arbitrator = ExhaustionAwareArbitrator()
    result = SemanticCompositionOrchestrator(
        arbitrator=arbitrator,  # type: ignore[arg-type]
        generators=SemanticCandidateGenerationRegistry(
            {"timeline-challenger": duplicate_timeline}
        ),
        max_rounds=2,
    ).run(document, artifact_root=tmp_path / "semantic")

    assert result.round_count == 2
    assert arbitrator.calls == 2
    generation = read_json_strict(result.generation_paths[0])
    assert generation["status"] == "partial"
    assert generation["supplementalGroups"] == []
    assert generation["fulfilledRequests"] == []
    assert generation["metrics"]["generatedCandidateCount"] == 0
    assert generation["metrics"]["unfulfilledRequestCount"] == 1
    assert generation["outputLattice"] == result.initial_lattice
    schema = json.loads(
        (
            ROOT
            / "contracts"
            / "semantic-candidate-generation.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(generation)


def test_orchestrator_accumulates_exhausted_requests_across_all_rounds(
    tmp_path: Path,
) -> None:
    document = _document()

    class AlternatingRequestArbitrator:
        def __init__(self) -> None:
            self.calls = 0
            self.exhausted_by_call: list[set[str]] = []

        def run(
            self,
            current_document,
            *,
            candidate_lattice,
            exhausted_request_group_ids=frozenset(),
            **_kwargs,
        ):
            assert current_document == document
            self.calls += 1
            self.exhausted_by_call.append(set(exhausted_request_group_ids))
            force_domain = (
                "speaker-cardinality-timeline"
                if self.calls == 1
                else "speech-disposition"
                if self.calls == 2
                else None
            )
            response = (
                _request_or_select_response(
                    candidate_lattice,
                    force_request_domains=frozenset({force_domain}),
                )
                if force_domain is not None
                else _select_current_response(candidate_lattice)
            )
            return _artifact(candidate_lattice, response)

        def release_resources(self) -> None:
            return None

    def duplicate_candidate(request, _document, current_lattice):
        group = next(
            group
            for domain in current_lattice["domains"]
            for group in domain["groups"]
            if group["groupId"] == request["groupId"]
        )
        current = next(
            candidate
            for candidate in group["candidates"]
            if candidate["candidateId"] == group["currentCandidateId"]
        )
        return {
            "producer": PRODUCER,
            "candidates": [{"payload": copy.deepcopy(current["payload"])}],
        }

    arbitrator = AlternatingRequestArbitrator()
    result = SemanticCompositionOrchestrator(
        arbitrator=arbitrator,  # type: ignore[arg-type]
        generators=SemanticCandidateGenerationRegistry(
            {
                "timeline-challenger": duplicate_candidate,
                "speech-disposition-challenger": duplicate_candidate,
            }
        ),
        max_rounds=3,
    ).run(document, artifact_root=tmp_path / "semantic")

    initial_groups = {
        domain["domain"]: group["groupId"]
        for domain in result.initial_lattice["domains"]
        for group in domain["groups"]
        if group["scopeId"] == "media"
    }
    timeline_group_id = initial_groups["speaker-cardinality-timeline"]
    speech_group_id = initial_groups["speech-disposition"]
    assert result.round_count == 3
    assert arbitrator.exhausted_by_call == [
        set(),
        {timeline_group_id},
        {timeline_group_id, speech_group_id},
    ]


def test_orchestrator_persists_redacted_arbitration_failure(
    tmp_path: Path,
) -> None:
    document = _document()

    class FailingArbitrator:
        model = "fixture-9b"

        def run(self, *args, **kwargs):
            del args, kwargs
            raise WorkerError(
                "SEMANTIC_JOB_PROVIDER_FAILED",
                "local semantic job arbitration failed closed",
                details={
                    "reason": "schema mismatch",
                    "attemptDiagnostics": [
                        {
                            "batchIndex": 0,
                            "attempt": 1,
                            "validationFailureCode": (
                                "STRICT_JSON_OR_SCHEMA_INVALID"
                            ),
                            "responseFields": ["decisions"],
                            "decisionCount": 1,
                            "translationCount": None,
                            "schemaErrorPath": "$",
                            "responseContentPersisted": False,
                        }
                    ],
                    "responseContentPersisted": False,
                },
            )

        def release_resources(self) -> None:
            return None

    orchestrator = SemanticCompositionOrchestrator(
        arbitrator=FailingArbitrator(),  # type: ignore[arg-type]
        generators=SemanticCandidateGenerationRegistry(
            {
                "timeline-challenger": (
                    lambda request, current_document, current_lattice: {}
                )
            }
        ),
    )

    with pytest.raises(WorkerError) as captured:
        orchestrator.run(document, artifact_root=tmp_path / "semantic")

    path = Path(captured.value.details["diagnosticArtifactPath"])
    artifact = read_json_strict(path)
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert artifact["artifactType"] == "semantic-arbitration-failure"
    assert artifact["responseContentPersisted"] is False
    assert artifact["error"]["details"]["attemptDiagnostics"][0][
        "schemaErrorPath"
    ] == "$"
    assert "Hello" not in serialized
    assert "World" not in serialized
    assert (
        captured.value.details["diagnosticArtifactSha256"]
        == canonical_json_sha256(artifact)
    )


def test_orchestrator_persists_redacted_candidate_generation_failure(
    tmp_path: Path,
) -> None:
    document = _document()

    class RequestingArbitrator:
        model = "fixture-9b"

        def run(self, current_document, *, candidate_lattice, **_kwargs):
            assert current_document == document
            return _artifact(
                candidate_lattice,
                _request_or_select_response(
                    candidate_lattice,
                    force_request_domains=frozenset(
                        {"speaker-cardinality-timeline"}
                    ),
                ),
            )

        def release_resources(self) -> None:
            return None

    def invalid_timeline(request, _document, current_lattice):
        group = next(
            group
            for domain in current_lattice["domains"]
            for group in domain["groups"]
            if group["groupId"] == request["groupId"]
        )
        current = next(
            candidate
            for candidate in group["candidates"]
            if candidate["candidateId"] == group["currentCandidateId"]
        )
        payload = copy.deepcopy(current["payload"])
        payload["speakerCount"] += 1
        return {
            "producer": PRODUCER,
            "candidates": [{"payload": payload}],
        }

    orchestrator = SemanticCompositionOrchestrator(
        arbitrator=RequestingArbitrator(),  # type: ignore[arg-type]
        generators=SemanticCandidateGenerationRegistry(
            {"timeline-challenger": invalid_timeline}
        ),
    )

    with pytest.raises(
        WorkerError,
        match="semantic candidate generation failed closed",
    ) as captured:
        orchestrator.run(document, artifact_root=tmp_path / "semantic")

    assert captured.value.code == "SEMANTIC_CANDIDATE_GENERATION_FAILED"
    assert "speakerCount" in captured.value.details["reason"]
    assert captured.value.details["exceptionType"] == (
        "SemanticCandidateLatticeError"
    )
    path = Path(captured.value.details["diagnosticArtifactPath"])
    artifact = read_json_strict(path)
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert artifact["artifactType"] == "semantic-candidate-generation-failure"
    assert artifact["sourceContentPersisted"] is False
    assert artifact["input"]["candidateGenerationRequests"][0][
        "minimumAlternativeCount"
    ] == 2
    assert "Hello" not in serialized
    assert "World" not in serialized
    assert (
        captured.value.details["diagnosticArtifactSha256"]
        == canonical_json_sha256(artifact)
    )


def test_composition_is_the_final_scoring_and_delivery_authority() -> None:
    document = _document()
    lattice = _full_lattice(document)
    arbitration = _artifact(lattice, _ready_response(lattice))
    composition = build_semantic_composition(
        document,
        lattice,
        arbitration,
        generated_at="2026-07-28T05:00:00Z",
    )
    review_queue = {
        "jobId": document["jobId"],
        "items": [],
        "decisions": [],
        "openCount": 0,
    }

    final = build_final_composed_transcript(
        document,
        review_queue,
        composition,
        arbitration,
        lattice,
        generated_at="2026-07-28T05:01:00Z",
    )

    assert final["schemaVersion"] == "1.2.0"
    assert final["acceptanceSubject"] == (
        "speech-speaker-timeline-language-final-text"
    )
    assert final["semantic"]["applicationPolicy"] == (
        "mandatory-candidate-selection"
    )
    assert final["semantic"]["requiresHumanApproval"] is False
    assert final["speakerPolicy"] == composition["speakerPolicy"]
    assert final["timeline"] == composition["timeline"]
    assert final["segments"][0]["finalText"] == (
        composition["segments"][0]["finalText"]
    )
    validate_final_composed_transcript(
        final,
        expected_document=document,
        expected_review_queue=review_queue,
        expected_composition_artifact=composition,
        expected_arbitration_artifact=arbitration,
        expected_input_lattice=lattice,
    )
    schema = json.loads(
        (
            ROOT
            / "contracts"
            / "final-adjudicated-transcript.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(final)


def test_manual_text_revision_overrides_composition_in_delivery_and_final() -> None:
    document = _document()
    document["segments"][0]["normalizedText"] = "Hello, manually reviewed."
    document["segments"][0]["displayText"] = "Hello, manually reviewed."
    document["segments"][0]["revisions"] = [
        {
            "id": "revision-manual-text-1",
            "type": "text",
            "source": "manual",
            "actor": "codex-semantic-adjudicator",
            "occurredAt": "2026-08-09T06:04:05Z",
            "before": "Hello",
            "after": "Hello, manually reviewed.",
            "reasonCode": "MANUAL_TEXT_REVIEW",
            "confidence": 1.0,
            "evidenceRefs": ["semantic-review:segment-1"],
        }
    ]
    lattice = _full_lattice(document)
    arbitration = _artifact(lattice, _ready_response(lattice))
    composition = build_semantic_composition(
        document,
        lattice,
        arbitration,
        generated_at="2026-08-09T06:05:00Z",
    )
    assert composition["segments"][0]["finalText"] != (
        "Hello, manually reviewed."
    )

    delivery = compose_transcript_document(
        document,
        composition,
        input_lattice=lattice,
        arbitration_artifact=arbitration,
    )
    assert delivery["segments"][0]["normalizedText"] == (
        "Hello, manually reviewed."
    )
    assert delivery["segments"][0]["displayText"] == (
        "Hello, manually reviewed."
    )

    review_queue = {
        "jobId": document["jobId"],
        "items": [],
        "decisions": [],
        "openCount": 0,
    }
    final = build_final_composed_transcript(
        document,
        review_queue,
        composition,
        arbitration,
        lattice,
        generated_at="2026-08-09T06:06:00Z",
    )
    assert final["schemaVersion"] == "1.3.0"
    assert final["finalTextAuthority"] == (
        "manual-text-revision-over-semantic-composition"
    )
    assert final["segments"][0]["finalText"] == "Hello, manually reviewed."
    assert final["segments"][1]["finalText"] == (
        composition["segments"][1]["finalText"]
    )

    schema = json.loads(
        (
            ROOT
            / "contracts"
            / "final-adjudicated-transcript.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(final)

    invalid_authority = copy.deepcopy(final)
    invalid_authority["finalTextAuthority"] = (
        "semantic-composition-selected-asr-text"
    )
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(invalid_authority)
