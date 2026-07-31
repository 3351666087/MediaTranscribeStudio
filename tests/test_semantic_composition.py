from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

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
    validate_semantic_composition,
    validate_semantic_candidate_generation,
    validate_semantic_job_arbitration,
    validate_final_composed_transcript,
)
from backend.persistence import canonical_json_sha256, read_json_strict
from backend.asr_evidence import build_asr_candidate_set
from backend.errors import WorkerError
from backend.semantic_composition import _bounded_transcript_context
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
    response: dict = {"choiceIndexes": choices}
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
    assert artifact["promptVersion"] == "semantic-job-candidate-arbitration-v8"
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert "total speaker count and complete timeline" in request["system_prompt"]
    assert "human locks" in request["system_prompt"]
    assert "current has no default priority" in request["system_prompt"]
    assert (
        "More or fewer speakers, turns, boundaries, or words are not quality evidence"
        in request["system_prompt"]
    )
    user_prompt = json.loads(request["user_prompt"])
    assert (
        user_prompt["outputRules"]["currentCandidateHasDefaultPriority"]
        is False
    )
    assert user_prompt["outputRules"]["requireCrossDomainConsistency"] is True
    assert user_prompt["decisionProtocol"] == {
        "choiceIndexesAlignWithGroupPositions": True,
        "candidateChoiceField": "choiceIndex",
        "requestDefaultChallengerIndex": -1,
    }
    assert request["response_schema"]["required"] == ["choiceIndexes"]


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


def test_semantic_transcript_context_keeps_targets_and_bounds_global_text() -> None:
    segments = [
        {"segmentId": f"segment-{index}", "text": str(index)}
        for index in range(20)
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
    assert global_sample[-1]["segmentId"] == "segment-19"
    assert global_policy["mode"] == "uniform-global-sample"


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
        "choiceIndexes",
        "translationTexts",
    ]
    assert set(request["response_schema"]["properties"]) == {
        "choiceIndexes",
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
        "semantic-job-candidate-arbitration-v8"
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
    assert diagnostic["responseFields"] == ["choiceIndexes"]


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
            response["choiceIndexes"][
                prompt["translationSlots"][0]["groupPosition"]
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
    lattice = build_semantic_candidate_lattice_from_document(document)
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
    lattice = build_semantic_candidate_lattice_from_document(document)
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


def test_runner_rearbitrates_unchanged_groups_when_cross_domain_lattice_changes() -> None:
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
            super().__init__([response])
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
    ).run(
        document,
        candidate_lattice=extended,
        carried_lattice=initial,
        carried_arbitration=previous,
    )

    assert artifact["metrics"]["selectedGroupCount"] == 8
    first_prompt = json.loads(provider.requests[0]["user_prompt"])
    assert set(_prompt_target_group_ids(extended, first_prompt)) == {
        selection["groupId"] for selection in complete["selections"]
    }
    selected = {
        item["groupId"]: item["selectedCandidateId"]
        for item in artifact["selections"]
    }
    assert selected[extended_timeline["groupId"]] == challenger_id


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
