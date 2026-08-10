from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from backend.asr_evidence import validate_asr_candidate_set
from backend.persistence import sha256_file
from backend.semantic_candidate_lattice import (
    build_semantic_candidate_lattice_from_document,
)
from backend.semantic_composition import (
    build_semantic_composition,
    build_semantic_job_arbitration,
)
from backend.semantic_processing import SemanticProcessingRunner, _segment_request
from tools.build_local_llm_semantic_trials import (
    GENERATOR_ID,
    CaseSpec,
    Recording,
    Utterance,
    assign_recording_splits,
    build_case,
    make_text_correction_candidate,
    make_unsafe_protected_candidate,
    protected_inventory,
)


def synthetic_recording(tmp_path: Path) -> Recording:
    tmp_path.mkdir(parents=True, exist_ok=True)
    audio = tmp_path / "recording.wav"
    textgrid = tmp_path / "recording.TextGrid"
    audio.write_bytes(b"fixture-audio")
    textgrid.write_text("fixture-textgrid", encoding="utf-8")
    texts = (
        "\u6211\u4eec\u5148\u8bf4\u4e00\u4e0b\u6574\u4f53\u65b9\u6848\u3002",
        "\u8fd9\u4e2a\u9879\u76ee\u9700\u8981\u7ee7\u7eed\u8ba8\u8bba\u3002",
        "\u6211\u4eec\u73b0\u5728\u53ef\u4ee5\u5f00\u59cb\u5206\u6790\u3002",
        (
            "\u6211\u4eec\u73b0\u5728\u4e0d\u80fd\u89e3\u51b33\u4e2a"
            "\u4ea7\u54c1\u95ee\u9898\u3002"
        ),
        "\u7136\u540e\u6211\u4eec\u518d\u786e\u8ba4\u65f6\u95f4\u3002",
        "\u8fd9\u4e2a\u4ea7\u54c1\u8fd8\u6709\u4e00\u4e9b\u95ee\u9898\u3002",
        "\u6700\u540e\u5927\u5bb6\u786e\u8ba4\u4e00\u4e0b\u7ed3\u8bba\u3002",
    )
    labels = ("B", "A", "A", "A", "B", "B", "B")
    utterances = tuple(
        Utterance(
            corpus="AliMeeting",
            recording_id="fixture-recording",
            speaker_label=label,
            start_ms=index * 2_000,
            end_ms=index * 2_000 + 1_500,
            text=text,
        )
        for index, (label, text) in enumerate(zip(labels, texts))
    )
    return Recording(
        corpus="AliMeeting",
        recording_id="fixture-recording",
        textgrid_path=textgrid,
        audio_path=audio,
        duration_ms=20_000,
        utterances=utterances,
        split="development",
        textgrid_sha256=sha256_file(textgrid),
        audio_sha256=sha256_file(audio),
    )


def _leaf_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {
            text
            for item in value.values()
            for text in _leaf_strings(item)
        }
    if isinstance(value, list):
        return {text for item in value for text in _leaf_strings(item)}
    return set()


def _compose_with(
    document: dict[str, Any],
    lattice: dict[str, Any],
    *,
    selected_by_group: dict[str, str],
) -> dict[str, Any]:
    selections = []
    for domain in lattice["domains"]:
        for group in domain["groups"]:
            selected = selected_by_group.get(
                group["groupId"],
                group["currentCandidateId"],
            )
            eligible = [
                item["candidateId"]
                for item in group["candidates"]
                if item["selectionEligible"] is True
            ]
            assert selected in eligible
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
    arbitration = build_semantic_job_arbitration(
        job_id=document["jobId"],
        lattice=lattice,
        response={
            "latticeId": lattice["latticeId"],
            "latticeSha256": lattice["latticeSha256"],
            "selections": selections,
            "candidateGenerationRequests": [],
        },
        model="fixture-local-llm",
        provider={
            "id": "fixture-provider",
            "version": "1",
            "networkPolicy": "loopback-only",
        },
        generated_at="2026-08-07T01:00:00Z",
    )
    return build_semantic_composition(
        document,
        lattice,
        arbitration,
        generated_at="2026-08-07T01:01:00Z",
    )


def test_parser_independent_partition_and_text_variants() -> None:
    assignments = assign_recording_splits(
        {
            "AliMeeting": [f"ali-{index}" for index in range(8)],
            "AISHELL-4": [f"aishell-{index}" for index in range(20)],
        }
    )
    assert sum(
        split == "held-out"
        for (corpus, _), split in assignments.items()
        if corpus == "AliMeeting"
    ) == 2
    assert sum(
        split == "held-out"
        for (corpus, _), split in assignments.items()
        if corpus == "AISHELL-4"
    ) == 5

    text = (
        "\u6211\u4eec\u73b0\u5728\u4e0d\u80fd\u89e3\u51b33\u4e2a"
        "\u4ea7\u54c1\u95ee\u9898\u3002"
    )
    correction = make_text_correction_candidate(text)
    unsafe = make_unsafe_protected_candidate(text)
    assert correction is not None and correction != text
    assert unsafe is not None and unsafe != text
    assert protected_inventory(correction) == protected_inventory(text)
    assert protected_inventory(unsafe) != protected_inventory(text)


def test_case_uses_hash_bound_deterministic_candidate_generator(
    tmp_path: Path,
) -> None:
    recording = synthetic_recording(tmp_path)
    target = recording.utterances[3]
    corrupted = make_text_correction_candidate(target.text)
    assert corrupted is not None
    case = build_case(
        CaseSpec(
            recording=recording,
            target_index=3,
            category="text-correction",
            variant_text=corrupted,
            competitor_label=None,
        )
    )

    target_segment = case["document"]["segments"][2]
    asr = target_segment["evidence"]["asr"]
    validated = validate_asr_candidate_set(
        asr,
        expected_text=target_segment["rawText"],
        expected_start_ms=target_segment["startMs"],
        expected_end_ms=target_segment["endMs"],
    )
    assert asr["evaluationGenerator"]["id"] == GENERATOR_ID
    assert validated["modelId"] == GENERATOR_ID
    assert validated["modelIdentityStatus"] == "manifest-bound"
    assert len(validated["nBest"]) == 2
    assert all(item["lexicalRepairEligible"] for item in validated["nBest"])
    assert validated["nBest"][0]["text"] == corrupted
    assert [item["acousticScore"] for item in validated["nBest"]] == [
        0.51,
        0.49,
    ]
    assert [item["decodeScore"] for item in validated["nBest"]] == [
        0.51,
        0.49,
    ]
    expected_candidate = next(
        item for item in validated["nBest"] if item["text"] == target.text
    )
    assert expected_candidate["rank"] == 2
    assert expected_candidate["lexicalRepairEligible"] is True
    assert case["expected"]["normalizedText"] == target.text
    assert len(case["caseSha256"]) == 64

    unsafe = make_unsafe_protected_candidate(target.text)
    assert unsafe is not None
    preservation = build_case(
        CaseSpec(
            recording=recording,
            target_index=3,
            category="lexical-protected-preservation",
            variant_text=unsafe,
            competitor_label=None,
        )
    )
    preservation_target = preservation["document"]["segments"][2]
    preservation_asr = validate_asr_candidate_set(
        preservation_target["evidence"]["asr"],
        expected_text=preservation_target["rawText"],
        expected_start_ms=preservation_target["startMs"],
        expected_end_ms=preservation_target["endMs"],
    )
    assert [item["acousticScore"] for item in preservation_asr["nBest"]] == [
        0.51,
        0.49,
    ]
    preservation_expected = next(
        item
        for item in preservation_asr["nBest"]
        if item["text"] == target.text
    )
    assert preservation_expected["rank"] == 1
    assert preservation_expected["lexicalRepairEligible"] is True
    for candidate_case in (case, preservation):
        candidate_target = candidate_case["document"]["segments"][2]
        ranked_speakers = sorted(
            candidate_target["speakerScores"],
            key=lambda item: (-item["score"], item["speakerId"]),
        )
        assert [item["score"] for item in ranked_speakers[:2]] == [0.51, 0.49]
        assert ranked_speakers[0]["speakerId"] == candidate_target["speakerId"]
        overlap = candidate_target["evidence"]["overlap"]
        assert overlap["provider"] == {
            "id": "deterministic-evaluation-top2-timeline",
            "version": "top2-symmetric-v1",
        }
        assert overlap["fullTimelineInference"]["scope"] == (
            "full-normalized-timeline"
        )


@pytest.mark.parametrize(
    "category",
    ["speaker-continuity-correction", "speaker-boundary-control"],
)
def test_speaker_cases_expose_only_current_state_and_neutral_turn_ids(
    tmp_path: Path,
    category: str,
) -> None:
    recording = synthetic_recording(tmp_path / category)
    case = build_case(
        CaseSpec(
            recording=recording,
            target_index=3,
            category=category,
            variant_text=None,
            competitor_label="B",
        )
    )
    document = case["document"]
    segments = document["segments"]
    target = segments[2]
    expected = case["expected"]
    timeline = document["speakerTimeline"]

    assert timeline["provider"] == {
        "id": "deterministic-evaluation-current-state",
        "version": "1",
    }
    assert [segment["turnId"] for segment in segments] == [
        f"turn-{index:04d}" for index in range(1, 6)
    ]
    assert timeline["regular"]["turns"] == [
        {
            "startMs": segment["startMs"],
            "endMs": segment["endMs"],
            "speakerId": segment["speakerId"],
        }
        for segment in segments
    ]
    target_timeline = next(
        turn
        for turn in timeline["regular"]["turns"]
        if turn["startMs"] == target["startMs"]
        and turn["endMs"] == target["endMs"]
    )
    assert target_timeline["speakerId"] == target["speakerId"]
    if category == "speaker-continuity-correction":
        assert target["speakerId"] != expected["speakerId"]
    else:
        assert target["speakerId"] == expected["speakerId"]

    lattice = build_semantic_candidate_lattice_from_document(document)
    timeline_domain = next(
        domain
        for domain in lattice["domains"]
        if domain["domain"] == "speaker-cardinality-timeline"
    )
    timeline_group = timeline_domain["groups"][0]
    current_candidate = next(
        candidate
        for candidate in timeline_group["candidates"]
        if candidate["candidateId"] == timeline_group["currentCandidateId"]
    )
    lattice_target_turn = next(
        turn
        for turn in current_candidate["payload"]["turns"]
        if turn["startMs"] == target["startMs"]
        and turn["endMs"] == target["endMs"]
    )
    assert lattice_target_turn["speakerId"] == target["speakerId"]
    if category == "speaker-continuity-correction":
        assert lattice_target_turn["speakerId"] != expected["speakerId"]

    ranked_speakers = sorted(
        target["speakerScores"],
        key=lambda item: (-item["score"], item["speakerId"]),
    )
    assert [item["score"] for item in ranked_speakers[:2]] == [0.51, 0.49]
    assert ranked_speakers[0]["speakerId"] == target["speakerId"]
    alternate_speaker = ranked_speakers[1]["speakerId"]
    alternate_timeline = next(
        candidate
        for candidate in timeline_group["candidates"]
        if candidate["selectionEligible"] is True
        and candidate["candidateId"] != timeline_group["currentCandidateId"]
        and any(
            turn["startMs"] == target["startMs"]
            and turn["endMs"] == target["endMs"]
            and turn["speakerId"] == alternate_speaker
            for turn in candidate["payload"]["turns"]
        )
    )
    assert re.fullmatch(r"candidate-[a-f0-9]{24}", alternate_timeline["candidateId"])

    assignment_domain = next(
        domain
        for domain in lattice["domains"]
        if domain["domain"] == "speaker-assignment"
    )
    assignment_group = next(
        group
        for group in assignment_domain["groups"]
        if group["scopeId"] == f"segment:{target['id']}"
    )
    alternate_assignment = next(
        candidate
        for candidate in assignment_group["candidates"]
        if candidate["payload"]["speakerId"] == alternate_speaker
        and candidate["selectionEligible"] is True
    )

    if category == "speaker-continuity-correction":
        assert alternate_speaker == expected["speakerId"]
        composition = _compose_with(
            document,
            lattice,
            selected_by_group={
                timeline_group["groupId"]: alternate_timeline["candidateId"],
                assignment_group["groupId"]: alternate_assignment["candidateId"],
            },
        )
        composed_target = next(
            item for item in composition["segments"] if item["id"] == target["id"]
        )
        assert composed_target["speakerId"] == expected["speakerId"]
    else:
        assert alternate_speaker != expected["speakerId"]
        current_composition = _compose_with(
            document,
            lattice,
            selected_by_group={},
        )
        current_target = next(
            item
            for item in current_composition["segments"]
            if item["id"] == target["id"]
        )
        assert current_target["speakerId"] == expected["speakerId"]
        alternate_composition = _compose_with(
            document,
            lattice,
            selected_by_group={
                timeline_group["groupId"]: alternate_timeline["candidateId"],
                assignment_group["groupId"]: alternate_assignment["candidateId"],
            },
        )
        alternate_target = next(
            item
            for item in alternate_composition["segments"]
            if item["id"] == target["id"]
        )
        assert alternate_target["speakerId"] == alternate_speaker

    requests = [
        _segment_request(
            document,
            segments,
            index,
            speaker_top_k=3,
            candidate_lattice=lattice,
        )[0]
        for index in range(len(segments))
    ]
    prompt = SemanticProcessingRunner._user_prompt(requests)
    prompt_payload = json.loads(prompt.split("input=", 1)[1])
    serialized_prompt = json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    for source_key in case["source"]["contextSourceKeys"]:
        assert source_key not in serialized_prompt
    source_labels = {utterance.speaker_label for utterance in recording.utterances}
    assert source_labels.isdisjoint(_leaf_strings(prompt_payload))
    assert case["source"]["targetTier"] not in _leaf_strings(prompt_payload)
    serialized_lattice = json.dumps(lattice, ensure_ascii=False, sort_keys=True)
    assert '"expected"' not in serialized_lattice
    assert all(
        source_key not in serialized_lattice
        for source_key in case["source"]["contextSourceKeys"]
    )
    assert source_labels.isdisjoint(_leaf_strings(lattice))
