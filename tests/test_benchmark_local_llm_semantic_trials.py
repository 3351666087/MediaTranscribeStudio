from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend import (
    LocalLLMConfig,
    MappingLocalLLMProvider,
    SemanticJobArbitrationRunner,
    SemanticProcessingRunner,
    build_semantic_candidate_lattice_from_document,
)
from backend.persistence import canonical_json_sha256, sha256_file
from tools.benchmark_local_llm_semantic_trials import (
    SemanticBenchmarkError,
    _InstrumentedProvider,
    _assert_report_redacted,
    _host_resource_state,
    _model_runtime_provenance,
    _provider_call_evidence_complete,
    _resource_summary,
    aggregate_results,
    analyze_mandatory_batch_sizes,
    build_parser,
    run_benchmark,
    score_case,
    score_mandatory_case,
)
from tools.build_local_llm_semantic_trials import (
    ARTIFACT_TYPE,
    GENERATOR_CONTRACT,
    GENERATOR_MANIFEST_SHA256,
    SCHEMA_VERSION,
    _ARCHIVE_EVIDENCE,
    CaseSpec,
    SemanticTrialError,
    Recording,
    Utterance,
    build_case,
    load_manifest,
    make_text_correction_candidate,
    make_unsafe_protected_candidate,
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


def _case(tmp_path: Path, category: str) -> dict:
    recording = synthetic_recording(tmp_path)
    target = recording.utterances[3]
    if category == "text-correction":
        variant = make_text_correction_candidate(target.text)
    else:
        variant = make_unsafe_protected_candidate(target.text)
    assert variant is not None
    return build_case(
        CaseSpec(
            recording=recording,
            target_index=3,
            category=category,
            variant_text=variant,
            competitor_label=None,
        )
    )


def _manifest(case: dict) -> dict:
    cases: list[dict] = []
    for index in range(96):
        row = copy.deepcopy(case)
        body = dict(row)
        body.pop("caseSha256")
        case_id = f"semantic-case-{index:024x}"
        body["caseId"] = case_id
        body["split"] = "development" if index < 72 else "held-out"
        body["recordingId"] = f"recording-{index}"
        document = copy.deepcopy(body["document"])
        document["jobId"] = case_id
        document["documentId"] = f"document-{case_id}"
        body["document"] = document
        body["documentCanonicalSha256"] = canonical_json_sha256(document)
        body["caseSha256"] = canonical_json_sha256(body)
        cases.append(body)
    partition_rows = [
        {
            "corpus": row["corpus"],
            "recordingId": row["recordingId"],
            "split": row["split"],
            "textGridPath": row["source"]["textGridPath"],
            "textGridSha256": row["source"]["textGridSha256"],
        }
        for row in cases
    ]
    source_corpora = []
    for corpus in ("AliMeeting", "AISHELL-4"):
        evidence = _ARCHIVE_EVIDENCE[corpus]
        source_corpora.append(
            {
                "corpus": corpus,
                **{
                    key: evidence[key]
                    for key in (
                        "openSlrId",
                        "openSlrPage",
                        "archiveUrl",
                        "archiveBytes",
                        "archiveSha256",
                        "retrievedAt",
                        "license",
                    )
                },
                "archivePath": f"D:/fixtures/{corpus}.tar.gz",
                "revision": None,
                "revisionStatus": "not-applicable-archive",
                "usagePolicy": "local-evaluation-only",
                "applicationBundlingAllowed": False,
            }
        )
    value = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": ARTIFACT_TYPE,
        "createdAt": "2026-08-07T00:00:00Z",
        "purpose": "production-path-local-llm-semantic-evaluation",
        "redistribution": {
            "derivedTextBundledWithApplication": False,
            "localEvaluationOnly": True,
        },
        "parser": {
            "package": "praatio",
            "version": "6.2.0",
            "includeEmptyIntervals": False,
        },
        "candidateGenerator": {
            **GENERATOR_CONTRACT,
            "manifestSha256": GENERATOR_MANIFEST_SHA256,
        },
        "sourceCorpora": source_corpora,
        "partition": {
            "method": "recording-isolated-sha256-rank-v1",
            "heldOutFractionPerCorpus": 0.25,
            "recordings": partition_rows,
        },
        "counts": {
            "cases": 96,
            "categories": {"text-correction": 96},
            "splits": {"development": 72, "held-out": 24},
            "corpora": {"AliMeeting": 96},
            "recordings": 96,
            "usedRecordings": 96,
        },
        "cases": cases,
    }
    return {**value, "canonicalSha256": canonical_json_sha256(value)}


def _artifact(case: dict, selected_text: str) -> dict:
    document = case["document"]
    target = document["segments"][2]
    gate_results = [
        {
            "segmentId": segment["id"],
            "decision": "propose" if segment["id"] == target["id"] else "abstain",
            "confidence": 0.8,
        }
        for segment in document["segments"]
    ]
    candidate = next(
        item
        for item in target["evidence"]["asr"]["nBest"]
        if item["text"] == selected_text
    )
    ranking = [
        item["speakerId"]
        for item in sorted(
            target["speakerScores"],
            key=lambda item: (-item["score"], item["speakerId"]),
        )
    ]
    proposal = {
        "segmentId": target["id"],
        "decision": "propose",
        "speakerRanking": ranking,
        "normalizedText": selected_text,
        "textEvidenceCandidateId": candidate["candidateId"],
        "confidence": 0.8,
        "reasonCodes": ["CONTEXTUAL_CANDIDATE_RERANK"],
        "evidenceRefs": [
            f"segment:{target['id']}",
            f"asr-nbest:{target['id']}:{candidate['candidateId']}",
        ],
    }
    return SemanticProcessingRunner(
        provider=MappingLocalLLMProvider(
            [{"results": gate_results}, {"results": [proposal]}]
        ),
        model="fixture",
        batch_size=5,
        context_tokens=32_768,
        output_tokens=4_096,
    ).run(document)


def _speaker_artifact(case: dict, selected_speaker: str) -> dict:
    document = case["document"]
    target = document["segments"][2]
    gate_results = [
        {
            "segmentId": segment["id"],
            "decision": "propose" if segment["id"] == target["id"] else "abstain",
            "confidence": 0.8,
        }
        for segment in document["segments"]
    ]
    ranked = sorted(
        target["speakerScores"],
        key=lambda item: (-item["score"], item["speakerId"]),
    )
    ranking = [selected_speaker] + [
        item["speakerId"]
        for item in ranked
        if item["speakerId"] != selected_speaker
    ]
    proposal = {
        "segmentId": target["id"],
        "decision": "propose",
        "speakerRanking": ranking,
        "normalizedText": target["normalizedText"],
        "textEvidenceCandidateId": "",
        "confidence": 0.8,
        "reasonCodes": ["SPEAKER_CONTINUITY"],
        "evidenceRefs": [
            f"segment:{target['id']}",
            f"speaker-score:{target['id']}:{selected_speaker}",
        ],
    }
    return SemanticProcessingRunner(
        provider=MappingLocalLLMProvider(
            [{"results": gate_results}, {"results": [proposal]}]
        ),
        model="fixture",
        batch_size=5,
        context_tokens=32_768,
        output_tokens=4_096,
    ).run(document)


def _abstention_artifact(case: dict) -> dict:
    document = case["document"]
    return SemanticProcessingRunner(
        provider=MappingLocalLLMProvider(
            [
                {
                    "results": [
                        {
                            "segmentId": segment["id"],
                            "decision": "abstain",
                            "confidence": 0.8,
                        }
                        for segment in document["segments"]
                    ]
                }
            ]
        ),
        model="fixture",
        batch_size=5,
        context_tokens=32_768,
        output_tokens=4_096,
    ).run(document)


class _ExpectedMandatoryProvider(MappingLocalLLMProvider):
    def __init__(
        self,
        case: dict,
        *,
        force_current_timeline: bool = False,
        force_current_choices: bool = False,
        force_alternate_text: bool = False,
        request_candidates: bool = False,
    ) -> None:
        super().__init__([])
        self.case = case
        self.force_current_timeline = force_current_timeline
        self.force_current_choices = force_current_choices
        self.force_alternate_text = force_alternate_text
        self.request_candidates = request_candidates

    def generate_json(self, **kwargs: Any) -> dict:
        prompt = json.loads(kwargs["user_prompt"])
        expected = self.case["expected"]
        target = next(
            item
            for item in self.case["document"]["segments"]
            if item["id"] == self.case["targetSegmentId"]
        )
        choices: list[int] = []
        groups = sorted(
            prompt["candidateLattice"]["targetGroups"],
            key=lambda item: item["groupPosition"],
        )
        for group in groups:
            candidates = group["candidates"]
            if self.request_candidates:
                choices.append(-1)
                continue
            selected = next(
                (item for item in candidates if item["current"] is True),
                candidates[0],
            )
            if self.force_current_choices:
                pass
            elif (
                self.force_alternate_text
                and group["scopeId"] == f"segment:{target['id']}"
                and group["domain"] == "asr-text"
            ):
                selected = next(
                    item for item in candidates if item["current"] is False
                )
            elif (
                group["scopeId"] == "media"
                and group["domain"] == "speaker-cardinality-timeline"
                and not self.force_current_timeline
            ):
                selected = next(
                    item
                    for item in candidates
                    if any(
                        turn["startMs"] == target["startMs"]
                        and turn["endMs"] == target["endMs"]
                        and turn["speakerId"] == expected["speakerId"]
                        for turn in item["summary"]["turns"]
                    )
                )
            elif group["scopeId"] == f"segment:{target['id']}":
                if group["domain"] == "speaker-assignment":
                    selected = next(
                        item
                        for item in candidates
                        if item["summary"]["speakerId"]
                        == expected["speakerId"]
                    )
                elif group["domain"] == "asr-text":
                    selected = next(
                        item
                        for item in candidates
                        if item["summary"]["text"]
                        == expected["normalizedText"]
                    )
            choices.append(int(selected["choiceIndex"]))
        return {"choiceIndexes": choices}


def _mandatory_run(
    case: dict,
    *,
    force_current_timeline: bool = False,
    force_current_choices: bool = False,
    force_alternate_text: bool = False,
    request_candidates: bool = False,
) -> tuple[dict, dict]:
    lattice = build_semantic_candidate_lattice_from_document(case["document"])
    artifact = SemanticJobArbitrationRunner(
        provider=_ExpectedMandatoryProvider(
            case,
            force_current_timeline=force_current_timeline,
            force_current_choices=force_current_choices,
            force_alternate_text=force_alternate_text,
            request_candidates=request_candidates,
        ),
        model="fixture",
        batch_size=8,
        context_tokens=32_768,
        output_tokens=4_096,
    ).run(case["document"], candidate_lattice=lattice)
    return lattice, artifact


def test_manifest_rejects_canonical_tampering(tmp_path: Path) -> None:
    manifest = _manifest(_case(tmp_path, "text-correction"))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    loaded = load_manifest(path)
    assert len(loaded["cases"]) == 96

    manifest["cases"][0]["expected"]["normalizedText"] += "tampered"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SemanticTrialError, match="canonical digest"):
        load_manifest(path)


def test_scoring_accepts_exact_correction_and_rejects_protected_change(
    tmp_path: Path,
) -> None:
    correction_case = _case(tmp_path / "correction", "text-correction")
    correction_artifact = _artifact(
        correction_case,
        correction_case["expected"]["normalizedText"],
    )
    passed = score_case(correction_case, correction_artifact)
    assert passed["schemaContractPassed"] is True
    assert passed["exactTextSelected"] is True
    assert passed["lexicalPreservationPassed"] is True
    assert passed["speakerAccuracyPassed"] is True
    assert passed["safeModificationPassed"] is True
    redacted_values = {
        correction_case["expected"]["normalizedText"],
        correction_case["source"]["targetTier"],
        *correction_case["source"]["contextSourceKeys"],
        *(
            text
            for segment in correction_case["document"]["segments"]
            for text in (
                segment["rawText"],
                segment["normalizedText"],
                segment["displayText"],
            )
        ),
    }
    assert redacted_values.isdisjoint(_leaf_strings(passed))

    preservation_case = _case(
        tmp_path / "preservation",
        "lexical-protected-preservation",
    )
    target = preservation_case["document"]["segments"][2]
    unsafe = target["evidence"]["asr"]["nBest"][1]["text"]
    unsafe_artifact = _artifact(preservation_case, unsafe)
    failed = score_case(preservation_case, unsafe_artifact)
    assert failed["schemaContractPassed"] is True
    assert failed["exactTextSelected"] is False
    assert failed["lexicalPreservationPassed"] is False
    assert failed["safeModificationPassed"] is False

    for row in (passed, failed):
        row["wallTimeSeconds"] = 1.0
        row["tokenCounters"] = {}
    aggregate = aggregate_results([passed, failed])
    assert aggregate["rates"]["schemaContractPassed"]["rate"] == 1.0
    assert aggregate["rates"]["safeModificationPassed"]["rate"] == 0.5


def test_scoring_handles_speaker_continuity_and_boundary_control(
    tmp_path: Path,
) -> None:
    continuity_recording = synthetic_recording(tmp_path / "continuity")
    continuity_case = build_case(
        CaseSpec(
            recording=continuity_recording,
            target_index=3,
            category="speaker-continuity-correction",
            variant_text=None,
            competitor_label="B",
        )
    )
    continuity_artifact = _speaker_artifact(
        continuity_case,
        continuity_case["expected"]["speakerId"],
    )
    continuity = score_case(continuity_case, continuity_artifact)
    assert continuity["schemaContractPassed"] is True
    assert continuity["speakerAccuracyPassed"] is True
    assert continuity["expectedSpeakerChangeMatched"] is True
    assert continuity["safeModificationPassed"] is True

    boundary_recording = synthetic_recording(tmp_path / "boundary")
    boundary_case = build_case(
        CaseSpec(
            recording=boundary_recording,
            target_index=3,
            category="speaker-boundary-control",
            variant_text=None,
            competitor_label="B",
        )
    )
    boundary = score_case(
        boundary_case,
        _abstention_artifact(boundary_case),
    )
    assert boundary["schemaContractPassed"] is True
    assert boundary["speakerAccuracyPassed"] is True
    assert boundary["speakerBoundaryControlPassed"] is True
    assert boundary["safeModificationPassed"] is True


def test_mandatory_scoring_requires_composable_cross_domain_selection(
    tmp_path: Path,
) -> None:
    recording = synthetic_recording(tmp_path / "mandatory")
    case = build_case(
        CaseSpec(
            recording=recording,
            target_index=3,
            category="speaker-continuity-correction",
            variant_text=None,
            competitor_label="B",
        )
    )
    lattice, artifact = _mandatory_run(case)
    passed = score_mandatory_case(case, lattice, artifact)
    assert passed["semanticPath"] == "production-mandatory"
    assert passed["schemaContractPassed"] is True
    assert passed["speakerAccuracyPassed"] is True
    assert passed["nonTargetSelectionsPreserved"] is True
    assert passed["safeModificationPassed"] is True
    assert passed["selectionOutcome"] == "expected-state-selected"
    sensitive_values = {
        text
        for segment in case["document"]["segments"]
        for text in (
            segment["rawText"],
            segment["normalizedText"],
            segment["displayText"],
        )
    }
    assert sensitive_values.isdisjoint(_leaf_strings(passed))

    repeated_lattice, repeated_artifact = _mandatory_run(case)
    repeated = score_mandatory_case(
        case,
        repeated_lattice,
        repeated_artifact,
    )
    assert repeated["decisionSha256"] == passed["decisionSha256"]

    inconsistent_lattice, inconsistent_artifact = _mandatory_run(
        case,
        force_current_timeline=True,
    )
    inconsistent = score_mandatory_case(
        case,
        inconsistent_lattice,
        inconsistent_artifact,
    )
    assert inconsistent["executionSucceeded"] is True
    assert inconsistent["schemaContractPassed"] is False
    assert inconsistent["failureCodes"] == ["MANDATORY_COMPOSITION_INVALID"]
    assert inconsistent["selectionOutcome"] == "composition-invalid"
    assert inconsistent["safeModificationPassed"] is False


def test_mandatory_candidate_requests_fail_closed(tmp_path: Path) -> None:
    case = _case(tmp_path, "text-correction")
    lattice, artifact = _mandatory_run(case, request_candidates=True)
    result = score_mandatory_case(case, lattice, artifact)
    assert result["executionSucceeded"] is True
    assert result["schemaContractPassed"] is False
    assert result["failureCodes"] == [
        "MANDATORY_CANDIDATE_GENERATION_REQUIRED"
    ]
    assert result["selectionOutcome"] == "candidate-generation-required"
    assert result["safeModificationPassed"] is False


def test_mandatory_scoring_distinguishes_current_and_reachable_wrong(
    tmp_path: Path,
) -> None:
    correction_case = _case(tmp_path / "current", "text-correction")
    current_lattice, current_artifact = _mandatory_run(
        correction_case,
        force_current_choices=True,
    )
    current = score_mandatory_case(
        correction_case,
        current_lattice,
        current_artifact,
    )
    assert current["executionSucceeded"] is True
    assert current["schemaContractPassed"] is True
    assert current["selectionOutcome"] == "current-state-selected"
    assert current["safeModificationPassed"] is False

    preservation_case = _case(
        tmp_path / "wrong",
        "lexical-protected-preservation",
    )
    wrong_lattice, wrong_artifact = _mandatory_run(
        preservation_case,
        force_alternate_text=True,
    )
    wrong = score_mandatory_case(
        preservation_case,
        wrong_lattice,
        wrong_artifact,
    )
    assert wrong["executionSucceeded"] is True
    assert wrong["schemaContractPassed"] is True
    assert wrong["selectionOutcome"] == "reachable-wrong-selection"
    assert wrong["safeModificationPassed"] is False

    for row in (current, wrong):
        row["wallTimeSeconds"] = 0.0
        row["tokenCounters"] = {}
    aggregate = aggregate_results([current, wrong])
    assert aggregate["selectionOutcomeCounts"] == {
        "current-state-selected": 1,
        "reachable-wrong-selection": 1,
    }


def test_runtime_provenance_binds_manifest_blob_and_model_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    blob = "b" * 64
    responses = {
        "/api/version": {"version": "0.32.6"},
        "/api/tags": {
            "models": [
                {
                    "name": "fixture:9b",
                    "model": "fixture:9b",
                    "digest": digest,
                    "size": 123,
                    "modified_at": "2026-08-07T00:00:00Z",
                }
            ]
        },
        "/api/show": {
            "modelfile": f"FROM D:\\models\\sha256-{blob}\n",
            "license": "Apache-2.0 fixture",
            "template": "fixture template",
            "parameters": "temperature 0",
            "details": {"parameter_size": "9B"},
            "capabilities": ["completion"],
        },
    }

    def fake_request(
        endpoint: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> dict:
        assert endpoint == "http://127.0.0.1:11434"
        if path == "/api/show":
            assert payload == {"model": "fixture:9b", "verbose": False}
        return copy.deepcopy(responses[path])

    monkeypatch.setattr(
        "tools.benchmark_local_llm_semantic_trials._ollama_request",
        fake_request,
    )
    evidence = _model_runtime_provenance(
        LocalLLMConfig(
            model="fixture:9b",
            expected_model_digest=digest,
        )
    )
    assert evidence["serverVersion"] == "0.32.6"
    assert evidence["actualManifestDigest"] == "sha256:" + digest
    assert evidence["manifestDigestVerified"] is True
    assert evidence["modelBlobSha256"] == blob
    assert evidence["licenseSha256"] == hashlib.sha256(
        b"Apache-2.0 fixture"
    ).hexdigest()
    assert len(evidence["modelfileSha256"]) == 64
    assert len(evidence["templateSha256"]) == 64


def test_resource_summary_requires_verified_absence_after_release() -> None:
    snapshots = [
        {
            "selectedModel": [
                {"sizeBytes": 100, "sizeVramBytes": 80}
            ],
            "host": {
                "ollamaAggregateRssBytes": 120,
                "availablePhysicalMemoryBytes": 1_000,
            },
        },
        {
            "ollamaApiAvailable": True,
            "selectedModelLoaded": False,
            "selectedModel": [],
            "host": {
                "ollamaAggregateRssBytes": 20,
                "availablePhysicalMemoryBytes": 1_100,
            },
        },
    ]
    summary = _resource_summary(snapshots)
    assert summary["peakSelectedModelBytes"] == 100
    assert summary["peakSelectedModelVramBytes"] == 80
    assert summary["peakOllamaAggregateRssBytes"] == 120
    assert summary["minimumAvailablePhysicalMemoryBytes"] == 1_000
    assert summary["modelAbsentAfterRelease"] is True

    failed = copy.deepcopy(snapshots)
    failed[-1]["selectedModelLoaded"] = True
    assert _resource_summary(failed)["modelAbsentAfterRelease"] is False


def test_host_resource_state_includes_windows_llama_server_rss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = SimpleNamespace(total=10_000, available=4_000, used=6_000)
    processes = [
        SimpleNamespace(
            info={
                "name": "ollama.exe",
                "memory_info": SimpleNamespace(rss=100),
            }
        ),
        SimpleNamespace(
            info={
                "name": "llama-server.exe",
                "memory_info": SimpleNamespace(rss=250),
            }
        ),
        SimpleNamespace(
            info={
                "name": "unrelated.exe",
                "memory_info": SimpleNamespace(rss=999),
            }
        ),
    ]
    fake_psutil = SimpleNamespace(
        Error=RuntimeError,
        virtual_memory=lambda: memory,
        process_iter=lambda fields: processes,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    state = _host_resource_state()
    assert state["ollamaProcessCount"] == 2
    assert state["ollamaAggregateRssBytes"] == 350
    assert state["observedProcessNames"] == [
        "llama-server.exe",
        "ollama.exe",
    ]


def test_instrumented_provider_persists_hashes_not_payload_bodies() -> None:
    provider = _InstrumentedProvider(
        MappingLocalLLMProvider([{"choiceIndexes": [0]}])
    )
    system_prompt = "private system prompt"
    user_prompt = "保密逐字稿：我们继续。"
    schema = {
        "type": "object",
        "properties": {"choiceIndexes": {"type": "array"}},
    }
    response = provider.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="fixture",
        temperature=0.0,
        response_schema=schema,
    )
    assert response == {"choiceIndexes": [0]}
    assert len(provider.call_evidence) == 1
    evidence = provider.call_evidence[0]
    assert evidence["outcome"] == "returned"
    assert len(evidence["systemPromptSha256"]) == 64
    assert len(evidence["userPromptSha256"]) == 64
    assert len(evidence["responseSchemaCanonicalSha256"]) == 64
    assert len(evidence["responseCanonicalSha256"]) == 64
    serialized = json.dumps(evidence, ensure_ascii=False)
    assert system_prompt not in serialized
    assert user_prompt not in serialized
    assert "choiceIndexes" not in serialized
    assert _provider_call_evidence_complete(provider) is True


def test_report_redaction_rejects_transcript_text_and_payload_fields(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "text-correction")
    _assert_report_redacted(
        {
            "promptVersion": "fixture-v1",
            "userPromptSha256": "a" * 64,
            "cases": [{"caseId": case["caseId"]}],
        },
        cases=[case],
    )
    with pytest.raises(SemanticBenchmarkError, match="transcript"):
        _assert_report_redacted(
            {"note": case["document"]["segments"][2]["normalizedText"]},
            cases=[case],
        )
    with pytest.raises(SemanticBenchmarkError, match="payload field"):
        _assert_report_redacted(
            {"userPrompt": "redacted or not, bodies are forbidden"},
            cases=[case],
        )


def test_benchmark_defaults_to_development_screening() -> None:
    args = build_parser().parse_args([])
    assert args.split == "development"
    assert args.evaluation_stage == "screening"
    assert args.unlock_held_out is False


@pytest.mark.parametrize(
    "digest",
    [None, "", "not-a-digest", "sha256:" + "g" * 64],
)
def test_run_benchmark_rejects_unpinned_digest_at_api_boundary(
    tmp_path: Path,
    digest: Any,
) -> None:
    with pytest.raises(SemanticBenchmarkError, match="pinned SHA-256"):
        run_benchmark(
            manifest_path=tmp_path / "missing.json",
            endpoint="http://127.0.0.1:11434",
            model="fixture:9b",
            model_digest=digest,
        )


def test_held_out_requires_promotion_and_explicit_unlock(tmp_path: Path) -> None:
    kwargs = {
        "manifest_path": tmp_path / "missing.json",
        "endpoint": "http://127.0.0.1:11434",
        "model": "fixture:9b",
        "model_digest": "a" * 64,
        "split": "held-out",
    }
    with pytest.raises(SemanticBenchmarkError, match="explicit unlock"):
        run_benchmark(**kwargs)
    with pytest.raises(SemanticBenchmarkError, match="explicit unlock"):
        run_benchmark(
            **kwargs,
            evaluation_stage="promotion",
            allow_held_out=False,
        )
    with pytest.raises(FileNotFoundError):
        run_benchmark(
            **kwargs,
            evaluation_stage="promotion",
            allow_held_out=True,
        )


def test_promotion_stage_requires_deployment_batch_size(tmp_path: Path) -> None:
    with pytest.raises(SemanticBenchmarkError, match="deployment batch size 8"):
        run_benchmark(
            manifest_path=tmp_path / "missing.json",
            endpoint="http://127.0.0.1:11434",
            model="fixture:9b",
            model_digest="a" * 64,
            evaluation_stage="promotion",
            batch_size=32,
        )


def test_static_preflight_is_structural_only_and_not_promotion_evidence(
    tmp_path: Path,
) -> None:
    manifest = _manifest(_case(tmp_path / "case", "text-correction"))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    report = analyze_mandatory_batch_sizes(
        manifest_path=path,
        batch_sizes=(32,),
        context_tokens=32_768,
        output_tokens=4_096,
    )
    assert report["evaluationPolicy"] == {
        "stage": "structural-preflight",
        "structuralPlanningOnly": True,
        "qualityScoringPerformed": False,
        "expectedAnswersUsedForPlanning": False,
        "heldOutQualityObserved": False,
        "allSplitsTraversedForStructuralSizing": True,
        "productionPromotionEvidence": False,
    }
    assert report["deploymentBatchSize"] == 8
    assert report["selection"]["selectedBatchSize"] == 32
    assert report["selection"]["configurationParity"] is False
    assert report["manifest"]["candidateGeneratorRevision"] == (
        manifest["candidateGenerator"]["revision"]
    )
