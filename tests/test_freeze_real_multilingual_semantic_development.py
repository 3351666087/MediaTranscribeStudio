from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from backend.persistence import canonical_json_sha256
from backend.semantic_candidate_lattice import (
    build_semantic_candidate_lattice_from_document,
)
from tools.benchmark_real_multilingual_semantic_models import (
    load_frozen_semantic_cases,
)
from tools.freeze_real_multilingual_semantic_development import (
    PRODUCTION_STATUS,
    REFERENCE_ONLY_STATUS,
    UNAVAILABLE_STATUS,
    build_real_multilingual_semantic_development_freeze,
    main,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _audio(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _segment(
    text: str,
    *,
    human_locked: bool,
    revisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "segment-1",
        "startMs": 0,
        "endMs": 1_000,
        "speakerId": "speaker-1",
        "rawText": "stale text",
        "normalizedText": text,
        "displayText": text,
        "confidence": 0.9,
        "speakerScores": [{"speakerId": "speaker-1", "score": 0.9}],
        "speakerMargin": 0.5,
        "overlapping": False,
        "humanLocked": human_locked,
        "revisions": revisions or [],
        "language": "en-US",
        "evidence": {"asr": {"provider": "fixture-asr"}},
    }


def _document(
    case_id: str,
    source_sha256: str,
    text: str,
    *,
    reviewed: bool,
) -> dict[str, Any]:
    return {
        "schemaVersion": "2.0.0",
        "documentId": f"document-{case_id}",
        "jobId": f"job-{case_id}",
        "generatedAt": "2026-08-09T00:00:00Z",
        "language": "en-US",
        "source": {
            "fileName": f"{case_id}.wav",
            "sha256": source_sha256,
            "durationMs": 1_000,
        },
        "speakerPolicy": {
            "mode": "manual",
            "resolvedCount": 1,
            "speakerIds": ["speaker-1"],
        },
        "speakers": [{"id": "speaker-1"}],
        "segments": [
            _segment(
                text,
                human_locked=reviewed,
                revisions=(
                    [
                        {
                            "id": "fixture-manual-revision-1",
                            "type": "text",
                            "source": "manual",
                            "actor": "fixture-manual-reviewer",
                            "occurredAt": "2026-08-09T00:01:00Z",
                            "before": "uncorrected asr text",
                            "after": text,
                            "reasonCode": "MANUAL_TEXT_REVIEW",
                            "confidence": 0.99,
                            "evidenceRefs": ["audio:0-1000ms"],
                        }
                    ]
                    if reviewed
                    else []
                ),
            )
        ],
        "provenance": {"offline": True, "models": []},
    }


def _review_queue(
    document: dict[str, Any],
    *,
    audit_source: str,
) -> dict[str, Any]:
    segment = document["segments"][0]
    decision = {
        "decisionId": "fixture-review-1",
        "command": "review.submit",
        "reason": "Full-clip manual review confirms the corrected transcript.",
        "evidence": ["audio:0-1000ms"],
        "confidence": 0.99,
        "audit": {
            "actor": "fixture-manual-reviewer",
            "source": audit_source,
        },
        "recordedAt": "2026-08-09T00:01:00Z",
    }
    return {
        "schemaVersion": "2.0.0",
        "jobId": document["jobId"],
        "items": [
            {
                "id": "segment-1:SEGMENT_LOW_CONFIDENCE",
                "scope": "segment",
                "segmentId": "segment-1",
                "reasonCode": "SEGMENT_LOW_CONFIDENCE",
                "status": "accepted",
                "timeRange": {"startMs": 0, "endMs": 1_000},
                "speakerId": "speaker-1",
                "text": {
                    "rawText": segment["rawText"],
                    "normalizedText": segment["normalizedText"],
                    "displayText": segment["displayText"],
                },
                "decision": decision,
            }
        ],
        "decisions": [{**decision, "item_id": "segment-1", "status": "accepted"}],
        "openCount": 0,
    }


def _production_output(
    *,
    eval_root: Path,
    run_id: str,
    case_id: str,
    source_sha256: str,
    text: str,
    generated_at: str,
    audit_source: str = "human",
    artifact_id: str | None = None,
) -> Path:
    output = (
        eval_root
        / "product-runs"
        / run_id
        / "outputs"
        / (artifact_id or case_id)
    )
    benchmark_document = _document(
        case_id,
        source_sha256,
        "uncorrected asr text",
        reviewed=False,
    )
    reviewed_document = _document(
        case_id,
        source_sha256,
        text,
        reviewed=True,
    )
    benchmark_lattice = build_semantic_candidate_lattice_from_document(
        benchmark_document
    )
    reviewed_lattice = build_semantic_candidate_lattice_from_document(
        reviewed_document
    )
    benchmark_document_sha256 = canonical_json_sha256(benchmark_document)
    reviewed_document_sha256 = canonical_json_sha256(reviewed_document)
    queue = _review_queue(reviewed_document, audit_source=audit_source)
    queue_sha256 = canonical_json_sha256(queue)
    stale_segment = {
        "id": "segment-1",
        "startMs": 0,
        "endMs": 1_000,
        "speakerId": "speaker-1",
        "language": "en-US",
        "finalText": "stale text",
        "overlapping": False,
    }
    final_artifact = {
        "schemaVersion": "1.2.0",
        "artifactType": "final-adjudicated-transcript",
        "artifactId": f"legacy-final-{case_id}",
        "jobId": reviewed_document["jobId"],
        "documentId": reviewed_document["documentId"],
        "generatedAt": generated_at,
        "status": "adjudication-complete",
        "disposition": "transcribable-speech",
        "input": {
            "sourceMediaSha256": source_sha256,
            "transcriptDocumentSha256": reviewed_document_sha256,
            "candidateLatticeSha256": reviewed_lattice["latticeSha256"],
            "reviewQueueSha256": queue_sha256,
        },
        "review": {
            "openCount": 0,
            "itemCount": 1,
            "acceptedCount": 1,
            "rejectedCount": 0,
            "decisionCount": 1,
        },
        "segments": [stale_segment],
    }
    _write_json(
        output / "semantic" / "input-transcript.v2.json", benchmark_document
    )
    _write_json(output / "transcript-document.v2.json", reviewed_document)
    _write_json(
        output
        / "semantic"
        / "composition-runs"
        / benchmark_document_sha256
        / "semantic-candidate-lattice.initial.v1.json",
        benchmark_lattice,
    )
    _write_json(
        output
        / "semantic"
        / "composition-runs"
        / reviewed_document_sha256
        / "semantic-candidate-lattice.initial.v1.json",
        reviewed_lattice,
    )
    _write_json(output / "review" / "review-queue.json", queue)
    _write_json(output / "final-adjudicated-transcript.v1.json", final_artifact)
    return output


def _rewrite_benchmark_input(
    output: Path,
    mutation: Any,
) -> dict[str, Any]:
    document_path = output / "semantic" / "input-transcript.v2.json"
    document = json.loads(document_path.read_text(encoding="utf-8"))
    mutation(document)
    lattice = build_semantic_candidate_lattice_from_document(document)
    document_sha256 = canonical_json_sha256(document)
    _write_json(document_path, document)
    _write_json(
        output
        / "semantic"
        / "composition-runs"
        / document_sha256
        / "semantic-candidate-lattice.initial.v1.json",
        lattice,
    )
    return document


def _fixture(
    tmp_path: Path,
    *,
    audit_source: str = "human",
) -> dict[str, Path | str]:
    project_root = tmp_path / "project"
    eval_root = tmp_path / "eval"
    eval_root.mkdir(parents=True)
    case_id = "fixture_en_production"
    audio_root = (
        project_root / ".runtime_cache" / "sample-library" / "global" / "audio"
    )
    production_sha = _audio(audio_root / f"{case_id}.wav", b"production-audio")
    reference_id = "fixture_es_reference"
    reference_sha = _audio(
        audio_root / f"{reference_id}.wav", b"reference-audio"
    )
    global_manifest = project_root / "sample_library" / "global-manifest.v1.json"
    _write_json(
        global_manifest,
        {
            "schemaVersion": "1.0.0",
            "cases": [
                {
                    "id": case_id,
                    "language": "en-US",
                    "evaluationSplit": "development",
                }
            ],
        },
    )
    reference_path = (
        project_root
        / ".runtime_cache"
        / "sample-library"
        / "global"
        / "fleurs-multilingual-frozen.v1.json"
    )
    reference = {
        "schemaVersion": "1.0.0",
        "artifactType": "fleurs-multilingual-frozen-reference",
        "cases": [
            {
                "id": reference_id,
                "language": "es-419",
                "evaluationSplit": "development",
                "tuningEligible": True,
                "path": f"audio/{reference_id}.wav",
                "sha256": reference_sha,
                "scoringTranscript": "REFERENCE_SECRET_TEXT",
                "expectedTranscript": "REFERENCE_SECRET_TEXT",
                "nativeTranscript": "Reference secret text.",
                "rawTranscript": "Reference secret text.",
            }
        ],
    }
    reference["canonicalSha256"] = canonical_json_sha256(reference)
    _write_json(reference_path, reference)
    _production_output(
        eval_root=eval_root,
        run_id="run-01",
        case_id=case_id,
        source_sha256=production_sha,
        text="corrected text",
        generated_at="2026-08-09T00:02:00Z",
        audit_source=audit_source,
    )
    return {
        "project_root": project_root,
        "eval_root": eval_root,
        "case_id": case_id,
        "production_sha": production_sha,
        "global_manifest": global_manifest,
        "reference_path": reference_path,
        "output": project_root / "benchmarks" / "freeze.v1.json",
    }


def _replay_fixture(tmp_path: Path) -> dict[str, Path | str]:
    fixture = _fixture(tmp_path)
    case_id = str(fixture["case_id"])
    source_output = (
        fixture["eval_root"]
        / "product-runs"
        / "run-01"
        / "outputs"
        / case_id
    )
    assert isinstance(source_output, Path)
    benchmark_path = source_output / "semantic" / "input-transcript.v2.json"
    benchmark_document = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_document["segments"][0]["rawText"] = (
        benchmark_document["segments"][0]["normalizedText"]
    )
    benchmark_document["segments"][0]["speakerMargin"] = 2.0
    _write_json(benchmark_path, benchmark_document)
    _write_json(source_output / "transcript-document.v2.json", benchmark_document)
    benchmark_document_sha256 = canonical_json_sha256(benchmark_document)
    _write_json(
        source_output
        / "semantic"
        / "composition-runs"
        / benchmark_document_sha256
        / "semantic-candidate-lattice.initial.v1.json",
        build_semantic_candidate_lattice_from_document(benchmark_document),
    )
    segment = benchmark_document["segments"][0]
    queue = {
        "schemaVersion": "2.0.0",
        "jobId": benchmark_document["jobId"],
        "items": [
            {
                "id": "segment-1:SEGMENT_LOW_CONFIDENCE",
                "scope": "segment",
                "segmentId": "segment-1",
                "reasonCode": "SEGMENT_LOW_CONFIDENCE",
                "status": "open",
                "timeRange": {"startMs": 0, "endMs": 1_000},
                "speakerId": "speaker-1",
                "text": {
                    "rawText": segment["rawText"],
                    "normalizedText": segment["normalizedText"],
                    "displayText": segment["displayText"],
                },
            }
        ],
        "decisions": [],
        "openCount": 1,
    }
    queue_path = source_output / "review" / "review-queue.json"
    _write_json(queue_path, queue)
    (source_output / "final-adjudicated-transcript.v1.json").unlink()

    decision_set = (
        fixture["project_root"]
        / "benchmarks"
        / "product_reviews"
        / "fleurs18-post-reference-manual-decisions-r2"
    )
    assert isinstance(decision_set, Path)
    decision_path = decision_set / f"{case_id}.review-decisions.json"
    decision = {
        "schemaVersion": "1.0.0",
        "artifactType": "production-review-decisions",
        "jobId": benchmark_document["jobId"],
        "automaticScoring": False,
        "decisions": [
            {
                "itemId": "segment-1:SEGMENT_LOW_CONFIDENCE",
                "action": "accept",
                "decisionId": "fixture-codex-replay-1",
                "reason": "Codex semantic adjudication corrects the transcript.",
                "evidence": ["audio:0-1000ms"],
                "confidence": 1.0,
                "normalizedText": "corrected text",
                "displayText": "corrected text",
                "audit": {
                    "actor": "fixture-codex-reviewer",
                    "source": "codex-agent",
                },
            }
        ],
    }
    _write_json(decision_path, decision)
    input_file_sha256 = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    queue_file_sha256 = hashlib.sha256(queue_path.read_bytes()).hexdigest()
    manifest = {
        "schemaVersion": "1.0.0",
        "artifactType": "fleurs-post-reference-codex-review-decision-set",
        "setId": "fleurs18-post-reference-manual-decisions-r2",
        "replay": {
            "api": "backend.review.validate_review_state then backend.review.resolve_review_item",
            "highMarginThreshold": 0.18,
            "allCaseOpenCountAfterReplay": 0,
        },
        "cases": [
            {
                "caseId": case_id,
                "decisionPath": decision_path.name,
                "decisionFileSha256": hashlib.sha256(
                    decision_path.read_bytes()
                ).hexdigest(),
                "decisionCanonicalSha256": canonical_json_sha256(decision),
                "sourceRunRoot": str(source_output.parents[1]),
                "sourceOutputDirectory": str(source_output),
                "sourceInputFileSha256": input_file_sha256,
                "sourceInputCanonicalSha256": canonical_json_sha256(
                    benchmark_document
                ),
                "sourceQueueFileSha256": queue_file_sha256,
                "sourceQueueCanonicalSha256": canonical_json_sha256(queue),
                "sourceMediaSha256": fixture["production_sha"],
                "observedOpenItemIds": [
                    "segment-1:SEGMENT_LOW_CONFIDENCE"
                ],
                "observedQueueOpenCount": 1,
                "replayDecisionIds": ["fixture-codex-replay-1"],
                "replayDecisionCount": 1,
                "replayOpenCount": 0,
                "replayApi": (
                    "backend.review.validate_review_state+resolve_review_item"
                ),
            }
        ],
    }
    manifest["canonicalSha256"] = canonical_json_sha256(manifest)
    manifest_path = decision_set / "MANIFEST.v1.json"
    _write_json(manifest_path, manifest)
    (decision_set / "MANIFEST.v1.json.sha256").write_text(
        f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}  MANIFEST.v1.json\n",
        encoding="utf-8",
    )
    (decision_set / "MANIFEST.v1.canonical.sha256").write_text(
        f"{canonical_json_sha256(manifest)}  MANIFEST.v1.json canonical-json\n",
        encoding="utf-8",
    )
    fixture["decision_set"] = decision_set
    fixture["pre_review_run_root"] = source_output.parents[1]
    return fixture


def test_freeze_is_benchmark_compatible_and_excludes_stale_final_text(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["output"]
    assert isinstance(output, Path)

    assert main(
        [
            "--project-root",
            str(fixture["project_root"]),
            "--eval-root",
            str(fixture["eval_root"]),
            "--global-manifest",
            str(fixture["global_manifest"]),
            "--fleurs-reference",
            str(fixture["reference_path"]),
            "--production-case",
            str(fixture["case_id"]),
            "--reference-language",
            "es-419",
            "--expected-production-count",
            "1",
            "--expected-reference-count",
            "1",
            "--output",
            str(output),
        ]
    ) == 0

    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["counts"] == {
        "benchmarkCases": 1,
        "inventoryRows": 2,
        PRODUCTION_STATUS: 1,
        REFERENCE_ONLY_STATUS: 1,
        UNAVAILABLE_STATUS: 0,
        "rejectedProductionCandidates": 0,
    }
    case_set = load_frozen_semantic_cases(output)
    assert len(case_set.cases) == 1
    case = case_set.cases[0]
    assert case.baseline["segments"][0]["finalText"] == "corrected text"
    assert case.document["segments"][0]["normalizedText"] == (
        "uncorrected asr text"
    )
    assert case.document["segments"][0]["humanLocked"] is False
    assert case.document["segments"][0]["revisions"] == []
    assert case.baseline["input"]["transcriptDocumentSha256"] == (
        case.document_sha256
    )
    assert case.baseline["input"]["candidateLatticeSha256"] == (
        case.lattice_sha256
    )
    assert case.lattice["binding"]["transcriptSha256"] == case.document_sha256
    assert "corrected text" not in json.dumps(case.document, ensure_ascii=False)
    assert "corrected text" not in json.dumps(case.lattice, ensure_ascii=False)
    target = value["cases"][0]["semanticCalibrationTarget"]
    assert target["counts"] == {
        "select": 0,
        "request-default-challenger": 1,
        "preservation-only": 4,
    }
    assert {
        (row["domain"], row["expectedAction"])
        for row in target["groups"]
    } == {
        ("speech-disposition", "preservation-only"),
        ("speaker-cardinality-timeline", "preservation-only"),
        ("speaker-assignment", "preservation-only"),
        ("language-span", "preservation-only"),
        ("asr-text", "request-default-challenger"),
    }
    assert "corrected text" not in json.dumps(target, ensure_ascii=False)
    production = next(
        row for row in value["inventory"] if row["status"] == PRODUCTION_STATUS
    )
    assert production["artifacts"]["legacyFinalArtifact"]["classification"] == (
        "known-invalid-pre-fix-evidence"
    )
    assert production["artifacts"]["legacyFinalArtifact"][
        "usedAsBenchmarkBaseline"
    ] is False
    assert "REFERENCE_SECRET_TEXT" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("contamination", ("human-lock", "manual-revision"))
def test_pre_review_authority_contamination_is_rejected(
    tmp_path: Path,
    contamination: str,
) -> None:
    fixture = _fixture(tmp_path)
    production_output = (
        fixture["eval_root"]
        / "product-runs"
        / "run-01"
        / "outputs"
        / str(fixture["case_id"])
    )
    assert isinstance(production_output, Path)

    def contaminate(document: dict[str, Any]) -> None:
        segment = document["segments"][0]
        if contamination == "human-lock":
            segment["humanLocked"] = True
        else:
            segment["revisions"].append(
                {
                    "id": "forbidden-pre-review-manual-revision",
                    "type": "text",
                    "source": "codex-agent",
                    "actor": "fixture-codex",
                    "occurredAt": "2026-08-09T00:01:00Z",
                    "before": "a",
                    "after": "b",
                    "reasonCode": "MANUAL_TEXT_REVIEW",
                    "confidence": 1.0,
                    "evidenceRefs": ["audio:0-1000ms"],
                }
            )

    _rewrite_benchmark_input(production_output, contaminate)
    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("es-419",),
    )

    assert value["cases"] == []
    assert value["counts"][UNAVAILABLE_STATUS] == 1


def test_duplicate_pre_review_lattice_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    production_output = (
        fixture["eval_root"]
        / "product-runs"
        / "run-01"
        / "outputs"
        / str(fixture["case_id"])
    )
    document = json.loads(
        (production_output / "semantic" / "input-transcript.v2.json").read_text(
            encoding="utf-8"
        )
    )
    lattice = build_semantic_candidate_lattice_from_document(document)
    _write_json(
        production_output
        / "semantic"
        / "composition-runs"
        / "duplicate-pre-review-binding"
        / "semantic-candidate-lattice.initial.v1.json",
        lattice,
    )

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("es-419",),
    )

    assert value["cases"] == []
    assert value["counts"][UNAVAILABLE_STATUS] == 1


def test_highest_contiguous_expanded_lattice_is_selected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    production_output = (
        fixture["eval_root"]
        / "product-runs"
        / "run-01"
        / "outputs"
        / str(fixture["case_id"])
    )
    document = json.loads(
        (production_output / "semantic" / "input-transcript.v2.json").read_text(
            encoding="utf-8"
        )
    )
    document_sha256 = canonical_json_sha256(document)
    lattice = build_semantic_candidate_lattice_from_document(document)
    run_directory = (
        production_output / "semantic" / "composition-runs" / document_sha256
    )
    for round_number in (1, 2):
        _write_json(
            run_directory
            / f"round-{round_number:02d}"
            / "semantic-candidate-lattice.output.v1.json",
            lattice,
        )

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("es-419",),
    )

    case = value["cases"][0]
    assert case["latticeSelection"]["selectedKind"] == "expanded-output"
    assert case["latticeSelection"]["selectedRound"] == 2
    production = next(
        row for row in value["inventory"] if row["status"] == PRODUCTION_STATUS
    )
    assert production["selection"]["candidateLattice"] == {
        "policy": "highest-contiguous-strictly-validated-expanded-output-else-initial-v1",
        "selectedKind": "expanded-output",
        "selectedRound": 2,
        "expandedRoundCount": 2,
        "transcriptSha256": case["documentSha256"],
    }
    assert "/round-02/" in production["artifacts"]["candidateLattice"]["path"]


def test_expanded_lattice_round_gap_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    production_output = (
        fixture["eval_root"]
        / "product-runs"
        / "run-01"
        / "outputs"
        / str(fixture["case_id"])
    )
    document = json.loads(
        (production_output / "semantic" / "input-transcript.v2.json").read_text(
            encoding="utf-8"
        )
    )
    document_sha256 = canonical_json_sha256(document)
    lattice = build_semantic_candidate_lattice_from_document(document)
    _write_json(
        production_output
        / "semantic"
        / "composition-runs"
        / document_sha256
        / "round-02"
        / "semantic-candidate-lattice.output.v1.json",
        lattice,
    )

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("es-419",),
    )

    assert value["cases"] == []
    assert value["counts"][UNAVAILABLE_STATUS] == 1


def test_current_text_equal_to_reference_remains_eligible_without_truth_fields(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    reference_path = fixture["reference_path"]
    assert isinstance(reference_path, Path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["cases"].append(
        {
            "id": fixture["case_id"],
            "language": "en-US",
            "evaluationSplit": "development",
            "tuningEligible": True,
            "path": f"audio/{fixture['case_id']}.wav",
            "sha256": fixture["production_sha"],
            "scoringTranscript": "uncorrected asr text",
            "expectedTranscript": "uncorrected asr text",
            "nativeTranscript": "uncorrected asr text",
            "rawTranscript": "uncorrected asr text",
        }
    )
    reference.pop("canonicalSha256")
    reference["canonicalSha256"] = canonical_json_sha256(reference)
    _write_json(reference_path, reference)

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=reference_path,
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("en-US",),
    )

    assert value["counts"][PRODUCTION_STATUS] == 1
    assert value["counts"][REFERENCE_ONLY_STATUS] == 0
    case = value["cases"][0]
    assert case["documentPath"]
    assert "scoringTranscript" not in json.dumps(case, ensure_ascii=False)


def test_decision_set_replay_is_deterministic_and_marks_final_artifact_absent(
    tmp_path: Path,
) -> None:
    fixture = _replay_fixture(tmp_path)
    arguments = {
        "project_root": fixture["project_root"],
        "eval_root": fixture["eval_root"],
        "global_manifest_path": fixture["global_manifest"],
        "fleurs_reference_path": fixture["reference_path"],
        "output_path": fixture["output"],
        "production_case_ids": (str(fixture["case_id"]),),
        "reference_languages": (),
        "codex_decision_set": fixture["decision_set"],
        "pre_review_run_roots": (fixture["pre_review_run_root"],),
    }

    first = build_real_multilingual_semantic_development_freeze(**arguments)
    second = build_real_multilingual_semantic_development_freeze(**arguments)

    assert first == second
    assert first["counts"][PRODUCTION_STATUS] == 1
    assert first["counts"][UNAVAILABLE_STATUS] == 0
    case = first["cases"][0]
    assert case["baselineAuditSource"] == "codex-agent"
    assert case["baseline"]["segments"][0]["finalText"] == "corrected text"
    assert case["documentSha256"] == case["baseline"]["input"][
        "transcriptDocumentSha256"
    ]
    assert case["latticeSha256"] == case["baseline"]["input"][
        "candidateLatticeSha256"
    ]
    document = json.loads(
        Path(fixture["pre_review_run_root"])
        .joinpath(
            "outputs",
            str(fixture["case_id"]),
            "semantic",
            "input-transcript.v2.json",
        )
        .read_text(encoding="utf-8")
    )
    assert document["segments"][0]["normalizedText"] == "uncorrected asr text"
    assert document["segments"][0]["humanLocked"] is False
    assert document["segments"][0]["revisions"] == []
    inventory = first["inventory"][0]
    assert inventory["selection"]["evidenceKind"] == "in-memory-review-replay"
    assert inventory["artifacts"]["legacyFinalArtifact"] == {
        "status": "absent",
        "classification": "not-applicable-in-memory-review-replay",
        "usedAsBenchmarkBaseline": False,
    }
    assert "reviewedTranscriptDocument" not in inventory["artifacts"]
    assert inventory["manualAdjudication"]["replayOpenCount"] == 0
    assert "reviewQueueSha256" not in case["baseline"]["input"]


def test_non_manual_review_is_classified_unavailable(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, audit_source="model-self-review")
    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("es-419",),
    )

    assert value["cases"] == []
    assert value["counts"][PRODUCTION_STATUS] == 0
    assert value["counts"][UNAVAILABLE_STATUS] == 1
    assert value["counts"]["rejectedProductionCandidates"] == 1


def test_codex_agent_review_is_preserved_as_manual_baseline_source(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, audit_source="codex-agent")
    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("es-419",),
    )

    assert value["counts"][PRODUCTION_STATUS] == 1
    assert value["cases"][0]["baselineAuditSource"] == "codex-agent"
    assert value["cases"][0]["baseline"]["adjudicationSource"] == (
        "codex-agent"
    )
    assert value["cases"][0]["baseline"]["review"]["source"] == (
        "codex-agent"
    )


def test_default_reference_languages_remain_enabled(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
    )

    references = [
        row
        for row in value["inventory"]
        if row["status"] == REFERENCE_ONLY_STATUS
    ]
    assert [row["caseId"] for row in references] == ["fixture_es_reference"]


def test_invalid_production_does_not_hide_matching_reference(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, audit_source="model-self-review")
    reference_path = fixture["reference_path"]
    assert isinstance(reference_path, Path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["cases"].append(
        {
            "id": fixture["case_id"],
            "language": "en-US",
            "evaluationSplit": "development",
            "tuningEligible": True,
            "path": f"audio/{fixture['case_id']}.wav",
            "sha256": fixture["production_sha"],
            "scoringTranscript": "FALLBACK_REFERENCE_SECRET",
            "expectedTranscript": "FALLBACK_REFERENCE_SECRET",
            "nativeTranscript": "Fallback reference secret.",
            "rawTranscript": "Fallback reference secret.",
        }
    )
    reference.pop("canonicalSha256")
    reference["canonicalSha256"] = canonical_json_sha256(reference)
    _write_json(reference_path, reference)

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=reference_path,
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("en-US",),
    )

    assert value["counts"][PRODUCTION_STATUS] == 0
    assert value["counts"][UNAVAILABLE_STATUS] == 1
    assert value["counts"][REFERENCE_ONLY_STATUS] == 1
    assert {
        (row["caseId"], row["status"]) for row in value["inventory"]
    } == {
        (str(fixture["case_id"]), UNAVAILABLE_STATUS),
        (str(fixture["case_id"]), REFERENCE_ONLY_STATUS),
    }


def test_latest_valid_manual_run_is_selected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _production_output(
        eval_root=fixture["eval_root"],
        run_id="run-02",
        case_id=str(fixture["case_id"]),
        source_sha256=str(fixture["production_sha"]),
        text="newer corrected text",
        generated_at="2026-08-09T00:03:00Z",
    )
    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("es-419",),
    )

    production = next(
        row for row in value["inventory"] if row["status"] == PRODUCTION_STATUS
    )
    assert production["selection"]["runId"] == "run-02"
    assert value["cases"][0]["baseline"]["segments"][0]["finalText"] == (
        "newer corrected text"
    )


@pytest.mark.parametrize(
    "artifact_id_suffix",
    ("", "-manual", "-manual-run2", "-hybrid", "-hybrid-run12"),
)
def test_scan_accepts_strict_sample_library_artifact_ids(
    tmp_path: Path,
    artifact_id_suffix: str,
) -> None:
    fixture = _fixture(tmp_path)
    case_id = str(fixture["case_id"])
    exact = (
        fixture["eval_root"]
        / "product-runs"
        / "run-01"
        / "outputs"
        / case_id
    )
    artifact_id = f"{case_id}{artifact_id_suffix}"
    if artifact_id_suffix:
        exact.rename(exact.with_name(artifact_id))

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(case_id,),
        reference_languages=("es-419",),
    )

    assert value["counts"][PRODUCTION_STATUS] == 1
    production = next(
        row for row in value["inventory"] if row["status"] == PRODUCTION_STATUS
    )
    assert f"/outputs/{artifact_id}/" in production["artifacts"][
        "benchmarkDocument"
    ]["path"]
    assert value["cases"][0]["caseId"] == case_id
    assert value["cases"][0]["language"] == "en-US"


@pytest.mark.parametrize(
    "collision_suffix",
    ("-manual-copy", "-hybrid-run2-extra", "x-manual", "-run2"),
)
def test_scan_rejects_artifact_id_prefix_collisions(
    tmp_path: Path,
    collision_suffix: str,
) -> None:
    fixture = _fixture(tmp_path)
    case_id = str(fixture["case_id"])
    exact = (
        fixture["eval_root"]
        / "product-runs"
        / "run-01"
        / "outputs"
        / case_id
    )
    exact.rename(exact.with_name(f"{case_id}{collision_suffix}"))

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(case_id,),
        reference_languages=("es-419",),
    )

    assert value["cases"] == []
    assert value["counts"][PRODUCTION_STATUS] == 0
    assert value["counts"][UNAVAILABLE_STATUS] == 1


def test_latest_complete_artifact_wins_within_one_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    case_id = str(fixture["case_id"])
    _production_output(
        eval_root=fixture["eval_root"],
        run_id="run-01",
        case_id=case_id,
        artifact_id=f"{case_id}-manual",
        source_sha256=str(fixture["production_sha"]),
        text="newer manual text",
        generated_at="2026-08-09T00:03:00Z",
    )

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=fixture["reference_path"],
        output_path=fixture["output"],
        production_case_ids=(case_id,),
        reference_languages=("es-419",),
    )

    assert value["cases"][0]["baseline"]["segments"][0]["finalText"] == (
        "newer manual text"
    )
    production = next(
        row for row in value["inventory"] if row["status"] == PRODUCTION_STATUS
    )
    assert "/outputs/fixture_en_production-manual/" in production[
        "artifacts"
    ]["benchmarkDocument"]["path"]


def test_reference_inventory_skips_case_selected_as_production(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    reference_path = fixture["reference_path"]
    assert isinstance(reference_path, Path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["cases"].append(
        {
            "id": fixture["case_id"],
            "language": "en-US",
            "evaluationSplit": "development",
            "tuningEligible": True,
            "path": f"audio/{fixture['case_id']}.wav",
            "sha256": fixture["production_sha"],
            "scoringTranscript": "PRODUCTION_REFERENCE_SECRET",
            "expectedTranscript": "PRODUCTION_REFERENCE_SECRET",
            "nativeTranscript": "Production reference secret.",
            "rawTranscript": "Production reference secret.",
        }
    )
    reference.pop("canonicalSha256")
    reference["canonicalSha256"] = canonical_json_sha256(reference)
    _write_json(reference_path, reference)

    value = build_real_multilingual_semantic_development_freeze(
        project_root=fixture["project_root"],
        eval_root=fixture["eval_root"],
        global_manifest_path=fixture["global_manifest"],
        fleurs_reference_path=reference_path,
        output_path=fixture["output"],
        production_case_ids=(str(fixture["case_id"]),),
        reference_languages=("en-US",),
    )

    assert [row["caseId"] for row in value["inventory"]] == [
        fixture["case_id"]
    ]
    assert value["inventory"][0]["status"] == PRODUCTION_STATUS
    assert value["counts"][REFERENCE_ONLY_STATUS] == 0


def test_cli_allows_explicit_empty_reference_inventory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["output"]
    assert isinstance(output, Path)

    assert main(
        [
            "--project-root",
            str(fixture["project_root"]),
            "--eval-root",
            str(fixture["eval_root"]),
            "--global-manifest",
            str(fixture["global_manifest"]),
            "--fleurs-reference",
            str(fixture["reference_path"]),
            "--production-case",
            str(fixture["case_id"]),
            "--no-reference-inventory",
            "--expected-production-count",
            "1",
            "--expected-reference-count",
            "0",
            "--output",
            str(output),
        ]
    ) == 0

    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["selection"]["referenceLanguages"] == []
    assert value["counts"][REFERENCE_ONLY_STATUS] == 0
    assert value["counts"]["inventoryRows"] == 1


def test_cli_accepts_strict_decision_set_replay_inputs(tmp_path: Path) -> None:
    fixture = _replay_fixture(tmp_path)
    output = fixture["output"]
    assert isinstance(output, Path)

    assert main(
        [
            "--project-root",
            str(fixture["project_root"]),
            "--eval-root",
            str(fixture["eval_root"]),
            "--global-manifest",
            str(fixture["global_manifest"]),
            "--fleurs-reference",
            str(fixture["reference_path"]),
            "--production-case",
            str(fixture["case_id"]),
            "--codex-decision-set",
            str(fixture["decision_set"]),
            "--pre-review-run-root",
            str(fixture["pre_review_run_root"]),
            "--no-reference-inventory",
            "--expected-production-count",
            "1",
            "--expected-reference-count",
            "0",
            "--output",
            str(output),
        ]
    ) == 0

    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["counts"][PRODUCTION_STATUS] == 1
    assert value["inventory"][0]["selection"]["evidenceKind"] == (
        "in-memory-review-replay"
    )


def test_cli_rejects_empty_reference_flag_with_reference_language(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(
            [
                "--project-root",
                str(fixture["project_root"]),
                "--eval-root",
                str(fixture["eval_root"]),
                "--global-manifest",
                str(fixture["global_manifest"]),
                "--fleurs-reference",
                str(fixture["reference_path"]),
                "--production-case",
                str(fixture["case_id"]),
                "--reference-language",
                "es-419",
                "--no-reference-inventory",
                "--output",
                str(fixture["output"]),
            ]
        )
