from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256
from tools.build_semantic_defect_audit import (
    SemanticDefectAuditError,
    build_semantic_defect_audit,
)


def _canonical(value: dict[str, object]) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop("canonicalSha256", None)
    body["canonicalSha256"] = canonical_json_sha256(body)
    return body


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    models = [
        {
            "alias": "candidate-a",
            "model": "qwen3.5:27b-q4_K_M",
            "registry": "ollama-qwen3.5-27b-q4-k-m",
            "digest": "sha256:" + "a" * 64,
        },
        {
            "alias": "candidate-b",
            "model": "qwen3.5:35b-a3b-q4_K_M",
            "registry": "ollama-qwen3.5-35b-a3b-q4-k-m",
            "digest": "sha256:" + "b" * 64,
        },
        {
            "alias": "candidate-c",
            "model": "qwen3.6:27b-q4_K_M",
            "registry": "ollama-qwen3.6-27b-q4-k-m",
            "digest": "sha256:" + "c" * 64,
        },
        {
            "alias": "candidate-d",
            "model": "qwen3.6:35b-a3b-q4_K_M",
            "registry": "ollama-qwen3.6-35b-a3b-q4-k-m",
            "digest": "sha256:" + "d" * 64,
        },
    ]
    vault = _canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "anonymous-semantic-blind-review-identity-vault",
            "seedSha256": "1" * 64,
            "inputSetCanonicalSha256": "2" * 64,
            "modelSetCanonicalSha256": "3" * 64,
            "cases": [
                {
                    "caseAlias": "case-alpha",
                    "caseId": "fixture-case-alpha",
                    "candidates": [
                        {
                            "candidateAlias": model["alias"],
                            "model": model["model"],
                            "modelId": model["registry"],
                            "digest": model["digest"],
                            "resultCanonicalSha256": "e" * 64,
                        }
                        for model in models
                    ],
                }
            ],
        }
    )
    comparison = _canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "semantic-model-blind-review-comparison",
            "comparisonId": "fixture-comparison-r1",
            "blindBatchId": "fixture-batch-r1",
            "uniqueWinnerRegistryModelId": models[0]["registry"],
            "models": [
                {
                    "model": model["model"],
                    "registryModelId": model["registry"],
                    "digest": model["digest"],
                }
                for model in models
            ],
        }
    )
    review = _canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "codex-semantic-blind-review",
            "reviewId": "fixture-review-r1",
            "cases": [
                {
                    "caseAlias": "case-alpha",
                    "assessmentGroups": [
                        {
                            "candidateAliases": ["candidate-a"],
                            "severity": "major",
                            "reason": "Speaker attribution splits one continuous turn.",
                            "evidence": {"observation": "visible split"},
                            "recommendedAction": "Resolve speaker continuity before publication.",
                        },
                        {
                            "candidateAliases": ["candidate-b"],
                            "severity": "blocker",
                            "reason": "Malformed lexical token requires ASR alternatives.",
                            "evidence": {"observation": "malformed token"},
                            "recommendedAction": "Request provider-native ASR alternatives.",
                        },
                        {
                            "candidateAliases": ["candidate-c"],
                            "severity": "major",
                            "reason": "Language and script conflict with a lexical defect.",
                            "evidence": {"observation": "script conflict"},
                            "recommendedAction": "Escalate language and ASR scope atomically.",
                        },
                        {
                            "candidateAliases": ["candidate-d"],
                            "severity": "major",
                            "reason": "The business scope is left unresolved.",
                            "evidence": {"observation": "scope omission"},
                            "recommendedAction": "Revisit the material semantic scope.",
                        },
                        {
                            "candidateAliases": ["candidate-a"],
                            "severity": "pass",
                            "reason": "No material defect in this observation.",
                            "evidence": {"observation": "pass"},
                            "recommendedAction": "Preserve the candidate.",
                        },
                    ],
                }
            ],
        }
    )
    benchmark = _canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "real-multilingual-semantic-benchmark",
            "benchmarkId": "fixture-benchmark-r1",
            "cases": [],
        }
    )
    sidecar = {
        "artifactType": "real-multilingual-semantic-model-result",
        "benchmarkArtifactSha256": benchmark["canonicalSha256"],
        "model": {
            "modelIdentity": {
                "modelId": models[0]["model"],
                "expectedDigest": models[0]["digest"],
            },
            "aggregate": {
                "calibration": {
                    "groupCount": 4,
                    "correctCount": 3,
                    "incorrectCount": 1,
                    "microAccuracy": 0.75,
                    "expectedActionCounts": {"select": 2},
                    "actualActionCounts": {"select": 1},
                }
            },
        },
    }

    paths = {
        "review": tmp_path / "review.json",
        "vault": tmp_path / "identity-vault.json",
        "comparison": tmp_path / "comparison.json",
        "benchmark": tmp_path / "benchmark.json",
        "sidecar": tmp_path / "model.json",
        "output": tmp_path / "audit.json",
    }
    _write(paths["review"], review)
    _write(paths["vault"], vault)
    _write(paths["comparison"], comparison)
    _write(paths["benchmark"], benchmark)
    _write(paths["sidecar"], sidecar)
    return paths


def test_audit_unblinds_and_classifies_major_blocker_groups(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = build_semantic_defect_audit(
        sealed_review_path=paths["review"],
        identity_vault_path=paths["vault"],
        comparison_path=paths["comparison"],
        output_path=paths["output"],
    )

    assert result["summary"] == {
        "findingCount": 4,
        "severityCounts": {"blocker": 1, "major": 3},
        "categoryCounts": {
            "asr-lexical": 1,
            "language-asr-scope": 1,
            "semantic-scope": 1,
            "speaker-continuity": 1,
        },
        "candidateModelCounts": {
            "qwen3.5:27b-q4_K_M": 1,
            "qwen3.5:35b-a3b-q4_K_M": 1,
            "qwen3.6:27b-q4_K_M": 1,
            "qwen3.6:35b-a3b-q4_K_M": 1,
        },
        "winnerFindingCount": 1,
        "winnerMajorCount": 1,
        "winnerBlockerCount": 0,
    }
    persisted = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert persisted["canonicalSha256"] == result["canonicalSha256"]
    assert persisted["findings"][0]["candidates"][0]["comparisonRegistryModelId"]


def test_audit_reads_result_sidecar_calibration(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = build_semantic_defect_audit(
        sealed_review_path=paths["review"],
        identity_vault_path=paths["vault"],
        comparison_path=paths["comparison"],
        benchmark_path=paths["benchmark"],
        model_report_paths=[paths["sidecar"]],
        output_path=paths["output"],
    )

    persisted = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert persisted["calibrationSidecars"] == [
        {
            "path": str(paths["sidecar"].resolve()),
            "fileSha256": persisted["calibrationSidecars"][0]["fileSha256"],
            "model": "qwen3.5:27b-q4_K_M",
            "digest": "sha256:" + "a" * 64,
            "calibration": {
                "groupCount": 4,
                "correctCount": 3,
                "incorrectCount": 1,
                "microAccuracy": 0.75,
                "expectedActionCounts": {"select": 2},
                "actualActionCounts": {"select": 1},
            },
        }
    ]


def test_audit_rejects_tampered_canonical_source(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    review = json.loads(paths["review"].read_text(encoding="utf-8"))
    review["cases"][0]["assessmentGroups"][0]["reason"] = "tampered"
    _write(paths["review"], review)

    with pytest.raises(SemanticDefectAuditError, match="canonical SHA-256"):
        build_semantic_defect_audit(
            sealed_review_path=paths["review"],
            identity_vault_path=paths["vault"],
            comparison_path=paths["comparison"],
            output_path=paths["output"],
        )


def test_audit_rejects_sidecar_not_bound_to_comparison(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sidecar = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    sidecar["model"]["modelIdentity"]["expectedDigest"] = "sha256:" + "f" * 64
    _write(paths["sidecar"], sidecar)

    with pytest.raises(SemanticDefectAuditError, match="not bound to comparison"):
        build_semantic_defect_audit(
            sealed_review_path=paths["review"],
            identity_vault_path=paths["vault"],
            comparison_path=paths["comparison"],
            benchmark_path=paths["benchmark"],
            model_report_paths=[paths["sidecar"]],
            output_path=paths["output"],
        )
