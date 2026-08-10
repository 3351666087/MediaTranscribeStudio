from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.build_semantic_promotion_decision import (
    SemanticPromotionDecisionBuildError,
    build_semantic_promotion_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = (
    "ollama-qwen3.5-9b",
    "ollama-qwen3.5-27b-q4-k-m",
    "ollama-qwen3.5-35b-a3b-q4-k-m",
    "ollama-qwen3.6-27b-q4-k-m",
    "ollama-qwen3.6-35b-a3b-q4-k-m",
    "ollama-glm-4.7-flash-q4-k-m",
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _canonical(value: dict[str, object]) -> dict[str, object]:
    body = copy.deepcopy(value)
    body["canonicalSha256"] = canonical_json_sha256(body)
    return body


def _registry_bindings(*, candidate_count: int = 6) -> list[dict[str, str]]:
    model_ids = MODEL_IDS[:candidate_count]
    registry = json.loads(
        (ROOT / "local-model-registry.json").read_text(encoding="utf-8")
    )
    selected = {
        str(model["id"]): model
        for model in registry["models"]
        if model.get("id") in model_ids
    }
    assert set(selected) == set(model_ids)
    bindings = []
    for model_id in model_ids:
        source = selected[model_id]["source"]
        bindings.append(
            {
                "registryModelId": model_id,
                "model": (
                    f"{str(source['repository']).removeprefix('library/')}:"
                    f"{source['tag']}"
                ),
                "digest": str(source["digest"]),
            }
        )
    return bindings


def _fixture(
    tmp_path: Path,
    *,
    candidate_count: int = 6,
    runtime_model_refs: bool = False,
) -> dict[str, object]:
    bindings = _registry_bindings(candidate_count=candidate_count)
    artifact_bindings = [
        {
            **binding,
            "registryModelId": (
                binding["model"]
                if runtime_model_refs
                else binding["registryModelId"]
            ),
        }
        for binding in bindings
    ]
    cases = []
    for case_index in range(1, 23):
        candidates = []
        for candidate_index, binding in enumerate(bindings, start=1):
            candidates.append(
                {
                    "candidateAlias": f"candidate-{candidate_index:02d}",
                    "model": binding["model"],
                    "modelId": (
                        binding["model"]
                        if runtime_model_refs
                        else binding["registryModelId"]
                    ),
                    "digest": binding["digest"],
                    "resultCanonicalSha256": (
                        f"{case_index:02x}{candidate_index:02x}" + "0" * 60
                    ),
                }
            )
        cases.append(
            {
                "caseAlias": f"case-{case_index:03d}",
                "caseId": f"fixture-case-{case_index:03d}",
                "caseSha256": f"{case_index:02x}" + "1" * 62,
                "candidates": candidates,
            }
        )
    vault = _canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "anonymous-semantic-blind-review-identity-vault",
            "seedSha256": "2" * 64,
            "inputSetCanonicalSha256": "3" * 64,
            "modelSetCanonicalSha256": "4" * 64,
            "cases": cases,
        }
    )
    vault_path = tmp_path / "blind" / "identity-vault.json"
    _write(vault_path, vault)

    manifest = _canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "anonymous-semantic-blind-review-manifest",
            "publicationPolicy": "atomic-directory-no-replace",
            "manifestWrittenLast": True,
            "packetCount": 22,
            "candidatesPerPacket": candidate_count,
            "files": [
                {
                    "relativePath": "identity-vault.json",
                    "role": "identity-vault",
                    "sizeBytes": vault_path.stat().st_size,
                    "fileSha256": sha256_file(vault_path),
                    "canonicalSha256": vault["canonicalSha256"],
                }
            ],
        }
    )
    manifest_path = tmp_path / "blind" / "manifest.json"
    _write(manifest_path, manifest)

    review_cases = []
    for case in cases:
        review_cases.append(
            {
                "caseAlias": case["caseAlias"],
                "preferredCandidateAliases": ["candidate-02"],
                "tie": False,
                "assessmentGroups": [
                    {
                        "candidateAliases": ["candidate-02"],
                        "severity": "pass",
                        "reason": "The candidate is faithful.",
                        "evidence": {
                            "startMs": 0,
                            "endMs": 1000,
                            "observation": "The visible output is coherent.",
                        },
                        "recommendedAction": "Use this candidate.",
                    },
                    {
                        "candidateAliases": ["candidate-04"],
                        "severity": "minor",
                        "reason": "The candidate is slightly over-conservative.",
                        "evidence": {
                            "startMs": 0,
                            "endMs": 1000,
                            "observation": "One choice is unnecessary.",
                        },
                        "recommendedAction": "Prefer the pass candidate.",
                    },
                    {
                        "candidateAliases": [
                            f"candidate-{index:02d}"
                            for index in range(1, candidate_count + 1)
                            if index not in {2, 4}
                        ],
                        "severity": "major",
                        "reason": "The candidate changes a material decision.",
                        "evidence": {
                            "startMs": 0,
                            "endMs": 1000,
                            "observation": "The visible decision is not faithful.",
                        },
                        "recommendedAction": "Do not select this group.",
                    },
                ],
            }
        )
    review = _canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "codex-semantic-blind-review",
            "reviewId": f"fixture-{candidate_count}-model-blind-review-r1",
            "blindPackage": {
                "manifestCanonicalSha256": manifest["canonicalSha256"],
                "manifestFileSha256": sha256_file(manifest_path),
                "reviewerPacketSetSha256": "5" * 64,
                "packetCount": 22,
                "candidatesPerPacket": candidate_count,
            },
            "bodySourceFileSha256": "6" * 64,
            "reviewPolicy": {
                "allowedSeverities": ["blocker", "major", "minor", "pass"],
                "automaticScoringUsed": False,
                "identityVaultReadBeforeSealing": False,
                "preferenceRule": "Prefer the faithful candidate.",
                "referenceTranscriptUsed": False,
            },
            "reviewer": {
                "actor": "fixture-codex-reviewer",
                "source": "codex-agent",
                "reviewedAt": "2026-08-09T20:18:57+08:00",
            },
            "validation": {
                "allCandidatesAssessedExactlyOnce": True,
                "assessedCandidateCount": 22 * candidate_count,
                "identityVaultRead": False,
                "reviewedCaseCount": 22,
                "sealedBeforeUnblind": True,
            },
            "cases": review_cases,
        }
    )
    review_path = tmp_path / "review" / "review.v1.json"
    _write(review_path, review)

    aggregate = []
    severity_by_candidate = tuple(
        {"blocker": 0, "major": 0, "minor": 0, "pass": 22}
        if index == 2
        else {"blocker": 0, "major": 0, "minor": 22, "pass": 0}
        if index == 4
        else {"blocker": 0, "major": 22, "minor": 0, "pass": 0}
        for index in range(1, candidate_count + 1)
    )
    ranks = tuple(
        1 if index == 2 else 2 if index == 4 else 3
        for index in range(1, candidate_count + 1)
    )
    for index, (binding, severity, rank) in enumerate(
        zip(artifact_bindings, severity_by_candidate, ranks, strict=True),
        start=1,
    ):
        preferred = index == 2
        aggregate.append(
            {
                **binding,
                "caseCount": 22,
                "severityCounts": severity,
                "preferredCaseCredit": {
                    "numerator": 22 if preferred else 0,
                    "denominator": 1,
                },
                "weightedPreferredShare": 1.0 if preferred else 0.0,
                "rank": rank,
            }
        )
    comparison = _canonical(
        {
            "schemaVersion": "1.0.0",
            "artifactType": "semantic-model-blind-review-comparison",
            "comparisonId": f"fixture-{candidate_count}-model-blind-review-r1.comparison",
            "reviewId": review["reviewId"],
            "blindBatchId": "semantic-blind-fixture-r1",
            "automaticScoringUsed": False,
            "candidateIdentitiesHiddenDuringReview": True,
            "sealedBeforeUnblind": True,
            "caseCount": 22,
            "candidateCount": candidate_count,
            "caseSetSha256": vault["inputSetCanonicalSha256"],
            "modelSetSha256": vault["modelSetCanonicalSha256"],
            "models": artifact_bindings,
            "rankingPolicy": {
                "automaticMetricsUsed": False,
                "order": [
                    "blocker-count-ascending",
                    "major-count-ascending",
                    "minor-count-ascending",
                    "pass-count-descending",
                    "weighted-preferred-share-descending",
                ],
                "tiePreferenceWeight": (
                    "one case credit divided equally among preferred candidates"
                ),
            },
            "aggregate": aggregate,
            "outcome": "unique-winner",
            "tiedRegistryModelIds": [
                artifact_bindings[1]["registryModelId"]
            ],
            "uniqueWinnerRegistryModelId": artifact_bindings[1][
                "registryModelId"
            ],
            "evidence": {
                "identityVaultCanonicalSha256": vault["canonicalSha256"],
                "identityVaultFileSha256": sha256_file(vault_path),
                "manifestCanonicalSha256": manifest["canonicalSha256"],
                "manifestFileSha256": sha256_file(manifest_path),
                "reviewerPacketSetSha256": "5" * 64,
                "sealedReviewCanonicalSha256": review["canonicalSha256"],
                "sealedReviewFileSha256": sha256_file(review_path),
                "sealedReviewPath": str(review_path),
            },
        }
    )
    comparison_path = tmp_path / "comparison" / "comparison.v1.json"
    _write(comparison_path, comparison)
    return {
        "manifest": manifest_path,
        "vault": vault_path,
        "review": review_path,
        "comparison": comparison_path,
        "comparison_value": comparison,
    }


def _build(tmp_path: Path, fixture: dict[str, object]) -> dict[str, object]:
    return build_semantic_promotion_decision(
        config_path=ROOT / "production.config.example.json",
        registry_path=ROOT / "local-model-registry.json",
        comparison_path=fixture["comparison"],
        blind_manifest_path=fixture["manifest"],
        identity_vault_path=fixture["vault"],
        sealed_review_path=fixture["review"],
        output_path=tmp_path / "decision.v1.json",
        expected_sealed_review_sha256=sha256_file(fixture["review"]),
        expected_comparison_canonical_sha256=fixture["comparison_value"][
            "canonicalSha256"
        ],
    )


def test_builds_human_only_decision_bound_to_real_comparison_file(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "evidence")
    config_path = ROOT / "production.config.example.json"
    before = config_path.read_bytes()

    result = _build(tmp_path, fixture)

    decision = result["decision"]
    assert decision["challengerRegistryModelId"] == MODEL_IDS[1]
    assert decision["incumbentRegistryModelId"] == MODEL_IDS[0]
    assert decision["automaticScoring"] is False
    assert decision["expectedConfigSha256"] == sha256_file(config_path)
    assert decision["evidence"]["comparisonArtifactSha256"] == sha256_file(
        fixture["comparison"]
    )
    assert decision["blindReview"]["caseSetSha256"] == "3" * 64
    assert config_path.read_bytes() == before


def test_builds_decision_from_four_candidate_comparison(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "evidence", candidate_count=4)

    result = _build(tmp_path, fixture)

    assert result["winnerRegistryModelId"] == MODEL_IDS[1]
    assert result["humanReviewCounts"] == {
        "reviewedCaseCount": 22,
        "candidateCount": 4,
    }
    assert len(result["humanReviewSummary"]) == 4


def test_normalizes_exact_runtime_model_refs_to_registry_ids(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path / "evidence",
        candidate_count=4,
        runtime_model_refs=True,
    )

    result = _build(tmp_path, fixture)

    assert result["winnerRegistryModelId"] == MODEL_IDS[1]
    assert result["decision"]["challengerRegistryModelId"] == MODEL_IDS[1]


def test_review_evidence_must_be_the_sealed_object_contract(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "evidence")
    review_path = fixture["review"]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["cases"][0]["assessmentGroups"][0]["evidence"] = ["invalid"]
    review.pop("canonicalSha256")
    review = _canonical(review)
    _write(review_path, review)

    with pytest.raises(
        SemanticPromotionDecisionBuildError,
        match="evidence must be an object",
    ):
        build_semantic_promotion_decision(
            config_path=ROOT / "production.config.example.json",
            registry_path=ROOT / "local-model-registry.json",
            comparison_path=fixture["comparison"],
            blind_manifest_path=fixture["manifest"],
            identity_vault_path=fixture["vault"],
            sealed_review_path=review_path,
            output_path=tmp_path / "decision.v1.json",
            expected_sealed_review_sha256=sha256_file(review_path),
            expected_comparison_canonical_sha256=fixture["comparison_value"][
                "canonicalSha256"
            ],
        )


def test_comparison_cannot_override_the_sealed_human_winner(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "evidence")
    comparison_path = fixture["comparison"]
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["uniqueWinnerRegistryModelId"] = MODEL_IDS[3]
    comparison["tiedRegistryModelIds"] = [MODEL_IDS[3]]
    comparison.pop("canonicalSha256")
    comparison = _canonical(comparison)
    _write(comparison_path, comparison)

    with pytest.raises(
        SemanticPromotionDecisionBuildError,
        match="winner differs from the sealed human review",
    ):
        build_semantic_promotion_decision(
            config_path=ROOT / "production.config.example.json",
            registry_path=ROOT / "local-model-registry.json",
            comparison_path=comparison_path,
            blind_manifest_path=fixture["manifest"],
            identity_vault_path=fixture["vault"],
            sealed_review_path=fixture["review"],
            output_path=tmp_path / "decision.v1.json",
            expected_sealed_review_sha256=sha256_file(fixture["review"]),
            expected_comparison_canonical_sha256=comparison["canonicalSha256"],
        )


def test_authorized_comparison_canonical_sha_is_required(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "evidence")

    with pytest.raises(
        SemanticPromotionDecisionBuildError,
        match="not the authorized digest",
    ):
        build_semantic_promotion_decision(
            config_path=ROOT / "production.config.example.json",
            registry_path=ROOT / "local-model-registry.json",
            comparison_path=fixture["comparison"],
            blind_manifest_path=fixture["manifest"],
            identity_vault_path=fixture["vault"],
            sealed_review_path=fixture["review"],
            output_path=tmp_path / "decision.v1.json",
            expected_sealed_review_sha256=sha256_file(fixture["review"]),
            expected_comparison_canonical_sha256="f" * 64,
        )
